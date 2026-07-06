package webgateway

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

const operatorBreakInDir = "runtime/audio/operator-breakin"
const operatorBreakInMaxChunkBytes = 512 << 10
const operatorBreakInMaxUploadBytes = 8 << 20
const operatorBreakInMaxSessions = 8

type OperatorBreakInManager struct {
	mu       sync.Mutex
	sessions map[string]*operatorBreakInSession
}

type operatorBreakInSession struct {
	ID          string
	AlertID     string
	FeedIDs     []string
	Title       string
	SampleRate  int
	Channels    int
	PrerollPath string
	StreamURL   string
	Publisher   *events.HostBridgePublisher
	Cancel      context.CancelFunc
	Bytes       int64
	Chunks      int
	StartedAt   time.Time
}

func NewOperatorBreakInManager() *OperatorBreakInManager {
	return &OperatorBreakInManager{sessions: map[string]*operatorBreakInSession{}}
}

func (s *wsSession) listOperatorBreakInPrerolls() (map[string]any, error) {
	files := []map[string]any{}
	for _, root := range []string{"audio", filepath.ToSlash(filepath.Join(operatorBreakInDir, "prerolls")), filepath.ToSlash(filepath.Join(operatorBreakInDir, "uploads"))} {
		dir := resolveConfigPath(s.configPath, root)
		entries, err := os.ReadDir(dir)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		for _, entry := range entries {
			if entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".wav") {
				continue
			}
			path := filepath.Join(dir, entry.Name())
			info, err := os.Stat(path)
			if err != nil {
				continue
			}
			rel := filepath.ToSlash(filepath.Join(root, entry.Name()))
			files = append(files, map[string]any{
				"path":     rel,
				"name":     entry.Name(),
				"size":     info.Size(),
				"modified": info.ModTime().UTC().Format(time.RFC3339Nano),
			})
		}
	}
	sort.Slice(files, func(i, j int) bool {
		return strings.ToLower(fmt.Sprint(files[i]["path"])) < strings.ToLower(fmt.Sprint(files[j]["path"]))
	})
	return map[string]any{"files": files}, nil
}

func (s *wsSession) uploadOperatorBreakInPreroll(payload map[string]any) (map[string]any, error) {
	name := safeID(firstNonBlank(stringValue(payload, "name"), fmt.Sprintf("upload-%d", time.Now().UnixNano())))
	if name == "" {
		name = fmt.Sprintf("upload-%d", time.Now().UnixNano())
	}
	if !strings.HasSuffix(strings.ToLower(name), ".wav") {
		name += ".wav"
	}
	data, err := decodeBase64Payload(stringValue(payload, "data"), operatorBreakInMaxUploadBytes)
	if err != nil {
		return nil, err
	}
	if _, _, err := wavPCM16Info(data); err != nil {
		return nil, fmt.Errorf("uploaded preroll must be PCM16 WAV: %w", err)
	}
	rel := filepath.ToSlash(filepath.Join(operatorBreakInDir, "uploads", name))
	path := resolveConfigPath(s.configPath, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := writeFileAtomic(path, data, 0o600); err != nil {
		return nil, err
	}
	return map[string]any{"path": rel, "name": name, "size": len(data)}, nil
}

func (s *wsSession) generateOperatorBreakInTone(payload map[string]any) (map[string]any, error) {
	frequency := intPayload(payload, "frequency_hz", 1050)
	durationMS := intPayload(payload, "duration_ms", 850)
	if frequency < 200 || frequency > 3000 {
		return nil, fmt.Errorf("tone frequency must be between 200 and 3000 Hz")
	}
	if durationMS < 100 || durationMS > 5000 {
		return nil, fmt.Errorf("tone duration must be between 100 and 5000 ms")
	}
	sampleRate := 48000
	channels := 1
	pcm := tonePCM16(frequency, durationMS, sampleRate, channels)
	wav := wavFromPCM16(pcm, sampleRate, channels)
	name := safeID(fmt.Sprintf("tone-%dhz-%dms", frequency, durationMS)) + ".wav"
	rel := filepath.ToSlash(filepath.Join(operatorBreakInDir, "prerolls", name))
	path := resolveConfigPath(s.configPath, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := writeFileAtomic(path, wav, 0o600); err != nil {
		return nil, err
	}
	return map[string]any{"path": rel, "name": name, "duration_ms": durationMS, "frequency_hz": frequency}, nil
}

func (s *wsSession) startOperatorBreakIn(payload map[string]any) (map[string]any, error) {
	targets, err := targetFeedIDs(s.configPath, payload)
	if err != nil {
		return nil, err
	}
	sampleRate := intPayload(payload, "sample_rate", 48000)
	channels := intPayload(payload, "channels", 1)
	if sampleRate < 8000 || sampleRate > 192000 {
		return nil, fmt.Errorf("sample_rate is outside the supported range")
	}
	if channels < 1 || channels > 2 {
		return nil, fmt.Errorf("channels must be 1 or 2")
	}
	prerollPath := strings.TrimSpace(stringValue(payload, "preroll_path"))
	if prerollPath != "" {
		if _, err := operatorBreakInWAVFromRelPath(s.configPath, prerollPath); err != nil {
			return nil, err
		}
	}
	result, err := s.server.breakIn.start(s.configPath, targets, strings.TrimSpace(stringValue(payload, "title")), sampleRate, channels, prerollPath)
	if err == nil {
		if id, _ := result["session_id"].(string); id != "" {
			s.trackOperatorBreakInSession(id)
		}
	}
	return result, err
}

func (s *wsSession) appendOperatorBreakInChunk(payload map[string]any) (map[string]any, error) {
	id := strings.TrimSpace(stringValue(payload, "session_id"))
	data, err := decodeBase64Payload(stringValue(payload, "data"), operatorBreakInMaxChunkBytes)
	if err != nil {
		return nil, err
	}
	return s.server.breakIn.appendChunk(id, data)
}

func (s *wsSession) finishOperatorBreakIn(payload map[string]any) (map[string]any, error) {
	id := strings.TrimSpace(stringValue(payload, "session_id"))
	defer s.untrackOperatorBreakInSession(id)
	result, err := s.server.breakIn.finish(id)
	if err != nil {
		return nil, err
	}
	return result, nil
}

func (s *wsSession) queueOperatorBreakInURL(payload map[string]any) (map[string]any, error) {
	targets, err := targetFeedIDs(s.configPath, payload)
	if err != nil {
		return nil, err
	}
	streamURL := strings.TrimSpace(firstNonBlank(stringValue(payload, "audio_url"), stringValue(payload, "stream_url"), stringValue(payload, "url")))
	if streamURL == "" {
		return nil, fmt.Errorf("media stream URL is required")
	}
	if err := validateOperatorBreakInStreamURL(streamURL); err != nil {
		return nil, err
	}
	title := fallbackText(strings.TrimSpace(stringValue(payload, "title")), "Operator Break-in Stream")
	ctx, cancel := context.WithCancel(context.Background())
	result, err := s.server.breakIn.startStream(s.configPath, targets, title, streamURL, cancel)
	if err != nil {
		cancel()
		return nil, err
	}
	id, _ := result["session_id"].(string)
	if id != "" {
		s.trackOperatorBreakInSession(id)
		go s.server.breakIn.streamURL(ctx, id, streamURL)
	}
	return result, nil
}

func (s *wsSession) cancelOperatorBreakIn(payload map[string]any) (map[string]any, error) {
	id := strings.TrimSpace(stringValue(payload, "session_id"))
	defer s.untrackOperatorBreakInSession(id)
	if err := s.server.breakIn.cancel(id); err != nil {
		return nil, err
	}
	return map[string]any{"cancelled": true, "session_id": id}, nil
}

func (m *OperatorBreakInManager) start(configPath string, feedIDs []string, title string, sampleRate int, channels int, prerollPath string) (map[string]any, error) {
	return m.startSession(configPath, feedIDs, title, sampleRate, channels, prerollPath, "", nil)
}

func (m *OperatorBreakInManager) startStream(configPath string, feedIDs []string, title string, streamURL string, cancel context.CancelFunc) (map[string]any, error) {
	return m.startSession(configPath, feedIDs, title, 48000, 1, "", streamURL, cancel)
}

func (m *OperatorBreakInManager) startSession(configPath string, feedIDs []string, title string, sampleRate int, channels int, prerollPath string, streamURL string, cancel context.CancelFunc) (map[string]any, error) {
	m.reapStale()
	m.mu.Lock()
	defer m.mu.Unlock()
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil, fmt.Errorf("event bridge is not available")
	}
	if len(m.sessions) >= operatorBreakInMaxSessions {
		return nil, fmt.Errorf("too many active break-in sessions")
	}
	id := safeID(fmt.Sprintf("breakin-%d", time.Now().UTC().UnixNano()))
	if id == "" {
		return nil, fmt.Errorf("unable to create break-in session id")
	}
	alertID := safeID("operator_breakin_" + id)
	publisher := events.NewHostBridgePublisher(bridgeAddr)
	session := &operatorBreakInSession{
		ID:          id,
		AlertID:     alertID,
		FeedIDs:     append([]string(nil), feedIDs...),
		Title:       fallbackText(title, "Operator Break-in"),
		SampleRate:  sampleRate,
		Channels:    channels,
		PrerollPath: prerollPath,
		StreamURL:   streamURL,
		Publisher:   publisher,
		Cancel:      cancel,
		StartedAt:   time.Now().UTC(),
	}
	if err := publishOperatorBreakInEvent(session, "operator.breakin.start", nil); err != nil {
		_ = publisher.Close()
		return nil, err
	}
	m.sessions[id] = &operatorBreakInSession{
		ID:          session.ID,
		AlertID:     session.AlertID,
		FeedIDs:     session.FeedIDs,
		Title:       session.Title,
		SampleRate:  session.SampleRate,
		Channels:    session.Channels,
		PrerollPath: session.PrerollPath,
		StreamURL:   session.StreamURL,
		Publisher:   session.Publisher,
		Cancel:      session.Cancel,
		StartedAt:   session.StartedAt,
	}
	if prerollPath != "" {
		preroll, err := operatorBreakInWAVFromRelPath(configPath, prerollPath)
		if err != nil {
			delete(m.sessions, id)
			_ = publisher.Close()
			return nil, err
		}
		if err := publishOperatorBreakInPCM(session, preroll.PCM, preroll.SampleRate, preroll.Channels); err != nil {
			delete(m.sessions, id)
			_ = publisher.Close()
			return nil, err
		}
		session.Bytes += int64(len(preroll.PCM))
		session.Chunks++
		m.sessions[id].Bytes = session.Bytes
		m.sessions[id].Chunks = session.Chunks
	}
	return map[string]any{
		"session_id": id,
		"alert_id":   alertID,
		"feed_ids":   feedIDs,
		"live":       true,
		"stream_url": streamURL,
	}, nil
}

func (m *OperatorBreakInManager) appendChunk(id string, data []byte) (map[string]any, error) {
	m.reapStale()
	if id == "" {
		return nil, fmt.Errorf("session_id is required")
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("audio chunk is empty")
	}
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("audio chunk must contain PCM16 samples")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	session := m.sessions[id]
	if session == nil || session.Publisher == nil {
		return nil, fmt.Errorf("break-in session is not active")
	}
	if err := publishOperatorBreakInPCM(session, data, session.SampleRate, session.Channels); err != nil {
		return nil, err
	}
	session.Bytes += int64(len(data))
	session.Chunks++
	return map[string]any{"session_id": id, "bytes": session.Bytes, "chunks": session.Chunks}, nil
}

func (m *OperatorBreakInManager) finish(id string) (map[string]any, error) {
	m.mu.Lock()
	session := m.sessions[id]
	if session == nil {
		m.mu.Unlock()
		return nil, fmt.Errorf("break-in session is not active")
	}
	delete(m.sessions, id)
	m.mu.Unlock()
	if session.Cancel != nil {
		session.Cancel()
	}
	err := publishOperatorBreakInEvent(session, "operator.breakin.finish", nil)
	_ = session.Publisher.Close()
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"live":       true,
		"finished":   true,
		"session_id": session.ID,
		"alert_id":   session.AlertID,
		"feed_ids":   session.FeedIDs,
		"bytes":      session.Bytes,
		"chunks":     session.Chunks,
	}, nil
}

func (m *OperatorBreakInManager) cancel(id string) error {
	if id == "" {
		return fmt.Errorf("session_id is required")
	}
	m.mu.Lock()
	session := m.sessions[id]
	delete(m.sessions, id)
	m.mu.Unlock()
	if session == nil {
		return fmt.Errorf("break-in session is not active")
	}
	if session.Cancel != nil {
		session.Cancel()
	}
	if session.Publisher != nil {
		_ = publishOperatorBreakInEvent(session, "operator.breakin.cancel", nil)
		_ = session.Publisher.Close()
	}
	return nil
}

func publishOperatorBreakInPCM(session *operatorBreakInSession, pcm []byte, sampleRate int, channels int) error {
	return publishOperatorBreakInEvent(session, "operator.breakin.chunk", map[string]any{
		"data":        base64.StdEncoding.EncodeToString(pcm),
		"sample_rate": sampleRate,
		"channels":    channels,
		"bytes":       len(pcm),
	})
}

func publishOperatorBreakInEvent(session *operatorBreakInSession, eventType string, extra map[string]any) error {
	if session == nil || session.Publisher == nil {
		return fmt.Errorf("break-in session is not active")
	}
	data := map[string]any{
		"session_id":  session.ID,
		"alert_id":    session.AlertID,
		"feed_ids":    session.FeedIDs,
		"title":       session.Title,
		"sample_rate": session.SampleRate,
		"channels":    session.Channels,
		"source":      "operator-breakin",
	}
	for key, value := range extra {
		data[key] = value
	}
	return session.Publisher.Publish(events.Event{
		Type:    eventType,
		Source:  "haze-web",
		Subject: session.ID,
		Data:    data,
	})
}

type wavPCM struct {
	PCM        []byte
	SampleRate int
	Channels   int
}

func operatorBreakInWAVFromRelPath(configPath string, rel string) (wavPCM, error) {
	rel = filepath.ToSlash(strings.TrimSpace(rel))
	if rel == "" || strings.Contains(rel, "..") || filepath.IsAbs(rel) {
		return wavPCM{}, fmt.Errorf("invalid preroll path")
	}
	if !(strings.HasPrefix(rel, "audio/") || strings.HasPrefix(rel, operatorBreakInDir+"/")) {
		return wavPCM{}, fmt.Errorf("preroll path must be under audio or operator break-in runtime audio")
	}
	raw, err := os.ReadFile(resolveConfigPath(configPath, rel))
	if err != nil {
		return wavPCM{}, err
	}
	pcm, info, err := wavPCM16Info(raw)
	if err != nil {
		return wavPCM{}, err
	}
	return wavPCM{PCM: pcm, SampleRate: info.SampleRate, Channels: info.Channels}, nil
}

func combineOperatorBreakInAudio(outputPath string, preroll []byte, prerollRate int, prerollChannels int, voicePath string, voiceRate int, voiceChannels int) error {
	if prerollRate != voiceRate || prerollChannels != voiceChannels {
		return fmt.Errorf("preroll WAV format must match microphone capture (%d Hz, %d channel)", voiceRate, voiceChannels)
	}
	voice, err := os.ReadFile(voicePath)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	tmp := outputPath + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}
	writeErr := func() error {
		if _, err := file.Write(preroll); err != nil {
			return err
		}
		if _, err := file.Write(silencePCM16(prerollRate, prerollChannels, 200*time.Millisecond)); err != nil {
			return err
		}
		if _, err := file.Write(voice); err != nil {
			return err
		}
		return nil
	}()
	closeErr := file.Close()
	if writeErr != nil {
		_ = os.Remove(tmp)
		return writeErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, outputPath)
}

func decodeBase64Payload(value string, maxBytes int) ([]byte, error) {
	value = strings.TrimSpace(value)
	if comma := strings.Index(value, ","); comma >= 0 && strings.Contains(value[:comma], "base64") {
		value = value[comma+1:]
	}
	if value == "" {
		return nil, fmt.Errorf("missing audio data")
	}
	if maxBytes > 0 && base64.StdEncoding.DecodedLen(len(value)) > maxBytes+2 {
		return nil, fmt.Errorf("audio data exceeds %d bytes", maxBytes)
	}
	data, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		return nil, fmt.Errorf("decode audio data: %w", err)
	}
	if len(data) > maxBytes {
		return nil, fmt.Errorf("audio data exceeds %d bytes", maxBytes)
	}
	return data, nil
}

func (s *wsSession) trackOperatorBreakInSession(id string) {
	id = strings.TrimSpace(id)
	if id == "" {
		return
	}
	s.mu.Lock()
	if s.breakIns == nil {
		s.breakIns = map[string]struct{}{}
	}
	s.breakIns[id] = struct{}{}
	s.mu.Unlock()
}

func (s *wsSession) untrackOperatorBreakInSession(id string) {
	id = strings.TrimSpace(id)
	if id == "" {
		return
	}
	s.mu.Lock()
	delete(s.breakIns, id)
	s.mu.Unlock()
}

func (s *wsSession) cancelOwnedOperatorBreakIns() {
	if s == nil || s.server == nil || s.server.breakIn == nil {
		return
	}
	s.mu.Lock()
	ids := make([]string, 0, len(s.breakIns))
	for id := range s.breakIns {
		ids = append(ids, id)
	}
	s.breakIns = map[string]struct{}{}
	s.mu.Unlock()
	for _, id := range ids {
		_ = s.server.breakIn.cancel(id)
	}
}

func (m *OperatorBreakInManager) streamURL(ctx context.Context, id string, streamURL string) {
	err := m.pipeStreamURL(ctx, id, streamURL)
	if ctx.Err() != nil {
		return
	}
	if err != nil {
		_ = m.cancelWithReason(id, "stream ended with error: "+err.Error())
		return
	}
	_, _ = m.finish(id)
}

func (m *OperatorBreakInManager) pipeStreamURL(ctx context.Context, id string, streamURL string) error {
	ffmpeg := strings.TrimSpace(os.Getenv("FFMPEG"))
	if ffmpeg == "" {
		ffmpeg = "ffmpeg"
	}
	cmd := exec.CommandContext(
		ctx,
		ffmpeg,
		"-hide_banner",
		"-loglevel", "error",
		"-re",
		"-i", streamURL,
		"-vn",
		"-sn",
		"-dn",
		"-ac", "1",
		"-ar", "48000",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"pipe:1",
	)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return err
	}
	readErr := m.readStreamPCM(ctx, id, stdout)
	waitErr := cmd.Wait()
	if errors.Is(ctx.Err(), context.Canceled) {
		return ctx.Err()
	}
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		return readErr
	}
	if waitErr != nil {
		return waitErr
	}
	return nil
}

func (m *OperatorBreakInManager) readStreamPCM(ctx context.Context, id string, reader io.Reader) error {
	buf := make([]byte, 48000*2/10)
	carry := make([]byte, 0, 1)
	for {
		n, err := reader.Read(buf)
		if n > 0 {
			chunk := buf[:n]
			if len(carry) > 0 {
				merged := make([]byte, 0, len(carry)+len(chunk))
				merged = append(merged, carry...)
				merged = append(merged, chunk...)
				chunk = merged
				carry = carry[:0]
			}
			if len(chunk)%2 != 0 {
				carry = append(carry[:0], chunk[len(chunk)-1])
				chunk = chunk[:len(chunk)-1]
			}
			if len(chunk) > 0 {
				if _, appendErr := m.appendChunk(id, append([]byte(nil), chunk...)); appendErr != nil {
					return appendErr
				}
			}
		}
		if err != nil {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
}

func (m *OperatorBreakInManager) cancelWithReason(id string, reason string) error {
	m.mu.Lock()
	session := m.sessions[id]
	delete(m.sessions, id)
	m.mu.Unlock()
	if session == nil {
		return nil
	}
	if session.Cancel != nil {
		session.Cancel()
	}
	if session.Publisher != nil {
		_ = publishOperatorBreakInEvent(session, "operator.breakin.cancel", map[string]any{"reason": reason})
		_ = session.Publisher.Close()
	}
	return nil
}

func validateOperatorBreakInStreamURL(raw string) error {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed == nil || parsed.Host == "" {
		return fmt.Errorf("media stream URL is invalid")
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https", "rtmp", "rtmps", "rtsp", "rtp", "srt", "udp":
		return nil
	default:
		return fmt.Errorf("media stream URL must use http, https, rtmp, rtsp, rtp, srt, or udp")
	}
}

func (m *OperatorBreakInManager) reapStale() {
	if m == nil {
		return
	}
	m.mu.Lock()
	for id, session := range m.sessions {
		if session == nil {
			delete(m.sessions, id)
		}
	}
	m.mu.Unlock()
}

type wavInfoLite struct {
	SampleRate int
	Channels   int
}

func wavPCM16Info(raw []byte) ([]byte, wavInfoLite, error) {
	if len(raw) < 44 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, wavInfoLite{}, fmt.Errorf("not a RIFF/WAVE file")
	}
	offset := 12
	var sampleRate int
	var channels int
	var bitsPerSample int
	var audioFormat int
	var pcm []byte
	for offset+8 <= len(raw) {
		id := string(raw[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(raw[offset+4 : offset+8]))
		offset += 8
		if size < 0 || offset+size > len(raw) {
			return nil, wavInfoLite{}, fmt.Errorf("invalid WAV chunk")
		}
		chunk := raw[offset : offset+size]
		switch id {
		case "fmt ":
			if len(chunk) < 16 {
				return nil, wavInfoLite{}, fmt.Errorf("invalid WAV fmt chunk")
			}
			audioFormat = int(binary.LittleEndian.Uint16(chunk[0:2]))
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(chunk[14:16]))
		case "data":
			pcm = append([]byte(nil), chunk...)
		}
		offset += size
		if offset%2 == 1 {
			offset++
		}
	}
	if audioFormat != 1 || bitsPerSample != 16 || sampleRate <= 0 || channels <= 0 || len(pcm) == 0 {
		return nil, wavInfoLite{}, fmt.Errorf("WAV is not PCM s16le")
	}
	return pcm, wavInfoLite{SampleRate: sampleRate, Channels: channels}, nil
}

func tonePCM16(frequency int, durationMS int, sampleRate int, channels int) []byte {
	samples := sampleRate * durationMS / 1000
	out := make([]byte, samples*channels*2)
	for i := 0; i < samples; i++ {
		ramp := 1.0
		fadeSamples := sampleRate / 100
		if i < fadeSamples {
			ramp = float64(i) / float64(fadeSamples)
		} else if tail := samples - i; tail < fadeSamples {
			ramp = float64(tail) / float64(fadeSamples)
		}
		value := int16(math.Sin(2*math.Pi*float64(frequency)*float64(i)/float64(sampleRate)) * 0.42 * ramp * math.MaxInt16)
		for ch := 0; ch < channels; ch++ {
			binary.LittleEndian.PutUint16(out[(i*channels+ch)*2:], uint16(value))
		}
	}
	return out
}

func silencePCM16(sampleRate int, channels int, duration time.Duration) []byte {
	samples := int(float64(sampleRate) * duration.Seconds())
	return make([]byte, samples*channels*2)
}

func wavFromPCM16(pcm []byte, sampleRate int, channels int) []byte {
	byteRate := sampleRate * channels * 2
	blockAlign := channels * 2
	out := make([]byte, 44+len(pcm))
	copy(out[0:4], "RIFF")
	binary.LittleEndian.PutUint32(out[4:8], uint32(36+len(pcm)))
	copy(out[8:12], "WAVE")
	copy(out[12:16], "fmt ")
	binary.LittleEndian.PutUint32(out[16:20], 16)
	binary.LittleEndian.PutUint16(out[20:22], 1)
	binary.LittleEndian.PutUint16(out[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(out[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(out[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(out[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(out[34:36], 16)
	copy(out[36:40], "data")
	binary.LittleEndian.PutUint32(out[40:44], uint32(len(pcm)))
	copy(out[44:], pcm)
	return out
}
