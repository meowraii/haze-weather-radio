package webgateway

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

const operatorBreakInDir = "runtime/audio/operator-breakin"
const operatorBreakInMaxPCMBytes = 48_000 * 2 * 180
const operatorBreakInMaxChunkBytes = 512 << 10
const operatorBreakInMaxUploadBytes = 8 << 20

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
	Publisher   *events.HostBridgePublisher
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
	return s.server.breakIn.start(s.configPath, targets, strings.TrimSpace(stringValue(payload, "title")), sampleRate, channels, prerollPath)
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
	url := strings.TrimSpace(firstNonBlank(stringValue(payload, "audio_url"), stringValue(payload, "stream_url"), stringValue(payload, "url")))
	if url == "" {
		return nil, fmt.Errorf("audio_url is required")
	}
	if !strings.HasPrefix(strings.ToLower(url), "http://") && !strings.HasPrefix(strings.ToLower(url), "https://") {
		return nil, fmt.Errorf("audio stream URL must use http or https")
	}
	alertID := safeID(firstNonBlank(stringValue(payload, "alert_id"), fmt.Sprintf("operator-stream-%d", time.Now().UTC().UnixNano())))
	title := fallbackText(strings.TrimSpace(stringValue(payload, "title")), "Operator Break-in Stream")
	data := map[string]any{
		"feed_ids":      targets,
		"alert_id":      alertID,
		"message_type":  "Operator",
		"title":         title,
		"event":         "OPR",
		"audio_url":     url,
		"alert_text":    title,
		"include_same":  false,
		"alert_sent_at": time.Now().UTC().Format(time.RFC3339Nano),
		"source":        "operator-breakin",
	}
	if err := publishAlertBroadcast(s.configPath, targets, data); err != nil {
		return nil, err
	}
	return map[string]any{"queued": true, "alert_id": alertID, "feed_ids": targets, "audio_url": url}, nil
}

func (s *wsSession) cancelOperatorBreakIn(payload map[string]any) (map[string]any, error) {
	id := strings.TrimSpace(stringValue(payload, "session_id"))
	if err := s.server.breakIn.cancel(id); err != nil {
		return nil, err
	}
	return map[string]any{"cancelled": true, "session_id": id}, nil
}

func (m *OperatorBreakInManager) start(configPath string, feedIDs []string, title string, sampleRate int, channels int, prerollPath string) (map[string]any, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil, fmt.Errorf("event bridge is not available")
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
		Publisher:   publisher,
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
		Publisher:   session.Publisher,
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
		"session_id":    id,
		"alert_id":      alertID,
		"feed_ids":      feedIDs,
		"max_pcm_bytes": operatorBreakInMaxPCMBytes,
		"live":          true,
	}, nil
}

func (m *OperatorBreakInManager) appendChunk(id string, data []byte) (map[string]any, error) {
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
	if session.Bytes+int64(len(data)) > operatorBreakInMaxPCMBytes {
		return nil, fmt.Errorf("break-in audio exceeds maximum duration")
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
	data, err := base64.StdEncoding.DecodeString(value)
	if err != nil {
		return nil, fmt.Errorf("decode audio data: %w", err)
	}
	if len(data) > maxBytes {
		return nil, fmt.Errorf("audio data exceeds %d bytes", maxBytes)
	}
	return data, nil
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
