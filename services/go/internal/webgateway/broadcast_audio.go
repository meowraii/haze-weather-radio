package webgateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	broadcastAlertAudioUploadDir = "runtime/audio/alerts/uploads"
	broadcastAlertMediaMaxBytes  = int64(20 << 20)
	broadcastAlertAudioTimeout   = 90 * time.Second
)

func (s *Server) alertAudioUpload(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		writer.Header().Set("Allow", "POST")
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if _, ok := s.requireOriginationRequest(writer, request); !ok {
		return
	}
	maxBytes := broadcastAlertMediaUploadBytes(s.config)
	request.Body = http.MaxBytesReader(writer, request.Body, maxBytes+(1<<20))
	if err := request.ParseMultipartForm(1 << 20); err != nil {
		writeJSONStatus(writer, http.StatusBadRequest, map[string]any{"detail": "media upload is invalid or too large"})
		return
	}
	file, header, err := request.FormFile("file")
	if err != nil {
		writeJSONStatus(writer, http.StatusBadRequest, map[string]any{"detail": "media file is required"})
		return
	}
	defer file.Close()
	result, err := prepareBroadcastAlertAudioReader(request.Context(), s.configPath, header.Filename, file, maxBytes)
	if err != nil {
		writeJSONStatus(writer, http.StatusBadRequest, map[string]any{"detail": err.Error()})
		return
	}
	writeJSON(writer, result)
}

func (s *wsSession) prepareBroadcastAlertAudioPayload(payload map[string]any, alertID string) (map[string]any, error) {
	mode := normalizeBroadcastAudioMode(stringPayload(payload, "audio_mode", "tts"))
	if mode != "file" {
		payload["audio_mode"] = mode
		return payload, nil
	}
	out := cloneBroadcastMap(payload)
	out["audio_mode"] = mode
	if strings.TrimSpace(stringPayload(out, "audio_path", "")) != "" {
		return out, nil
	}
	audioURL := strings.TrimSpace(firstNonBlank(
		stringPayload(out, "audio_url", ""),
		stringPayload(out, "url", ""),
	))
	if audioURL == "" {
		return out, nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), broadcastAlertAudioTimeout)
	defer cancel()
	result, err := prepareBroadcastAlertAudioURL(ctx, s.configPath, audioURL, alertID, broadcastAlertMediaUploadBytes(s.config))
	if err != nil {
		return nil, err
	}
	out["audio_path"] = result["path"]
	out["audio_format"] = result["format"]
	out["audio_sample_rate"] = result["sample_rate"]
	out["audio_channels"] = result["channels"]
	out["authoritative_url"] = audioURL
	return out, nil
}

func prepareBroadcastAlertAudioReader(ctx context.Context, configPath string, filename string, reader io.Reader, maxBytes int64) (map[string]any, error) {
	ext := broadcastAlertMediaExtension(filename)
	id := safeID(firstNonBlank(strings.TrimSuffix(filepath.Base(filename), ext), fmt.Sprintf("upload-%d", time.Now().UnixNano())))
	if id == "" {
		id = safeID(fmt.Sprintf("upload-%d", time.Now().UnixNano()))
	}
	id = safeID(fmt.Sprintf("%s-%d", id, time.Now().UnixNano()))
	inputRel := filepath.ToSlash(filepath.Join(broadcastAlertAudioUploadDir, id+ext))
	inputPath := resolveConfigPath(configPath, inputRel)
	if err := os.MkdirAll(filepath.Dir(inputPath), 0o755); err != nil {
		return nil, err
	}
	maxBytes = normalizeBroadcastAlertMediaLimit(maxBytes)
	limited := io.LimitReader(reader, maxBytes+1)
	var buf bytes.Buffer
	n, err := io.Copy(&buf, limited)
	if err != nil {
		return nil, fmt.Errorf("read media upload failed")
	}
	if n > maxBytes {
		return nil, fmt.Errorf("media file is too large; maximum is %s", byteLimitLabel(maxBytes))
	}
	if n == 0 {
		return nil, fmt.Errorf("media file is empty")
	}
	if err := writeFileAtomic(inputPath, buf.Bytes(), 0o600); err != nil {
		return nil, err
	}
	defer os.Remove(inputPath)
	return prepareBroadcastAlertAudioInput(ctx, configPath, inputPath, id, "")
}

func prepareBroadcastAlertAudioURL(ctx context.Context, configPath string, rawURL string, alertID string, maxBytes int64) (map[string]any, error) {
	parsed, err := validateBroadcastAlertAudioURL(rawURL)
	if err != nil {
		return nil, err
	}
	name := filepath.Base(parsed.Path)
	ext := broadcastAlertMediaExtension(name)
	id := safeID(firstNonBlank(alertID, strings.TrimSuffix(name, ext), fmt.Sprintf("url-%d", time.Now().UnixNano())))
	if id == "" {
		id = safeID(fmt.Sprintf("url-%d", time.Now().UnixNano()))
	}
	id = safeID(fmt.Sprintf("%s-%d", id, time.Now().UnixNano()))
	inputRel := filepath.ToSlash(filepath.Join(broadcastAlertAudioUploadDir, id+ext))
	inputPath := resolveConfigPath(configPath, inputRel)
	if err := downloadBroadcastAlertAudio(ctx, parsed.String(), inputPath, maxBytes); err != nil {
		return nil, err
	}
	defer os.Remove(inputPath)
	return prepareBroadcastAlertAudioInput(ctx, configPath, inputPath, id, parsed.String())
}

func prepareBroadcastAlertAudioInput(ctx context.Context, configPath string, inputPath string, id string, sourceURL string) (map[string]any, error) {
	outputRel := filepath.ToSlash(filepath.Join(broadcastAlertAudioUploadDir, id+".pcm16le"))
	outputPath := resolveConfigPath(configPath, outputRel)
	convertCtx, cancel := context.WithTimeout(ctx, broadcastAlertAudioTimeout)
	defer cancel()
	if err := convertBroadcastAlertAudioToPCM(convertCtx, inputPath, outputPath); err != nil {
		return nil, err
	}
	info, err := os.Stat(outputPath)
	if err != nil {
		return nil, err
	}
	result := map[string]any{
		"path":        outputRel,
		"audio_path":  outputRel,
		"format":      "pcm_s16le",
		"sample_rate": 48000,
		"channels":    1,
		"bytes":       info.Size(),
		"source":      "webpanel-normalized-media",
	}
	if sourceURL != "" {
		result["authoritative_url"] = sourceURL
	}
	return result, nil
}

func convertBroadcastAlertAudioToPCM(ctx context.Context, inputPath string, outputPath string) error {
	ffmpeg, err := resolveFFmpegExecutable()
	if err != nil {
		return fmt.Errorf("audio conversion backend is unavailable")
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	tmp := outputPath + ".tmp"
	_ = os.Remove(tmp)
	cmd := exec.CommandContext(
		ctx,
		ffmpeg,
		"-hide_banner",
		"-loglevel", "error",
		"-nostdin",
		"-y",
		"-i", inputPath,
		"-vn",
		"-sn",
		"-dn",
		"-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
		"-ac", "1",
		"-ar", "48000",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		tmp,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		_ = os.Remove(tmp)
		detail := strings.TrimSpace(string(output))
		if detail == "" {
			detail = "conversion failed"
		}
		return fmt.Errorf("media conversion failed: %s", detail)
	}
	return os.Rename(tmp, outputPath)
}

func downloadBroadcastAlertAudio(ctx context.Context, sourceURL string, outputPath string, maxBytes int64) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return fmt.Errorf("media URL is invalid")
	}
	request.Header.Set("User-Agent", "HazeWeatherRadio/26.06 alert-media")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return fmt.Errorf("media URL could not be fetched")
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("media URL returned %s", response.Status)
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	tmp := outputPath + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}
	maxBytes = normalizeBroadcastAlertMediaLimit(maxBytes)
	_, copyErr := io.Copy(file, io.LimitReader(response.Body, maxBytes+1))
	closeErr := file.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("media URL download failed")
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if info, err := os.Stat(tmp); err == nil && info.Size() > maxBytes {
		_ = os.Remove(tmp)
		return fmt.Errorf("media resource is too large; maximum is %s", byteLimitLabel(maxBytes))
	}
	return os.Rename(tmp, outputPath)
}

func validateBroadcastAlertAudioURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed == nil || parsed.Host == "" {
		return nil, fmt.Errorf("media URL is invalid")
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
		return parsed, nil
	default:
		return nil, fmt.Errorf("media URL must use http or https")
	}
}

func broadcastAlertMediaUploadBytes(config Config) int64 {
	return normalizeBroadcastAlertMediaLimit(config.Webpanel.Authentication.MaxAudioUploadBytes)
}

func normalizeBroadcastAlertMediaLimit(maxBytes int64) int64 {
	if maxBytes <= 0 {
		return broadcastAlertMediaMaxBytes
	}
	if maxBytes < 1<<20 {
		return 1 << 20
	}
	return maxBytes
}

func byteLimitLabel(bytes int64) string {
	if bytes >= 1<<20 && bytes%(1<<20) == 0 {
		return fmt.Sprintf("%d MB", bytes/(1<<20))
	}
	if bytes >= 1<<20 {
		return fmt.Sprintf("%.1f MB", float64(bytes)/(1<<20))
	}
	if bytes >= 1<<10 {
		return fmt.Sprintf("%d KB", bytes/(1<<10))
	}
	return fmt.Sprintf("%d bytes", bytes)
}

func broadcastAlertMediaExtension(filename string) string {
	ext := strings.ToLower(filepath.Ext(filepath.Base(filename)))
	if len(ext) < 2 || len(ext) > 16 {
		return ".bin"
	}
	for _, ch := range ext[1:] {
		if ch >= 'a' && ch <= 'z' || ch >= '0' && ch <= '9' {
			continue
		}
		return ".bin"
	}
	return ext
}

func normalizeBroadcastAudioMode(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "audio_file", "audio-file", "file":
		return "file"
	case "operator", "operator_breakin", "operator-break-in", "breakin":
		return "operator"
	case "audio_stream", "audio-stream", "stream":
		return "stream"
	default:
		return "tts"
	}
}

func writeJSONStatus(writer http.ResponseWriter, status int, value map[string]any) {
	writer.Header().Set("Content-Type", "application/json")
	if writer.Header().Get("Cache-Control") == "" {
		writer.Header().Set("Cache-Control", "no-store")
	}
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}
