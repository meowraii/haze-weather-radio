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
	broadcastAlertAudioMaxBytes  = 20 << 20
	broadcastAlertAudioTimeout   = 90 * time.Second
)

var broadcastAlertAudioExtensions = map[string]bool{
	".wav":  true,
	".mp3":  true,
	".ogg":  true,
	".opus": true,
	".m4a":  true,
	".aac":  true,
	".flac": true,
	".webm": true,
}

func (s *Server) alertAudioUpload(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		writer.Header().Set("Allow", "POST")
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.auth.Authenticated(request) {
		http.Error(writer, "unauthorized", http.StatusUnauthorized)
		return
	}
	request.Body = http.MaxBytesReader(writer, request.Body, broadcastAlertAudioMaxBytes+1<<20)
	if err := request.ParseMultipartForm(1 << 20); err != nil {
		writeJSONStatus(writer, http.StatusBadRequest, map[string]any{"detail": "audio upload is invalid or too large"})
		return
	}
	file, header, err := request.FormFile("file")
	if err != nil {
		writeJSONStatus(writer, http.StatusBadRequest, map[string]any{"detail": "audio file is required"})
		return
	}
	defer file.Close()
	result, err := prepareBroadcastAlertAudioReader(request.Context(), s.configPath, header.Filename, file)
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
	result, err := prepareBroadcastAlertAudioURL(ctx, s.configPath, audioURL, alertID)
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

func prepareBroadcastAlertAudioReader(ctx context.Context, configPath string, filename string, reader io.Reader) (map[string]any, error) {
	ext := strings.ToLower(filepath.Ext(filepath.Base(filename)))
	if !broadcastAlertAudioExtensions[ext] {
		return nil, fmt.Errorf("unsupported audio file type")
	}
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
	limited := io.LimitReader(reader, broadcastAlertAudioMaxBytes+1)
	var buf bytes.Buffer
	n, err := io.Copy(&buf, limited)
	if err != nil {
		return nil, fmt.Errorf("read audio upload failed")
	}
	if n > broadcastAlertAudioMaxBytes {
		return nil, fmt.Errorf("audio file is too large; maximum is 20 MB")
	}
	if n == 0 {
		return nil, fmt.Errorf("audio file is empty")
	}
	if err := writeFileAtomic(inputPath, buf.Bytes(), 0o600); err != nil {
		return nil, err
	}
	defer os.Remove(inputPath)
	return prepareBroadcastAlertAudioInput(ctx, configPath, inputPath, id, "")
}

func prepareBroadcastAlertAudioURL(ctx context.Context, configPath string, rawURL string, alertID string) (map[string]any, error) {
	parsed, err := validateBroadcastAlertAudioURL(rawURL)
	if err != nil {
		return nil, err
	}
	name := filepath.Base(parsed.Path)
	ext := strings.ToLower(filepath.Ext(name))
	if !broadcastAlertAudioExtensions[ext] {
		ext = ".bin"
	}
	id := safeID(firstNonBlank(alertID, strings.TrimSuffix(name, ext), fmt.Sprintf("url-%d", time.Now().UnixNano())))
	if id == "" {
		id = safeID(fmt.Sprintf("url-%d", time.Now().UnixNano()))
	}
	id = safeID(fmt.Sprintf("%s-%d", id, time.Now().UnixNano()))
	inputRel := filepath.ToSlash(filepath.Join(broadcastAlertAudioUploadDir, id+ext))
	inputPath := resolveConfigPath(configPath, inputRel)
	if err := downloadBroadcastAlertAudio(ctx, parsed.String(), inputPath); err != nil {
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
		"source":      "webpanel-normalized-audio",
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
		return fmt.Errorf("audio conversion failed: %s", detail)
	}
	return os.Rename(tmp, outputPath)
}

func downloadBroadcastAlertAudio(ctx context.Context, sourceURL string, outputPath string) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return fmt.Errorf("audio URL is invalid")
	}
	request.Header.Set("User-Agent", "HazeWeatherRadio/26.06 alert-audio")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return fmt.Errorf("audio URL could not be fetched")
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("audio URL returned %s", response.Status)
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	tmp := outputPath + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(file, io.LimitReader(response.Body, broadcastAlertAudioMaxBytes+1))
	closeErr := file.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("audio URL download failed")
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if info, err := os.Stat(tmp); err == nil && info.Size() > broadcastAlertAudioMaxBytes {
		_ = os.Remove(tmp)
		return fmt.Errorf("audio file is too large; maximum is 20 MB")
	}
	return os.Rename(tmp, outputPath)
}

func validateBroadcastAlertAudioURL(raw string) (*url.URL, error) {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed == nil || parsed.Host == "" {
		return nil, fmt.Errorf("audio URL is invalid")
	}
	switch strings.ToLower(parsed.Scheme) {
	case "http", "https":
		return parsed, nil
	default:
		return nil, fmt.Errorf("audio URL must use http or https")
	}
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
