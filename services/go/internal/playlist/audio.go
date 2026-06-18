package playlist

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type audioInfo struct {
	SampleRate int
	Channels   int
	DurationMS int64
	Bytes      int64
}

func wavInfo(path string) (audioInfo, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return audioInfo{}, err
	}
	if len(raw) < 44 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return audioInfo{}, fmt.Errorf("not a RIFF/WAVE file")
	}
	offset := 12
	var sampleRate int
	var channels int
	var bitsPerSample int
	var dataBytes int
	for offset+8 <= len(raw) {
		id := string(raw[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(raw[offset+4 : offset+8]))
		offset += 8
		if size < 0 || offset+size > len(raw) {
			return audioInfo{}, fmt.Errorf("invalid WAV chunk size")
		}
		chunk := raw[offset : offset+size]
		switch id {
		case "fmt ":
			if len(chunk) < 16 {
				return audioInfo{}, fmt.Errorf("invalid WAV fmt chunk")
			}
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(chunk[14:16]))
		case "data":
			dataBytes = len(chunk)
		}
		offset += size
		if offset%2 == 1 {
			offset++
		}
	}
	if sampleRate <= 0 || channels <= 0 || bitsPerSample <= 0 {
		return audioInfo{}, fmt.Errorf("WAV format is incomplete")
	}
	bytesPerSecond := int64(sampleRate * channels * bitsPerSample / 8)
	if bytesPerSecond <= 0 {
		return audioInfo{}, fmt.Errorf("invalid WAV byte rate")
	}
	return audioInfo{
		SampleRate: sampleRate,
		Channels:   channels,
		DurationMS: int64(dataBytes) * 1000 / bytesPerSecond,
		Bytes:      int64(dataBytes),
	}, nil
}

func pcmInfo(path string, sampleRate int, channels int) (audioInfo, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return audioInfo{}, err
	}
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if channels <= 0 {
		channels = 1
	}
	bytesPerSecond := int64(sampleRate * channels * 2)
	if bytesPerSecond <= 0 {
		return audioInfo{}, fmt.Errorf("invalid PCM byte rate")
	}
	return audioInfo{
		SampleRate: sampleRate,
		Channels:   channels,
		DurationMS: int64(len(raw)) * 1000 / bytesPerSecond,
		Bytes:      int64(len(raw)),
	}, nil
}

func wavToPCM16File(inputPath string, outputPath string, sampleRate int, channels int) error {
	raw, err := os.ReadFile(inputPath)
	if err != nil {
		return err
	}
	pcm, info, err := wavPCM16(raw)
	if err != nil {
		return err
	}
	if info.SampleRate != sampleRate || info.Channels != channels {
		return fmt.Errorf("WAV format requires resampling")
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	tmp := outputPath + ".tmp"
	if err := os.WriteFile(tmp, pcm, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, outputPath)
}

func wavPCM16(raw []byte) ([]byte, audioInfo, error) {
	if len(raw) < 44 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, audioInfo{}, fmt.Errorf("not a RIFF/WAVE file")
	}
	offset := 12
	var sampleRate int
	var channels int
	var bitsPerSample int
	var audioFormat int
	var data []byte
	for offset+8 <= len(raw) {
		id := string(raw[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(raw[offset+4 : offset+8]))
		offset += 8
		if size < 0 || offset+size > len(raw) {
			return nil, audioInfo{}, fmt.Errorf("invalid WAV chunk size")
		}
		chunk := raw[offset : offset+size]
		switch id {
		case "fmt ":
			if len(chunk) < 16 {
				return nil, audioInfo{}, fmt.Errorf("invalid WAV fmt chunk")
			}
			audioFormat = int(binary.LittleEndian.Uint16(chunk[0:2]))
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(chunk[14:16]))
		case "data":
			data = append([]byte(nil), chunk...)
		}
		offset += size
		if offset%2 == 1 {
			offset++
		}
	}
	if audioFormat != 1 || bitsPerSample != 16 || sampleRate <= 0 || channels <= 0 || len(data) == 0 {
		return nil, audioInfo{}, fmt.Errorf("WAV is not PCM s16le")
	}
	bytesPerSecond := int64(sampleRate * channels * 2)
	return data, audioInfo{
		SampleRate: sampleRate,
		Channels:   channels,
		DurationMS: int64(len(data)) * 1000 / bytesPerSecond,
		Bytes:      int64(len(data)),
	}, nil
}

func downloadFile(ctx context.Context, sourceURL string, outputPath string, maxBytes int64) error {
	if strings.TrimSpace(sourceURL) == "" {
		return fmt.Errorf("missing audio URL")
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return err
	}
	request.Header.Set("User-Agent", "HazeWeatherRadio/26.06 alert-audio")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return err
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
	_, copyErr := io.Copy(file, io.LimitReader(response.Body, maxBytes+1))
	closeErr := file.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if info, err := os.Stat(tmp); err == nil && info.Size() > maxBytes {
		_ = os.Remove(tmp)
		return fmt.Errorf("audio resource exceeds %d bytes", maxBytes)
	}
	return os.Rename(tmp, outputPath)
}

func convertAudioToPCM(ctx context.Context, inputPath string, outputPath string, sampleRate int, channels int) error {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if channels <= 0 {
		channels = 1
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	tmp := outputPath + ".tmp"
	_ = os.Remove(tmp)
	ffmpeg := strings.TrimSpace(os.Getenv("FFMPEG"))
	if ffmpeg == "" {
		ffmpeg = "ffmpeg"
	}
	cmd := exec.CommandContext(
		ctx,
		ffmpeg,
		"-hide_banner",
		"-loglevel", "error",
		"-y",
		"-i", inputPath,
		"-vn",
		"-ac", fmt.Sprintf("%d", channels),
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		tmp,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("ffmpeg audio conversion failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return os.Rename(tmp, outputPath)
}

func writePriorityAlertManifest(path string, manifest priorityAlertManifest) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
