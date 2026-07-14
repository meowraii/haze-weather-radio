package playlist

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
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
	file, err := os.Open(path)
	if err != nil {
		return audioInfo{}, err
	}
	defer file.Close()
	stat, err := file.Stat()
	if err != nil {
		return audioInfo{}, err
	}
	var header [12]byte
	if _, err := io.ReadFull(file, header[:]); err != nil || string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return audioInfo{}, fmt.Errorf("not a RIFF/WAVE file")
	}
	remaining := stat.Size() - int64(len(header))
	var sampleRate int
	var channels int
	var bitsPerSample int
	var dataBytes int64
	for remaining >= 8 {
		var chunkHeader [8]byte
		if _, err := io.ReadFull(file, chunkHeader[:]); err != nil {
			return audioInfo{}, err
		}
		remaining -= int64(len(chunkHeader))
		size := int64(binary.LittleEndian.Uint32(chunkHeader[4:8]))
		paddedSize := size + size%2
		if paddedSize > remaining {
			return audioInfo{}, fmt.Errorf("invalid WAV chunk size")
		}
		switch string(chunkHeader[0:4]) {
		case "fmt ":
			if size < 16 {
				return audioInfo{}, fmt.Errorf("invalid WAV fmt chunk")
			}
			var format [16]byte
			if _, err := io.ReadFull(file, format[:]); err != nil {
				return audioInfo{}, err
			}
			channels = int(binary.LittleEndian.Uint16(format[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(format[4:8]))
			bitsPerSample = int(binary.LittleEndian.Uint16(format[14:16]))
			if _, err := file.Seek(paddedSize-16, io.SeekCurrent); err != nil {
				return audioInfo{}, err
			}
		case "data":
			dataBytes = size
			if sampleRate > 0 && channels > 0 && bitsPerSample > 0 {
				remaining = 0
				continue
			}
			if _, err := file.Seek(paddedSize, io.SeekCurrent); err != nil {
				return audioInfo{}, err
			}
		default:
			if _, err := file.Seek(paddedSize, io.SeekCurrent); err != nil {
				return audioInfo{}, err
			}
		}
		remaining -= paddedSize
	}
	if sampleRate <= 0 || channels <= 0 || bitsPerSample <= 0 || dataBytes <= 0 {
		return audioInfo{}, fmt.Errorf("WAV format is incomplete")
	}
	bytesPerSecond := int64(sampleRate * channels * bitsPerSample / 8)
	if bytesPerSecond <= 0 {
		return audioInfo{}, fmt.Errorf("invalid WAV byte rate")
	}
	return audioInfo{
		SampleRate: sampleRate,
		Channels:   channels,
		DurationMS: dataBytes * 1000 / bytesPerSecond,
		Bytes:      dataBytes,
	}, nil
}

func pcmInfo(path string, sampleRate int, channels int) (audioInfo, error) {
	stat, err := os.Stat(path)
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
		DurationMS: stat.Size() * 1000 / bytesPerSecond,
		Bytes:      stat.Size(),
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
			// raw remains owned by the caller for the lifetime of the returned PCM.
			// Keeping this slice avoids copying the entire audio payload on each parse.
			data = chunk
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

func writePCM16WAV(path string, pcm []byte, sampleRate int, channels int) error {
	stream, err := newPCM16WAVStream(path, sampleRate, channels)
	if err != nil {
		return err
	}
	defer stream.Abort()
	if err := stream.Append(pcm); err != nil {
		return err
	}
	return stream.Commit()
}

type pcm16WAVStream struct {
	path       string
	tmp        string
	file       *os.File
	sampleRate int
	channels   int
	dataBytes  uint64
}

func newPCM16WAVStream(path string, sampleRate int, channels int) (*pcm16WAVStream, error) {
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if channels <= 0 {
		channels = 1
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	tmp := path + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return nil, err
	}
	stream := &pcm16WAVStream{
		path:       path,
		tmp:        tmp,
		file:       file,
		sampleRate: sampleRate,
		channels:   channels,
	}
	if err := stream.writeHeader(0); err != nil {
		_ = file.Close()
		_ = os.Remove(tmp)
		return nil, err
	}
	return stream, nil
}

func (s *pcm16WAVStream) Append(pcm []byte) error {
	if s.file == nil {
		return fmt.Errorf("PCM WAV stream is closed")
	}
	frameBytes := s.channels * 2
	if len(pcm)%frameBytes != 0 {
		return fmt.Errorf("PCM payload must contain complete s16le frames")
	}
	if uint64(len(pcm)) > uint64(math.MaxUint32)-s.dataBytes {
		return fmt.Errorf("PCM WAV payload exceeds RIFF size limit")
	}
	written, err := s.file.Write(pcm)
	s.dataBytes += uint64(written)
	if err != nil {
		return err
	}
	if written != len(pcm) {
		return io.ErrShortWrite
	}
	return nil
}

func (s *pcm16WAVStream) Commit() error {
	if s.file == nil {
		return fmt.Errorf("PCM WAV stream is closed")
	}
	if _, err := s.file.Seek(0, io.SeekStart); err != nil {
		return err
	}
	if err := s.writeHeader(uint32(s.dataBytes)); err != nil {
		return err
	}
	if err := s.file.Close(); err != nil {
		s.file = nil
		return err
	}
	s.file = nil
	if err := os.Rename(s.tmp, s.path); err != nil {
		_ = os.Remove(s.tmp)
		return err
	}
	return nil
}

func (s *pcm16WAVStream) Abort() {
	if s.file != nil {
		_ = s.file.Close()
		s.file = nil
	}
	_ = os.Remove(s.tmp)
}

func (s *pcm16WAVStream) writeHeader(dataSize uint32) error {
	riffSize := uint32(36) + dataSize
	byteRate := uint32(s.sampleRate * s.channels * 2)
	blockAlign := uint16(s.channels * 2)
	if _, err := s.file.Write([]byte("RIFF")); err != nil {
		return err
	}
	for _, value := range []any{
		riffSize,
		[]byte("WAVEfmt "),
		uint32(16),
		uint16(1),
		uint16(s.channels),
		uint32(s.sampleRate),
		byteRate,
		blockAlign,
		uint16(16),
		[]byte("data"),
		dataSize,
	} {
		if err := binary.Write(s.file, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	return nil
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

func convertAlertAudioToPCM(ctx context.Context, inputPath string, outputPath string, sampleRate int, channels int) error {
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
		"-nostdin",
		"-y",
		"-i", inputPath,
		"-vn",
		"-sn",
		"-dn",
		"-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
		"-ac", fmt.Sprintf("%d", channels),
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		tmp,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("ffmpeg alert audio conversion failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return os.Rename(tmp, outputPath)
}

func convertRawPCM16FileToPCM(ctx context.Context, inputPath string, outputPath string, sourceRate int, sourceChannels int, sampleRate int, channels int) error {
	if sourceRate <= 0 {
		sourceRate = 48000
	}
	if sourceChannels <= 0 {
		sourceChannels = 1
	}
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
		"-nostdin",
		"-y",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"-ar", fmt.Sprintf("%d", sourceRate),
		"-ac", fmt.Sprintf("%d", sourceChannels),
		"-i", inputPath,
		"-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
		"-f", "s16le",
		"-acodec", "pcm_s16le",
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-ac", fmt.Sprintf("%d", channels),
		tmp,
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("ffmpeg alert PCM conversion failed: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return os.Rename(tmp, outputPath)
}

func convertAudioToWAV(ctx context.Context, inputPath string, outputPath string, sampleRate int, channels int) error {
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
		"-f", "wav",
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
	tmp := path + "." + queueID("manifest") + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		return err
	}
	if err := replaceFileAtomically(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func replaceFileAtomically(sourcePath string, targetPath string) error {
	if _, err := os.Stat(sourcePath); err != nil {
		return err
	}
	firstErr := os.Rename(sourcePath, targetPath)
	if firstErr == nil {
		return nil
	}
	if _, err := os.Stat(targetPath); err != nil {
		return firstErr
	}
	backupPath := targetPath + "." + queueID("replace") + ".bak"
	if err := os.Rename(targetPath, backupPath); err != nil {
		return err
	}
	if err := os.Rename(sourcePath, targetPath); err != nil {
		_ = os.Rename(backupPath, targetPath)
		return err
	}
	_ = os.Remove(backupPath)
	return nil
}
