package tts

import (
	"archive/tar"
	"compress/bzip2"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const kokoroProviderID = "kokoro"
const defaultKokoroArchiveURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
const defaultKokoroArchiveSHA256 = "912804855a04745fa77a30be545b3f9a5d15c4d66db00b88cbcd4921df605ac7"
const defaultKokoroArchiveSize = 319625534

type kokoroOptions struct {
	ModelDir        string
	ModelPath       string
	VoicesPath      string
	TokensPath      string
	DataDir         string
	LexiconPath     string
	RuleFsts        string
	RuleFars        string
	RuntimeProvider string
	Lang            string
	Threads         int
	MaxSentences    int
	LengthScale     float32
	SilenceScale    float32
	Speed           float32
	Debug           bool
	ArchiveURL      string
	ArchiveSHA256   string
	ArchiveSize     int64
}

func defaultKokoroOptions() kokoroOptions {
	modelDir := envOrDefault("HAZE_KOKORO_MODEL_DIR", filepath.Join("managed", "voices", "kokoro"))
	lexiconPath := strings.TrimSpace(os.Getenv("HAZE_KOKORO_LEXICON"))
	if lexiconPath == "" {
		lexiconPath = existingKokoroPath(filepath.Join(modelDir, "lexicon.txt"))
	}
	return kokoroOptions{
		ModelDir:        modelDir,
		ModelPath:       envOrDefault("HAZE_KOKORO_MODEL", filepath.Join(modelDir, "model.onnx")),
		VoicesPath:      envOrDefault("HAZE_KOKORO_VOICES", filepath.Join(modelDir, "voices.bin")),
		TokensPath:      envOrDefault("HAZE_KOKORO_TOKENS", filepath.Join(modelDir, "tokens.txt")),
		DataDir:         envOrDefault("HAZE_KOKORO_DATA_DIR", filepath.Join(modelDir, "espeak-ng-data")),
		LexiconPath:     lexiconPath,
		RuleFsts:        strings.TrimSpace(os.Getenv("HAZE_KOKORO_RULE_FSTS")),
		RuleFars:        strings.TrimSpace(os.Getenv("HAZE_KOKORO_RULE_FARS")),
		RuntimeProvider: envOrDefault("HAZE_KOKORO_PROVIDER", "cpu"),
		Lang:            NormalizeLanguage(os.Getenv("HAZE_KOKORO_LANG")),
		Threads:         kokoroEnvInt("HAZE_KOKORO_THREADS", defaultKokoroThreads()),
		MaxSentences:    kokoroEnvInt("HAZE_KOKORO_MAX_SENTENCES", 1),
		LengthScale:     kokoroEnvFloat32("HAZE_KOKORO_LENGTH_SCALE", 1.0),
		SilenceScale:    kokoroEnvFloat32("HAZE_KOKORO_SILENCE_SCALE", 0.2),
		Speed:           kokoroEnvFloat32("HAZE_KOKORO_SPEED", 1.0),
		Debug:           kokoroEnvBool("HAZE_KOKORO_DEBUG", false),
		ArchiveURL:      envOrDefault("HAZE_KOKORO_ARCHIVE_URL", defaultKokoroArchiveURL),
		ArchiveSHA256:   envOrDefault("HAZE_KOKORO_ARCHIVE_SHA256", defaultKokoroArchiveSHA256),
		ArchiveSize:     int64(kokoroEnvInt("HAZE_KOKORO_ARCHIVE_SIZE", defaultKokoroArchiveSize)),
	}
}

func (o kokoroOptions) validate() error {
	for _, item := range []struct {
		name string
		path string
		dir  bool
	}{
		{name: "model", path: o.ModelPath},
		{name: "voices", path: o.VoicesPath},
		{name: "tokens", path: o.TokensPath},
		{name: "espeak-ng-data", path: o.DataDir, dir: true},
	} {
		if strings.TrimSpace(item.path) == "" {
			return fmt.Errorf("%w: kokoro %s path is not configured", ErrProviderUnavailable, item.name)
		}
		info, err := os.Stat(item.path)
		if err != nil {
			return fmt.Errorf("%w: kokoro %s not found at %s", ErrProviderUnavailable, item.name, item.path)
		}
		if item.dir && !info.IsDir() {
			return fmt.Errorf("%w: kokoro %s is not a directory: %s", ErrProviderUnavailable, item.name, item.path)
		}
		if !item.dir && info.IsDir() {
			return fmt.Errorf("%w: kokoro %s is not a file: %s", ErrProviderUnavailable, item.name, item.path)
		}
	}
	return nil
}

func (o kokoroOptions) ensureModelFiles(ctx context.Context) error {
	if o.validate() == nil {
		return nil
	}
	if strings.TrimSpace(o.ArchiveURL) == "" {
		return o.validate()
	}
	if strings.TrimSpace(o.ModelDir) == "" {
		return fmt.Errorf("%w: kokoro model directory is not configured", ErrProviderUnavailable)
	}
	if err := os.MkdirAll(o.ModelDir, 0o755); err != nil {
		return err
	}
	archivePath := filepath.Join(o.ModelDir, filepath.Base(o.ArchiveURL))
	if err := o.downloadArchive(ctx, archivePath); err != nil {
		return err
	}
	if err := o.extractArchive(archivePath); err != nil {
		return err
	}
	return o.validate()
}

func (o kokoroOptions) downloadArchive(ctx context.Context, archivePath string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if ok, err := kokoroFileMatchesSHA256(archivePath, o.ArchiveSHA256); ok && err == nil {
		return nil
	}
	client := http.DefaultClient
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, o.ArchiveURL, nil)
	if err != nil {
		return err
	}
	response, err := client.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("download %s failed: %s", o.ArchiveURL, response.Status)
	}
	tmp, err := os.CreateTemp(o.ModelDir, "kokoro-*.tar.bz2")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	removeTmp := true
	defer func() {
		if removeTmp {
			_ = os.Remove(tmpPath)
		}
	}()
	hasher := sha256.New()
	written, err := io.Copy(tmp, io.TeeReader(response.Body, hasher))
	closeErr := tmp.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if o.ArchiveSize > 0 && written != o.ArchiveSize {
		return fmt.Errorf("downloaded kokoro archive size = %d, want %d", written, o.ArchiveSize)
	}
	if expected := strings.TrimSpace(o.ArchiveSHA256); expected != "" {
		sum := hex.EncodeToString(hasher.Sum(nil))
		if !strings.EqualFold(sum, expected) {
			return fmt.Errorf("downloaded kokoro archive sha256 = %s, want %s", sum, expected)
		}
	}
	_ = os.Remove(archivePath)
	if err := os.Rename(tmpPath, archivePath); err != nil {
		return err
	}
	removeTmp = false
	return nil
}

func (o kokoroOptions) extractArchive(archivePath string) error {
	file, err := os.Open(filepath.Clean(archivePath))
	if err != nil {
		return err
	}
	defer file.Close()
	reader := tar.NewReader(bzip2.NewReader(file))
	for {
		header, err := reader.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		relative := kokoroArchiveRelativePath(header.Name)
		if relative == "" {
			continue
		}
		target := filepath.Join(o.ModelDir, filepath.FromSlash(relative))
		if !kokoroPathWithin(o.ModelDir, target) {
			return fmt.Errorf("kokoro archive entry escapes model directory: %s", header.Name)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg, tar.TypeRegA:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			if err := kokoroWriteArchiveFile(target, reader, header.FileInfo().Mode()); err != nil {
				return err
			}
		default:
			continue
		}
	}
}

func kokoroSpeakerID(voiceID string) (int, error) {
	raw := strings.TrimSpace(voiceID)
	if raw == "" {
		return 0, nil
	}
	value := strings.ToLower(raw)
	for _, prefix := range []string{"kokoro:", "sid:", "speaker:", "voice:"} {
		value = strings.TrimPrefix(value, prefix)
	}
	value = strings.TrimPrefix(value, "kokoro-")
	sid, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || sid < 0 {
		return 0, fmt.Errorf("kokoro voice_id %q must be a non-negative speaker id", voiceID)
	}
	return sid, nil
}

func kokoroPCM16LE(samples []float32) []byte {
	data := make([]byte, len(samples)*2)
	for i, sample := range samples {
		clamped := kokoroClampFloat32(sample, -1, 1)
		var value int16
		if clamped < 0 {
			value = int16(clamped * 32768)
		} else {
			value = int16(clamped * 32767)
		}
		binary.LittleEndian.PutUint16(data[i*2:], uint16(value))
	}
	return data
}

func kokoroEnvInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func kokoroEnvFloat32(key string, fallback float32) float32 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseFloat(value, 32)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return float32(parsed)
}

func kokoroEnvBool(key string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "":
		return fallback
	case "1", "true", "yes", "on", "enabled":
		return true
	case "0", "false", "no", "off", "disabled":
		return false
	default:
		return fallback
	}
}

func existingKokoroPath(path string) string {
	if _, err := os.Stat(path); err == nil {
		return path
	}
	return ""
}

func kokoroFileMatchesSHA256(path string, expected string) (bool, error) {
	expected = strings.TrimSpace(expected)
	if expected == "" {
		_, err := os.Stat(path)
		return err == nil, err
	}
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return false, err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return false, err
	}
	return strings.EqualFold(hex.EncodeToString(hash.Sum(nil)), expected), nil
}

func kokoroArchiveRelativePath(name string) string {
	cleaned := filepath.ToSlash(filepath.Clean(strings.TrimSpace(name)))
	cleaned = strings.TrimPrefix(cleaned, "./")
	if cleaned == "." || cleaned == "" || strings.HasPrefix(cleaned, "../") || strings.HasPrefix(cleaned, "/") {
		return ""
	}
	parts := strings.Split(cleaned, "/")
	if len(parts) <= 1 {
		return ""
	}
	return strings.Join(parts[1:], "/")
}

func kokoroPathWithin(root string, path string) bool {
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return false
	}
	pathAbs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(rootAbs, pathAbs)
	if err != nil {
		return false
	}
	return rel == "." || rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func kokoroWriteArchiveFile(path string, reader io.Reader, mode os.FileMode) error {
	file, err := os.OpenFile(filepath.Clean(path), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(file, reader)
	closeErr := file.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func defaultKokoroThreads() int {
	cpus := runtime.NumCPU()
	if cpus < 1 {
		return 1
	}
	if cpus > 4 {
		return 4
	}
	return cpus
}

func kokoroClampFloat32(value float32, low float32, high float32) float32 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
