package tts

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const kokoroProviderID = "kokoro"

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
