package tts

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// NeuralCLIProvider delegates heavier voice-cloning synthesis to an external
// model worker while keeping Haze's hot playout path provider-agnostic.
type NeuralCLIProvider struct {
	ProviderID  string
	Executable  string
	ProfilesDir string
}

func NewF5TTSProvider() *NeuralCLIProvider {
	return &NeuralCLIProvider{
		ProviderID:  "f5tts",
		Executable:  envOrDefault("HAZE_F5TTS_PYTHON", ""),
		ProfilesDir: envOrDefault("HAZE_TTS_PROFILES_DIR", filepath.Join("managed", "voices", "profiles")),
	}
}

func NewChatterboxProvider() *NeuralCLIProvider {
	return &NeuralCLIProvider{
		ProviderID:  "chatterbox",
		Executable:  envOrDefault("HAZE_CHATTERBOX_PYTHON", ""),
		ProfilesDir: envOrDefault("HAZE_TTS_PROFILES_DIR", filepath.Join("managed", "voices", "profiles")),
	}
}

func (p *NeuralCLIProvider) ID() string { return p.ProviderID }

func (p *NeuralCLIProvider) ListVoices(ctx context.Context) ([]Voice, error) {
	entries, err := os.ReadDir(p.ProfilesDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: no neural voice profiles in %s", ErrProviderUnavailable, p.ProfilesDir)
		}
		return nil, err
	}
	voices := []Voice{}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		id := entry.Name()
		if _, err := os.Stat(filepath.Join(p.ProfilesDir, id, "ref.wav")); err != nil {
			continue
		}
		voices = append(voices, Voice{ID: id, Name: id, Provider: p.ID()})
	}
	if len(voices) == 0 {
		return nil, fmt.Errorf("%w: no usable neural voice profiles in %s", ErrProviderUnavailable, p.ProfilesDir)
	}
	return voices, nil
}

func (p *NeuralCLIProvider) Synthesize(ctx context.Context, req Request) (Audio, error) {
	if strings.TrimSpace(req.Text) == "" {
		return Audio{}, errors.New("empty synthesis text")
	}
	profile, err := p.profile(req.VoiceID)
	if err != nil {
		return Audio{}, err
	}
	output, err := os.CreateTemp("", "haze-neural-*.wav")
	if err != nil {
		return Audio{}, err
	}
	outputPath := output.Name()
	_ = output.Close()
	defer os.Remove(outputPath)

	executable, args, err := p.command(profile, req.Text, outputPath)
	if err != nil {
		return Audio{}, err
	}
	cmd := exec.CommandContext(ctx, executable, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail != "" {
			return Audio{}, fmt.Errorf("%s failed: %s", p.ID(), detail)
		}
		return Audio{}, err
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		return Audio{}, err
	}
	return Audio{Format: FormatWAV, Data: data}, nil
}

type neuralProfile struct {
	ID       string
	Dir      string
	RefAudio string
	RefText  string
}

func (p *NeuralCLIProvider) profile(voiceID string) (neuralProfile, error) {
	id := safeProfileID(voiceID)
	if id == "" {
		return neuralProfile{}, fmt.Errorf("%w: %s voice_id is required", ErrProviderUnavailable, p.ID())
	}
	dir := filepath.Join(p.ProfilesDir, id)
	refAudio := filepath.Join(dir, "ref.wav")
	if _, err := os.Stat(refAudio); err != nil {
		return neuralProfile{}, fmt.Errorf("%w: missing %s", ErrProviderUnavailable, refAudio)
	}
	var rawRefText []byte
	for _, name := range []string{"ref.txt", "transcript.txt"} {
		if raw, err := os.ReadFile(filepath.Join(dir, name)); err == nil {
			rawRefText = raw
			break
		}
	}
	return neuralProfile{ID: id, Dir: dir, RefAudio: refAudio, RefText: strings.TrimSpace(string(rawRefText))}, nil
}

func (p *NeuralCLIProvider) command(profile neuralProfile, text string, outputPath string) (string, []string, error) {
	switch p.ID() {
	case "f5tts":
		script, err := neuralScript("HAZE_F5TTS_SCRIPT", "f5_infer.py")
		if err != nil {
			return "", nil, err
		}
		executable, err := pythonExecutable(strings.TrimSpace(p.Executable))
		if err != nil {
			return "", nil, err
		}
		args := []string{
			script,
			"--ref-audio", profile.RefAudio,
			"--ref-text", profile.RefText,
			"--text", text,
			"--output", outputPath,
		}
		if model := strings.TrimSpace(os.Getenv("HAZE_F5TTS_MODEL")); model != "" {
			args = append(args, "--model", model)
		}
		return executable, args, nil
	case "chatterbox":
		script, err := neuralScript("HAZE_CHATTERBOX_SCRIPT", "chatterbox_infer.py")
		if err != nil {
			return "", nil, err
		}
		executable, err := pythonExecutable(strings.TrimSpace(p.Executable))
		if err != nil {
			return "", nil, err
		}
		return executable, []string{
			script,
			"--audio-prompt", profile.RefAudio,
			"--text", text,
			"--output", outputPath,
		}, nil
	default:
		return "", nil, fmt.Errorf("%w: unknown neural provider %s", ErrProviderUnavailable, p.ID())
	}
}

func neuralScript(envKey string, filename string) (string, error) {
	if configured := strings.TrimSpace(os.Getenv(envKey)); configured != "" {
		if _, err := os.Stat(configured); err == nil {
			return filepath.Clean(configured), nil
		}
		return "", fmt.Errorf("%w: neural TTS script %q", ErrProviderUnavailable, configured)
	}
	candidates := []string{
		filepath.Join("managed", "scripts", filename),
		filepath.Join("scripts", "tts", filename),
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return filepath.Clean(candidate), nil
		}
	}
	return "", fmt.Errorf("%w: %s not found", ErrProviderUnavailable, filename)
}

func pythonExecutable(configured string) (string, error) {
	if strings.TrimSpace(configured) != "" {
		if path, err := exec.LookPath(configured); err == nil {
			return path, nil
		}
		return "", fmt.Errorf("%w: python executable %q", ErrProviderUnavailable, configured)
	}
	for _, candidate := range []string{
		filepath.Join("managed", "venvs", "neural-tts", "Scripts", "python.exe"),
		filepath.Join("managed", "venvs", "neural-tts", "bin", "python"),
	} {
		if _, err := os.Stat(candidate); err == nil {
			return filepath.Clean(candidate), nil
		}
	}
	for _, candidate := range []string{"py", "python", "python3"} {
		if path, err := exec.LookPath(candidate); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("%w: python for chatterbox", ErrProviderUnavailable)
}

func safeProfileID(value string) string {
	var builder strings.Builder
	for _, ch := range strings.TrimSpace(value) {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '-' || ch == '_' || ch == '.' {
			builder.WriteRune(ch)
		}
	}
	return builder.String()
}

func envOrDefault(key string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}
