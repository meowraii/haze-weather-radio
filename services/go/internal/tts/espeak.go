package tts

import (
	"bytes"
	"context"
	"errors"
	"os/exec"
	"strconv"
	"strings"
)

// ESpeakProvider uses an espeak-ng executable to synthesize WAV audio.
type ESpeakProvider struct {
	Executable string
}

// NewESpeakProvider creates an eSpeak NG provider.
func NewESpeakProvider(executable string) *ESpeakProvider {
	if strings.TrimSpace(executable) == "" {
		executable = "espeak-ng"
	}
	return &ESpeakProvider{Executable: executable}
}

func (p *ESpeakProvider) ID() string { return "espeak" }

func (p *ESpeakProvider) ListVoices(ctx context.Context) ([]Voice, error) {
	output, err := exec.CommandContext(ctx, p.Executable, "--voices").Output()
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(output), "\n")
	voices := make([]Voice, 0, len(lines))
	for _, line := range lines {
		fields := strings.Fields(line)
		if len(fields) < 4 || fields[0] == "Pty" {
			continue
		}
		lang := fields[1]
		name := fields[3]
		voices = append(voices, Voice{
			ID:       name,
			Name:     name,
			Provider: p.ID(),
			Language: []string{NormalizeLanguage(lang)},
		})
	}
	return voices, nil
}

func (p *ESpeakProvider) Synthesize(ctx context.Context, req Request) (Audio, error) {
	if strings.TrimSpace(req.Text) == "" {
		return Audio{}, errors.New("empty synthesis text")
	}
	args := []string{"--stdout"}
	if req.VoiceID != "" {
		args = append(args, "-v", req.VoiceID)
	} else if req.Language != "" {
		args = append(args, "-v", req.Language)
	}
	if req.Rate > 0 {
		args = append(args, "-s", strconv.Itoa(req.Rate))
	}
	args = append(args, req.Text)

	cmd := exec.CommandContext(ctx, p.Executable, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		if stderr.Len() > 0 {
			return Audio{}, errors.New(strings.TrimSpace(stderr.String()))
		}
		return Audio{}, err
	}
	return Audio{
		Format: FormatWAV,
		Data:   output,
	}, nil
}
