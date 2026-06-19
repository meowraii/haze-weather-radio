package tts

import (
	"context"
	"errors"
)

// AudioFormat describes the synthesized audio payload.
type AudioFormat string

const (
	// FormatWAV is a RIFF/WAVE payload.
	FormatWAV AudioFormat = "wav"
	// FormatPCM16LE is signed 16-bit little-endian PCM.
	FormatPCM16LE AudioFormat = "pcm_s16le"
)

// Voice describes one available TTS voice.
type Voice struct {
	ID       string   `json:"id"`
	Name     string   `json:"name,omitempty"`
	Provider string   `json:"provider"`
	Language []string `json:"language,omitempty"`
}

// Request describes a synthesis request.
type Request struct {
	Text            string
	VoiceID         string
	Language        string
	OutputFormat    AudioFormat
	Rate            int
	Volume          int
	SentenceSilence float64
}

// Audio is a synthesized audio result.
type Audio struct {
	Format     AudioFormat
	SampleRate int
	Channels   int
	Data       []byte
}

// Provider is implemented by concrete TTS engines.
type Provider interface {
	ID() string
	ListVoices(ctx context.Context) ([]Voice, error)
	Synthesize(ctx context.Context, req Request) (Audio, error)
}

var ErrProviderUnavailable = errors.New("tts provider unavailable")
