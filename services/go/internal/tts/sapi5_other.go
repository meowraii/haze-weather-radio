//go:build !windows

package tts

import (
	"context"
	"fmt"
)

// SAPI5Provider is unavailable on non-Windows platforms.
type SAPI5Provider struct{}

// NewSAPI5Provider creates an unavailable SAPI5 provider.
func NewSAPI5Provider() *SAPI5Provider {
	return &SAPI5Provider{}
}

func (p *SAPI5Provider) ID() string { return "sapi5" }

func (p *SAPI5Provider) ListVoices(ctx context.Context) ([]Voice, error) {
	_ = ctx
	return nil, fmt.Errorf("%w: sapi5 requires Windows", ErrProviderUnavailable)
}

func (p *SAPI5Provider) Synthesize(ctx context.Context, req Request) (Audio, error) {
	_ = ctx
	_ = req
	return Audio{}, fmt.Errorf("%w: sapi5 requires Windows", ErrProviderUnavailable)
}
