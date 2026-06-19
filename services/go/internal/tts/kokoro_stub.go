//go:build !cgo || (!windows && !darwin && !linux) || (windows && !amd64 && !386) || (darwin && !amd64 && !arm64) || (linux && !amd64 && !arm64 && !arm && !386 && !mips && !mips64 && !mips64le && !mipsle)

package tts

import (
	"context"
	"fmt"
)

// KokoroProvider is present in non-native builds so provider selection fails cleanly.
type KokoroProvider struct {
	options kokoroOptions
}

// NewKokoroProvider creates a Kokoro provider placeholder when the native binding is unavailable.
func NewKokoroProvider() *KokoroProvider {
	return &KokoroProvider{options: defaultKokoroOptions()}
}

func (p *KokoroProvider) ID() string { return kokoroProviderID }

func (p *KokoroProvider) ListVoices(ctx context.Context) ([]Voice, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return nil, kokoroNativeUnavailable()
}

func (p *KokoroProvider) Synthesize(ctx context.Context, req Request) (Audio, error) {
	if err := ctx.Err(); err != nil {
		return Audio{}, err
	}
	return Audio{}, kokoroNativeUnavailable()
}

func kokoroNativeUnavailable() error {
	return fmt.Errorf("%w: kokoro native runtime requires a supported CGO build with sherpa-onnx-go", ErrProviderUnavailable)
}
