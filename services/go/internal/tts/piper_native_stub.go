//go:build !cgo || (!windows && !darwin && !linux) || (windows && !amd64 && !386) || (darwin && !amd64 && !arm64) || (linux && !amd64 && !arm64 && !arm && !386 && !mips && !mips64 && !mips64le && !mipsle)

package tts

import (
	"context"
	"fmt"
)

func (p *PiperProvider) synthesizeWithNative(ctx context.Context, voice resolvedPiperVoice, req Request) (Audio, error) {
	return Audio{}, fmt.Errorf("%w: native piper requires a supported CGO build with sherpa-onnx-go", ErrProviderUnavailable)
}
