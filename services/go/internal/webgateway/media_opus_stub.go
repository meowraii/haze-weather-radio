//go:build !opus_cgo || !cgo

package webgateway

import "fmt"

func opusBackendAvailable() bool {
	return false
}

func newOpusFrameEncoder(sampleRate int, channels int) (opusFrameEncoder, error) {
	return nil, fmt.Errorf("native Opus encoder is not available in this build")
}
