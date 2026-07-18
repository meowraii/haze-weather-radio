//go:build opus_cgo && cgo

package webgateway

import "github.com/hraban/opus"

type libOpusFrameEncoder struct {
	encoder *opus.Encoder
}

func opusBackendAvailable() bool {
	return true
}

func newOpusFrameEncoder(sampleRate int, channels int) (opusFrameEncoder, error) {
	encoder, err := opus.NewEncoder(sampleRate, channels, opus.AppAudio)
	if err != nil {
		return nil, err
	}
	_ = encoder.SetBitrate(opusBitrateBPS)
	_ = encoder.SetComplexity(8)
	_ = encoder.SetDTX(false)
	return &libOpusFrameEncoder{encoder: encoder}, nil
}

func (e *libOpusFrameEncoder) Encode(samples []int16) ([]byte, error) {
	out := make([]byte, 1275)
	n, err := e.encoder.Encode(samples, out)
	if err != nil {
		return nil, err
	}
	return out[:n], nil
}
