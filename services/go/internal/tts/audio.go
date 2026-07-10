package tts

import (
	"encoding/binary"
	"fmt"
	"math"
)

// NormalizeAudio converts PCM16 audio to the requested sample rate and channel count.
// A zero target leaves that dimension unchanged.
func NormalizeAudio(audio Audio, targetSampleRate int, targetChannels int) (Audio, error) {
	if targetSampleRate <= 0 && targetChannels <= 0 {
		return audio, nil
	}

	pcm := audio.Data
	sampleRate := audio.SampleRate
	channels := audio.Channels
	if audio.Format == FormatWAV {
		var err error
		pcm, sampleRate, channels, err = pcm16WAV(audio.Data)
		if err != nil {
			return Audio{}, err
		}
	} else if audio.Format != FormatPCM16LE {
		return Audio{}, fmt.Errorf("unsupported audio format %q", audio.Format)
	}
	if sampleRate <= 0 || channels <= 0 {
		return Audio{}, fmt.Errorf("audio format is missing sample rate or channels")
	}
	if targetSampleRate <= 0 {
		targetSampleRate = sampleRate
	}
	if targetChannels <= 0 {
		targetChannels = channels
	}
	if sampleRate == targetSampleRate && channels == targetChannels {
		return audio, nil
	}

	normalized, err := normalizePCM16(pcm, sampleRate, channels, targetSampleRate, targetChannels)
	if err != nil {
		return Audio{}, err
	}
	audio.SampleRate = targetSampleRate
	audio.Channels = targetChannels
	if audio.Format == FormatWAV {
		audio.Data, err = encodePCM16WAV(normalized, targetSampleRate, targetChannels)
		if err != nil {
			return Audio{}, err
		}
	} else {
		audio.Data = normalized
	}
	return audio, nil
}

func pcm16WAV(raw []byte) ([]byte, int, int, error) {
	if len(raw) < 12 || string(raw[:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return nil, 0, 0, fmt.Errorf("not a RIFF/WAVE file")
	}
	var sampleRate int
	var channels int
	var format int
	var bits int
	var pcm []byte
	for offset := 12; offset+8 <= len(raw); {
		size := int(binary.LittleEndian.Uint32(raw[offset+4 : offset+8]))
		start := offset + 8
		end := start + size
		if size < 0 || end < start || end > len(raw) {
			return nil, 0, 0, fmt.Errorf("invalid WAV chunk length")
		}
		switch string(raw[offset : offset+4]) {
		case "fmt ":
			if size < 16 {
				return nil, 0, 0, fmt.Errorf("invalid WAV fmt chunk")
			}
			format = int(binary.LittleEndian.Uint16(raw[start : start+2]))
			channels = int(binary.LittleEndian.Uint16(raw[start+2 : start+4]))
			sampleRate = int(binary.LittleEndian.Uint32(raw[start+4 : start+8]))
			bits = int(binary.LittleEndian.Uint16(raw[start+14 : start+16]))
		case "data":
			pcm = raw[start:end]
		}
		offset = end + size%2
	}
	if format != 1 || bits != 16 || sampleRate <= 0 || channels <= 0 || len(pcm) == 0 {
		return nil, 0, 0, fmt.Errorf("WAV is not PCM s16le")
	}
	if len(pcm)%(channels*2) != 0 {
		return nil, 0, 0, fmt.Errorf("WAV contains an incomplete PCM frame")
	}
	return pcm, sampleRate, channels, nil
}

func normalizePCM16(pcm []byte, inputRate int, inputChannels int, outputRate int, outputChannels int) ([]byte, error) {
	if inputRate <= 0 || outputRate <= 0 || inputChannels <= 0 || outputChannels <= 0 {
		return nil, fmt.Errorf("invalid PCM format")
	}
	inputFrames := len(pcm) / (inputChannels * 2)
	if inputFrames == 0 {
		return nil, fmt.Errorf("PCM payload is empty")
	}
	outputFrames := int(math.Round(float64(inputFrames) * float64(outputRate) / float64(inputRate)))
	if outputFrames < 1 {
		outputFrames = 1
	}
	out := make([]byte, outputFrames*outputChannels*2)
	for outputFrame := 0; outputFrame < outputFrames; outputFrame++ {
		source := float64(outputFrame) * float64(inputRate) / float64(outputRate)
		left := int(source)
		if left >= inputFrames {
			left = inputFrames - 1
		}
		right := left + 1
		if right >= inputFrames {
			right = inputFrames - 1
		}
		fraction := source - float64(left)
		for outputChannel := 0; outputChannel < outputChannels; outputChannel++ {
			sample := interpolatedChannel(pcm, left, right, fraction, inputChannels, outputChannel, outputChannels)
			offset := (outputFrame*outputChannels + outputChannel) * 2
			binary.LittleEndian.PutUint16(out[offset:offset+2], uint16(int16(math.Round(sample))))
		}
	}
	return out, nil
}

func interpolatedChannel(pcm []byte, left int, right int, fraction float64, inputChannels int, outputChannel int, outputChannels int) float64 {
	if inputChannels == outputChannels {
		return interpolateSample(pcm, left, right, fraction, inputChannels, outputChannel)
	}
	if inputChannels == 1 {
		return interpolateSample(pcm, left, right, fraction, 1, 0)
	}
	var mixed float64
	for channel := 0; channel < inputChannels; channel++ {
		mixed += interpolateSample(pcm, left, right, fraction, inputChannels, channel)
	}
	return mixed / float64(inputChannels)
}

func interpolateSample(pcm []byte, left int, right int, fraction float64, channels int, channel int) float64 {
	leftOffset := (left*channels + channel) * 2
	rightOffset := (right*channels + channel) * 2
	a := float64(int16(binary.LittleEndian.Uint16(pcm[leftOffset : leftOffset+2])))
	b := float64(int16(binary.LittleEndian.Uint16(pcm[rightOffset : rightOffset+2])))
	return a + (b-a)*fraction
}

func encodePCM16WAV(pcm []byte, sampleRate int, channels int) ([]byte, error) {
	if len(pcm) == 0 || len(pcm)%(channels*2) != 0 || uint64(len(pcm)) > uint64(math.MaxUint32)-36 {
		return nil, fmt.Errorf("invalid PCM payload for WAV")
	}
	out := make([]byte, 44+len(pcm))
	copy(out[0:4], "RIFF")
	binary.LittleEndian.PutUint32(out[4:8], uint32(36+len(pcm)))
	copy(out[8:16], "WAVEfmt ")
	binary.LittleEndian.PutUint32(out[16:20], 16)
	binary.LittleEndian.PutUint16(out[20:22], 1)
	binary.LittleEndian.PutUint16(out[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(out[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(out[28:32], uint32(sampleRate*channels*2))
	binary.LittleEndian.PutUint16(out[32:34], uint16(channels*2))
	binary.LittleEndian.PutUint16(out[34:36], 16)
	copy(out[36:40], "data")
	binary.LittleEndian.PutUint32(out[40:44], uint32(len(pcm)))
	copy(out[44:], pcm)
	return out, nil
}
