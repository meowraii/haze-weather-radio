package ivr

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type wavPCM struct {
	SampleRate int
	Channels   int
	Samples    []int16
}

func readWAVPCM16(path string) (wavPCM, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return wavPCM{}, err
	}
	if len(raw) < 44 || string(raw[0:4]) != "RIFF" || string(raw[8:12]) != "WAVE" {
		return wavPCM{}, fmt.Errorf("unsupported wav container")
	}
	offset := 12
	var sampleRate int
	var channels int
	var bits int
	var data []byte
	for offset+8 <= len(raw) {
		id := string(raw[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(raw[offset+4 : offset+8]))
		offset += 8
		if offset+size > len(raw) {
			break
		}
		chunk := raw[offset : offset+size]
		switch id {
		case "fmt ":
			if len(chunk) < 16 {
				return wavPCM{}, fmt.Errorf("short wav fmt chunk")
			}
			format := binary.LittleEndian.Uint16(chunk[0:2])
			if format != 1 {
				return wavPCM{}, fmt.Errorf("unsupported wav format %d", format)
			}
			channels = int(binary.LittleEndian.Uint16(chunk[2:4]))
			sampleRate = int(binary.LittleEndian.Uint32(chunk[4:8]))
			bits = int(binary.LittleEndian.Uint16(chunk[14:16]))
		case "data":
			data = chunk
		}
		offset += size
		if offset%2 != 0 {
			offset++
		}
	}
	if sampleRate <= 0 || channels <= 0 || bits != 16 || len(data) == 0 {
		return wavPCM{}, fmt.Errorf("wav must be 16-bit PCM")
	}
	samples := make([]int16, len(data)/2)
	for index := range samples {
		samples[index] = int16(binary.LittleEndian.Uint16(data[index*2 : index*2+2]))
	}
	return wavPCM{SampleRate: sampleRate, Channels: channels, Samples: samples}, nil
}

func writePCMUFromWAV(wavPath string, pcmuPath string, targetRate int) error {
	wav, err := readWAVPCM16(wavPath)
	if err != nil {
		return err
	}
	if targetRate <= 0 {
		targetRate = 8000
	}
	mono := monoSamples(wav.Samples, wav.Channels)
	resampled := resampleLinear(mono, wav.SampleRate, targetRate)
	out := make([]byte, len(resampled))
	for index, sample := range resampled {
		out[index] = linearToULaw(sample)
	}
	return os.WriteFile(pcmuPath, out, 0o644)
}

func monoSamples(samples []int16, channels int) []int16 {
	if channels <= 1 {
		return samples
	}
	out := make([]int16, len(samples)/channels)
	for frame := range out {
		sum := 0
		for ch := 0; ch < channels; ch++ {
			sum += int(samples[frame*channels+ch])
		}
		out[frame] = int16(sum / channels)
	}
	return out
}

func resampleLinear(samples []int16, sourceRate int, targetRate int) []int16 {
	if sourceRate <= 0 || targetRate <= 0 || sourceRate == targetRate || len(samples) == 0 {
		return samples
	}
	ratio := float64(sourceRate) / float64(targetRate)
	outLen := int(math.Ceil(float64(len(samples)) / ratio))
	out := make([]int16, outLen)
	for index := range out {
		pos := float64(index) * ratio
		left := int(pos)
		if left >= len(samples)-1 {
			out[index] = samples[len(samples)-1]
			continue
		}
		frac := pos - float64(left)
		value := float64(samples[left])*(1-frac) + float64(samples[left+1])*frac
		out[index] = int16(math.Round(value))
	}
	return out
}

func writeWAV(path string, sampleRate int, samples []int16) error {
	var buf bytes.Buffer
	dataSize := uint32(len(samples) * 2)
	_, _ = buf.WriteString("RIFF")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(36)+dataSize)
	_, _ = buf.WriteString("WAVEfmt ")
	_ = binary.Write(&buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(1))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(sampleRate))
	_ = binary.Write(&buf, binary.LittleEndian, uint32(sampleRate*2))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(2))
	_ = binary.Write(&buf, binary.LittleEndian, uint16(16))
	_, _ = buf.WriteString("data")
	_ = binary.Write(&buf, binary.LittleEndian, dataSize)
	for _, sample := range samples {
		_ = binary.Write(&buf, binary.LittleEndian, sample)
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func linearToULaw(sample int16) byte {
	const bias = 0x84
	const clip = 32635
	pcm := int(sample)
	mask := 0xFF
	if pcm < 0 {
		pcm = -pcm
		mask = 0x7F
	}
	if pcm > clip {
		pcm = clip
	}
	pcm += bias
	segment := 7
	for expMask := 0x4000; (pcm&expMask) == 0 && segment > 0; expMask >>= 1 {
		segment--
	}
	mantissa := (pcm >> (segment + 3)) & 0x0F
	return byte(^((segment << 4) | mantissa) & mask)
}
