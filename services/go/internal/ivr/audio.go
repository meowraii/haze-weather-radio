package ivr

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/gotranspile/g722"
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
	return writePCMUFromPCM(wav, pcmuPath, targetRate)
}

func writeG722FromWAV(wavPath string, g722Path string) error {
	wav, err := readWAVPCM16(wavPath)
	if err != nil {
		return err
	}
	return writeG722FromPCM(wav, g722Path)
}

func writeTelephoneAudioFromWAV(wavPath string, pcmuPath string, g722Path string, targetRate int) error {
	wav, err := readWAVPCM16(wavPath)
	if err != nil {
		return err
	}
	if err := writePCMUFromPCM(wav, pcmuPath, targetRate); err != nil {
		return err
	}
	return writeG722FromPCM(wav, g722Path)
}

func writeTelephoneAudioFromPCMFile(pcmPath string, sampleRate int, channels int, wavPath string, pcmuPath string, g722Path string, targetRate int) error {
	raw, err := os.ReadFile(pcmPath)
	if err != nil {
		return err
	}
	pcm, err := pcm16LE(raw, sampleRate, channels)
	if err != nil {
		return err
	}
	mono := monoSamples(pcm.Samples, pcm.Channels)
	monoPCM := wavPCM{SampleRate: pcm.SampleRate, Channels: 1, Samples: mono}
	if err := writeWAV(wavPath, monoPCM.SampleRate, monoPCM.Samples); err != nil {
		return err
	}
	if err := writePCMUFromPCM(monoPCM, pcmuPath, targetRate); err != nil {
		return err
	}
	return writeG722FromPCM(monoPCM, g722Path)
}

func pcm16LE(raw []byte, sampleRate int, channels int) (wavPCM, error) {
	if sampleRate <= 0 {
		return wavPCM{}, fmt.Errorf("pcm sample rate is required")
	}
	if channels <= 0 {
		channels = 1
	}
	if len(raw) == 0 || len(raw)%2 != 0 {
		return wavPCM{}, fmt.Errorf("pcm data must be non-empty 16-bit little-endian samples")
	}
	samples := make([]int16, len(raw)/2)
	for index := range samples {
		samples[index] = int16(binary.LittleEndian.Uint16(raw[index*2 : index*2+2]))
	}
	return wavPCM{SampleRate: sampleRate, Channels: channels, Samples: samples}, nil
}

func writePCMUFromPCM(pcm wavPCM, pcmuPath string, targetRate int) error {
	if targetRate <= 0 {
		targetRate = 8000
	}
	mono := monoSamples(pcm.Samples, pcm.Channels)
	resampled := resampleLinear(mono, pcm.SampleRate, targetRate)
	out := make([]byte, len(resampled))
	for index, sample := range resampled {
		out[index] = linearToULaw(sample)
	}
	return os.WriteFile(pcmuPath, out, 0o644)
}

func writeG722FromPCM(pcm wavPCM, g722Path string) error {
	mono := monoSamples(pcm.Samples, pcm.Channels)
	resampled := resampleLinear(mono, pcm.SampleRate, sipG722SampleRate)
	out := encodeG722Samples(g722.NewEncoder(g722.Rate64000, 0), resampled)
	if len(out) == 0 {
		out = g722SilenceFrame()
	}
	return os.WriteFile(g722Path, out, 0o644)
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
	sign := 0
	if pcm < 0 {
		sign = 0x80
		pcm = -pcm
	}
	if pcm > clip {
		pcm = clip
	}
	pcm += bias
	exponent := 7
	for mask := 0x4000; exponent > 0 && pcm&mask == 0; exponent-- {
		mask >>= 1
	}
	mantissa := (pcm >> (exponent + 3)) & 0x0F
	return ^byte(sign | (exponent << 4) | mantissa)
}

func encodeG722Samples(encoder *g722.Encoder, samples []int16) []byte {
	if len(samples)%2 != 0 {
		padded := make([]int16, len(samples)+1)
		copy(padded, samples)
		samples = padded
	}
	out := make([]byte, len(samples))
	n := encoder.Encode(out, samples)
	return out[:n]
}

func g722SilenceFrame() []byte {
	return encodeG722Samples(g722.NewEncoder(g722.Rate64000, 0), make([]int16, sipG722FrameSamples))
}
