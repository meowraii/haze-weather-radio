package tts

import (
	"encoding/binary"
	"testing"
)

func TestNormalizeAudioResamplesPCM16WAV(t *testing.T) {
	pcm := make([]byte, 4)
	binary.LittleEndian.PutUint16(pcm[0:2], uint16(int16(0)))
	binary.LittleEndian.PutUint16(pcm[2:4], uint16(int16(1000)))
	wav, err := encodePCM16WAV(pcm, 22050, 1)
	if err != nil {
		t.Fatal(err)
	}

	normalized, err := NormalizeAudio(Audio{
		Format:     FormatWAV,
		SampleRate: 22050,
		Channels:   1,
		Data:       wav,
	}, 44100, 1)
	if err != nil {
		t.Fatal(err)
	}
	gotPCM, rate, channels, err := pcm16WAV(normalized.Data)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 44100 || channels != 1 || len(gotPCM) != 8 {
		t.Fatalf("normalized format rate=%d channels=%d bytes=%d", rate, channels, len(gotPCM))
	}
}

func TestNormalizeAudioMixesStereoToMono(t *testing.T) {
	pcm := make([]byte, 4)
	binary.LittleEndian.PutUint16(pcm[0:2], uint16(int16(1000)))
	binary.LittleEndian.PutUint16(pcm[2:4], 0xfe0c)

	normalized, err := NormalizeAudio(Audio{
		Format:     FormatPCM16LE,
		SampleRate: 48000,
		Channels:   2,
		Data:       pcm,
	}, 48000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := int16(binary.LittleEndian.Uint16(normalized.Data)); got != 250 {
		t.Fatalf("mixed sample = %d, want 250", got)
	}
}
