package playlist

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkWAVPCM16OneMinuteMono(b *testing.B) {
	pcm := bytes.Repeat([]byte{0x34, 0x12}, 48_000*60)
	raw := benchmarkPCM16WAV(pcm, 48_000, 1)
	b.SetBytes(int64(len(pcm)))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		got, info, err := wavPCM16(raw)
		if err != nil {
			b.Fatal(err)
		}
		if len(got) != len(pcm) || info.SampleRate != 48_000 || info.Channels != 1 {
			b.Fatal("unexpected WAV PCM result")
		}
	}
}

func BenchmarkCombineAlertAudioTenMiBVoice(b *testing.B) {
	dir := b.TempDir()
	voice := filepath.Join(dir, "voice.pcm16le")
	voicePCM := bytes.Repeat([]byte{0x34, 0x12}, 5*1024*1024)
	if err := os.WriteFile(voice, voicePCM, 0o600); err != nil {
		b.Fatal(err)
	}
	output := filepath.Join(dir, "alert.pcm16le")
	b.SetBytes(int64(len(voicePCM)))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := combineAlertAudio(output, []byte{1, 2, 3, 4}, voice, []byte{5, 6}, 48_000, 1); err != nil {
			b.Fatal(err)
		}
	}
}

func TestWAVPCM16DataIsUsableWhileSourceBufferIsRetained(t *testing.T) {
	want := []byte{0x01, 0x00, 0x02, 0x00}
	raw := benchmarkPCM16WAV(want, 48_000, 1)

	got, info, err := wavPCM16(raw)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("PCM = %v, want %v", got, want)
	}
	if info.SampleRate != 48_000 || info.Channels != 1 {
		t.Fatalf("WAV info = %#v", info)
	}
}

func benchmarkPCM16WAV(pcm []byte, sampleRate int, channels int) []byte {
	raw := make([]byte, 44+len(pcm))
	copy(raw[0:4], "RIFF")
	binary.LittleEndian.PutUint32(raw[4:8], uint32(36+len(pcm)))
	copy(raw[8:16], "WAVEfmt ")
	binary.LittleEndian.PutUint32(raw[16:20], 16)
	binary.LittleEndian.PutUint16(raw[20:22], 1)
	binary.LittleEndian.PutUint16(raw[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(raw[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(raw[28:32], uint32(sampleRate*channels*2))
	binary.LittleEndian.PutUint16(raw[32:34], uint16(channels*2))
	binary.LittleEndian.PutUint16(raw[34:36], 16)
	copy(raw[36:40], "data")
	binary.LittleEndian.PutUint32(raw[40:44], uint32(len(pcm)))
	copy(raw[44:], pcm)
	return raw
}
