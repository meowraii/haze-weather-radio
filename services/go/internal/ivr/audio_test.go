package ivr

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/gotranspile/g722"
)

func TestWritePCMUFromWAVCreatesTelephonyPayload(t *testing.T) {
	dir := t.TempDir()
	wavPath := filepath.Join(dir, "tone.wav")
	pcmuPath := filepath.Join(dir, "tone.pcmu")
	samples := make([]int16, 480)
	for i := range samples {
		samples[i] = int16((i % 64) * 400)
	}
	if err := writeWAV(wavPath, 48000, samples); err != nil {
		t.Fatalf("writeWAV: %v", err)
	}
	if err := writePCMUFromWAV(wavPath, pcmuPath, 8000); err != nil {
		t.Fatalf("writePCMUFromWAV: %v", err)
	}
	raw, err := os.ReadFile(pcmuPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(raw) == 0 {
		t.Fatal("PCMU output is empty")
	}
	if len(raw) > len(samples) {
		t.Fatalf("PCMU output was not downsampled: got %d bytes for %d samples", len(raw), len(samples))
	}
}

func TestWriteG722FromWAVCreatesWidebandPayload(t *testing.T) {
	dir := t.TempDir()
	wavPath := filepath.Join(dir, "tone.wav")
	g722Path := filepath.Join(dir, "tone.g722")
	samples := make([]int16, 960)
	for i := range samples {
		samples[i] = int16((i%64 - 32) * 400)
	}
	if err := writeWAV(wavPath, 48000, samples); err != nil {
		t.Fatalf("writeWAV: %v", err)
	}
	if err := writeG722FromWAV(wavPath, g722Path); err != nil {
		t.Fatalf("writeG722FromWAV: %v", err)
	}
	raw, err := os.ReadFile(g722Path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if len(raw) == 0 {
		t.Fatal("G.722 output is empty")
	}
	if len(raw) > len(samples) {
		t.Fatalf("G.722 output was not encoded compactly: got %d bytes for %d samples", len(raw), len(samples))
	}
}

func TestWriteTelephoneAudioFromPCMFileWritesAllCodecs(t *testing.T) {
	dir := t.TempDir()
	pcmPath := filepath.Join(dir, "tone.pcm16")
	wavPath := filepath.Join(dir, "tone.wav")
	pcmuPath := filepath.Join(dir, "tone.pcmu")
	g722Path := filepath.Join(dir, "tone.g722")
	samples := make([]int16, 960)
	for i := range samples {
		samples[i] = int16((i%96 - 48) * 300)
	}
	rawPCM := make([]byte, len(samples)*2)
	for index, sample := range samples {
		binary.LittleEndian.PutUint16(rawPCM[index*2:index*2+2], uint16(sample))
	}
	if err := os.WriteFile(pcmPath, rawPCM, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	if err := writeTelephoneAudioFromPCMFile(pcmPath, 24000, 1, wavPath, pcmuPath, g722Path, 8000); err != nil {
		t.Fatalf("writeTelephoneAudioFromPCMFile: %v", err)
	}

	wav, err := readWAVPCM16(wavPath)
	if err != nil {
		t.Fatalf("readWAVPCM16: %v", err)
	}
	if wav.SampleRate != 24000 {
		t.Fatalf("WAV sample rate = %d, want 24000", wav.SampleRate)
	}
	if len(wav.Samples) != len(samples) {
		t.Fatalf("WAV samples = %d, want %d", len(wav.Samples), len(samples))
	}
	for _, path := range []string{pcmuPath, g722Path} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile(%s): %v", path, err)
		}
		if len(raw) == 0 {
			t.Fatalf("%s output is empty", filepath.Base(path))
		}
	}
}

func TestEncodeG722SamplesPadsOddLengthInput(t *testing.T) {
	out := encodeG722Samples(g722.NewEncoder(g722.Rate64000, 0), []int16{1, -1, 1})
	if len(out) == 0 {
		t.Fatal("G.722 output is empty")
	}
}

func TestLinearToULawKnownSamples(t *testing.T) {
	cases := map[int16]byte{
		0:     0xff,
		1000:  0xce,
		-1000: 0x4e,
	}
	for sample, expected := range cases {
		if got := linearToULaw(sample); got != expected {
			t.Fatalf("linearToULaw(%d) = 0x%02x, want 0x%02x", sample, got, expected)
		}
	}
}
