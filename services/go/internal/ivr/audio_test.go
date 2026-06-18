package ivr

import (
	"os"
	"path/filepath"
	"testing"
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
