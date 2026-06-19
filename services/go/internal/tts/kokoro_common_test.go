package tts

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestKokoroSpeakerID(t *testing.T) {
	tests := []struct {
		name    string
		voiceID string
		want    int
		wantErr bool
	}{
		{name: "empty default", voiceID: "", want: 0},
		{name: "numeric", voiceID: "12", want: 12},
		{name: "sid prefix", voiceID: "sid:4", want: 4},
		{name: "kokoro dash prefix", voiceID: "kokoro-7", want: 7},
		{name: "invalid", voiceID: "af_sky", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kokoroSpeakerID(tt.voiceID)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("speaker id = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestKokoroPCM16LEClampsSamples(t *testing.T) {
	got := kokoroPCM16LE([]float32{-2, -1, -0.5, 0, 0.5, 1, 2})
	want := []byte{
		0x00, 0x80,
		0x00, 0x80,
		0x00, 0xc0,
		0x00, 0x00,
		0xff, 0x3f,
		0xff, 0x7f,
		0xff, 0x7f,
	}
	if string(got) != string(want) {
		t.Fatalf("pcm = %v, want %v", got, want)
	}
}

func TestKokoroOptionsValidate(t *testing.T) {
	dir := t.TempDir()
	dataDir := filepath.Join(dir, "espeak-ng-data")
	if err := os.Mkdir(dataDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"model.onnx", "voices.bin", "tokens.txt"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	options := kokoroOptions{
		ModelPath:  filepath.Join(dir, "model.onnx"),
		VoicesPath: filepath.Join(dir, "voices.bin"),
		TokensPath: filepath.Join(dir, "tokens.txt"),
		DataDir:    dataDir,
	}
	if err := options.validate(); err != nil {
		t.Fatal(err)
	}

	options.TokensPath = filepath.Join(dir, "missing.txt")
	err := options.validate()
	if !errors.Is(err, ErrProviderUnavailable) {
		t.Fatalf("err = %v, want ErrProviderUnavailable", err)
	}
}
