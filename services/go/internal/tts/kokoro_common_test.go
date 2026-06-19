package tts

import (
	"archive/tar"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
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

func TestDefaultKokoroOptionsIncludesDownloadManifest(t *testing.T) {
	t.Setenv("HAZE_KOKORO_ARCHIVE_URL", "")
	t.Setenv("HAZE_KOKORO_ARCHIVE_SHA256", "")
	options := defaultKokoroOptions()
	if options.ArchiveURL != defaultKokoroArchiveURL {
		t.Fatalf("ArchiveURL = %q", options.ArchiveURL)
	}
	if options.ArchiveSHA256 != defaultKokoroArchiveSHA256 {
		t.Fatalf("ArchiveSHA256 = %q", options.ArchiveSHA256)
	}
	if options.ArchiveSize != defaultKokoroArchiveSize {
		t.Fatalf("ArchiveSize = %d", options.ArchiveSize)
	}
}

func TestKokoroArchiveRelativePathStripsTopLevelDirectory(t *testing.T) {
	tests := map[string]string{
		"kokoro-en-v0_19/model.onnx":                "model.onnx",
		"./kokoro-en-v0_19/espeak-ng-data/en_dict":  "espeak-ng-data/en_dict",
		"kokoro-en-v0_19/nested/tokens.txt":         "nested/tokens.txt",
		"kokoro-en-v0_19":                           "",
		"../kokoro-en-v0_19/model.onnx":             "",
		"/tmp/kokoro-en-v0_19/model.onnx":           "",
		"kokoro-en-v0_19/../../evil/model.onnx":     "",
		"kokoro-en-v0_19/../model.onnx":             "",
		"kokoro-en-v0_19/espeak-ng-data/../tokens":  "tokens",
		"kokoro-en-v0_19/espeak-ng-data/./phonemes": "espeak-ng-data/phonemes",
	}
	for input, want := range tests {
		if got := kokoroArchiveRelativePath(input); got != want {
			t.Fatalf("kokoroArchiveRelativePath(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestKokoroFileMatchesSHA256(t *testing.T) {
	path := filepath.Join(t.TempDir(), "file")
	raw := []byte("kokoro")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	ok, err := kokoroFileMatchesSHA256(path, hex.EncodeToString(sum[:]))
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Fatal("expected sha256 match")
	}
	ok, err = kokoroFileMatchesSHA256(path, strings.Repeat("0", 64))
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Fatal("expected sha256 mismatch")
	}
}

func TestKokoroPathWithin(t *testing.T) {
	root := t.TempDir()
	if !kokoroPathWithin(root, filepath.Join(root, "model.onnx")) {
		t.Fatal("expected child path to be within root")
	}
	if kokoroPathWithin(root, filepath.Join(root, "..", "model.onnx")) {
		t.Fatal("expected parent path to be outside root")
	}
}

func TestKokoroWriteArchiveFile(t *testing.T) {
	target := filepath.Join(t.TempDir(), "nested", "file.txt")
	var archive bytes.Buffer
	writer := tar.NewWriter(&archive)
	if err := writer.WriteHeader(&tar.Header{
		Name: "kokoro-en-v0_19/nested/file.txt",
		Mode: 0o644,
		Size: int64(len("hello")),
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := writer.Write([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	reader := tar.NewReader(&archive)
	header, err := reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := kokoroWriteArchiveFile(target, reader, header.FileInfo().Mode()); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "hello" {
		t.Fatalf("file = %q", got)
	}
}
