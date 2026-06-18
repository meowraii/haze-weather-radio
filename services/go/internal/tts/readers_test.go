package tts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadReadersAcceptsVoiceIDAndLegacyPath(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "readers.xml")
	raw := `<Readers>
  <reader id="tom" provider="sapi5">
    <gender>male</gender>
    <language>en-CA</language>
    <voice_id>Nuance Tom</voice_id>
  </reader>
  <reader id="ava" provider="sapi5">
    <gender>female</gender>
    <language>en_CA</language>
    <path>Nuance Ava</path>
  </reader>
</Readers>`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}

	readers, err := LoadReaders(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(readers) != 2 {
		t.Fatalf("len = %d", len(readers))
	}
	if readers[0].Provider != "sapi5" || readers[0].VoiceID != "Nuance Tom" {
		t.Fatalf("unexpected reader: %+v", readers[0])
	}
	if readers[1].Provider != "sapi5" || readers[1].VoiceID != "Nuance Ava" || readers[1].Language != "en-ca" {
		t.Fatalf("unexpected reader: %+v", readers[1])
	}
}

func TestLoadReadersAcceptsAutoReaderWithoutVoiceID(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "readers.xml")
	raw := `<Readers>
  <reader id="00" provider="auto">
    <gender>male</gender>
    <language>en-CA</language>
  </reader>
</Readers>`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}

	readers, err := LoadReaders(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(readers) != 1 {
		t.Fatalf("len = %d", len(readers))
	}
	if readers[0].Provider != "auto" || readers[0].VoiceID != "" {
		t.Fatalf("unexpected reader: %+v", readers[0])
	}
}

func TestNormalizeProviderFastAliases(t *testing.T) {
	for _, provider := range []string{"fast", "ivr-fast", "low_latency"} {
		if got := NormalizeProvider(provider); got != "fast" {
			t.Fatalf("NormalizeProvider(%q) = %q", provider, got)
		}
	}
}

func TestSelectReaderPrefersExplicitReaderID(t *testing.T) {
	readers := []Reader{
		{ID: "00", Provider: "auto", Gender: "male", Language: "en-ca"},
		{ID: "wxr_tom", Provider: "sapi5", Gender: "male", Language: "en-ca", VoiceID: "Nuance Tom"},
	}

	reader, ok := SelectReader(readers, "wxr_tom", "en-CA", "")
	if !ok {
		t.Fatal("reader not found")
	}
	if reader.Provider != "sapi5" || reader.VoiceID != "Nuance Tom" {
		t.Fatalf("reader = %+v", reader)
	}
}

func TestSelectReaderFallsBackByLanguageAndGender(t *testing.T) {
	readers := []Reader{
		{ID: "male", Provider: "auto", Gender: "male", Language: "en-ca"},
		{ID: "female", Provider: "auto", Gender: "female", Language: "en"},
	}

	reader, ok := SelectReader(readers, "female", "en-CA", "")
	if !ok {
		t.Fatal("reader not found")
	}
	if reader.ID != "female" {
		t.Fatalf("reader = %+v", reader)
	}
}
