package webgateway

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestTTSPayloadRoundTripReadersXML(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, `services:
  go:
    tts:
      readers: managed/configs/readers.xml
`)

	payload, err := saveTTSPayload(configPath, map[string]any{
		"readers": []any{
			map[string]any{
				"id":       "00",
				"provider": "sapi",
				"gender":   "female",
				"language": "en_CA",
				"voice_id": "Microsoft Linda",
			},
			map[string]any{
				"id":       "01",
				"provider": "piper",
				"gender":   "male",
				"language": "en-US",
				"voice_id": "en_US-hfc_male-medium",
			},
		},
	})
	if err != nil {
		t.Fatalf("saveTTSPayload: %v", err)
	}
	if payload["configured"] != "managed/configs/readers.xml" {
		t.Fatalf("configured path = %#v", payload["configured"])
	}

	readers, ok := payload["readers"].([]map[string]any)
	if !ok || len(readers) != 2 {
		t.Fatalf("readers payload = %#v", payload["readers"])
	}
	if readers[0]["provider"] != "sapi5" || readers[0]["language"] != "en-ca" {
		t.Fatalf("reader normalization failed: %#v", readers[0])
	}

	raw, err := os.ReadFile(filepath.Join(dir, "managed", "configs", "readers.xml"))
	if err != nil {
		t.Fatalf("read readers.xml: %v", err)
	}
	text := string(raw)
	if !containsAll(text, "<Readers>", `provider="sapi5"`, "<voice_id>Microsoft Linda</voice_id>") {
		t.Fatalf("unexpected XML:\n%s", text)
	}
}

func TestTTSSaveRejectsDuplicateReaders(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, `services:
  go:
    tts:
      readers: managed/configs/readers.xml
`)
	_, err := saveTTSPayload(configPath, map[string]any{
		"readers": []any{
			map[string]any{"id": "00", "provider": "piper"},
			map[string]any{"id": "00", "provider": "kokoro"},
		},
	})
	if err == nil {
		t.Fatal("expected duplicate reader error")
	}
}

func containsAll(text string, needles ...string) bool {
	for _, needle := range needles {
		if !strings.Contains(text, needle) {
			return false
		}
	}
	return true
}
