package webgateway

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestDictionaryPayloadLoadsManagedDictionary(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "version: test\n")
	mustWrite(t, filepath.Join(dir, "managed", "dictionary.json"), `{
  "en-*": {
    "AAFC": "A A F C",
    "kPa": "kilopascals"
  }
}`)

	payload, err := loadDictionaryPayload(configPath)
	if err != nil {
		t.Fatalf("loadDictionaryPayload() error = %v", err)
	}
	summary := payload["summary"].(map[string]any)
	if summary["group_count"] != 1 {
		t.Fatalf("group_count = %v", summary["group_count"])
	}
	if summary["entry_count"] != 2 {
		t.Fatalf("entry_count = %v", summary["entry_count"])
	}
}

func TestWriteDictionaryPayloadNormalizesAndPersists(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "version: test\n")

	payload, err := writeDictionaryPayload(configPath, map[string]any{
		"groups": map[string]any{
			" en-CA ": map[string]any{
				" AAFC ": " A A F C ",
				"drop":   "   ",
			},
		},
	})
	if err != nil {
		t.Fatalf("writeDictionaryPayload() error = %v", err)
	}
	summary := payload["summary"].(map[string]any)
	if summary["entry_count"] != 1 {
		t.Fatalf("entry_count = %v", summary["entry_count"])
	}

	raw, err := os.ReadFile(filepath.Join(dir, "managed", "dictionary.json"))
	if err != nil {
		t.Fatalf("read dictionary: %v", err)
	}
	var stored map[string]map[string]string
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatalf("stored dictionary invalid json: %v", err)
	}
	if stored["en-CA"]["AAFC"] != "A A F C" {
		t.Fatalf("stored entry = %q", stored["en-CA"]["AAFC"])
	}
	if _, ok := stored["en-CA"]["drop"]; ok {
		t.Fatal("blank replacement should be removed")
	}
}

func TestWriteDictionaryPayloadRejectsInvalidGroup(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "version: test\n")

	_, err := writeDictionaryPayload(configPath, map[string]any{
		"groups": map[string]any{
			"en-*": "not an object",
		},
	})
	if err == nil {
		t.Fatal("expected invalid group error")
	}
}
