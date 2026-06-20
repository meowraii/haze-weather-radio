package webgateway

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBulletinsImportSaveExportXML(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "storage:\n  sqlite:\n    enabled: true\n")

	imported, err := importBulletinsPayload(configPath, map[string]any{"xml": `<?xml version="1.0"?>
<bulletins>
  <bulletin id="test" enabled="true">
    <title>Road Closure</title>
    <active start="2026-06-20T12:00" expire="2026-06-21T12:00" />
    <schedule mode="hours" end_of_cycle="true"><hours><hour>9</hour><hour>21</hour></hours></schedule>
    <target><feed id="sk-0001" /></target>
    <content type="tts"><lang code="en-CA">Highway 1 is closed.</lang></content>
  </bulletin>
</bulletins>`})
	if err != nil {
		t.Fatalf("import bulletins: %v", err)
	}
	rows, _ := imported["bulletins"].([]map[string]any)
	if len(rows) != 1 {
		t.Fatalf("expected one bulletin row, got %#v", imported["bulletins"])
	}
	exported, err := exportBulletinsPayload(configPath, map[string]any{"id": "test"})
	if err != nil {
		t.Fatalf("export bulletin: %v", err)
	}
	xmlText, _ := exported["xml"].(string)
	if !strings.Contains(xmlText, `<bulletin id="test" enabled="true">`) || !strings.Contains(xmlText, "Highway 1 is closed.") {
		t.Fatalf("exported XML missing bulletin fields:\n%s", xmlText)
	}
}

func TestBulletinAudioUploadRejectsUnsafeType(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "")

	_, err := uploadBulletinAudio(configPath, map[string]any{
		"filename":     "bad.exe",
		"audio_base64": base64.StdEncoding.EncodeToString([]byte("not really audio")),
	})
	if err == nil {
		t.Fatal("expected unsupported extension error")
	}
	if _, statErr := os.Stat(filepath.Join(dir, "managed", "audio", "bulletins", "bad.exe")); !os.IsNotExist(statErr) {
		t.Fatalf("unsafe upload was written: %v", statErr)
	}
}
