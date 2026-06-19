package webgateway

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestOperatorBreakInFinishWritesAlertQueueItem(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	manager := NewOperatorBreakInManager()

	started, err := manager.start(configPath, []string{"sk-0001"}, "Desk Mic", 48000, 1, "")
	if err != nil {
		t.Fatal(err)
	}
	sessionID, _ := started["session_id"].(string)
	if sessionID == "" {
		t.Fatal("session id was not returned")
	}
	if _, err := manager.appendChunk(sessionID, []byte{0x00, 0x00, 0x10, 0x00}); err != nil {
		t.Fatal(err)
	}
	item, err := manager.finish(configPath, sessionID)
	if err != nil {
		t.Fatal(err)
	}
	if item.Type != "operator_breakin" || item.Status != "pending" || item.Priority != "operator" {
		t.Fatalf("unexpected manifest item: %#v", item)
	}
	if item.AudioBytes != 4 {
		t.Fatalf("audio bytes = %d, want 4", item.AudioBytes)
	}
	if _, err := os.Stat(resolveConfigPath(configPath, item.AudioPath)); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(resolveConfigPath(configPath, item.ManifestPath))
	if err != nil {
		t.Fatal(err)
	}
	var manifest sameQueueItem
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.ID != item.ID || manifest.Header != "Desk Mic" {
		t.Fatalf("unexpected persisted manifest: %#v", manifest)
	}
}

func TestOperatorBreakInRejectsUnsafePrerollPath(t *testing.T) {
	if _, err := operatorBreakInWAVFromRelPath(filepath.Join(t.TempDir(), "config.yaml"), "../secret.wav"); err == nil {
		t.Fatal("expected unsafe preroll path to be rejected")
	}
}
