package webgateway

import (
	"bufio"
	"encoding/json"
	"net"
	"path/filepath"
	"testing"
)

func TestOperatorBreakInPublishesLiveEvents(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	t.Setenv("HAZE_HOST_BRIDGE_ADDR", listener.Addr().String())
	received := make(chan map[string]any, 4)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		for scanner.Scan() {
			var event map[string]any
			if err := json.Unmarshal(scanner.Bytes(), &event); err == nil {
				received <- event
			}
		}
	}()
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
	result, err := manager.finish(sessionID)
	if err != nil {
		t.Fatal(err)
	}
	if live, _ := result["live"].(bool); !live {
		t.Fatalf("finish result did not report live mode: %#v", result)
	}
	wantTypes := []string{"operator.breakin.start", "operator.breakin.chunk", "operator.breakin.finish"}
	for _, want := range wantTypes {
		event := <-received
		if event["type"] != want {
			t.Fatalf("event type = %v, want %s", event["type"], want)
		}
	}
}

func TestOperatorBreakInRejectsUnsafePrerollPath(t *testing.T) {
	if _, err := operatorBreakInWAVFromRelPath(filepath.Join(t.TempDir(), "config.yaml"), "../secret.wav"); err == nil {
		t.Fatal("expected unsafe preroll path to be rejected")
	}
}
