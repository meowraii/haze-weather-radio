package productrender

import (
	"encoding/json"
	"net"
	"testing"
	"time"
)

func TestBridgeRetainsCAPAlertWhenBoundedQueueIsFull(t *testing.T) {
	serverConn, clientConn := net.Pipe()
	bridge := &bridgeClient{
		conn:   clientConn,
		done:   make(chan struct{}),
		events: make(chan map[string]any, 1),
	}
	bridge.events <- map[string]any{"type": "ordinary.event"}
	go bridge.readLoop()
	defer bridge.Close()
	defer serverConn.Close()

	writeDone := make(chan error, 1)
	go func() {
		writeDone <- json.NewEncoder(serverConn).Encode(map[string]any{
			"type":    "cap.alert.received",
			"subject": "urn:test:critical",
		})
	}()

	select {
	case message := <-bridge.Events():
		if stringAt(message, "type") != "ordinary.event" {
			t.Fatalf("first event = %#v, want the prefilled ordinary event", message)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out draining the prefilled bridge event")
	}

	select {
	case message := <-bridge.Events():
		if stringAt(message, "type") != "cap.alert.received" {
			t.Fatalf("critical event = %#v", message)
		}
		if stringAt(message, "subject") != "urn:test:critical" {
			t.Fatalf("critical subject = %q", stringAt(message, "subject"))
		}
	case <-time.After(time.Second):
		t.Fatal("cap.alert.received was lost while the bounded queue was full")
	}

	select {
	case err := <-writeDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out writing the critical bridge event")
	}
}
