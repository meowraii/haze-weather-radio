package events

import (
	"bufio"
	"encoding/json"
	"net"
	"testing"
	"time"
)

func TestHostBridgePublisherPublishesJSONL(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer func() {
		_ = listener.Close()
	}()

	received := make(chan Event, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer func() {
			_ = conn.Close()
		}()
		line, err := bufio.NewReader(conn).ReadBytes('\n')
		if err != nil {
			return
		}
		var event Event
		if err := json.Unmarshal(line, &event); err != nil {
			return
		}
		received <- event
	}()

	publisher := NewHostBridgePublisher(listener.Addr().String())
	if err := publisher.Publish(Event{
		Type:    "cap.alert.received",
		Source:  "test",
		Subject: "abc",
	}); err != nil {
		t.Fatalf("publish: %v", err)
	}

	event := <-received
	if event.Type != "cap.alert.received" {
		t.Fatalf("type = %q", event.Type)
	}
	if event.Timestamp.IsZero() {
		t.Fatal("timestamp was not populated")
	}
}

func TestHostBridgePublisherWriteDeadlineBoundsBlockedConnection(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	addr := listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatalf("close listener: %v", err)
	}

	serverConn, clientConn := net.Pipe()
	defer func() {
		_ = serverConn.Close()
	}()

	publisher := NewHostBridgePublisher(addr)
	publisher.conn = clientConn
	publisher.dialTimeout = 25 * time.Millisecond
	publisher.writeTimeout = 25 * time.Millisecond

	started := time.Now()
	err = publisher.Publish(Event{Type: "data.ready", Source: "test"})
	if err == nil {
		t.Fatal("publish unexpectedly succeeded")
	}
	if elapsed := time.Since(started); elapsed > time.Second {
		t.Fatalf("publish blocked for %s", elapsed)
	}
}
