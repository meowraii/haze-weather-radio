package events

import (
	"encoding/json"
	"io"
	"net"
	"sync"
	"time"
)

// HostBridgePublisher emits events to the Haze loopback event bridge.
type HostBridgePublisher struct {
	addr string
	mu   sync.Mutex
	conn net.Conn
}

// NewHostBridgePublisher creates a publisher for the Haze event bridge.
func NewHostBridgePublisher(addr string) *HostBridgePublisher {
	return &HostBridgePublisher{addr: addr}
}

// Close releases the current bridge connection.
func (p *HostBridgePublisher) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.conn == nil {
		return nil
	}
	err := p.conn.Close()
	p.conn = nil
	return err
}

// Publish writes an event to the host bridge, reconnecting once if needed.
func (p *HostBridgePublisher) Publish(event Event) error {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}
	raw, err := json.Marshal(event)
	if err != nil {
		return err
	}
	raw = append(raw, '\n')

	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.write(raw); err == nil {
		return nil
	}
	p.closeLocked()
	return p.write(raw)
}

func (p *HostBridgePublisher) write(raw []byte) error {
	if p.conn == nil {
		conn, err := net.DialTimeout("tcp", p.addr, 3*time.Second)
		if err != nil {
			return err
		}
		p.conn = conn
		go drainBridge(conn)
	}
	_, err := p.conn.Write(raw)
	return err
}

func (p *HostBridgePublisher) closeLocked() {
	if p.conn != nil {
		_ = p.conn.Close()
		p.conn = nil
	}
}

func drainBridge(conn net.Conn) {
	_, _ = io.Copy(io.Discard, conn)
}
