package webgateway

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sync"
	"time"
)

const (
	tlsProtocolQueueSize   = 128
	tlsProtocolSniffLimit  = 128
	defaultTLSSniffTimeout = 10 * time.Second
)

// ListenAndServeTLS serves HTTPS and redirects plaintext HTTP received on the
// same listener to HTTPS.
func (t *TLSRuntime) ListenAndServeTLS(server *http.Server) error {
	if server == nil {
		return fmt.Errorf("HTTPS server is required")
	}
	addr := server.Addr
	if addr == "" {
		addr = ":https"
	}
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	return t.ServeTLS(server, listener)
}

// ServeTLS serves HTTPS and plaintext HTTP redirects on an existing listener.
func (t *TLSRuntime) ServeTLS(server *http.Server, listener net.Listener) error {
	if t == nil || !t.Enabled {
		return fmt.Errorf("TLS runtime is not enabled")
	}
	if server == nil {
		return fmt.Errorf("HTTPS server is required")
	}
	if listener == nil {
		return fmt.Errorf("HTTPS listener is required")
	}

	sniffTimeout := server.ReadHeaderTimeout
	if sniffTimeout <= 0 {
		sniffTimeout = defaultTLSSniffTimeout
	}
	protocols := newTLSProtocolMux(listener, sniffTimeout)
	defer protocols.Close()

	redirectServer := &http.Server{
		Handler:           t.HTTPSRedirectHandler(listener.Addr().String()),
		ReadTimeout:       server.ReadTimeout,
		ReadHeaderTimeout: server.ReadHeaderTimeout,
		WriteTimeout:      server.WriteTimeout,
		IdleTimeout:       server.IdleTimeout,
		MaxHeaderBytes:    server.MaxHeaderBytes,
	}
	redirectErrors := make(chan error, 1)
	go func() {
		err := redirectServer.Serve(protocols.httpListener())
		if err != nil && !errors.Is(err, http.ErrServerClosed) && !errors.Is(err, net.ErrClosed) {
			redirectErrors <- fmt.Errorf("HTTP-to-HTTPS redirect listener: %w", err)
		}
		close(redirectErrors)
	}()
	defer redirectServer.Close()

	certFile := t.CertFile
	keyFile := t.KeyFile
	if t.Mode == tlsModeACME {
		certFile = ""
		keyFile = ""
	}
	serveErr := server.ServeTLS(protocols.tlsListener(), certFile, keyFile)
	_ = protocols.Close()
	if redirectErr := <-redirectErrors; redirectErr != nil {
		return redirectErr
	}
	return serveErr
}

type tlsProtocolMux struct {
	listener net.Listener
	timeout  time.Duration

	tlsConnections  chan net.Conn
	httpConnections chan net.Conn
	sniffSlots      chan struct{}
	done            chan struct{}

	closeOnce sync.Once
	wait      sync.WaitGroup

	pendingMu sync.Mutex
	pending   map[net.Conn]struct{}
}

func newTLSProtocolMux(listener net.Listener, timeout time.Duration) *tlsProtocolMux {
	mux := &tlsProtocolMux{
		listener:        listener,
		timeout:         timeout,
		tlsConnections:  make(chan net.Conn, tlsProtocolQueueSize),
		httpConnections: make(chan net.Conn, tlsProtocolQueueSize),
		sniffSlots:      make(chan struct{}, tlsProtocolSniffLimit),
		done:            make(chan struct{}),
		pending:         make(map[net.Conn]struct{}),
	}
	mux.wait.Add(1)
	go mux.accept()
	return mux
}

func (m *tlsProtocolMux) accept() {
	defer m.wait.Done()
	for {
		connection, err := m.listener.Accept()
		if err != nil {
			m.signalClosed()
			return
		}
		if !m.trackPending(connection) {
			_ = connection.Close()
			return
		}
		select {
		case m.sniffSlots <- struct{}{}:
			m.wait.Add(1)
			go m.classify(connection)
		default:
			m.untrackPending(connection)
			_ = connection.Close()
		}
	}
}

func (m *tlsProtocolMux) classify(connection net.Conn) {
	defer m.wait.Done()
	defer func() { <-m.sniffSlots }()
	defer m.untrackPending(connection)

	if err := connection.SetReadDeadline(time.Now().Add(m.timeout)); err != nil {
		_ = connection.Close()
		return
	}
	var prefix [1]byte
	if _, err := io.ReadFull(connection, prefix[:]); err != nil {
		_ = connection.Close()
		return
	}
	if err := connection.SetReadDeadline(time.Time{}); err != nil {
		_ = connection.Close()
		return
	}

	replayed := &prefixedConnection{
		Conn:   connection,
		prefix: prefix[:],
	}
	queue := m.httpConnections
	if prefix[0] == 0x16 {
		queue = m.tlsConnections
	}
	select {
	case queue <- replayed:
	case <-m.done:
		_ = connection.Close()
	default:
		_ = connection.Close()
	}
}

func (m *tlsProtocolMux) trackPending(connection net.Conn) bool {
	m.pendingMu.Lock()
	defer m.pendingMu.Unlock()
	select {
	case <-m.done:
		return false
	default:
		m.pending[connection] = struct{}{}
		return true
	}
}

func (m *tlsProtocolMux) untrackPending(connection net.Conn) {
	m.pendingMu.Lock()
	delete(m.pending, connection)
	m.pendingMu.Unlock()
}

func (m *tlsProtocolMux) signalClosed() {
	m.closeOnce.Do(func() {
		close(m.done)
		_ = m.listener.Close()

		m.pendingMu.Lock()
		for connection := range m.pending {
			_ = connection.Close()
		}
		m.pendingMu.Unlock()
	})
}

func (m *tlsProtocolMux) Close() error {
	m.signalClosed()
	m.wait.Wait()
	drainConnections(m.tlsConnections)
	drainConnections(m.httpConnections)
	return nil
}

func (m *tlsProtocolMux) tlsListener() net.Listener {
	return &routedListener{
		parent:      m,
		connections: m.tlsConnections,
	}
}

func (m *tlsProtocolMux) httpListener() net.Listener {
	return &routedListener{
		parent:      m,
		connections: m.httpConnections,
	}
}

type routedListener struct {
	parent      *tlsProtocolMux
	connections <-chan net.Conn
}

func (l *routedListener) Accept() (net.Conn, error) {
	for {
		select {
		case <-l.parent.done:
			return nil, net.ErrClosed
		case connection := <-l.connections:
			select {
			case <-l.parent.done:
				_ = connection.Close()
				return nil, net.ErrClosed
			default:
				return connection, nil
			}
		}
	}
}

func (l *routedListener) Close() error {
	return l.parent.Close()
}

func (l *routedListener) Addr() net.Addr {
	return l.parent.listener.Addr()
}

type prefixedConnection struct {
	net.Conn
	prefix []byte
}

func (c *prefixedConnection) Read(buffer []byte) (int, error) {
	if len(c.prefix) == 0 {
		return c.Conn.Read(buffer)
	}
	count := copy(buffer, c.prefix)
	c.prefix = c.prefix[count:]
	return count, nil
}

func drainConnections(connections <-chan net.Conn) {
	for {
		select {
		case connection := <-connections:
			_ = connection.Close()
		default:
			return
		}
	}
}
