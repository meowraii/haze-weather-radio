package webgateway

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestTLSStatusPromptsForActualDomain(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "http://panel.example.com/admin", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Host = "panel.example.com"

	status := tlsStatus(Config{}, request)

	if status["actual_domain"] != true {
		t.Fatalf("actual_domain = %v", status["actual_domain"])
	}
	if status["needs_setup"] != true {
		t.Fatalf("needs_setup = %v", status["needs_setup"])
	}
	if status["https"] != false {
		t.Fatalf("https = %v", status["https"])
	}
}

func TestTLSStatusDoesNotPromptForLocalhost(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "http://localhost:6444/admin", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Host = "localhost:6444"

	status := tlsStatus(Config{}, request)

	if status["actual_domain"] != false {
		t.Fatalf("actual_domain = %v", status["actual_domain"])
	}
	if status["needs_setup"] != false {
		t.Fatalf("needs_setup = %v", status["needs_setup"])
	}
}

func TestPublicTLSNoticeTreatsLocalhostSubdomainAsActualDomain(t *testing.T) {
	request, err := http.NewRequest(http.MethodGet, "http://localhost.example.com/", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Host = "localhost.example.com"

	status := tlsStatePayload(Config{}, request)

	if status["actual_domain"] != true {
		t.Fatalf("actual_domain = %v", status["actual_domain"])
	}
	if status["needs_setup"] != true {
		t.Fatalf("needs_setup = %v", status["needs_setup"])
	}
}

func TestACMERuntimeRequiresWhitelistedDomains(t *testing.T) {
	var config Config
	config.Webpanel.TLS.Enabled = true
	config.Webpanel.TLS.Mode = "acme"

	if _, err := NewTLSRuntime(config, "config.yaml"); err == nil {
		t.Fatal("expected missing domain error")
	}
}

func TestACMERuntimeNormalizesDomains(t *testing.T) {
	var config Config
	config.Webpanel.TLS.Enabled = true
	config.Webpanel.TLS.Mode = "letsencrypt"
	config.Webpanel.TLS.Domains = []string{"Panel.Example.COM.", "panel.example.com"}
	config.Webpanel.TLS.CacheDir = t.TempDir()

	runtime, err := NewTLSRuntime(config, "config.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if runtime.Mode != tlsModeACME {
		t.Fatalf("mode = %s", runtime.Mode)
	}
	if len(runtime.Domains) != 1 || runtime.Domains[0] != "panel.example.com" {
		t.Fatalf("domains = %#v", runtime.Domains)
	}
	if runtime.Manager == nil {
		t.Fatal("ACME manager was nil")
	}
}

func TestHTTPSRedirectHandler(t *testing.T) {
	tests := []struct {
		name         string
		host         string
		localAddress string
		httpsAddress string
		domains      []string
		target       string
		status       int
	}{
		{
			name:         "configured domain and nonstandard port",
			host:         "panel.example.com:80",
			localAddress: "192.168.50.10:6444",
			httpsAddress: "0.0.0.0:6444",
			domains:      []string{"panel.example.com"},
			target:       "https://panel.example.com:6444/listen/feed%201?format=opus",
			status:       http.StatusPermanentRedirect,
		},
		{
			name:         "matching local IPv4 address",
			host:         "192.168.50.10:80",
			localAddress: "192.168.50.10:6444",
			httpsAddress: "0.0.0.0:6444",
			target:       "https://192.168.50.10:6444/listen/feed%201?format=opus",
			status:       http.StatusPermanentRedirect,
		},
		{
			name:         "matching local IPv6 address",
			host:         "[fd00::10]:80",
			localAddress: "[fd00::10]:6444",
			httpsAddress: "[::]:6444",
			target:       "https://[fd00::10]:6444/listen/feed%201?format=opus",
			status:       http.StatusPermanentRedirect,
		},
		{
			name:         "untrusted host uses configured domain",
			host:         "attacker.example",
			localAddress: "192.168.50.10:6444",
			httpsAddress: "0.0.0.0:6444",
			domains:      []string{"panel.example.com"},
			target:       "https://panel.example.com:6444/listen/feed%201?format=opus",
			status:       http.StatusPermanentRedirect,
		},
		{
			name:         "malformed host is rejected",
			host:         "panel.example.com/attacker",
			localAddress: "192.168.50.10:6444",
			httpsAddress: "0.0.0.0:6444",
			domains:      []string{"panel.example.com"},
			status:       http.StatusBadRequest,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtime := &TLSRuntime{Domains: test.domains}
			request := httptest.NewRequest(http.MethodGet, "http://placeholder/listen/feed%201?format=opus", nil)
			request.Host = test.host
			request = request.WithContext(context.WithValue(
				request.Context(),
				http.LocalAddrContextKey,
				testAddress(test.localAddress),
			))
			recorder := httptest.NewRecorder()

			runtime.HTTPSRedirectHandler(test.httpsAddress).ServeHTTP(recorder, request)

			if recorder.Code != test.status {
				t.Fatalf("status = %d, want %d", recorder.Code, test.status)
			}
			if got := recorder.Header().Get("Location"); got != test.target {
				t.Fatalf("Location = %q, want %q", got, test.target)
			}
		})
	}
}

func TestHTTPChallengeHandlerHonorsRedirectSetting(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "http://panel.example.com/status", nil)
	request.Host = "panel.example.com"

	runtime := &TLSRuntime{
		Domains:      []string{"panel.example.com"},
		RedirectHTTP: false,
	}
	recorder := httptest.NewRecorder()
	runtime.HTTPChallengeHandler("0.0.0.0:6444").ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("redirect disabled status = %d, want %d", recorder.Code, http.StatusNotFound)
	}

	runtime.RedirectHTTP = true
	recorder = httptest.NewRecorder()
	runtime.HTTPChallengeHandler("0.0.0.0:6444").ServeHTTP(recorder, request)
	if recorder.Code != http.StatusPermanentRedirect {
		t.Fatalf("redirect enabled status = %d, want %d", recorder.Code, http.StatusPermanentRedirect)
	}
}

func TestServeTLSRedirectsPlainHTTPOnHTTPSPort(t *testing.T) {
	certificateServer := httptest.NewTLSServer(http.NotFoundHandler())
	serverCertificate := certificateServer.TLS.Certificates[0]
	rootCertificate := certificateServer.Certificate()
	certificateServer.Close()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	runtime := &TLSRuntime{
		Enabled: true,
		Mode:    tlsModeManual,
	}
	server := &http.Server{
		Handler: http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
			_, _ = writer.Write([]byte("secure"))
		}),
		ReadHeaderTimeout: time.Second,
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{serverCertificate},
			MinVersion:   tls.VersionTLS12,
		},
	}
	errorsChannel := make(chan error, 1)
	go func() {
		errorsChannel <- runtime.ServeTLS(server, listener)
	}()

	address := listener.Addr().String()
	connection, err := net.DialTimeout("tcp", address, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	_, err = fmt.Fprintf(
		connection,
		"GET /listen/feed?format=opus HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n",
		address,
	)
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.ReadResponse(bufio.NewReader(connection), nil)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusPermanentRedirect {
		t.Fatalf("plaintext status = %d, want %d", response.StatusCode, http.StatusPermanentRedirect)
	}
	if got, wanted := response.Header.Get("Location"), "https://"+address+"/listen/feed?format=opus"; got != wanted {
		t.Fatalf("Location = %q, want %q", got, wanted)
	}

	roots := x509.NewCertPool()
	roots.AddCert(rootCertificate)
	client := &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				RootCAs:    roots,
				ServerName: "example.com",
				MinVersion: tls.VersionTLS12,
			},
		},
	}
	secureResponse, err := client.Get("https://" + address + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer secureResponse.Body.Close()
	if secureResponse.StatusCode != http.StatusOK {
		t.Fatalf("HTTPS status = %d, want %d", secureResponse.StatusCode, http.StatusOK)
	}

	shutdownContext, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownContext); err != nil {
		t.Fatal(err)
	}
	if err := <-errorsChannel; !errors.Is(err, http.ErrServerClosed) {
		t.Fatalf("ServeTLS error = %v, want %v", err, http.ErrServerClosed)
	}
}

type testAddress string

func (a testAddress) Network() string {
	return "tcp"
}

func (a testAddress) String() string {
	return string(a)
}
