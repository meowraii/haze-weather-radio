package webgateway

import (
	"net/http"
	"testing"
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
	request, err := http.NewRequest(http.MethodGet, "http://localhost:8086/admin", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Host = "localhost:8086"

	status := tlsStatus(Config{}, request)

	if status["actual_domain"] != false {
		t.Fatalf("actual_domain = %v", status["actual_domain"])
	}
	if status["needs_setup"] != false {
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

func TestRedirectHTTPSHostUsesConfiguredTLSPort(t *testing.T) {
	got := redirectHTTPSHost("panel.example.com", "0.0.0.0:8086")
	if got != "panel.example.com:8086" {
		t.Fatalf("redirect host = %s", got)
	}
	got = redirectHTTPSHost("panel.example.com", "0.0.0.0:443")
	if got != "panel.example.com" {
		t.Fatalf("redirect host on 443 = %s", got)
	}
}
