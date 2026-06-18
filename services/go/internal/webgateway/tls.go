package webgateway

import (
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"os"
	"strconv"
	"strings"

	"golang.org/x/crypto/acme"
	"golang.org/x/crypto/acme/autocert"
)

const (
	tlsModeManual            = "manual"
	tlsModeACME              = "acme"
	defaultACMECacheDir      = "runtime/tls/acme"
	letsencryptStagingCA     = "https://acme-staging-v02.api.letsencrypt.org/directory"
	defaultHTTPChallengeHost = "0.0.0.0"
	defaultHTTPChallengePort = 80
)

// TLSRuntime contains the prepared HTTPS and ACME state for the web gateway.
type TLSRuntime struct {
	Enabled           bool
	Mode              string
	Domains           []string
	CertFile          string
	KeyFile           string
	CacheDir          string
	HTTPChallengeAddr string
	RedirectHTTP      bool
	HSTS              bool
	Staging           bool
	Manager           *autocert.Manager
}

// NewTLSRuntime prepares the configured TLS runtime. ACME is deliberately
// domain-whitelisted so Host headers cannot trigger arbitrary certificate
// issuance.
func NewTLSRuntime(config Config, configPath string) (*TLSRuntime, error) {
	tlsConfig := config.Webpanel.TLS
	mode := normalizeTLSMode(tlsConfig.Mode)
	runtime := &TLSRuntime{
		Enabled:           tlsConfig.Enabled,
		Mode:              mode,
		Domains:           normalizeDomains(tlsConfig.Domains),
		CertFile:          strings.TrimSpace(tlsConfig.CertFile),
		KeyFile:           strings.TrimSpace(tlsConfig.KeyFile),
		CacheDir:          resolveConfigPath(configPath, fallbackText(tlsConfig.CacheDir, defaultACMECacheDir)),
		HTTPChallengeAddr: httpChallengeAddress(config),
		RedirectHTTP:      tlsConfig.RedirectHTTP,
		HSTS:              tlsConfig.HSTS,
		Staging:           tlsConfig.Staging,
	}
	if !runtime.Enabled {
		return runtime, nil
	}
	switch mode {
	case tlsModeACME:
		if len(runtime.Domains) == 0 {
			return nil, fmt.Errorf("webpanel.tls.domains must include at least one domain when ACME is enabled")
		}
		if err := os.MkdirAll(runtime.CacheDir, 0o700); err != nil {
			return nil, fmt.Errorf("create ACME cache directory: %w", err)
		}
		manager := &autocert.Manager{
			Prompt:     autocert.AcceptTOS,
			HostPolicy: autocert.HostWhitelist(runtime.Domains...),
			Cache:      autocert.DirCache(runtime.CacheDir),
			Email:      strings.TrimSpace(tlsConfig.Email),
		}
		if runtime.Staging {
			manager.Client = &acme.Client{DirectoryURL: letsencryptStagingCA}
		}
		runtime.Manager = manager
	case tlsModeManual:
		if runtime.CertFile == "" || runtime.KeyFile == "" {
			return nil, fmt.Errorf("webpanel.tls.cert_file and key_file are required for manual TLS")
		}
		runtime.CertFile = resolveConfigPath(configPath, runtime.CertFile)
		runtime.KeyFile = resolveConfigPath(configPath, runtime.KeyFile)
	}
	return runtime, nil
}

// TLSConfig returns a secure server TLS configuration for the selected mode.
func (t *TLSRuntime) TLSConfig() *tls.Config {
	if t == nil || !t.Enabled {
		return nil
	}
	if t.Manager != nil {
		config := t.Manager.TLSConfig()
		config.MinVersion = tls.VersionTLS12
		return config
	}
	return &tls.Config{MinVersion: tls.VersionTLS12}
}

// HTTPChallengeEnabled reports whether the ACME HTTP-01 helper should run.
func (t *TLSRuntime) HTTPChallengeEnabled(surface WebSurface) bool {
	if t == nil || !t.Enabled || t.Mode != tlsModeACME || t.Manager == nil {
		return false
	}
	if surface == SurfaceAdmin {
		return false
	}
	return t.HTTPChallengeAddr != ""
}

// HTTPChallengeHandler serves ACME HTTP-01 challenges and optionally redirects
// ordinary HTTP requests to the HTTPS listener.
func (t *TLSRuntime) HTTPChallengeHandler(httpsAddr string) http.Handler {
	fallback := http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if t == nil || !t.RedirectHTTP {
			http.NotFound(writer, request)
			return
		}
		target := "https://" + redirectHTTPSHost(request.Host, httpsAddr) + request.URL.RequestURI()
		http.Redirect(writer, request, target, http.StatusPermanentRedirect)
	})
	if t == nil || t.Manager == nil {
		return fallback
	}
	return t.Manager.HTTPHandler(fallback)
}

func normalizeTLSMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case tlsModeACME, "letsencrypt", "lets_encrypt":
		return tlsModeACME
	default:
		return tlsModeManual
	}
}

func normalizeDomains(domains []string) []string {
	seen := map[string]struct{}{}
	out := []string{}
	for _, domain := range domains {
		domain = normalizeDomain(domain)
		if domain == "" {
			continue
		}
		if _, ok := seen[domain]; ok {
			continue
		}
		seen[domain] = struct{}{}
		out = append(out, domain)
	}
	return out
}

func normalizeDomain(host string) string {
	host = strings.ToLower(strings.TrimSpace(host))
	host = strings.TrimSuffix(host, ".")
	if host == "" {
		return ""
	}
	if parsedHost, _, err := net.SplitHostPort(host); err == nil {
		host = parsedHost
	}
	if strings.Contains(host, "/") {
		return ""
	}
	return host
}

func httpChallengeAddress(config Config) string {
	challenge := config.Webpanel.TLS.HTTPChallenge
	enabled := true
	if challenge.Enabled != nil {
		enabled = *challenge.Enabled
	}
	if !enabled {
		return ""
	}
	if addr := strings.TrimSpace(challenge.Addr); addr != "" {
		return addr
	}
	host := strings.TrimSpace(challenge.Host)
	if host == "" {
		host = defaultHTTPChallengeHost
	}
	port := challenge.Port
	if port <= 0 {
		port = defaultHTTPChallengePort
	}
	return net.JoinHostPort(host, strconv.Itoa(port))
}

func redirectHTTPSHost(requestHost string, httpsAddr string) string {
	host := strings.TrimSpace(requestHost)
	if host == "" {
		return host
	}
	name, _, err := net.SplitHostPort(host)
	if err == nil {
		host = name
	}
	_, httpsPort, err := net.SplitHostPort(httpsAddr)
	if err != nil || httpsPort == "" || httpsPort == "443" {
		return host
	}
	return net.JoinHostPort(host, httpsPort)
}

func tlsStatus(config Config, request *http.Request) map[string]any {
	host := requestHostName(request)
	domains := normalizeDomains(config.Webpanel.TLS.Domains)
	mode := normalizeTLSMode(config.Webpanel.TLS.Mode)
	actualDomain := isActualDomain(host)
	https := requestIsHTTPS(request)
	domainConfigured := containsDomain(domains, normalizeDomain(host))
	acmeConfigured := config.Webpanel.TLS.Enabled && mode == tlsModeACME && len(domains) > 0
	needsSetup := actualDomain && (!https || !config.Webpanel.TLS.Enabled || (mode == tlsModeACME && !domainConfigured))
	message := "Local HTTP is active."
	if actualDomain && https {
		message = "Panel is being accessed over HTTPS on a domain."
	} else if actualDomain && !config.Webpanel.TLS.Enabled {
		message = "Domain access detected. Enable ACME to issue a Let's Encrypt certificate."
	} else if actualDomain && mode == tlsModeACME && !domainConfigured {
		message = "Domain access detected, but this hostname is not in webpanel.tls.domains."
	} else if config.Webpanel.TLS.Enabled {
		message = "HTTPS is configured for the web panel."
	}
	return map[string]any{
		"enabled":             config.Webpanel.TLS.Enabled,
		"mode":                mode,
		"https":               https,
		"host":                host,
		"actual_domain":       actualDomain,
		"domain_configured":   domainConfigured,
		"configured_domains":  domains,
		"acme_configured":     acmeConfigured,
		"acme_cache_dir":      fallbackText(config.Webpanel.TLS.CacheDir, defaultACMECacheDir),
		"http_challenge_addr": httpChallengeAddress(config),
		"redirect_http":       config.Webpanel.TLS.RedirectHTTP,
		"hsts":                config.Webpanel.TLS.HSTS,
		"staging":             config.Webpanel.TLS.Staging,
		"needs_setup":         needsSetup,
		"message":             message,
	}
}

func requestIsHTTPS(request *http.Request) bool {
	if request == nil {
		return false
	}
	if request.TLS != nil {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(request.Header.Get("X-Forwarded-Proto")), "https")
}

func requestHostName(request *http.Request) string {
	if request == nil {
		return ""
	}
	host := strings.TrimSpace(request.Host)
	if host == "" {
		host = strings.TrimSpace(request.URL.Host)
	}
	return normalizeDomain(host)
}

func isActualDomain(host string) bool {
	host = normalizeDomain(host)
	if host == "" || !strings.Contains(host, ".") {
		return false
	}
	if strings.EqualFold(host, "localhost") || strings.HasSuffix(host, ".localhost") {
		return false
	}
	if strings.HasSuffix(host, ".local") || strings.HasSuffix(host, ".lan") {
		return false
	}
	ip := net.ParseIP(host)
	return ip == nil
}

func containsDomain(values []string, wanted string) bool {
	wanted = normalizeDomain(wanted)
	for _, value := range values {
		if normalizeDomain(value) == wanted {
			return true
		}
	}
	return false
}
