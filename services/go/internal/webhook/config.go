package webhook

import (
	"encoding/xml"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

type Options struct {
	ConfigPath   string
	BridgeAddr   string
	WebhooksPath string
	Timeout      time.Duration
}

type rootConfig struct {
	FeedsFile string `yaml:"feeds_file"`
	Services  struct {
		Go struct {
			Webhook webhookServiceConfig `yaml:"webhook"`
		} `yaml:"go"`
	} `yaml:"services"`
	Webpanel webpanelConfig `yaml:"webpanel"`
}

type webhookServiceConfig struct {
	Webhooks      string `yaml:"webhooks"`
	Timeout       string `yaml:"timeout"`
	ListenBaseURL string `yaml:"listen_base_url"`
}

type webpanelConfig struct {
	Enabled bool     `yaml:"enabled"`
	Host    string   `yaml:"host"`
	Port    yamlPort `yaml:"port"`
	Public  struct {
		Enabled bool     `yaml:"enabled"`
		Host    string   `yaml:"host"`
		Port    yamlPort `yaml:"port"`
	} `yaml:"public"`
	TLS struct {
		Enabled bool `yaml:"enabled"`
	} `yaml:"tls"`
}

type yamlPort int

func (p *yamlPort) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.ScalarNode {
		return nil
	}
	raw := strings.TrimSpace(value.Value)
	if raw == "" {
		*p = 0
		return nil
	}
	port, err := strconv.Atoi(raw)
	if err != nil {
		return err
	}
	*p = yamlPort(port)
	return nil
}

type loadedConfig struct {
	Root          rootConfig
	BaseDir       string
	WebhooksPath  string
	ListenBaseURL string
}

type webhooksXML struct {
	Webhooks []WebhookConfig `xml:"Webhook"`
}

type WebhookConfig struct {
	FeedID         string     `xml:"feed-id,attr"`
	EnabledRaw     string     `xml:"enabled,attr"`
	EmbedAudio     EmbedAudio `xml:"EmbedAudio"`
	Username       string     `xml:"Username"`
	IconURL        string     `xml:"IconURL"`
	LogTestAlerts  string     `xml:"LogTestAlerts"`
	LogAdminAlerts string     `xml:"LogAdminAlerts"`
	WebhookURL     string     `xml:"WebhookURL"`
}

type EmbedAudio struct {
	EnabledRaw string `xml:"enabled,attr"`
	Codec      string `xml:"codec,attr"`
}

type configCache struct {
	mu      sync.Mutex
	path    string
	mtime   time.Time
	configs []WebhookConfig
}

var envRefRE = regexp.MustCompile(`^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$`)

func loadConfig(options Options) (loadedConfig, error) {
	if strings.TrimSpace(options.ConfigPath) == "" {
		options.ConfigPath = "config.yaml"
	}
	raw, err := os.ReadFile(filepath.Clean(options.ConfigPath))
	if err != nil {
		return loadedConfig{}, err
	}
	raw = []byte(os.ExpandEnv(string(raw)))
	var root rootConfig
	if err := yaml.Unmarshal(raw, &root); err != nil {
		return loadedConfig{}, err
	}
	baseDir := filepath.Dir(filepath.Clean(options.ConfigPath))
	webhooksPath := firstNonBlank(options.WebhooksPath, root.Services.Go.Webhook.Webhooks, "managed/configs/webhooks.xml")
	return loadedConfig{
		Root:          root,
		BaseDir:       baseDir,
		WebhooksPath:  resolvePath(baseDir, webhooksPath),
		ListenBaseURL: publicListenBaseURL(root),
	}, nil
}

func (c *configCache) load(path string) ([]WebhookConfig, error) {
	path = filepath.Clean(path)
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.path == path && c.mtime.Equal(info.ModTime()) {
		return append([]WebhookConfig{}, c.configs...), nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	raw = []byte(os.ExpandEnv(string(raw)))
	var parsed webhooksXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parse webhooks XML: %w", err)
	}
	configs := make([]WebhookConfig, 0, len(parsed.Webhooks))
	for _, cfg := range parsed.Webhooks {
		cfg.FeedID = strings.TrimSpace(cfg.FeedID)
		cfg.Username = firstNonBlank(cfg.Username, "Haze Weather Radio")
		cfg.IconURL = strings.TrimSpace(cfg.IconURL)
		cfg.WebhookURL = resolveEnvValue(cfg.WebhookURL)
		cfg.EmbedAudio.Codec = firstNonBlank(cfg.EmbedAudio.Codec, "libopus")
		if cfg.FeedID == "" || cfg.WebhookURL == "" {
			continue
		}
		configs = append(configs, cfg)
	}
	c.path = path
	c.mtime = info.ModTime()
	c.configs = configs
	return append([]WebhookConfig{}, configs...), nil
}

func (cfg WebhookConfig) enabled() bool {
	return xmlBool(cfg.EnabledRaw, false)
}

func (cfg WebhookConfig) logTestAlerts() bool {
	return xmlBool(cfg.LogTestAlerts, false)
}

func (cfg WebhookConfig) logAdminAlerts() bool {
	return xmlBool(cfg.LogAdminAlerts, false)
}

func (cfg WebhookConfig) audioEnabled() bool {
	return xmlBool(cfg.EmbedAudio.EnabledRaw, false)
}

func xmlBool(raw string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return fallback
	case "1", "true", "yes", "on", "enabled":
		return true
	case "0", "false", "no", "off", "disabled":
		return false
	default:
		return fallback
	}
}

func resolvePath(base string, value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return filepath.Clean(base)
	}
	if filepath.IsAbs(value) {
		return filepath.Clean(value)
	}
	return filepath.Clean(filepath.Join(base, value))
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if text := strings.TrimSpace(value); text != "" {
			return text
		}
	}
	return ""
}

func loadDotEnv(path string) {
	path = filepath.Clean(path)
	raw, err := os.ReadFile(path)
	if err != nil {
		if !os.IsNotExist(err) || filepath.Base(path) != ".env" {
			return
		}
		examplePath := filepath.Join(filepath.Dir(path), ".env.example")
		exampleRaw, readErr := os.ReadFile(examplePath)
		if readErr != nil {
			log.Printf("WARN .env file not found and no .env.example is available: %s", path)
			return
		}
		if writeErr := os.WriteFile(path, exampleRaw, 0o600); writeErr != nil {
			log.Printf("WARN .env file not found and could not create %s: %v", path, writeErr)
			return
		}
		log.Printf("WARN .env file not found: created %s from %s", path, examplePath)
		raw = exampleRaw
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		if key == "" || os.Getenv(key) != "" {
			continue
		}
		value = strings.Trim(strings.TrimSpace(value), `"'`)
		_ = os.Setenv(key, value)
	}
}

func resolveEnvValue(raw string) string {
	text := strings.TrimSpace(raw)
	if text == "" {
		return ""
	}
	if matches := envRefRE.FindStringSubmatch(text); len(matches) == 2 {
		return strings.TrimSpace(os.Getenv(matches[1]))
	}
	return os.Expand(text, func(key string) string {
		return strings.TrimSpace(os.Getenv(key))
	})
}

func publicListenBaseURL(root rootConfig) string {
	if value := firstNonBlank(os.Getenv("HAZE_PUBLIC_BASE_URL"), root.Services.Go.Webhook.ListenBaseURL); value != "" {
		return strings.TrimRight(value, "/")
	}
	host := firstNonBlank(root.Webpanel.Public.Host, root.Webpanel.Host, "127.0.0.1")
	if host == "0.0.0.0" || host == "::" || host == "[::]" {
		host = "127.0.0.1"
	}
	port := root.Webpanel.Public.Port
	if port <= 0 {
		port = root.Webpanel.Port
	}
	if port <= 0 {
		port = 6444
	}
	scheme := "http"
	if root.Webpanel.TLS.Enabled {
		scheme = "https"
	}
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") {
		host = "[" + host + "]"
	}
	defaultPort := 80
	if scheme == "https" {
		defaultPort = 443
	}
	if int(port) == defaultPort {
		return scheme + "://" + host
	}
	return fmt.Sprintf("%s://%s:%d", scheme, host, int(port))
}
