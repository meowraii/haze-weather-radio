package ivr

import (
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

var requiredMenus = []string{
	"entry",
	"language_select",
	"location_code",
	"location_menu",
	"weather_product",
	"broadcast_menu",
	"geophysical_alert",
	"operator",
	"error",
}

var requiredMenuLines = map[string][]string{
	"entry":             {"main"},
	"location_code":     {"main"},
	"location_menu":     {"main"},
	"weather_product":   {"unavailable"},
	"broadcast_menu":    {"main"},
	"geophysical_alert": {"main"},
	"operator":          {"main"},
	"error":             {"invalid_code", "timeout"},
}

type promptConfigXML struct {
	XMLName  xml.Name       `xml:"ivr"`
	Defaults promptDefaults `xml:"defaults"`
	Menus    []promptMenu   `xml:"menu"`
}

type promptDefaults struct {
	TTSProfile
	Lines []promptLine `xml:"line"`
}

type promptMenu struct {
	ID          string        `xml:"id,attr"`
	TTSProfile                // menu-level TTS overrides
	TimeoutRaw  string        `xml:"timeout,attr"`
	Timeout     time.Duration `xml:"-"`
	RetriesRaw  string        `xml:"retries,attr"`
	Retries     int           `xml:"-"`
	TransferURI string        `xml:"transfer_uri,attr"`
	Lines       []promptLine  `xml:"line"`
	Options     []menuOption  `xml:"option"`
}

type promptLine struct {
	Key  string `xml:"key,attr"`
	Text string `xml:",chardata"`
}

type menuOption struct {
	Digit    string `xml:"digit,attr"`
	Action   string `xml:"action,attr"`
	Next     string `xml:"next,attr"`
	Language string `xml:"language,attr"`
	Packages string `xml:"packages,attr"`
}

// TTSProfile describes menu-specific synthesis overrides.
type TTSProfile struct {
	ReaderID           string        `xml:"reader_id,attr"`
	Provider           string        `xml:"provider,attr"`
	VoiceID            string        `xml:"voice_id,attr"`
	Language           string        `xml:"language,attr"`
	ExplicitLanguage   bool          `xml:"-"`
	RateRaw            string        `xml:"rate,attr"`
	Rate               int           `xml:"-"`
	VolumeRaw          string        `xml:"volume,attr"`
	Volume             int           `xml:"-"`
	SentenceSilenceRaw string        `xml:"sentence_silence,attr"`
	SentenceSilence    float64       `xml:"-"`
	CacheTTLRaw        string        `xml:"cache_ttl,attr"`
	CacheTTL           time.Duration `xml:"-"`
}

type PromptConfig struct {
	Defaults TTSProfile
	Lines    map[string]string
	Menus    map[string]promptMenu
}

func loadPromptConfig(path string) (PromptConfig, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return defaultPromptConfig(), nil
		}
		return PromptConfig{}, err
	}
	var parsed promptConfigXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return PromptConfig{}, fmt.Errorf("parse IVR XML: %w", err)
	}
	return normalizePromptConfig(parsed)
}

func normalizePromptConfig(parsed promptConfigXML) (PromptConfig, error) {
	defaults := normalizeTTSProfile(parsed.Defaults.TTSProfile, TTSProfile{
		Provider: "fast",
		Language: "en-CA",
		Volume:   100,
		CacheTTL: 24 * time.Hour,
	})
	cfg := PromptConfig{
		Defaults: defaults,
		Lines:    map[string]string{},
		Menus:    map[string]promptMenu{},
	}
	for _, line := range parsed.Defaults.Lines {
		if key := strings.ToLower(strings.TrimSpace(line.Key)); key != "" {
			cfg.Lines[key] = cleanPromptText(line.Text)
		}
	}
	for _, menu := range parsed.Menus {
		menu.ID = strings.ToLower(strings.TrimSpace(menu.ID))
		if menu.ID == "" {
			continue
		}
		menu.TTSProfile = normalizeTTSProfile(menu.TTSProfile, defaults)
		menu.Timeout = parseDuration(menu.TimeoutRaw, 8*time.Second)
		menu.Retries = parseInt(menu.RetriesRaw, 2)
		for index := range menu.Lines {
			menu.Lines[index].Key = strings.ToLower(strings.TrimSpace(menu.Lines[index].Key))
			menu.Lines[index].Text = cleanPromptText(menu.Lines[index].Text)
		}
		for index := range menu.Options {
			menu.Options[index].Digit = strings.TrimSpace(menu.Options[index].Digit)
			menu.Options[index].Action = strings.ToLower(strings.TrimSpace(menu.Options[index].Action))
			menu.Options[index].Next = strings.ToLower(strings.TrimSpace(menu.Options[index].Next))
			menu.Options[index].Language = strings.TrimSpace(menu.Options[index].Language)
			menu.Options[index].Packages = strings.TrimSpace(menu.Options[index].Packages)
		}
		cfg.Menus[menu.ID] = menu
	}
	for _, id := range requiredMenus {
		if _, ok := cfg.Menus[id]; !ok {
			return PromptConfig{}, fmt.Errorf("IVR XML missing required menu %q", id)
		}
	}
	for menuID, lines := range requiredMenuLines {
		for _, lineKey := range lines {
			if cfg.MenuLine(menuID, lineKey, nil) == "" {
				return PromptConfig{}, fmt.Errorf("IVR XML missing required line %q in menu %q", lineKey, menuID)
			}
		}
	}
	return cfg, nil
}

func defaultPromptConfig() PromptConfig {
	raw := promptConfigXML{
		Defaults: promptDefaults{
			TTSProfile: TTSProfile{Provider: "fast", Language: "en-CA", VolumeRaw: "100", CacheTTLRaw: "24h"},
			Lines: []promptLine{
				{Key: "one_moment", Text: "One moment."},
				{Key: "enter_code", Text: "Enter your weather code, followed by pound."},
				{Key: "no_entry", Text: "No entry was received. Goodbye."},
				{Key: "goodbye", Text: "Goodbye."},
			},
		},
		Menus: []promptMenu{
			{ID: "entry", Lines: []promptLine{{Key: "main", Text: "This is the Canada TeleMET weather service. 1 For service in English, 2 for French, 3 for Spanish, 4 to listen to a Canada RadioMET broadcast. 5 for the NOAA Geophysical Alert Message, or 0 to yell at an operator."}}},
			{ID: "language_select", Lines: []promptLine{{Key: "main", Text: "1 for English. 2 for French. 3 for Spanish."}}},
			{ID: "location_code", Lines: []promptLine{{Key: "main", Text: "Enter your weather code, followed by pound."}}},
			{ID: "location_menu", Lines: []promptLine{{Key: "main", Text: "You have reached {location}. 1 for regional observations, 2 for your 7 day outlook, 3 for air quality indicies, or 4 to listen to a corresponding, 10 minute Canada RadioMET broadcast."}}},
			{ID: "weather_product", Lines: []promptLine{{Key: "unavailable", Text: "Weather is unavailable for that code."}}},
			{ID: "broadcast_menu", Lines: []promptLine{{Key: "main", Text: "The Canada RadioMET broadcast is not available by telephone yet."}}},
			{ID: "geophysical_alert", Lines: []promptLine{{Key: "main", Text: "NOAA Geophysical Alert Message."}}},
			{ID: "operator", Lines: []promptLine{{Key: "main", Text: "Operator transfer is not configured."}}},
			{ID: "error", Lines: []promptLine{{Key: "invalid_code", Text: "No match. Try again."}, {Key: "timeout", Text: "No entry. Goodbye."}}},
		},
	}
	cfg, _ := normalizePromptConfig(raw)
	return cfg
}

func normalizeTTSProfile(profile TTSProfile, fallback TTSProfile) TTSProfile {
	out := profile
	out.ExplicitLanguage = strings.TrimSpace(profile.Language) != ""
	if strings.TrimSpace(out.ReaderID) == "" {
		out.ReaderID = fallback.ReaderID
	}
	if strings.TrimSpace(out.Provider) == "" {
		out.Provider = fallback.Provider
	}
	if strings.TrimSpace(out.VoiceID) == "" {
		out.VoiceID = fallback.VoiceID
	}
	if strings.TrimSpace(out.Language) == "" {
		out.Language = fallback.Language
	}
	out.Rate = parseIntWithFallback(out.RateRaw, out.Rate, fallback.Rate)
	out.Volume = parseIntWithFallback(out.VolumeRaw, out.Volume, fallback.Volume)
	out.SentenceSilence = parseFloatWithFallback(out.SentenceSilenceRaw, out.SentenceSilence, fallback.SentenceSilence)
	out.CacheTTL = parseDuration(out.CacheTTLRaw, fallback.CacheTTL)
	if out.CacheTTL <= 0 {
		out.CacheTTL = 24 * time.Hour
	}
	return out
}

func (cfg PromptConfig) Menu(id string) (promptMenu, bool) {
	menu, ok := cfg.Menus[strings.ToLower(strings.TrimSpace(id))]
	return menu, ok
}

func (cfg PromptConfig) MenuLine(menuID string, key string, values map[string]string) string {
	menu, ok := cfg.Menu(menuID)
	if ok {
		key = strings.ToLower(strings.TrimSpace(key))
		for _, line := range menu.Lines {
			if line.Key == key && line.Text != "" {
				return renderPromptText(line.Text, values)
			}
		}
	}
	if text := cfg.Lines[strings.ToLower(strings.TrimSpace(key))]; text != "" {
		return renderPromptText(text, values)
	}
	return ""
}

func (cfg PromptConfig) Option(menuID string, digit string) (menuOption, bool) {
	menu, ok := cfg.Menu(menuID)
	if !ok {
		return menuOption{}, false
	}
	digit = strings.TrimSpace(digit)
	for _, option := range menu.Options {
		if option.Digit == digit {
			return option, true
		}
	}
	return menuOption{}, false
}

func (cfg PromptConfig) TTSForMenu(menuID string) TTSProfile {
	if menu, ok := cfg.Menu(menuID); ok {
		return menu.TTSProfile
	}
	return cfg.Defaults
}

func renderPromptText(text string, values map[string]string) string {
	text = cleanPromptText(text)
	for key, value := range values {
		text = strings.ReplaceAll(text, "{"+key+"}", strings.TrimSpace(value))
	}
	return strings.Join(strings.Fields(text), " ")
}

func cleanPromptText(text string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
}

func parseDuration(raw string, fallback time.Duration) time.Duration {
	if strings.TrimSpace(raw) == "" {
		return fallback
	}
	value, err := time.ParseDuration(strings.TrimSpace(raw))
	if err != nil {
		return fallback
	}
	return value
}

func parseInt(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return fallback
	}
	return value
}

func parseIntWithFallback(raw string, value int, fallback int) int {
	if strings.TrimSpace(raw) != "" {
		return parseInt(raw, fallback)
	}
	if value != 0 {
		return value
	}
	return fallback
}

func parseFloatWithFallback(raw string, value float64, fallback float64) float64 {
	if strings.TrimSpace(raw) != "" {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
		if err == nil {
			return parsed
		}
		return fallback
	}
	if value != 0 {
		return value
	}
	return fallback
}
