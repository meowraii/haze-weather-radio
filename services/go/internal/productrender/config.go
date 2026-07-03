package productrender

import (
	"encoding/csv"
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
	"github.com/meowraii/haze-weather-radio/services/go/internal/locationdb"
	"gopkg.in/yaml.v3"
)

type Options struct {
	ConfigPath string
	BridgeAddr string
	Refresh    time.Duration
}

type rootConfig struct {
	Version   string                  `yaml:"version"`
	FeedsFile string                  `yaml:"feeds_file"`
	Operator  operatorConfig          `yaml:"operator"`
	Storage   datastore.StorageConfig `yaml:"storage"`
	Services  struct {
		Go struct {
			ProductRender productRenderConfig `yaml:"product_render"`
		} `yaml:"go"`
	} `yaml:"services"`
}

type productRenderConfig struct {
	Enabled bool          `yaml:"enabled"`
	Cleanup cleanupConfig `yaml:"cleanup"`
}

type cleanupConfig struct {
	Enabled *bool `yaml:"enabled"`
	Hour    *int  `yaml:"hour"`
	Minute  *int  `yaml:"minute"`
}

type operatorConfig struct {
	OnAirName any `yaml:"on_air_name"`
}

type feedsXML struct {
	Feeds []feedXML `xml:"feed"`
}

type feedXML struct {
	ID         string `xml:"id,attr"`
	EnabledRaw string `xml:"enabled,attr"`
	Timezone   string `xml:"timezone,attr"`
	Playout    struct {
		Routine           string `xml:"routine,attr"`
		SAME              string `xml:"same,attr"`
		SAMEOriginator    string `xml:"same_originator,attr"`
		SAMEAttentionTone string `xml:"same_attention_tone,attr"`
	} `xml:"playout"`
	Alerts struct {
		CapCP  feedAlertSourceXML `xml:"cap_cp"`
		NWSCAP feedAlertSourceXML `xml:"nws_cap"`
	} `xml:"alerts"`
	Languages struct {
		Langs []struct {
			Code string `xml:"code,attr"`
		} `xml:"lang"`
	} `xml:"languages"`
	Description struct {
		Langs []struct {
			Code   string `xml:"code,attr"`
			Text   string `xml:"text,attr"`
			Suffix string `xml:"suffix,attr"`
		} `xml:"lang"`
	} `xml:"description"`
	Locations struct {
		Coverage struct {
			Regions []coverageRegionXML `xml:"region"`
		} `xml:"coverage"`
		ObservationLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"observationLocations"`
		AirQualityLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"airQualityLocations"`
		ClimateLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"climateLocations"`
		HydrometricLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"hydrometricLocations"`
	} `xml:"locations"`
	Transmitter struct {
		Transmitters []transmitterXML `xml:"transmitter"`
	} `xml:"transmitter_metadata"`
}

type feedAlertSourceXML struct {
	EnabledRaw string         `xml:"enabled,attr"`
	Filter     alertFilterXML `xml:"filter"`
}

type alertFilterXML struct {
	UseFeedLocations string             `xml:"use_feed_locations,attr"`
	Allowlist        alertFilterListXML `xml:"allowlist"`
	Blocklist        alertFilterListXML `xml:"blocklist"`
}

type alertFilterListXML struct {
	Severities   []string              `xml:"severity"`
	Urgencies    []string              `xml:"urgency"`
	Certainties  []string              `xml:"certainty"`
	MessageTypes []string              `xml:"message_type"`
	Events       []string              `xml:"event"`
	NAADSEvents  []string              `xml:"naads_event"`
	Others       []alertFilterOtherXML `xml:"other"`
}

type alertFilterOtherXML struct {
	ValueName string `xml:"value_name,attr"`
	Value     string `xml:"value,attr"`
}

type coverageRegionXML struct {
	ID             string                 `xml:"id,attr"`
	Source         string                 `xml:"source,attr"`
	Name           string                 `xml:"name,attr"`
	DeriveForecast string                 `xml:"derive_forecast,attr"`
	Subregions     []coverageSubregionXML `xml:"subregion"`
}

type coverageSubregionXML struct {
	ID string `xml:"id,attr"`
}

type locationXML struct {
	ID           string `xml:"id,attr"`
	Source       string `xml:"source,attr"`
	NameOverride string `xml:"name_override,attr"`
	Latitude     string `xml:"latitude,attr"`
	Longitude    string `xml:"longitude,attr"`
	NormalID     string `xml:"normal_id,attr"`
}

type transmitterXML struct {
	Network      transmitterNetworkXML   `xml:"network"`
	HostName     string                  `xml:"host_name"`
	SiteName     string                  `xml:"site_name"`
	Callsign     string                  `xml:"callsign"`
	Relationship string                  `xml:"relationship"`
	FrequencyMHz transmitterFrequencyXML `xml:"frequency_mhz"`
}

type transmitterNetworkXML struct {
	Name           string `xml:"name"`
	Pronounciation string `xml:"pronounciation"`
	Pronunciation  string `xml:"pronunciation"`
}

type transmitterFrequencyXML struct {
	GPCLK string `xml:"gpclk,attr"`
	Value string `xml:",chardata"`
}

type packagesXML struct {
	Defaults packageDefaultsXML `xml:"defaults"`
	Packages []packageXML       `xml:"package"`
}

type productTextXML struct {
	Packages []productTextPackageXML `xml:"package"`
}

type productTextPackageXML struct {
	ID    string                `xml:"id,attr"`
	Texts []productTextEntryXML `xml:"text"`
}

type productTextEntryXML struct {
	Key  string `xml:"key,attr"`
	Lang string `xml:"lang,attr"`
	Text string `xml:",chardata"`
}

type packageDefaultsXML struct {
	Enabled  string `xml:"enabled"`
	ReaderID string `xml:"reader_id"`
}

type packageXML struct {
	ID         string              `xml:"id,attr"`
	EnabledRaw string              `xml:"enabled,attr"`
	ReaderID   string              `xml:"reader_id"`
	Locations  packageLocationsXML `xml:"locations"`
}

type packageProfile struct {
	Enabled   bool
	ReaderID  string
	Locations packageLocations
}

type packageLocationsXML struct {
	StateProv string   `xml:"stateProv,attr"`
	Mentions  []string `xml:"mention"`
}

type packageLocations struct {
	StateProv string
	Mentions  []string
}

type loadedConfig struct {
	Root          rootConfig
	Feeds         []feedXML
	Packages      map[string]packageProfile
	ProductText   map[string]map[string]map[string]string
	ForecastNames map[string]forecastRegionName
	BaseDir       string
	Store         datastore.Store
}

type forecastRegionName struct {
	English string
	French  string
}

func loadConfig(configPath string) (loadedConfig, error) {
	raw, err := os.ReadFile(filepath.Clean(configPath))
	if err != nil {
		return loadedConfig{}, err
	}
	raw = []byte(os.ExpandEnv(string(raw)))
	var root rootConfig
	if err := yaml.Unmarshal(raw, &root); err != nil {
		return loadedConfig{}, err
	}
	baseDir := filepath.Dir(filepath.Clean(configPath))
	feeds, err := loadFeeds(resolvePath(baseDir, fallbackText(root.FeedsFile, "managed/configs/feeds.xml")))
	if err != nil {
		return loadedConfig{}, err
	}
	packages, err := loadPackages(resolvePath(baseDir, "managed/configs/packages.xml"))
	if err != nil {
		return loadedConfig{}, err
	}
	productText, err := loadProductText(resolvePath(baseDir, "managed/configs/product_text.xml"))
	if err != nil {
		return loadedConfig{}, err
	}
	forecastNames := loadForecastRegionNamesFromSQLite(baseDir)
	if len(forecastNames) == 0 {
		forecastNames = loadForecastRegionNames(resolvePath(baseDir, "managed/csv/FORECAST_LOCATIONS.csv"))
	}
	return loadedConfig{
		Root:          root,
		Feeds:         feeds,
		Packages:      packages,
		ProductText:   productText,
		ForecastNames: forecastNames,
		BaseDir:       baseDir,
	}, nil
}

func loadForecastRegionNamesFromSQLite(baseDir string) map[string]forecastRegionName {
	snap, ok := locationdb.Load(baseDir)
	if !ok {
		return map[string]forecastRegionName{}
	}
	out := map[string]forecastRegionName{}
	for _, place := range snap.PlacesBySource("forecast") {
		if strings.TrimSpace(place.Code) == "" {
			continue
		}
		out[place.Code] = forecastRegionName{English: place.Name, French: place.NameFR}
	}
	return out
}

func loadForecastRegionNames(path string) map[string]forecastRegionName {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return map[string]forecastRegionName{}
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	_, _ = reader.Read()
	header, err := reader.Read()
	if err != nil {
		return map[string]forecastRegionName{}
	}
	codeIndex := csvHeaderIndex(header, "CODE")
	nameIndex := csvHeaderIndex(header, "NAME")
	nomIndex := csvHeaderIndex(header, "NOM")
	if codeIndex < 0 || nameIndex < 0 || nomIndex < 0 {
		return map[string]forecastRegionName{}
	}
	names := map[string]forecastRegionName{}
	for {
		row, err := reader.Read()
		if err != nil {
			break
		}
		maxIndex := maxInt(codeIndex, nameIndex, nomIndex)
		if len(row) <= maxIndex {
			continue
		}
		code := strings.TrimSpace(strings.Trim(row[codeIndex], `"`))
		if code == "" {
			continue
		}
		if _, exists := names[code]; exists {
			continue
		}
		names[code] = forecastRegionName{
			English: strings.TrimSpace(strings.Trim(row[nameIndex], `"`)),
			French:  strings.TrimSpace(strings.Trim(row[nomIndex], `"`)),
		}
	}
	return names
}

func csvHeaderIndex(header []string, name string) int {
	for index, value := range header {
		if strings.EqualFold(strings.TrimSpace(value), name) {
			return index
		}
	}
	return -1
}

func maxInt(values ...int) int {
	maxValue := values[0]
	for _, value := range values[1:] {
		if value > maxValue {
			maxValue = value
		}
	}
	return maxValue
}

func loadFeeds(path string) ([]feedXML, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	raw = []byte(os.ExpandEnv(string(raw)))
	var parsed feedsXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parse feeds XML: %w", err)
	}
	return parsed.Feeds, nil
}

func loadPackages(path string) (map[string]packageProfile, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]packageProfile{}, nil
		}
		return nil, err
	}
	raw = []byte(os.ExpandEnv(string(raw)))
	var parsed packagesXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parse packages XML: %w", err)
	}
	defaultEnabled := xmlBool(parsed.Defaults.Enabled, true)
	defaultReader := strings.TrimSpace(parsed.Defaults.ReaderID)
	profiles := map[string]packageProfile{}
	for _, item := range parsed.Packages {
		id := strings.ToLower(strings.TrimSpace(item.ID))
		if id == "" {
			continue
		}
		reader := strings.TrimSpace(item.ReaderID)
		if reader == "" {
			reader = defaultReader
		}
		profiles[id] = packageProfile{
			Enabled:   xmlBool(item.EnabledRaw, defaultEnabled),
			ReaderID:  reader,
			Locations: normalizePackageLocations(item.Locations),
		}
	}
	return profiles, nil
}

func normalizePackageLocations(raw packageLocationsXML) packageLocations {
	stateProv := strings.ToUpper(strings.TrimSpace(raw.StateProv))
	seen := map[string]struct{}{}
	mentions := make([]string, 0, len(raw.Mentions)+4)
	addMention := func(value string) {
		value = strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
		if value == "" {
			return
		}
		key := strings.ToUpper(value)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		mentions = append(mentions, value)
	}
	for _, mention := range raw.Mentions {
		addMention(mention)
	}
	switch stateProv {
	case "SK":
		addMention("SK")
		addMention("Saskatchewan")
		addMention("Southern SK")
		addMention("Southern Saskatchewan")
	}
	return packageLocations{StateProv: stateProv, Mentions: mentions}
}

func loadProductText(path string) (map[string]map[string]map[string]string, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]map[string]map[string]string{}, nil
		}
		return nil, err
	}
	raw = []byte(os.ExpandEnv(string(raw)))
	var parsed productTextXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parse product text XML: %w", err)
	}
	out := map[string]map[string]map[string]string{}
	for _, pkg := range parsed.Packages {
		pkgID := strings.ToLower(strings.TrimSpace(pkg.ID))
		if pkgID == "" {
			continue
		}
		if out[pkgID] == nil {
			out[pkgID] = map[string]map[string]string{}
		}
		for _, entry := range pkg.Texts {
			key := strings.ToLower(strings.TrimSpace(entry.Key))
			if key == "" {
				continue
			}
			lang := normalizeLangKey(entry.Lang)
			if out[pkgID][key] == nil {
				out[pkgID][key] = map[string]string{}
			}
			out[pkgID][key][lang] = strings.TrimSpace(entry.Text)
		}
	}
	return out, nil
}

func (cfg loadedConfig) feedByID(feedID string) (feedXML, bool) {
	for _, feed := range cfg.Feeds {
		if strings.EqualFold(strings.TrimSpace(feed.ID), strings.TrimSpace(feedID)) {
			return feed, true
		}
	}
	return feedXML{}, false
}

func normalizeLangKey(lang string) string {
	lang = strings.ToLower(strings.ReplaceAll(strings.TrimSpace(lang), "_", "-"))
	if lang == "" || lang == "*" {
		return "*"
	}
	return lang
}

func (cfg loadedConfig) readerID(pkgID string) string {
	if profile, ok := cfg.Packages[pkgID]; ok {
		return profile.ReaderID
	}
	return ""
}

func (cfg loadedConfig) packageEnabled(pkgID string) bool {
	profile, ok := cfg.Packages[pkgID]
	return !ok || profile.Enabled
}

func (cfg loadedConfig) packageLocations(pkgID string) packageLocations {
	if profile, ok := cfg.Packages[strings.ToLower(strings.TrimSpace(pkgID))]; ok {
		return profile.Locations
	}
	return packageLocations{}
}

func feedLanguage(feed feedXML) string {
	for _, lang := range feed.Languages.Langs {
		if code := strings.TrimSpace(lang.Code); code != "" {
			return code
		}
	}
	return "en-CA"
}

func feedSiteName(feed feedXML) string {
	return fallbackText(stationTransmitter(feed).SiteName, feed.ID)
}

func feedCallsign(feed feedXML) string {
	return strings.TrimSpace(stationTransmitter(feed).Callsign)
}

func feedFrequencyMHz(feed feedXML) string {
	return stationTransmitter(feed).frequencyText()
}

func transmitterList(feed feedXML) []transmitterXML {
	out := make([]transmitterXML, 0, len(feed.Transmitter.Transmitters))
	for _, transmitter := range feed.Transmitter.Transmitters {
		if transmitter.empty() {
			continue
		}
		out = append(out, transmitter)
	}
	return out
}

func stationTransmitter(feed feedXML) transmitterXML {
	transmitters := transmitterList(feed)
	for _, transmitter := range transmitters {
		if transmitter.isRelationship("primary") {
			return transmitter
		}
	}
	for _, transmitter := range transmitters {
		if !transmitter.isRelationship("replaces") && !transmitter.isRelationship("ip") &&
			(strings.TrimSpace(transmitter.Callsign) != "" || strings.TrimSpace(transmitter.SiteName) != "") {
			return transmitter
		}
	}
	for _, transmitter := range transmitters {
		if strings.TrimSpace(transmitter.Callsign) != "" || strings.TrimSpace(transmitter.SiteName) != "" {
			return transmitter
		}
	}
	if len(transmitters) > 0 {
		return transmitters[0]
	}
	return transmitterXML{}
}

func (t transmitterXML) empty() bool {
	return strings.TrimSpace(t.SiteName) == "" &&
		strings.TrimSpace(t.Callsign) == "" &&
		strings.TrimSpace(t.Relationship) == "" &&
		strings.TrimSpace(t.HostName) == "" &&
		t.frequencyText() == ""
}

func (t transmitterXML) relationship() string {
	relationship := strings.ToLower(strings.TrimSpace(t.Relationship))
	if relationship == "secondary/repeater" {
		return "secondary"
	}
	if relationship == "" {
		return "unknown"
	}
	return relationship
}

func (t transmitterXML) isRelationship(relationship string) bool {
	current := t.relationship()
	wanted := strings.ToLower(strings.TrimSpace(relationship))
	return current == wanted || (wanted == "repeater" && current == "secondary")
}

func (t transmitterXML) frequencyText() string {
	return strings.TrimSpace(t.FrequencyMHz.Value)
}

func replacementTransmitter(feed feedXML) (transmitterXML, bool) {
	for _, transmitter := range transmitterList(feed) {
		if transmitter.isRelationship("replaces") {
			return transmitter, true
		}
	}
	return transmitterXML{}, false
}

func feedCoverageCodes(feed feedXML) map[string]struct{} {
	codes := map[string]struct{}{}
	for _, region := range feed.Locations.Coverage.Regions {
		addCoverageCode(codes, region.ID)
		for _, subregion := range region.Subregions {
			addCoverageCode(codes, subregion.ID)
		}
	}
	return codes
}

func addCoverageCode(codes map[string]struct{}, raw string) {
	code := strings.TrimSpace(raw)
	if code == "" {
		return
	}
	codes[code] = struct{}{}
}

func displayText(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	case []any:
		for _, item := range typed {
			if text := displayText(item); text != "" {
				return text
			}
		}
	case map[string]any:
		for _, key := range []string{"text", "name", "value", "address"} {
			if text := displayText(typed[key]); text != "" {
				return text
			}
		}
		for _, child := range typed {
			if text := displayText(child); text != "" {
				return text
			}
		}
	}
	return strings.TrimSpace(fmt.Sprint(value))
}

func spokenText(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	case []any:
		merged := map[string]any{}
		for _, item := range typed {
			if child, ok := item.(map[string]any); ok {
				for key, value := range child {
					merged[key] = value
				}
			} else if text := spokenText(item); text != "" {
				return text
			}
		}
		for _, key := range []string{"pronunciation", "pronounciation", "text", "name", "value", "address"} {
			if text := spokenText(merged[key]); text != "" {
				return text
			}
		}
	case map[string]any:
		for _, key := range []string{"pronunciation", "pronounciation", "text", "name", "value", "address"} {
			if text := spokenText(typed[key]); text != "" {
				return text
			}
		}
		for _, child := range typed {
			if text := spokenText(child); text != "" {
				return text
			}
		}
	}
	return strings.TrimSpace(fmt.Sprint(value))
}

func spokenNetworkName(network transmitterNetworkXML) string {
	for _, value := range []string{network.Pronunciation, network.Pronounciation, network.Name} {
		if text := strings.TrimSpace(value); text != "" {
			return text
		}
	}
	return ""
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

func intText(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return fallback
	}
	return value
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

func fallbackText(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return strings.TrimSpace(fallback)
	}
	return value
}
