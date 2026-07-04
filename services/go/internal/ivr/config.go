package ivr

import (
	"encoding/csv"
	"encoding/xml"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
	"github.com/meowraii/haze-weather-radio/services/go/internal/locationdb"
	ttspkg "github.com/meowraii/haze-weather-radio/services/go/internal/tts"
	"gopkg.in/yaml.v3"
)

// Options are command-line/runtime settings for the IVR edge service.
type Options struct {
	ConfigPath      string
	BridgeAddr      string
	MediaBridgeAddr string
	HTTPAddr        string
	SIPAddr         string
	CacheDir        string
}

type rootConfig struct {
	FeedsFile string                  `yaml:"feeds_file"`
	Storage   datastore.StorageConfig `yaml:"storage"`
	Operator  struct {
		OnAirName     any `yaml:"on_air_name"`
		TelephoneName any `yaml:"telephone_name"`
	} `yaml:"operator"`
	Services struct {
		Rust struct {
			Media struct {
				Enabled bool   `yaml:"enabled"`
				Addr    string `yaml:"addr"`
				Listen  string `yaml:"listen"`
			} `yaml:"media"`
		} `yaml:"rust"`
		Go struct {
			TTS struct {
				Readers string `yaml:"readers"`
			} `yaml:"tts"`
			IVR Config `yaml:"ivr"`
		} `yaml:"go"`
	} `yaml:"services"`
}

// Config controls the scalable telephone IVR edge.
type Config struct {
	Enabled             *bool         `yaml:"enabled"`
	Mode                string        `yaml:"mode"`
	HTTP                httpConfig    `yaml:"http"`
	SIP                 sipConfig     `yaml:"sip"`
	RTP                 rtpConfig     `yaml:"rtp"`
	Cache               cacheConfig   `yaml:"cache"`
	PromptsFile         string        `yaml:"prompts_file"`
	DefaultLanguage     string        `yaml:"default_language"`
	DefaultReaderID     string        `yaml:"default_reader_id"`
	DefaultPackages     []string      `yaml:"default_packages"`
	BroadcastPackages   []string      `yaml:"broadcast_packages"`
	MaxCallSeconds      int           `yaml:"max_call_seconds"`
	DigitTimeoutSeconds int           `yaml:"digit_timeout_seconds"`
	RenderTimeout       time.Duration `yaml:"-"`
	RenderTimeoutRaw    string        `yaml:"render_timeout"`
	MaxConcurrentCalls  int           `yaml:"max_concurrent_calls"`
	MaxRenderInflight   int           `yaml:"max_render_inflight"`
}

type httpConfig struct {
	Enabled bool   `yaml:"enabled"`
	Addr    string `yaml:"addr"`
}

type sipConfig struct {
	Enabled        bool           `yaml:"enabled"`
	Listen         string         `yaml:"listen"`
	ListenPorts    sipListenPorts `yaml:"listen_ports"`
	Domain         string         `yaml:"domain"`
	PublicHost     string         `yaml:"public_host"`
	AllowedSources []string       `yaml:"allowed_sources"`
	Auth           struct {
		Enabled     bool   `yaml:"enabled"`
		Username    string `yaml:"username"`
		PasswordEnv string `yaml:"password_env"`
	} `yaml:"auth"`
	Registration sipRegistrationConfig `yaml:"registration"`
}

type sipListenPort struct {
	Port   int    `yaml:"port"`
	Domain string `yaml:"domain"`
}

type sipListenPorts []sipListenPort

type sipListenBinding struct {
	Addr   string
	Domain string
}

func (ports *sipListenPorts) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.SequenceNode {
		return fmt.Errorf("listen_ports must be a sequence")
	}
	out := make([]sipListenPort, 0, len(value.Content))
	for _, item := range value.Content {
		switch item.Kind {
		case yaml.ScalarNode:
			port, err := strconv.Atoi(strings.TrimSpace(item.Value))
			if err != nil {
				return fmt.Errorf("invalid SIP listen port %q", item.Value)
			}
			out = append(out, sipListenPort{Port: port})
		case yaml.MappingNode:
			var port sipListenPort
			if err := item.Decode(&port); err != nil {
				return err
			}
			out = append(out, port)
		default:
			return fmt.Errorf("invalid SIP listen_ports entry")
		}
	}
	*ports = out
	return nil
}

type sipRegistrationConfig struct {
	Enabled       bool   `yaml:"enabled"`
	Server        string `yaml:"server"`
	Domain        string `yaml:"domain"`
	RegisterURI   string `yaml:"register_uri"`
	Username      string `yaml:"username"`
	AuthUsername  string `yaml:"auth_username"`
	PasswordEnv   string `yaml:"password_env"`
	FromUser      string `yaml:"from_user"`
	ContactUser   string `yaml:"contact_user"`
	ContactHost   string `yaml:"contact_host"`
	ViaHost       string `yaml:"via_host"`
	UserAgent     string `yaml:"user_agent"`
	SupportedPath *bool  `yaml:"supported_path"`
	Expires       int    `yaml:"expires"`
	RetrySeconds  int    `yaml:"retry_seconds"`
}

type rtpConfig struct {
	PortMin int `yaml:"port_min"`
	PortMax int `yaml:"port_max"`
}

type cacheConfig struct {
	Dir              string   `yaml:"dir"`
	TTL              string   `yaml:"ttl"`
	PrewarmCodes     []string `yaml:"prewarm_codes"`
	PhoneSampleRate  int      `yaml:"phone_sample_rate"`
	PhoneCodec       string   `yaml:"phone_codec"`
	MaxEntries       int      `yaml:"max_entries"`
	StampedeWaiters  int      `yaml:"stampede_waiters"`
	RefreshOnStartup bool     `yaml:"refresh_on_startup"`
	StaticOnStartup  bool     `yaml:"static_prompts_on_startup"`
}

type loadedConfig struct {
	Root              rootConfig
	IVR               Config
	BaseDir           string
	PromptsPath       string
	Feeds             []feedXML
	ForecastLocations map[string]locationRecord
	CLCs              map[string]locationRecord
	Geocodes          map[string]locationRecord
	NWS               map[string]locationRecord
	Prompts           PromptConfig
}

type feedsXML struct {
	Feeds []feedXML `xml:"feed"`
}

type feedXML struct {
	ID         string `xml:"id,attr"`
	EnabledRaw string `xml:"enabled,attr"`
	Timezone   string `xml:"timezone,attr"`
	Languages  struct {
		Langs []struct {
			Code string `xml:"code,attr"`
		} `xml:"lang"`
	} `xml:"languages"`
	Locations struct {
		Coverage struct {
			Regions []coverageRegionXML `xml:"region"`
		} `xml:"coverage"`
		ObservationLocations struct {
			Locations []feedLocationXML `xml:"location"`
		} `xml:"observationLocations"`
	} `xml:"locations"`
	Transmitter struct {
		Transmitters []transmitterXML `xml:"transmitter"`
	} `xml:"transmitter_metadata"`
}

type transmitterXML struct {
	Network      transmitterNetworkXML   `xml:"network"`
	SiteName     string                  `xml:"site_name"`
	Callsign     string                  `xml:"callsign"`
	Relationship string                  `xml:"relationship"`
	FrequencyMHz transmitterFrequencyXML `xml:"frequency_mhz"`
	HostName     string                  `xml:"host_name"`
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

type feedLocationXML struct {
	ID           string `xml:"id,attr"`
	Source       string `xml:"source,attr"`
	NameOverride string `xml:"name_override,attr"`
}

type locationRecord struct {
	Code      string `json:"code"`
	Source    string `json:"source"`
	Name      string `json:"name"`
	Province  string `json:"province,omitempty"`
	FeedID    string `json:"feed_id,omitempty"`
	Forecast  string `json:"forecast_id,omitempty"`
	StationID string `json:"station_id,omitempty"`
	Latitude  string `json:"latitude,omitempty"`
	Longitude string `json:"longitude,omitempty"`
}

func loadConfig(path string, overrides Options) (loadedConfig, error) {
	if strings.TrimSpace(path) == "" {
		path = "config.yaml"
	}
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return loadedConfig{}, err
	}
	raw = []byte(os.ExpandEnv(string(raw)))
	var root rootConfig
	if err := yaml.Unmarshal(raw, &root); err != nil {
		return loadedConfig{}, err
	}
	baseDir := filepath.Dir(filepath.Clean(path))
	cfg := root.Services.Go.IVR
	normalizeIVRConfig(&cfg)
	if overrides.HTTPAddr != "" {
		cfg.HTTP.Addr = overrides.HTTPAddr
	}
	if overrides.SIPAddr != "" {
		cfg.SIP.Listen = overrides.SIPAddr
		cfg.SIP.ListenPorts = nil
	}
	if overrides.CacheDir != "" {
		cfg.Cache.Dir = overrides.CacheDir
	}
	feeds, err := loadFeeds(resolvePath(baseDir, fallbackText(root.FeedsFile, "managed/configs/feeds.xml")))
	if err != nil {
		return loadedConfig{}, err
	}
	promptsPath := resolvePath(baseDir, cfg.PromptsFile)
	prompts, err := loadPromptConfig(promptsPath)
	if err != nil {
		return loadedConfig{}, err
	}
	return loadedConfig{
		Root:              root,
		IVR:               cfg,
		BaseDir:           baseDir,
		PromptsPath:       promptsPath,
		Feeds:             feeds,
		ForecastLocations: loadForecastLocationsForBase(baseDir),
		CLCs:              loadCLCsForBase(baseDir),
		Geocodes:          loadGeocodesForBase(baseDir),
		NWS:               loadNWSForBase(baseDir),
		Prompts:           prompts,
	}, nil
}

func normalizeIVRConfig(cfg *Config) {
	if cfg.Mode == "" {
		cfg.Mode = "sip-edge"
	}
	if cfg.HTTP.Addr == "" {
		cfg.HTTP.Addr = "127.0.0.1:8096"
	}
	if cfg.SIP.Listen == "" {
		cfg.SIP.Listen = "0.0.0.0:5060"
	}
	if cfg.SIP.Registration.Expires == 0 {
		cfg.SIP.Registration.Expires = 300
	}
	if cfg.SIP.Registration.RetrySeconds == 0 {
		cfg.SIP.Registration.RetrySeconds = 30
	}
	if cfg.SIP.Registration.UserAgent == "" {
		cfg.SIP.Registration.UserAgent = "Haze Weather Radio IVR"
	}
	if cfg.SIP.Registration.SupportedPath == nil {
		defaultSupportedPath := true
		cfg.SIP.Registration.SupportedPath = &defaultSupportedPath
	}
	if cfg.RTP.PortMin == 0 {
		cfg.RTP.PortMin = 30000
	}
	if cfg.RTP.PortMax == 0 {
		cfg.RTP.PortMax = 39999
	}
	if cfg.Cache.Dir == "" {
		cfg.Cache.Dir = "runtime/ivr/cache"
	}
	if cfg.Cache.TTL == "" {
		cfg.Cache.TTL = "10m"
	}
	if cfg.Cache.PhoneSampleRate == 0 {
		cfg.Cache.PhoneSampleRate = 8000
	}
	if cfg.Cache.PhoneCodec == "" {
		cfg.Cache.PhoneCodec = "pcmu"
	}
	if cfg.Cache.StampedeWaiters == 0 {
		cfg.Cache.StampedeWaiters = 64
	}
	if cfg.PromptsFile == "" {
		cfg.PromptsFile = "managed/configs/ivr.xml"
	}
	if cfg.DefaultLanguage == "" {
		cfg.DefaultLanguage = "en-CA"
	}
	if len(cfg.DefaultPackages) == 0 {
		cfg.DefaultPackages = []string{"current_conditions", "forecast"}
	}
	if len(cfg.BroadcastPackages) == 0 {
		cfg.BroadcastPackages = []string{"alerts", "current_conditions", "air_quality", "forecast", "geophysical_alert"}
	}
	if cfg.MaxCallSeconds == 0 {
		cfg.MaxCallSeconds = 240
	}
	if cfg.DigitTimeoutSeconds == 0 {
		cfg.DigitTimeoutSeconds = 8
	}
	if cfg.RenderTimeoutRaw == "" {
		cfg.RenderTimeoutRaw = "60s"
	}
	timeout, err := time.ParseDuration(cfg.RenderTimeoutRaw)
	if err != nil || timeout <= 0 {
		timeout = 60 * time.Second
	}
	cfg.RenderTimeout = timeout
	if cfg.MaxConcurrentCalls == 0 {
		cfg.MaxConcurrentCalls = 256
	}
	if cfg.MaxRenderInflight == 0 {
		cfg.MaxRenderInflight = 8
	}
}

func (cfg sipConfig) listenBindings() []sipListenBinding {
	host, fallbackPort := sipListenHostPort(cfg.Listen)
	if len(cfg.ListenPorts) == 0 {
		return []sipListenBinding{{
			Addr:   net.JoinHostPort(host, strconv.Itoa(fallbackPort)),
			Domain: normalizeSIPDomain(cfg.Domain),
		}}
	}
	seen := map[string]struct{}{}
	bindings := make([]sipListenBinding, 0, len(cfg.ListenPorts))
	for _, configured := range cfg.ListenPorts {
		port := configured.Port
		if port <= 0 || port > 65535 {
			continue
		}
		addr := net.JoinHostPort(host, strconv.Itoa(port))
		if _, ok := seen[addr]; ok {
			continue
		}
		seen[addr] = struct{}{}
		bindings = append(bindings, sipListenBinding{
			Addr:   addr,
			Domain: normalizeSIPDomain(firstNonBlank(configured.Domain, cfg.Domain)),
		})
	}
	if len(bindings) == 0 {
		return []sipListenBinding{{
			Addr:   net.JoinHostPort(host, strconv.Itoa(fallbackPort)),
			Domain: normalizeSIPDomain(cfg.Domain),
		}}
	}
	return bindings
}

func (cfg sipConfig) listenAddrs() []string {
	bindings := cfg.listenBindings()
	addrs := make([]string, 0, len(bindings))
	for _, binding := range bindings {
		addrs = append(addrs, binding.Addr)
	}
	return addrs
}

func sipListenHostPort(listen string) (string, int) {
	listen = strings.TrimSpace(listen)
	if listen == "" {
		return "0.0.0.0", 5060
	}
	host, portText, err := net.SplitHostPort(listen)
	if err == nil {
		port, portErr := strconv.Atoi(portText)
		if portErr == nil && port > 0 && port <= 65535 {
			if strings.TrimSpace(host) == "" {
				host = "0.0.0.0"
			}
			return host, port
		}
	}
	if port, err := strconv.Atoi(listen); err == nil && port > 0 && port <= 65535 {
		return "0.0.0.0", port
	}
	return "0.0.0.0", 5060
}

func normalizeSIPDomain(value string) string {
	value = strings.TrimSpace(value)
	value = strings.Trim(value, "<>[]")
	value = strings.TrimSuffix(value, ".")
	return strings.ToLower(value)
}

func (cfg loadedConfig) enabled() bool {
	if cfg.IVR.Enabled == nil {
		return false
	}
	return *cfg.IVR.Enabled
}

func (cfg loadedConfig) cacheDir() string {
	return resolvePath(cfg.BaseDir, cfg.IVR.Cache.Dir)
}

func (cfg loadedConfig) ttsReadersPath() string {
	return resolvePath(cfg.BaseDir, fallbackText(cfg.Root.Services.Go.TTS.Readers, "managed/configs/readers.xml"))
}

func (cfg loadedConfig) mediaServiceBaseURL() string {
	media := cfg.Root.Services.Rust.Media
	if !media.Enabled {
		return ""
	}
	addr := strings.TrimSpace(firstNonBlank(media.Addr, media.Listen))
	if addr == "" {
		addr = "127.0.0.1:8097"
	}
	if strings.HasPrefix(addr, "http://") || strings.HasPrefix(addr, "https://") {
		return strings.TrimRight(addr, "/")
	}
	return "http://" + addr
}

func (cfg loadedConfig) ttsReaderFingerprint(readerID string, language string, gender string) string {
	readerID = strings.TrimSpace(readerID)
	if readerID == "" {
		return ""
	}
	readers, err := ttspkg.LoadReaders(cfg.ttsReadersPath())
	if err != nil {
		return "error:" + err.Error()
	}
	reader, ok := ttspkg.SelectReader(readers, readerID, language, gender)
	if !ok {
		return "missing:" + readerID
	}
	return strings.Join([]string{
		reader.ID,
		reader.Provider,
		reader.VoiceID,
		reader.Language,
		reader.Gender,
	}, "|")
}

func (cfg loadedConfig) cacheTTL() time.Duration {
	ttl, err := time.ParseDuration(cfg.IVR.Cache.TTL)
	if err != nil || ttl <= 0 {
		return 10 * time.Minute
	}
	return ttl
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

func loadForecastLocations(path string) map[string]locationRecord {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return map[string]locationRecord{}
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	_, _ = reader.Read()
	header, err := reader.Read()
	if err != nil {
		return map[string]locationRecord{}
	}
	codeIndex := csvIndex(header, "CODE")
	nameIndex := csvIndex(header, "NAME")
	provIndex := csvIndex(header, "PROVINCE/WATERBODY 2 PROVINCE/PLAN D'EAU 2")
	out := map[string]locationRecord{}
	for {
		row, err := reader.Read()
		if err != nil {
			break
		}
		if codeIndex < 0 || nameIndex < 0 || len(row) <= maxInt(codeIndex, nameIndex) {
			continue
		}
		code := strings.Trim(strings.TrimSpace(row[codeIndex]), `"`)
		if code == "" {
			continue
		}
		province := ""
		if provIndex >= 0 && len(row) > provIndex {
			province = strings.Trim(strings.TrimSpace(row[provIndex]), `"`)
		}
		if _, exists := out[code]; !exists {
			out[code] = locationRecord{
				Code:     code,
				Source:   "eccc_forecast",
				Name:     strings.Trim(strings.TrimSpace(row[nameIndex]), `"`),
				Province: province,
			}
		}
	}
	return out
}

func loadForecastLocationsForBase(baseDir string) map[string]locationRecord {
	if snap, ok := locationdb.Load(baseDir); ok {
		out := map[string]locationRecord{}
		for _, place := range snap.PlacesBySource("forecast") {
			out[place.Code] = locationRecord{
				Code:      place.Code,
				Source:    "eccc_forecast",
				Name:      place.Name,
				Province:  place.Region,
				Latitude:  floatText(place.Lat),
				Longitude: floatText(place.Lon),
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return loadForecastLocations(resolvePath(baseDir, "managed/csv/FORECAST_LOCATIONS.csv"))
}

func loadCLCsForBase(baseDir string) map[string]locationRecord {
	if snap, ok := locationdb.Load(baseDir); ok {
		out := map[string]locationRecord{}
		for _, place := range snap.PlacesBySource("clc") {
			out[place.Code] = locationRecord{
				Code:      place.Code,
				Source:    "clc",
				Name:      place.Name,
				Province:  place.Region,
				Latitude:  floatText(place.Lat),
				Longitude: floatText(place.Lon),
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return loadCommaCSV(resolvePath(baseDir, "managed/csv/CLC_Base_Zone.csv"), "CLC", "NAME", "PROVINCE_C", "clc")
}

func loadGeocodesForBase(baseDir string) map[string]locationRecord {
	if snap, ok := locationdb.Load(baseDir); ok {
		out := map[string]locationRecord{}
		for _, place := range snap.PlacesBySource("sgc") {
			out[place.Code] = locationRecord{
				Code:      place.Code,
				Source:    "capcp_geocode",
				Name:      place.Name,
				Province:  place.Region,
				Latitude:  floatText(place.Lat),
				Longitude: floatText(place.Lon),
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	return loadCommaCSV(resolvePath(baseDir, "managed/csv/CAP-CP_Geocodes.csv"), "CAPCPGCODE", "NAME", "PROVINCE_C", "capcp_geocode")
}

func loadNWSForBase(baseDir string) map[string]locationRecord {
	if snap, ok := locationdb.Load(baseDir); ok {
		out := map[string]locationRecord{}
		for _, place := range snap.PlacesBySource("nws_zone") {
			out[place.Code] = locationRecord{Code: place.Code, Source: "nws_zone", Name: place.Name, Province: place.Region, Forecast: place.Code}
		}
		for _, place := range snap.PlacesBySource("nws_same") {
			out[place.Code] = locationRecord{Code: place.Code, Source: "nws_same", Name: place.Name, Province: place.Region}
		}
		for _, place := range snap.PlacesBySource("nws_marine_zone") {
			out[place.Code] = locationRecord{Code: place.Code, Source: "nws_marine_zone", Name: place.Name, Province: place.Region, Forecast: place.Code}
		}
		for _, place := range snap.PlacesBySource("nws_marine_same") {
			out[place.Code] = locationRecord{Code: place.Code, Source: "nws_marine_same", Name: place.Name, Province: place.Region}
		}
		if len(out) > 0 {
			return out
		}
	}
	return loadPipeCSV(resolvePath(baseDir, "managed/csv/NWS_ZONE_COUNTY_CORRELATION.csv"))
}

func floatText(value float64) string {
	if value == 0 {
		return ""
	}
	return fmt.Sprintf("%.8f", value)
}

func loadCommaCSV(path string, codeHeader string, nameHeader string, provinceHeader string, source string) map[string]locationRecord {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return map[string]locationRecord{}
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	header, err := reader.Read()
	if err != nil {
		return map[string]locationRecord{}
	}
	codeIndex := csvIndex(header, codeHeader)
	nameIndex := csvIndex(header, nameHeader)
	provIndex := csvIndex(header, provinceHeader)
	latIndex := csvIndex(header, "LAT_DD")
	lonIndex := csvIndex(header, "LON_DD")
	out := map[string]locationRecord{}
	for {
		row, err := reader.Read()
		if err != nil {
			break
		}
		if codeIndex < 0 || nameIndex < 0 || len(row) <= maxInt(codeIndex, nameIndex) {
			continue
		}
		code := strings.Trim(strings.TrimSpace(row[codeIndex]), `"`)
		if code == "" {
			continue
		}
		province := ""
		if provIndex >= 0 && len(row) > provIndex {
			province = strings.Trim(strings.TrimSpace(row[provIndex]), `"`)
		}
		latitude := ""
		if latIndex >= 0 && len(row) > latIndex {
			latitude = strings.Trim(strings.TrimSpace(row[latIndex]), `"`)
		}
		longitude := ""
		if lonIndex >= 0 && len(row) > lonIndex {
			longitude = strings.Trim(strings.TrimSpace(row[lonIndex]), `"`)
		}
		out[code] = locationRecord{
			Code:      code,
			Source:    source,
			Name:      strings.Trim(strings.TrimSpace(row[nameIndex]), `"`),
			Province:  province,
			Latitude:  latitude,
			Longitude: longitude,
		}
	}
	return out
}

func loadPipeCSV(path string) map[string]locationRecord {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return map[string]locationRecord{}
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.Comma = '|'
	reader.FieldsPerRecord = -1
	header, err := reader.Read()
	if err != nil {
		return map[string]locationRecord{}
	}
	zoneIndex := csvIndex(header, "STATE+ZONE")
	nameIndex := csvIndex(header, "ZONE_NAME")
	countyIndex := csvIndex(header, "COUNTY_NAME")
	fipsIndex := csvIndex(header, "FIPS/SAME")
	stateIndex := csvIndex(header, "STATE")
	out := map[string]locationRecord{}
	for {
		row, err := reader.Read()
		if err != nil {
			break
		}
		if zoneIndex >= 0 && len(row) > zoneIndex {
			code := strings.ToUpper(strings.TrimSpace(row[zoneIndex]))
			if code != "" {
				out[code] = locationRecord{Code: code, Source: "nws_zone", Name: valueAt(row, nameIndex), Province: valueAt(row, stateIndex), Forecast: code}
			}
		}
		if fipsIndex >= 0 && len(row) > fipsIndex {
			code := strings.TrimSpace(row[fipsIndex])
			if code != "" {
				out[code] = locationRecord{Code: code, Source: "nws_same", Name: valueAt(row, countyIndex), Province: valueAt(row, stateIndex)}
			}
		}
	}
	return out
}

func csvIndex(header []string, name string) int {
	for index, value := range header {
		if strings.EqualFold(strings.TrimSpace(value), name) {
			return index
		}
	}
	return -1
}

func valueAt(row []string, index int) string {
	if index < 0 || index >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[index])
}

func maxInt(values ...int) int {
	max := values[0]
	for _, value := range values[1:] {
		if value > max {
			max = value
		}
	}
	return max
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
	if strings.TrimSpace(value) == "" {
		return strings.TrimSpace(fallback)
	}
	return strings.TrimSpace(value)
}

func displayText(value any) string {
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
			}
		}
		for _, key := range []string{"pronunciation", "text", "name"} {
			if text := displayText(merged[key]); text != "" {
				return text
			}
		}
	case map[string]any:
		for _, key := range []string{"pronunciation", "text", "name", "value"} {
			if text := displayText(typed[key]); text != "" {
				return text
			}
		}
	default:
		return strings.TrimSpace(fmt.Sprint(value))
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
