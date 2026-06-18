package dataingest

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
	"gopkg.in/yaml.v3"
)

const serviceID = "haze-data-ingest"

type Options struct {
	ConfigPath string
	Interval   time.Duration
	Timeout    time.Duration
	Once       bool
}

type rootConfig struct {
	FeedsFile string                  `yaml:"feeds_file"`
	Storage   datastore.StorageConfig `yaml:"storage"`
	Services  struct {
		Go struct {
			DataIngest struct {
				Interval string `yaml:"interval"`
				Timeout  string `yaml:"timeout"`
			} `yaml:"data_ingest"`
		} `yaml:"go"`
	} `yaml:"services"`
}

type feedsXML struct {
	Feeds []feedXML `xml:"feed"`
}

type feedXML struct {
	ID         string `xml:"id,attr"`
	EnabledRaw string `xml:"enabled,attr"`
	Timezone   string `xml:"timezone,attr"`
	Locations  struct {
		Coverage struct {
			Regions []coverageRegionXML `xml:"region"`
		} `xml:"coverage"`
		ObservationLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"observationLocations"`
		AirQualityLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"airQualityLocations"`
	} `xml:"locations"`
}

type coverageRegionXML struct {
	ID             string `xml:"id,attr"`
	Source         string `xml:"source,attr"`
	Name           string `xml:"name,attr"`
	DeriveForecast string `xml:"derive_forecast,attr"`
}

type locationXML struct {
	ID           string `xml:"id,attr"`
	Source       string `xml:"source,attr"`
	NameOverride string `xml:"name_override,attr"`
}

type loadedConfig struct {
	Root          rootConfig
	Feeds         []feedXML
	ForecastNames map[string]forecastRegionName
	BaseDir       string
}

type forecastRegionName struct {
	English string
	French  string
}

func Run(ctx context.Context, options Options) error {
	if strings.TrimSpace(options.ConfigPath) == "" {
		options.ConfigPath = "config.yaml"
	}
	if options.Interval <= 0 {
		options.Interval = 45 * time.Minute
	}
	if options.Timeout <= 0 {
		options.Timeout = 20 * time.Second
	}
	loadDotEnv(filepath.Join(filepath.Dir(filepath.Clean(options.ConfigPath)), ".env"))
	loadDotEnv(".env")
	cfg, err := loadConfig(options.ConfigPath)
	if err != nil {
		return err
	}
	interval := configDuration(cfg.Root.Services.Go.DataIngest.Interval, options.Interval)
	timeout := configDuration(cfg.Root.Services.Go.DataIngest.Timeout, options.Timeout)
	publisher := events.Publisher(events.NewJSONLPublisher(os.Stdout))
	if bridgeAddr := os.Getenv("HAZE_HOST_BRIDGE_ADDR"); bridgeAddr != "" {
		publisher = events.NewHostBridgePublisher(bridgeAddr)
	}
	_ = publisher.Publish(events.Event{
		Type:   "service.ready",
		Source: serviceID,
		Data: map[string]any{
			"service": serviceID,
			"feeds":   len(cfg.enabledFeeds()),
		},
	})

	store, err := openStore(ctx, cfg)
	if err != nil {
		return err
	}
	defer store.Close()

	client := &http.Client{Timeout: timeout}
	for {
		if err := fetchOnce(ctx, cfg, client, publisher, store); err != nil {
			log.Printf("data ingest cycle failed: %v", err)
		}
		if options.Once {
			return nil
		}
		timer := time.NewTimer(interval)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func fetchOnce(ctx context.Context, cfg loadedConfig, client *http.Client, publisher events.Publisher, store datastore.Store) error {
	ecccCache := map[string]map[string]any{}
	for _, feed := range cfg.enabledFeeds() {
		for _, loc := range feed.Locations.ObservationLocations.Locations {
			if strings.TrimSpace(loc.ID) == "" {
				continue
			}
			payload, err := fetchObservation(ctx, client, loc)
			if err != nil {
				log.Printf("observation fetch failed for %s/%s: %v", sourceKind(loc.Source), loc.ID, err)
				continue
			}
			payload["station_id"] = loc.ID
			if sourceKind(loc.Source) == "eccc" {
				if raw, ok := payload["_raw_citypage"].(map[string]any); ok {
					ecccCache[loc.ID] = raw
				}
				delete(payload, "_raw_citypage")
			}
			if err := persistObservation(ctx, store, loc, payload); err != nil {
				log.Printf("observation store failed for %s: %v", loc.ID, err)
				continue
			}
			publishDataReady(publisher, feed.ID, "current_conditions", loc.ID)
		}
		for _, region := range feed.Locations.Coverage.Regions {
			if sourceKind(region.Source) != "eccc" {
				continue
			}
			forecastID := fallbackText(region.DeriveForecast, region.ID)
			if strings.TrimSpace(forecastID) == "" {
				continue
			}
			raw := ecccCache[forecastID]
			if raw == nil {
				var err error
				raw, err = fetchECCCCitypage(ctx, client, forecastID)
				if err != nil {
					log.Printf("ECCC forecast fetch failed for %s: %v", forecastID, err)
					continue
				}
				ecccCache[forecastID] = raw
			}
			payload, ok := buildECCCForecast(raw, region, cfg.ForecastNames)
			if !ok {
				continue
			}
			if err := persistForecast(ctx, store, region, forecastID, payload); err != nil {
				log.Printf("forecast store failed for %s: %v", forecastID, err)
				continue
			}
			publishDataReady(publisher, feed.ID, "forecast", forecastID)
		}
		for _, loc := range feed.Locations.AirQualityLocations.Locations {
			if sourceKind(loc.Source) != "eccc" || strings.TrimSpace(loc.ID) == "" {
				continue
			}
			payload, err := fetchAQHI(ctx, client, loc.ID)
			if err != nil {
				log.Printf("ECCC AQHI fetch failed for %s: %v", loc.ID, err)
				continue
			}
			if err := persistAirQuality(ctx, store, loc, payload); err != nil {
				log.Printf("air quality store failed for %s: %v", loc.ID, err)
				continue
			}
			publishDataReady(publisher, feed.ID, "air_quality", loc.ID)
		}
	}
	if text, err := fetchText(ctx, client, "https://services.swpc.noaa.gov/text/wwv.txt"); err == nil && strings.TrimSpace(text) != "" {
		if err := store.StoreTextProduct(ctx, datastore.TextProductRecord{Source: "nws", ID: "wwv", Text: text, Metadata: map[string]any{"source_url": "https://services.swpc.noaa.gov/text/wwv.txt"}}); err == nil {
			publishDataReady(publisher, "", "geophysical_alert", "wwv")
		} else {
			log.Printf("geophysical alert store failed: %v", err)
		}
	}
	if text, sourceURL, err := fetchLatestECCCDiscussion(ctx, client, time.Now().UTC()); err == nil && strings.TrimSpace(text) != "" {
		if err := store.StoreTextProduct(ctx, datastore.TextProductRecord{Source: "eccc", ID: "focn45.cwwg", Text: text, Metadata: map[string]any{"source_url": sourceURL}}); err == nil {
			publishDataReady(publisher, "", "eccc_discussion", sourceURL)
		} else {
			log.Printf("ECCC discussion store failed: %v", err)
		}
	} else if err != nil {
		log.Printf("ECCC discussion fetch failed: %v", err)
	}
	return nil
}

func loadConfig(configPath string) (loadedConfig, error) {
	raw, err := os.ReadFile(filepath.Clean(configPath))
	if err != nil {
		return loadedConfig{}, err
	}
	var root rootConfig
	if err := yaml.Unmarshal(raw, &root); err != nil {
		return loadedConfig{}, err
	}
	baseDir := filepath.Dir(filepath.Clean(configPath))
	feedsPath := resolvePath(baseDir, fallbackText(root.FeedsFile, "managed/configs/feeds.xml"))
	feedsRaw, err := os.ReadFile(filepath.Clean(feedsPath))
	if err != nil {
		return loadedConfig{}, err
	}
	var feeds feedsXML
	if err := xml.Unmarshal(feedsRaw, &feeds); err != nil {
		return loadedConfig{}, err
	}
	return loadedConfig{
		Root:          root,
		Feeds:         feeds.Feeds,
		ForecastNames: loadForecastRegionNames(resolvePath(baseDir, "managed/csv/FORECAST_LOCATIONS.csv")),
		BaseDir:       baseDir,
	}, nil
}

func (cfg loadedConfig) enabledFeeds() []feedXML {
	out := make([]feedXML, 0, len(cfg.Feeds))
	for _, feed := range cfg.Feeds {
		if strings.TrimSpace(feed.ID) != "" && xmlBool(feed.EnabledRaw, true) {
			out = append(out, feed)
		}
	}
	return out
}

func fetchECCCCitypage(ctx context.Context, client *http.Client, locationID string) (map[string]any, error) {
	url := fmt.Sprintf("https://api.weather.gc.ca/collections/citypageweather-realtime/items/%s?f=json", locationID)
	var raw map[string]any
	return raw, fetchJSON(ctx, client, url, &raw)
}

func fetchObservation(ctx context.Context, client *http.Client, loc locationXML) (map[string]any, error) {
	switch sourceKind(loc.Source) {
	case "eccc":
		raw, err := fetchECCCCitypage(ctx, client, loc.ID)
		if err != nil {
			return nil, err
		}
		payload := buildECCCConditions(raw)
		payload["_raw_citypage"] = raw
		return payload, nil
	case "nws":
		if raw, err := fetchTWCObservationWithMetadata(ctx, client, loc.ID); err == nil {
			payload := buildTWCConditions(raw, loc)
			payload["source"] = "nws"
			payload["_source_api"] = "api.weather.com"
			return payload, nil
		}
		raw, err := fetchNWSObservation(ctx, client, loc.ID)
		if err != nil {
			return nil, err
		}
		return buildNWSConditions(raw, loc), nil
	case "twc":
		raw, err := fetchTWCObservationWithMetadata(ctx, client, loc.ID)
		if err != nil {
			return nil, err
		}
		return buildTWCConditions(raw, loc), nil
	default:
		return nil, fmt.Errorf("unsupported observation source %q", loc.Source)
	}
}

func buildECCCConditions(raw map[string]any) map[string]any {
	props := mapAt(raw, "properties")
	cc := mapAt(props, "currentConditions")
	wind := mapAt(cc, "wind")
	return map[string]any{
		"source":      "eccc",
		"observed_at": ev(cc, "timestamp"),
		"station":     bilingual(cc, "station", "value"),
		"properties": map[string]any{
			"temp":       ev(cc, "temperature", "value"),
			"condition":  bilingual(cc, "condition"),
			"wind":       map[string]any{"speed": ev(wind, "speed", "value"), "direction": ev(wind, "direction", "value"), "gust": ev(wind, "gust", "value")},
			"humidity":   ev(cc, "relativeHumidity", "value"),
			"dewpoint":   ev(cc, "dewpoint", "value"),
			"visibility": ev(cc, "visibility", "value"),
			"pressure":   map[string]any{"value": ev(cc, "pressure", "value"), "tendency": bilingual(cc, "pressure", "tendency")},
			"windChill":  ev(cc, "windChill", "value"),
			"humidex":    ev(cc, "humidex", "value"),
			"heatIndex":  nil,
		},
	}
}

func fetchNWSObservation(ctx context.Context, client *http.Client, stationID string) (map[string]any, error) {
	url := fmt.Sprintf("https://api.weather.gov/stations/%s/observations/latest", stationID)
	var raw map[string]any
	return raw, fetchJSON(ctx, client, url, &raw)
}

func buildNWSConditions(raw map[string]any, loc locationXML) map[string]any {
	props := mapAt(raw, "properties")
	windDirection, hasWindDirection := numberAt(mapAt(props, "windDirection"), "value")
	station := fallbackText(loc.NameOverride, firstNonBlank(textAt(props, "name"), textAt(props, "stationIdentifier"), loc.ID))
	return map[string]any{
		"source":      "nws",
		"observed_at": props["timestamp"],
		"station":     map[string]any{"en": station},
		"properties": map[string]any{
			"temp":      roundedNullable(unitValue(props, "temperature")),
			"condition": map[string]any{"en": textAt(props, "textDescription")},
			"wind": map[string]any{
				"speed":     roundedNullable(speedKPH(props, "windSpeed")),
				"direction": degreesToCardinal(windDirection, hasWindDirection),
				"gust":      roundedNullable(speedKPH(props, "windGust")),
			},
			"humidity":   roundedNullable(unitValue(props, "relativeHumidity")),
			"dewpoint":   roundedNullable(unitValue(props, "dewpoint")),
			"visibility": roundedNullable(visibilityKM(props)),
			"pressure":   map[string]any{"value": roundedNullable(pressureKPA(props)), "tendency": nil},
			"windChill":  roundedNullable(unitValue(props, "windChill")),
			"humidex":    nil,
			"heatIndex":  roundedNullable(unitValue(props, "heatIndex")),
		},
	}
}

func fetchTWCObservation(ctx context.Context, client *http.Client, stationID string) (map[string]any, error) {
	apiKey := strings.TrimSpace(os.Getenv("TWC_API_KEY"))
	if apiKey == "" {
		return nil, fmt.Errorf("TWC_API_KEY is not configured")
	}
	url := fmt.Sprintf("https://api.weather.com/v3/wx/observations/current?icaoCode=%s&units=m&language=en-CA&format=json&apiKey=%s", url.QueryEscape(stationID), url.QueryEscape(apiKey))
	var raw map[string]any
	return raw, fetchJSON(ctx, client, url, &raw)
}

func fetchTWCObservationWithMetadata(ctx context.Context, client *http.Client, stationID string) (map[string]any, error) {
	raw, err := fetchTWCObservation(ctx, client, stationID)
	if err != nil {
		return nil, err
	}
	if name, nameErr := fetchTWCLocationName(ctx, client, stationID); nameErr == nil && strings.TrimSpace(name) != "" {
		raw["_twc_location_name"] = strings.TrimSpace(name)
	}
	return raw, nil
}

func fetchTWCLocationName(ctx context.Context, client *http.Client, stationID string) (string, error) {
	apiKey := strings.TrimSpace(os.Getenv("TWC_API_KEY"))
	if apiKey == "" {
		return "", fmt.Errorf("TWC_API_KEY is not configured")
	}
	url := fmt.Sprintf("https://api.weather.com/v3/location/point?icaoCode=%s&language=en-CA&format=json&apiKey=%s", url.QueryEscape(stationID), url.QueryEscape(apiKey))
	var raw map[string]any
	if err := fetchJSON(ctx, client, url, &raw); err != nil {
		return "", err
	}
	location := mapAt(raw, "location")
	return firstNonBlank(
		textAt(location, "displayName"),
		textAt(location, "city"),
		textAt(location, "neighborhood"),
		textAt(location, "postalCode"),
	), nil
}

func buildTWCConditions(raw map[string]any, loc locationXML) map[string]any {
	pressureHPA, _ := numberAt(raw, "pressureMeanSeaLevel")
	var pressureKPA any
	if pressureHPA > 0 {
		pressureKPA = math.Round(pressureHPA) / 10
	}
	return map[string]any{
		"source":      "twc",
		"observed_at": raw["validTimeLocal"],
		"station":     map[string]any{"en": twcStationName(raw, loc)},
		"properties": map[string]any{
			"temp":       raw["temperature"],
			"condition":  map[string]any{"en": firstNonBlank(textAt(raw, "wxPhraseLong"), textAt(raw, "wxPhraseMedium"))},
			"wind":       map[string]any{"speed": raw["windSpeed"], "direction": raw["windDirectionCardinal"], "gust": raw["windGust"]},
			"humidity":   raw["relativeHumidity"],
			"dewpoint":   raw["temperatureDewPoint"],
			"visibility": raw["visibility"],
			"pressure": map[string]any{
				"value":    pressureKPA,
				"tendency": raw["pressureTendencyTrend"],
			},
			"windChill": raw["temperatureWindChill"],
			"humidex":   nil,
			"heatIndex": raw["temperatureHeatIndex"],
		},
	}
}

func twcStationName(raw map[string]any, loc locationXML) string {
	location := mapAt(raw, "location")
	return firstNonBlank(
		loc.NameOverride,
		textAt(raw, "obsName"),
		textAt(raw, "stationName"),
		textAt(raw, "station_name"),
		textAt(raw, "displayName"),
		textAt(raw, "display_name"),
		textAt(raw, "name"),
		textAt(raw, "_twc_location_name"),
		textAt(location, "displayName"),
		textAt(location, "city"),
		textAt(location, "neighborhood"),
		loc.ID,
	)
}

func buildECCCForecast(raw map[string]any, region coverageRegionXML, forecastNames map[string]forecastRegionName) (map[string]any, bool) {
	props := mapAt(raw, "properties")
	group := mapAt(props, "forecastGroup")
	rawForecasts, _ := group["forecasts"].([]any)
	periods := make([]map[string]any, 0, len(rawForecasts))
	for _, item := range rawForecasts {
		forecast, ok := item.(map[string]any)
		if !ok {
			continue
		}
		enParts := []string{}
		frParts := []string{}
		if text := textLang(mapAt(forecast, "cloudPrecip"), "en"); text != "" {
			enParts = append(enParts, text)
		}
		if text := textLang(mapAt(forecast, "cloudPrecip"), "fr"); text != "" {
			frParts = append(frParts, text)
		}
		for _, key := range []string{"temperatures", "windChill"} {
			summary := mapAt(mapAt(forecast, key), "textSummary")
			if text := textLang(summary, "en"); text != "" {
				enParts = append(enParts, text)
			}
			if text := textLang(summary, "fr"); text != "" {
				frParts = append(frParts, text)
			}
		}
		periods = append(periods, map[string]any{
			"period":      bilingual(forecast, "period", "textForecastName"),
			"textSummary": map[string]any{"en": strings.Join(enParts, " "), "fr": strings.Join(frParts, " ")},
		})
	}
	if len(periods) == 0 {
		return nil, false
	}
	return map[string]any{
		"source":          "eccc",
		"forecast":        periods,
		"forecast_region": region.ID,
		"name":            forecastLocationNameBlock(region, forecastNames),
	}, true
}

func forecastLocationNameBlock(region coverageRegionXML, forecastNames map[string]forecastRegionName) map[string]any {
	if names, ok := forecastNames[forecastRegionBaseCode(region.ID)]; ok && (names.English != "" || names.French != "") {
		fallback := firstNonBlank(names.English, names.French, region.ID)
		return map[string]any{
			"en": pauseForecastRegionName(firstNonBlank(names.English, fallback), "en"),
			"fr": pauseForecastRegionName(firstNonBlank(names.French, fallback), "fr"),
		}
	}
	fallback := fallbackText(region.Name, region.ID)
	return map[string]any{"en": pauseForecastRegionName(fallback, "en"), "fr": pauseForecastRegionName(fallback, "fr")}
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

func pauseForecastRegionName(value string, language string) string {
	cleaned := strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
	if cleaned == "" {
		return ""
	}
	parts := strings.Split(cleaned, " - ")
	if len(parts) <= 1 {
		return cleaned
	}
	cleanParts := make([]string, 0, len(parts))
	for index, part := range parts {
		part = strings.TrimSpace(strings.Trim(part, " ,."))
		part = strings.TrimSpace(strings.TrimSuffix(part, " region"))
		part = strings.TrimSpace(strings.TrimSuffix(part, " Region"))
		part = strings.TrimSpace(strings.TrimPrefix(part, "and "))
		if index == len(parts)-1 && strings.Contains(part, " and ") {
			for _, child := range strings.Split(part, " and ") {
				child = strings.TrimSpace(strings.Trim(child, " ,."))
				if child != "" {
					cleanParts = append(cleanParts, child)
				}
			}
			continue
		}
		if part != "" {
			cleanParts = append(cleanParts, part)
		}
	}
	if len(cleanParts) <= 1 {
		return cleaned
	}
	if language == "fr" {
		return strings.Join(cleanParts, ". ")
	}
	if len(cleanParts) == 2 {
		return cleanParts[0] + " and " + cleanParts[1] + " region"
	}
	return strings.Join(cleanParts[:len(cleanParts)-1], ", ") + ", and " + cleanParts[len(cleanParts)-1] + " region"
}

func forecastRegionBaseCode(region string) string {
	region = strings.TrimSpace(region)
	if before, _, ok := strings.Cut(region, "-"); ok {
		return strings.TrimSpace(before)
	}
	return region
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

func fetchAQHI(ctx context.Context, client *http.Client, stationID string) (map[string]any, error) {
	obsURL := fmt.Sprintf("https://api.weather.gc.ca/collections/aqhi-observations-realtime/items?offset=0&limit=1000&sortby=-latest&location_id=%s&f=json", stationID)
	fcstURL := fmt.Sprintf("https://api.weather.gc.ca/collections/aqhi-forecasts-realtime/items?limit=1000&offset=0&f=json&sortby=-publication_datetime&aqhi_type=-Period&location_id=%s", stationID)
	var obsRaw, fcstRaw map[string]any
	_ = fetchJSON(ctx, client, obsURL, &obsRaw)
	_ = fetchJSON(ctx, client, fcstURL, &fcstRaw)
	payload := map[string]any{"source": "eccc"}
	if props := firstFeatureProperties(obsRaw); props != nil {
		payload["location"] = map[string]any{"en": textAt(props, "location_name_en"), "fr": textAt(props, "location_name_fr")}
		payload["observed_at"] = props["observation_datetime"]
		payload["aqhi"] = props["aqhi"]
		payload["special_notes"] = map[string]any{"en": textAt(props, "special_notes_en"), "fr": textAt(props, "special_notes_fr")}
	}
	if props := firstFeatureProperties(fcstRaw); props != nil {
		payload["forecast"] = map[string]any{"published_at": props["publication_datetime"], "periods": aqhiPeriods(props)}
	}
	if len(payload) <= 1 {
		return nil, fmt.Errorf("no AQHI data for %s", stationID)
	}
	return payload, nil
}

func aqhiPeriods(props map[string]any) []map[string]any {
	group := mapAt(props, "forecast_period")
	keys := []string{"period1", "period2", "period3", "period4", "period5", "period6"}
	out := []map[string]any{}
	for _, key := range keys {
		period := mapAt(group, key)
		if len(period) == 0 {
			continue
		}
		out = append(out, map[string]any{
			"period":       map[string]any{"en": textAt(period, "forecast_period_en"), "fr": textAt(period, "forecast_period_fr")},
			"aqhi":         period["aqhi"],
			"aqhi_insmoke": period["aqhi_insmoke"],
		})
	}
	return out
}

func fetchJSON(ctx context.Context, client *http.Client, url string, target any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "HazeWeatherRadio/26.06 data-ingest")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("%s returned %s", url, resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(target)
}

func fetchText(ctx context.Context, client *http.Client, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "HazeWeatherRadio/26.06 data-ingest")
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("%s returned %s", url, resp.Status)
	}
	raw, err := io.ReadAll(resp.Body)
	return string(raw), err
}

var ecccDiscussionLinkPattern = regexp.MustCompile(`href="([^"]*FOCN45_CWWG[^"]*)"`)

func fetchLatestECCCDiscussion(ctx context.Context, client *http.Client, now time.Time) (string, string, error) {
	var lastErr error
	for _, day := range []time.Time{now, now.Add(-24 * time.Hour)} {
		date := day.Format("20060102")
		for hour := 23; hour >= 0; hour-- {
			dirURL := fmt.Sprintf("https://dd.weather.gc.ca/today/bulletins/alphanumeric/%s/FO/CWWG/%02d/", date, hour)
			index, err := fetchText(ctx, client, dirURL)
			if err != nil {
				lastErr = err
				continue
			}
			links := ecccDiscussionLinks(index)
			if len(links) == 0 {
				continue
			}
			sort.Strings(links)
			for i := len(links) - 1; i >= 0; i-- {
				sourceURL := dirURL + strings.TrimLeft(links[i], "/")
				text, err := fetchText(ctx, client, sourceURL)
				if err != nil {
					lastErr = err
					continue
				}
				if strings.TrimSpace(text) != "" {
					return text, sourceURL, nil
				}
			}
		}
	}
	if lastErr != nil {
		return "", "", lastErr
	}
	return "", "", fmt.Errorf("no FOCN45 CWWG discussion bulletin found")
}

func ecccDiscussionLinks(index string) []string {
	matches := ecccDiscussionLinkPattern.FindAllStringSubmatch(index, -1)
	links := make([]string, 0, len(matches))
	seen := map[string]struct{}{}
	for _, match := range matches {
		if len(match) < 2 {
			continue
		}
		link := strings.TrimSpace(match[1])
		if link == "" || strings.Contains(link, "/") {
			continue
		}
		if _, ok := seen[link]; ok {
			continue
		}
		seen[link] = struct{}{}
		links = append(links, link)
	}
	return links
}

func firstFeatureProperties(raw map[string]any) map[string]any {
	features, _ := raw["features"].([]any)
	if len(features) == 0 {
		return nil
	}
	feature, _ := features[0].(map[string]any)
	return mapAt(feature, "properties")
}

func unitValue(props map[string]any, key string) (float64, bool) {
	return numberAt(mapAt(props, key), "value")
}

func speedKPH(props map[string]any, key string) (float64, bool) {
	item := mapAt(props, key)
	value, ok := numberAt(item, "value")
	if !ok {
		return 0, false
	}
	unit := strings.ToLower(textAt(item, "unitCode"))
	if strings.Contains(unit, "m_s-1") || strings.Contains(unit, "m/s") {
		value *= 3.6
	}
	return value, true
}

func visibilityKM(props map[string]any) (float64, bool) {
	item := mapAt(props, "visibility")
	value, ok := numberAt(item, "value")
	if !ok {
		return 0, false
	}
	unit := strings.ToLower(textAt(item, "unitCode"))
	if strings.Contains(unit, "m") && !strings.Contains(unit, "km") {
		value /= 1000
	}
	return value, true
}

func pressureKPA(props map[string]any) (float64, bool) {
	item := mapAt(props, "barometricPressure")
	value, ok := numberAt(item, "value")
	if !ok {
		return 0, false
	}
	unit := strings.ToLower(textAt(item, "unitCode"))
	if strings.Contains(unit, "pa") {
		value /= 1000
	}
	return value, true
}

func numberAt(source map[string]any, key string) (float64, bool) {
	if source == nil {
		return 0, false
	}
	switch value := source[key].(type) {
	case float64:
		return value, true
	case int:
		return float64(value), true
	case json.Number:
		parsed, err := value.Float64()
		return parsed, err == nil
	case string:
		parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		return parsed, err == nil
	default:
		return 0, false
	}
}

func roundedNullable(value float64, ok bool) any {
	if !ok {
		return nil
	}
	return math.Round(value*10) / 10
}

func degreesToCardinal(value float64, ok bool) any {
	if !ok {
		return nil
	}
	dirs := []string{"N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"}
	return dirs[int(math.Round(value/22.5))%len(dirs)]
}

func loadDotEnv(path string) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return
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

func publishDataReady(publisher events.Publisher, feedID string, kind string, subject string) {
	_ = publisher.Publish(events.Event{
		Type:    "data.ready",
		Source:  serviceID,
		Subject: subject,
		Data:    map[string]any{"feed_id": feedID, "kind": kind, "id": subject},
	})
}

func openStore(ctx context.Context, cfg loadedConfig) (datastore.Store, error) {
	store, err := datastore.Open(ctx, cfg.Root.Storage, cfg.BaseDir)
	if err != nil {
		return nil, err
	}
	log.Printf("datastore connected")
	return store, nil
}

func persistObservation(ctx context.Context, store datastore.Store, loc locationXML, payload map[string]any) error {
	if store == nil {
		return datastore.ErrNotConfigured
	}
	source := sourceKind(loc.Source)
	name := firstNonBlank(loc.NameOverride, localizedText(payload["station"], "en"), loc.ID)
	stationID := fallbackText(textValue(payload["station_id"]), loc.ID)
	if err := store.UpsertLocation(ctx, datastore.LocationRecord{
		Source:     source,
		LocationID: loc.ID,
		Kind:       "observation_station",
		NameEN:     name,
		NameFR:     firstNonBlank(localizedText(payload["station"], "fr"), name),
		StationID:  stationID,
		CityPageID: cityPageID(source, loc.ID),
		Metadata: map[string]any{
			"configured_source": loc.Source,
			"name_override":     loc.NameOverride,
		},
	}); err != nil {
		return err
	}
	return store.UpsertObservation(ctx, datastore.ObservationRecord{
		Source:        source,
		LocationID:    loc.ID,
		StationID:     stationID,
		ObservedAtRaw: textValue(payload["observed_at"]),
		Payload:       payload,
	})
}

func persistForecast(ctx context.Context, store datastore.Store, region coverageRegionXML, forecastID string, payload map[string]any) error {
	if store == nil {
		return datastore.ErrNotConfigured
	}
	source := sourceKind(region.Source)
	nameBlock, _ := payload["name"].(map[string]any)
	nameEN := firstNonBlank(localizedText(nameBlock, "en"), region.Name, region.ID)
	if err := store.UpsertLocation(ctx, datastore.LocationRecord{
		Source:     source,
		LocationID: region.ID,
		Kind:       "forecast_region",
		NameEN:     nameEN,
		NameFR:     firstNonBlank(localizedText(nameBlock, "fr"), nameEN),
		CityPageID: forecastID,
		CLC:        region.ID,
		Metadata: map[string]any{
			"derive_forecast": forecastID,
			"configured_name": region.Name,
		},
	}); err != nil {
		return err
	}
	return store.UpsertForecast(ctx, datastore.ForecastRecord{
		Source:       source,
		ForecastID:   forecastID,
		RegionID:     region.ID,
		IssuedAtRaw:  firstNonBlank(textValue(payload["issued_at"]), textValue(payload["published_at"])),
		UpdatedAtRaw: firstNonBlank(textValue(payload["updated_at"]), textValue(payload["last_updated"])),
		Payload:      payload,
	})
}

func persistAirQuality(ctx context.Context, store datastore.Store, loc locationXML, payload map[string]any) error {
	if store == nil {
		return datastore.ErrNotConfigured
	}
	source := sourceKind(loc.Source)
	name := firstNonBlank(loc.NameOverride, localizedText(payload["location"], "en"), loc.ID)
	if err := store.UpsertLocation(ctx, datastore.LocationRecord{
		Source:     source,
		LocationID: loc.ID,
		Kind:       "air_quality_location",
		NameEN:     name,
		NameFR:     firstNonBlank(localizedText(payload["location"], "fr"), name),
		Metadata: map[string]any{
			"configured_source": loc.Source,
			"name_override":     loc.NameOverride,
		},
	}); err != nil {
		return err
	}
	return store.StoreProductPayload(ctx, datastore.ProductPayloadRecord{
		Kind:    "air_quality",
		Source:  source,
		ID:      loc.ID,
		Payload: payload,
	})
}

func cityPageID(source string, id string) string {
	if source == "eccc" {
		return strings.TrimSpace(id)
	}
	return ""
}

func localizedText(value any, lang string) string {
	switch typed := value.(type) {
	case map[string]any:
		return textValue(typed[lang])
	case map[string]string:
		return strings.TrimSpace(typed[lang])
	default:
		return textValue(value)
	}
}

func textValue(value any) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case fmt.Stringer:
		return strings.TrimSpace(typed.String())
	case nil:
		return ""
	default:
		return strings.TrimSpace(fmt.Sprint(typed))
	}
}

func mapAt(source map[string]any, key string) map[string]any {
	if source == nil {
		return nil
	}
	value, _ := source[key].(map[string]any)
	return value
}

func bilingual(source map[string]any, keys ...string) map[string]any {
	current := source
	for _, key := range keys {
		current = mapAt(current, key)
	}
	out := map[string]any{}
	for _, lang := range []string{"en", "fr"} {
		if value, ok := current[lang]; ok {
			out[lang] = value
		}
	}
	return out
}

func ev(source map[string]any, keys ...string) any {
	current := source
	for _, key := range keys {
		current = mapAt(current, key)
	}
	if value, ok := current["en"]; ok {
		return value
	}
	return nil
}

func textLang(source map[string]any, lang string) string {
	if source == nil {
		return ""
	}
	value, _ := source[lang].(string)
	return strings.TrimSpace(value)
}

func textAt(source map[string]any, key string) string {
	value, _ := source[key].(string)
	return strings.TrimSpace(value)
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if text := strings.TrimSpace(value); text != "" {
			return text
		}
	}
	return ""
}

func configDuration(raw string, fallback time.Duration) time.Duration {
	if parsed, err := time.ParseDuration(strings.TrimSpace(raw)); err == nil && parsed > 0 {
		return parsed
	}
	return fallback
}

func sourceKind(raw string) string {
	value := strings.ToLower(strings.TrimSpace(raw))
	if value == "" {
		return "eccc"
	}
	if value == "weather.com" || value == "weatherdotcom" {
		return "twc"
	}
	return value
}

func resolvePath(base string, value string) string {
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

func safeID(value string) string {
	var builder strings.Builder
	for _, ch := range value {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '-' || ch == '_' || ch == '.' {
			builder.WriteRune(ch)
		}
	}
	if builder.Len() == 0 {
		return "item"
	}
	return builder.String()
}
