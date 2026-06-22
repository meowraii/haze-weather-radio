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
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
	"github.com/meowraii/haze-weather-radio/services/go/internal/locationdb"
	"gopkg.in/yaml.v3"
)

const serviceID = "haze-data-ingest"
const thunderstormOutlookNearbyKM = 175.0

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
		ClimateLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"climateLocations"`
		HydrometricLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"hydrometricLocations"`
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
	Latitude     string `xml:"latitude,attr"`
	Longitude    string `xml:"longitude,attr"`
	NormalID     string `xml:"normal_id,attr"`
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
		for _, loc := range feed.Locations.ClimateLocations.Locations {
			if sourceKind(loc.Source) != "eccc" || strings.TrimSpace(loc.ID) == "" {
				continue
			}
			payload, err := fetchClimateSummary(ctx, client, loc, feed.Timezone, time.Now().UTC())
			if err != nil {
				log.Printf("ECCC climate summary fetch failed for %s: %v", loc.ID, err)
				continue
			}
			if err := persistClimateSummary(ctx, store, loc, payload); err != nil {
				log.Printf("climate summary store failed for %s: %v", loc.ID, err)
				continue
			}
			publishDataReady(publisher, feed.ID, "climate_summary", loc.ID)
		}
		fetchFeedSpecialtyProducts(ctx, client, publisher, store, feed, ecccCache)
	}
	if text, err := fetchText(ctx, client, "https://services.swpc.noaa.gov/text/wwv.txt"); err == nil && strings.TrimSpace(text) != "" {
		if err := store.StoreTextProduct(ctx, datastore.TextProductRecord{Source: "nws", ID: "wwv", Text: text, Metadata: map[string]any{"source_url": "https://services.swpc.noaa.gov/text/wwv.txt"}}); err == nil {
			publishDataReady(publisher, "", "geophysical_alert", "wwv")
		} else {
			log.Printf("geophysical alert store failed: %v", err)
		}
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
		ForecastNames: loadForecastRegionNamesForBase(baseDir),
		BaseDir:       baseDir,
	}, nil
}

func loadForecastRegionNamesForBase(baseDir string) map[string]forecastRegionName {
	names := loadForecastRegionNamesFromSQLite(baseDir)
	if len(names) > 0 {
		return names
	}
	return loadForecastRegionNames(resolvePath(baseDir, "managed/csv/FORECAST_LOCATIONS.csv"))
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
		"issued_at":       textValue(ev(group, "timestamp")),
		"updated_at":      props["lastUpdated"],
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

func fetchClimateSummary(ctx context.Context, client *http.Client, loc locationXML, timezone string, now time.Time) (map[string]any, error) {
	daily, timestamp, err := fetchLatestClimateDaily(ctx, client, loc.ID, now)
	if err != nil {
		return nil, err
	}
	props := daily.Properties
	month := intValue(props["LOCAL_MONTH"])
	day := intValue(props["LOCAL_DAY"])
	var climateDate time.Time
	if month == 0 {
		if parsed, parseErr := parseClimateDate(textValue(props["LOCAL_DATE"])); parseErr == nil {
			month = int(parsed.Month())
			day = parsed.Day()
			climateDate = parsed
		}
	}
	if climateDate.IsZero() {
		if parsed, parseErr := parseClimateDate(textValue(props["LOCAL_DATE"])); parseErr == nil {
			climateDate = parsed
		}
	}
	normalID := firstNonBlank(loc.NormalID, textValue(props["CLIMATE_IDENTIFIER"]), loc.ID)
	normals := map[string]any{}
	if month > 0 {
		if fetched, normalErr := fetchClimateNormals(ctx, client, normalID, month); normalErr == nil {
			normals = fetched
		} else {
			log.Printf("ECCC climate normals fetch failed for %s: %v", normalID, normalErr)
		}
	}
	stationName := firstNonBlank(loc.NameOverride, titleText(textValue(props["STATION_NAME"])), loc.ID)
	obs := map[string]any{
		"station":             map[string]any{"en": stationName, "fr": stationName},
		"date":                textValue(props["LOCAL_DATE"]),
		"high":                climateNumber(props, "MAX_TEMPERATURE"),
		"low":                 climateNumber(props, "MIN_TEMPERATURE"),
		"mean":                climateNumber(props, "MEAN_TEMPERATURE"),
		"precipitation":       climateNumber(props, "TOTAL_PRECIPITATION"),
		"precipitation_trace": climateTrace(props, "TOTAL_PRECIPITATION"),
		"rain":                climateNumber(props, "TOTAL_RAIN"),
		"rain_trace":          climateTrace(props, "TOTAL_RAIN"),
		"snowfall":            climateNumber(props, "TOTAL_SNOW"),
		"snowfall_trace":      climateTrace(props, "TOTAL_SNOW"),
		"snow_on_ground":      climateNumber(props, "SNOW_ON_GROUND"),
		"max_gust_speed":      climateNumber(props, "SPEED_MAX_GUST"),
		"max_gust_direction":  climateGustDirection(props),
		"heating_degree_days": climateNumber(props, "HEATING_DEGREE_DAYS"),
		"cooling_degree_days": climateNumber(props, "COOLING_DEGREE_DAYS"),
		"min_humidity":        climateNumber(props, "MIN_REL_HUMIDITY"),
	}
	lon, lat, hasPoint := configuredPoint(loc)
	if !hasPoint {
		lon, lat, hasPoint = featurePoint(daily)
	}
	records := map[string]any{}
	if hasPoint && month > 0 && day > 0 {
		if fetched, recordsErr := fetchClimateRecords(ctx, client, month, day, lon, lat); recordsErr == nil {
			records = fetched
		} else {
			log.Printf("ECCC climate records fetch failed for %s: %v", loc.ID, recordsErr)
		}
	}
	astronomy := map[string]any{}
	if hasPoint && !climateDate.IsZero() {
		if sun, ok := climateSunriseSunset(climateDate, lon, lat, timezone); ok {
			astronomy = sun
		}
	}
	return map[string]any{
		"source":       "eccc",
		"name":         map[string]any{"en": stationName, "fr": stationName},
		"last_updated": firstNonBlank(timestamp, now.Format(time.RFC3339)),
		"observations": obs,
		"normals":      normals,
		"records":      records,
		"astronomy":    astronomy,
		"metadata": map[string]any{
			"collection":          "climate-daily",
			"station_id":          textValue(props["STN_ID"]),
			"climate_identifier":  firstNonBlank(textValue(props["CLIMATE_IDENTIFIER"]), loc.ID),
			"normal_identifier":   normalID,
			"acceptable_fields":   climateAcceptableFieldList(),
			"accepted_flag_notes": "Only blank-value flags and trace precipitation flags are rendered.",
			"records_collection":  "ltce-temperature, ltce-precipitation, ltce-snowfall",
			"astronomy_source":    "computed from the climate station coordinates",
		},
	}, nil
}

func fetchLatestClimateDaily(ctx context.Context, client *http.Client, stationID string, now time.Time) (geoFeature, string, error) {
	query := "limit=14&sortby=-LOCAL_DATE&CLIMATE_IDENTIFIER=" + url.QueryEscape(strings.TrimSpace(stationID))
	features, timestamp, err := fetchCollectionFeatures(ctx, client, "climate-daily", query)
	if err != nil {
		return geoFeature{}, "", err
	}
	for _, feature := range features {
		if climateDailyUsable(feature.Properties, now) {
			return feature, timestamp, nil
		}
	}
	return geoFeature{}, "", fmt.Errorf("no usable climate daily rows for %s", stationID)
}

func climateDailyUsable(props map[string]any, now time.Time) bool {
	if props == nil {
		return false
	}
	if parsed, err := parseClimateDate(textValue(props["LOCAL_DATE"])); err == nil && parsed.After(now.Add(24*time.Hour)) {
		return false
	}
	for _, field := range climateAcceptableFieldList() {
		if climateNumber(props, field) != nil || climateTrace(props, field) {
			return true
		}
	}
	return false
}

func climateAcceptableFieldList() []string {
	return []string{
		"MAX_TEMPERATURE",
		"MIN_TEMPERATURE",
		"MEAN_TEMPERATURE",
		"TOTAL_PRECIPITATION",
		"TOTAL_RAIN",
		"TOTAL_SNOW",
		"SNOW_ON_GROUND",
		"SPEED_MAX_GUST",
		"DIRECTION_MAX_GUST",
		"HEATING_DEGREE_DAYS",
		"COOLING_DEGREE_DAYS",
		"MIN_REL_HUMIDITY",
	}
}

func fetchClimateNormals(ctx context.Context, client *http.Client, stationID string, month int) (map[string]any, error) {
	query := fmt.Sprintf("limit=120&CLIMATE_IDENTIFIER=%s&MONTH=%d", url.QueryEscape(strings.TrimSpace(stationID)), month)
	features, _, err := fetchCollectionFeatures(ctx, client, "climate-normals", query)
	if err != nil {
		return nil, err
	}
	values := map[int]float64{}
	station := ""
	periodBegin := 0
	periodEnd := 0
	for _, feature := range features {
		props := feature.Properties
		id := intValue(props["NORMAL_ID"])
		if id == 0 || !climateNormalUsable(props) {
			continue
		}
		value, ok := numberValue(props["VALUE"])
		if !ok {
			continue
		}
		values[id] = math.Round(value*10) / 10
		if station == "" {
			station = titleText(textValue(props["STATION_NAME"]))
		}
		if periodBegin == 0 {
			periodBegin = intValue(props["PERIOD_BEGIN"])
		}
		if periodEnd == 0 {
			periodEnd = intValue(props["PERIOD_END"])
		}
	}
	if len(values) == 0 {
		return nil, fmt.Errorf("no usable climate normals for %s month %d", stationID, month)
	}
	temperature := map[string]any{}
	if value, ok := values[5]; ok {
		temperature["high"] = value
	}
	if value, ok := values[8]; ok {
		temperature["low"] = value
	}
	if value, ok := values[1]; ok {
		temperature["mean"] = value
	}
	normals := map[string]any{
		"station":     map[string]any{"en": station, "fr": station},
		"month":       month,
		"temperature": temperature,
	}
	if periodBegin > 0 && periodEnd > 0 {
		normals["period"] = fmt.Sprintf("%d to %d", periodBegin, periodEnd)
	}
	if value, ok := values[56]; ok {
		normals["precipitation"] = value
	}
	if value, ok := values[52]; ok {
		normals["rainfall"] = value
	}
	if value, ok := values[54]; ok {
		normals["snowfall"] = value
	}
	if value, ok := values[90]; ok {
		normals["wind_speed"] = value
	}
	return normals, nil
}

func fetchClimateRecords(ctx context.Context, client *http.Client, month int, day int, lon float64, lat float64) (map[string]any, error) {
	records := map[string]any{}
	var firstErr error
	if feature, ok, err := fetchClosestClimateRecordFeature(ctx, client, "ltce-temperature", month, day, lon, lat); err != nil {
		firstErr = err
	} else if ok {
		props := feature.Properties
		addClimateRecord(records, "high_temperature", props, "RECORD_HIGH_MAX_TEMP", "RECORD_HIGH_MAX_TEMP_YR", true)
		addClimateRecord(records, "low_temperature", props, "RECORD_LOW_MIN_TEMP", "RECORD_LOW_MIN_TEMP_YR", true)
	}
	if feature, ok, err := fetchClosestClimateRecordFeature(ctx, client, "ltce-precipitation", month, day, lon, lat); err != nil {
		if firstErr == nil {
			firstErr = err
		}
	} else if ok {
		addClimateRecord(records, "precipitation", feature.Properties, "RECORD_PRECIPITATION", "RECORD_PRECIPITATION_YR", false)
	}
	if feature, ok, err := fetchClosestClimateRecordFeature(ctx, client, "ltce-snowfall", month, day, lon, lat); err != nil {
		if firstErr == nil {
			firstErr = err
		}
	} else if ok {
		addClimateRecord(records, "snowfall", feature.Properties, "RECORD_SNOWFALL", "RECORD_SNOWFALL_YR", false)
	}
	if len(records) == 0 && firstErr != nil {
		return nil, firstErr
	}
	return records, nil
}

func fetchClosestClimateRecordFeature(ctx context.Context, client *http.Client, collection string, month int, day int, lon float64, lat float64) (geoFeature, bool, error) {
	for _, radius := range []float64{0.75, 1.5} {
		query := fmt.Sprintf(
			"limit=50&LOCAL_MONTH=%d&LOCAL_DAY=%d&bbox=%.5f,%.5f,%.5f,%.5f",
			month,
			day,
			lon-radius,
			lat-radius,
			lon+radius,
			lat+radius,
		)
		features, _, err := fetchCollectionFeatures(ctx, client, collection, query)
		if err != nil {
			return geoFeature{}, false, err
		}
		if feature, ok := closestFeature(features, lon, lat); ok {
			return feature, true, nil
		}
	}
	return geoFeature{}, false, nil
}

func closestFeature(features []geoFeature, lon float64, lat float64) (geoFeature, bool) {
	var best geoFeature
	bestDistance := math.Inf(1)
	for _, feature := range features {
		featureLon, featureLat, ok := featurePoint(feature)
		if !ok {
			continue
		}
		dLon := featureLon - lon
		dLat := featureLat - lat
		distance := dLon*dLon + dLat*dLat
		if distance < bestDistance {
			best = feature
			bestDistance = distance
		}
	}
	return best, !math.IsInf(bestDistance, 1)
}

func featurePoint(feature geoFeature) (float64, float64, bool) {
	coordinates, _ := feature.Geometry["coordinates"].([]any)
	if len(coordinates) < 2 {
		return 0, 0, false
	}
	lon, lonOK := numberValue(coordinates[0])
	lat, latOK := numberValue(coordinates[1])
	if !lonOK || !latOK {
		return 0, 0, false
	}
	return lon, lat, true
}

func geoFeatureDistanceToPointKM(feature geoFeature, lon float64, lat float64) (float64, bool) {
	return geoGeometryDistanceToPointKM(feature.Geometry, lon, lat)
}

func geoFeatureDirectionFromPoint(feature geoFeature, lon float64, lat float64) (string, bool) {
	targetLon, targetLat, ok := geoGeometryCentroid(feature.Geometry)
	if !ok {
		return "", false
	}
	return cardinalDirection(lon, lat, targetLon, targetLat), true
}

func geoGeometryDistanceToPointKM(geometry map[string]any, lon float64, lat float64) (float64, bool) {
	geometryType := strings.ToLower(strings.TrimSpace(textValue(geometry["type"])))
	coordinates, _ := geometry["coordinates"].([]any)
	switch geometryType {
	case "point":
		if len(coordinates) < 2 {
			return 0, false
		}
		pointLon, lonOK := numberValue(coordinates[0])
		pointLat, latOK := numberValue(coordinates[1])
		if !lonOK || !latOK {
			return 0, false
		}
		return haversineKM(lon, lat, pointLon, pointLat), true
	case "linestring":
		return lineStringDistanceToPointKM(coordinates, lon, lat)
	case "polygon":
		return polygonDistanceToPointKM(coordinates, lon, lat)
	case "multipolygon":
		best := math.Inf(1)
		for _, polygonRaw := range coordinates {
			polygon, _ := polygonRaw.([]any)
			if distance, ok := polygonDistanceToPointKM(polygon, lon, lat); ok && distance < best {
				best = distance
			}
		}
		return best, !math.IsInf(best, 1)
	default:
		return 0, false
	}
}

func geoGeometryCentroid(geometry map[string]any) (float64, float64, bool) {
	geometryType := strings.ToLower(strings.TrimSpace(textValue(geometry["type"])))
	coordinates, _ := geometry["coordinates"].([]any)
	switch geometryType {
	case "point":
		if len(coordinates) < 2 {
			return 0, 0, false
		}
		lon, lonOK := numberValue(coordinates[0])
		lat, latOK := numberValue(coordinates[1])
		return lon, lat, lonOK && latOK
	case "linestring":
		return coordinateListCentroid(coordinates)
	case "polygon":
		if len(coordinates) == 0 {
			return 0, 0, false
		}
		outer, _ := coordinates[0].([]any)
		return coordinateListCentroid(outer)
	case "multipolygon":
		lons := []float64{}
		lats := []float64{}
		for _, polygonRaw := range coordinates {
			polygon, _ := polygonRaw.([]any)
			if len(polygon) == 0 {
				continue
			}
			outer, _ := polygon[0].([]any)
			if lon, lat, ok := coordinateListCentroid(outer); ok {
				lons = append(lons, lon)
				lats = append(lats, lat)
			}
		}
		return averageLonLat(lons, lats)
	default:
		return 0, 0, false
	}
}

func coordinateListCentroid(points []any) (float64, float64, bool) {
	lons := []float64{}
	lats := []float64{}
	for index, raw := range points {
		if index == len(points)-1 && len(points) > 1 {
			if sameCoordinate(points[0], raw) {
				continue
			}
		}
		point, _ := raw.([]any)
		if len(point) < 2 {
			continue
		}
		lon, lonOK := numberValue(point[0])
		lat, latOK := numberValue(point[1])
		if lonOK && latOK {
			lons = append(lons, lon)
			lats = append(lats, lat)
		}
	}
	return averageLonLat(lons, lats)
}

func sameCoordinate(left any, right any) bool {
	leftPoint, _ := left.([]any)
	rightPoint, _ := right.([]any)
	if len(leftPoint) < 2 || len(rightPoint) < 2 {
		return false
	}
	leftLon, leftLonOK := numberValue(leftPoint[0])
	leftLat, leftLatOK := numberValue(leftPoint[1])
	rightLon, rightLonOK := numberValue(rightPoint[0])
	rightLat, rightLatOK := numberValue(rightPoint[1])
	return leftLonOK && leftLatOK && rightLonOK && rightLatOK && math.Abs(leftLon-rightLon) < 1e-9 && math.Abs(leftLat-rightLat) < 1e-9
}

func averageLonLat(lons []float64, lats []float64) (float64, float64, bool) {
	if len(lons) == 0 || len(lons) != len(lats) {
		return 0, 0, false
	}
	lonTotal := 0.0
	latTotal := 0.0
	for index := range lons {
		lonTotal += lons[index]
		latTotal += lats[index]
	}
	count := float64(len(lons))
	return lonTotal / count, latTotal / count, true
}

func polygonDistanceToPointKM(rings []any, lon float64, lat float64) (float64, bool) {
	if len(rings) == 0 {
		return 0, false
	}
	outer, ok := rings[0].([]any)
	if !ok || len(outer) < 4 {
		return 0, false
	}
	if pointInRing(lon, lat, outer) {
		inHole := false
		for _, holeRaw := range rings[1:] {
			hole, _ := holeRaw.([]any)
			if len(hole) >= 4 && pointInRing(lon, lat, hole) {
				inHole = true
				break
			}
		}
		if !inHole {
			return 0, true
		}
	}
	return lineStringDistanceToPointKM(outer, lon, lat)
}

func lineStringDistanceToPointKM(points []any, lon float64, lat float64) (float64, bool) {
	if len(points) == 0 {
		return 0, false
	}
	best := math.Inf(1)
	var prevLon, prevLat float64
	hasPrev := false
	for _, raw := range points {
		point, _ := raw.([]any)
		if len(point) < 2 {
			continue
		}
		pointLon, lonOK := numberValue(point[0])
		pointLat, latOK := numberValue(point[1])
		if !lonOK || !latOK {
			continue
		}
		if !hasPrev {
			distance := haversineKM(lon, lat, pointLon, pointLat)
			if distance < best {
				best = distance
			}
			prevLon, prevLat = pointLon, pointLat
			hasPrev = true
			continue
		}
		distance := segmentDistanceToPointKM(prevLon, prevLat, pointLon, pointLat, lon, lat)
		if distance < best {
			best = distance
		}
		prevLon, prevLat = pointLon, pointLat
	}
	return best, !math.IsInf(best, 1)
}

func pointInRing(lon float64, lat float64, ring []any) bool {
	inside := false
	j := len(ring) - 1
	for i := 0; i < len(ring); i++ {
		left, _ := ring[i].([]any)
		right, _ := ring[j].([]any)
		if len(left) < 2 || len(right) < 2 {
			j = i
			continue
		}
		leftLon, leftLonOK := numberValue(left[0])
		leftLat, leftLatOK := numberValue(left[1])
		rightLon, rightLonOK := numberValue(right[0])
		rightLat, rightLatOK := numberValue(right[1])
		if !leftLonOK || !leftLatOK || !rightLonOK || !rightLatOK {
			j = i
			continue
		}
		if (leftLat > lat) != (rightLat > lat) {
			crossLon := (rightLon-leftLon)*(lat-leftLat)/(rightLat-leftLat) + leftLon
			if lon < crossLon {
				inside = !inside
			}
		}
		j = i
	}
	return inside
}

func segmentDistanceToPointKM(lon1 float64, lat1 float64, lon2 float64, lat2 float64, lon float64, lat float64) float64 {
	const kmPerDegreeLat = 111.32
	cosLat := math.Cos(degreesToRadians(lat))
	x1 := lon1 * cosLat * kmPerDegreeLat
	y1 := lat1 * kmPerDegreeLat
	x2 := lon2 * cosLat * kmPerDegreeLat
	y2 := lat2 * kmPerDegreeLat
	px := lon * cosLat * kmPerDegreeLat
	py := lat * kmPerDegreeLat
	dx := x2 - x1
	dy := y2 - y1
	if math.Abs(dx) < 1e-9 && math.Abs(dy) < 1e-9 {
		return math.Hypot(px-x1, py-y1)
	}
	t := ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
	t = math.Max(0, math.Min(1, t))
	closestX := x1 + t*dx
	closestY := y1 + t*dy
	return math.Hypot(px-closestX, py-closestY)
}

func haversineKM(lon1 float64, lat1 float64, lon2 float64, lat2 float64) float64 {
	const earthRadiusKM = 6371.0
	dLat := degreesToRadians(lat2 - lat1)
	dLon := degreesToRadians(lon2 - lon1)
	lat1Rad := degreesToRadians(lat1)
	lat2Rad := degreesToRadians(lat2)
	a := math.Sin(dLat/2)*math.Sin(dLat/2) + math.Cos(lat1Rad)*math.Cos(lat2Rad)*math.Sin(dLon/2)*math.Sin(dLon/2)
	return earthRadiusKM * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

func cardinalDirection(fromLon float64, fromLat float64, toLon float64, toLat float64) string {
	dx := (toLon - fromLon) * math.Cos(degreesToRadians(fromLat))
	dy := toLat - fromLat
	if math.Abs(dx) < 1e-9 && math.Abs(dy) < 1e-9 {
		return ""
	}
	degrees := math.Mod(radiansToDegrees(math.Atan2(dx, dy))+360, 360)
	directions := []string{"north", "north east", "east", "south east", "south", "south west", "west", "north west"}
	index := int(math.Floor((degrees+22.5)/45.0)) % len(directions)
	return directions[index]
}

func addClimateRecord(records map[string]any, key string, props map[string]any, valueField string, yearField string, includeZero bool) {
	value, valueOK := numberValue(props[valueField])
	year := intValue(props[yearField])
	if !valueOK || year <= 0 {
		return
	}
	if !includeZero && math.Abs(value) < 0.05 {
		return
	}
	records[key] = map[string]any{
		"value": math.Round(value*10) / 10,
		"year":  year,
	}
}

func climateSunriseSunset(date time.Time, lon float64, lat float64, timezone string) (map[string]any, bool) {
	loc, err := time.LoadLocation(fallbackText(timezone, "UTC"))
	if err != nil {
		loc = time.UTC
	}
	localDate := time.Date(date.Year(), date.Month(), date.Day(), 0, 0, 0, 0, loc)
	sunrise, sunriseOK := solarEvent(localDate, lat, lon, true, loc)
	sunset, sunsetOK := solarEvent(localDate, lat, lon, false, loc)
	if !sunriseOK || !sunsetOK {
		return nil, false
	}
	return map[string]any{
		"sunrise":  sunrise.Format(time.RFC3339),
		"sunset":   sunset.Format(time.RFC3339),
		"timezone": timezone,
	}, true
}

func solarEvent(localDate time.Time, lat float64, lon float64, sunrise bool, loc *time.Location) (time.Time, bool) {
	day := float64(localDate.YearDay())
	longitudeHour := lon / 15
	targetHour := 18.0
	if sunrise {
		targetHour = 6
	}
	t := day + ((targetHour - longitudeHour) / 24)
	meanAnomaly := normalizeDegrees((0.9856 * t) - 3.289)
	trueLongitude := normalizeDegrees(meanAnomaly + (1.916 * sinDegrees(meanAnomaly)) + (0.020 * sinDegrees(2*meanAnomaly)) + 282.634)
	rightAscension := normalizeDegrees(radiansToDegrees(math.Atan(0.91764 * tanDegrees(trueLongitude))))
	rightAscension += math.Floor(trueLongitude/90)*90 - math.Floor(rightAscension/90)*90
	rightAscension /= 15
	sinDeclination := 0.39782 * sinDegrees(trueLongitude)
	cosDeclination := math.Cos(math.Asin(sinDeclination))
	cosHour := (cosDegrees(90.833) - (sinDeclination * sinDegrees(lat))) / (cosDeclination * cosDegrees(lat))
	if cosHour > 1 || cosHour < -1 {
		return time.Time{}, false
	}
	hourAngle := radiansToDegrees(math.Acos(cosHour))
	if sunrise {
		hourAngle = 360 - hourAngle
	}
	hourAngle /= 15
	localMeanTime := hourAngle + rightAscension - (0.06571 * t) - 6.622
	utcHour := normalizeHours(localMeanTime - longitudeHour)
	event := time.Date(localDate.Year(), localDate.Month(), localDate.Day(), 0, 0, 0, 0, time.UTC).
		Add(time.Duration(math.Round(utcHour*3600)) * time.Second)
	return alignSolarEventToLocalDate(event, localDate, loc), true
}

func alignSolarEventToLocalDate(event time.Time, localDate time.Time, loc *time.Location) time.Time {
	target := dateOnly(localDate, loc)
	for i := 0; i < 3; i++ {
		current := dateOnly(event.In(loc), loc)
		switch {
		case current.Equal(target):
			return event
		case current.Before(target):
			event = event.Add(24 * time.Hour)
		default:
			event = event.Add(-24 * time.Hour)
		}
	}
	return event
}

func dateOnly(value time.Time, loc *time.Location) time.Time {
	year, month, day := value.In(loc).Date()
	return time.Date(year, month, day, 0, 0, 0, 0, loc)
}

func normalizeDegrees(value float64) float64 {
	out := math.Mod(value, 360)
	if out < 0 {
		out += 360
	}
	return out
}

func normalizeHours(value float64) float64 {
	out := math.Mod(value, 24)
	if out < 0 {
		out += 24
	}
	return out
}

func sinDegrees(value float64) float64 {
	return math.Sin(degreesToRadians(value))
}

func cosDegrees(value float64) float64 {
	return math.Cos(degreesToRadians(value))
}

func tanDegrees(value float64) float64 {
	return math.Tan(degreesToRadians(value))
}

func degreesToRadians(value float64) float64 {
	return value * math.Pi / 180
}

func radiansToDegrees(value float64) float64 {
	return value * 180 / math.Pi
}

func climateNormalUsable(props map[string]any) bool {
	if strings.EqualFold(strings.TrimSpace(textValue(props["CURRENT_FLAG"])), "N") {
		return false
	}
	period := strings.ToUpper(strings.TrimSpace(textValue(props["PERIOD"])))
	if period != "" && period != "NORM" {
		return false
	}
	if years := intValue(props["YEAR_COUNT_NORMAL_PERIOD"]); years > 0 && years < 15 {
		return false
	}
	if percent, ok := numberValue(props["PERCENT_OF_POSSIBLE_OBS"]); ok && percent < 80 {
		return false
	}
	return true
}

func climateNumber(props map[string]any, field string) any {
	if !climateFlagOK(textValue(props[field+"_FLAG"]), field) {
		return nil
	}
	value, ok := numberValue(props[field])
	if !ok {
		return nil
	}
	return math.Round(value*10) / 10
}

func climateTrace(props map[string]any, field string) bool {
	if !climateTraceCapable(field) {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(textValue(props[field+"_FLAG"])), "T")
}

func climateFlagOK(flag string, field string) bool {
	flag = strings.ToUpper(strings.TrimSpace(flag))
	if flag == "" {
		return true
	}
	return flag == "T" && climateTraceCapable(field)
}

func climateTraceCapable(field string) bool {
	switch field {
	case "TOTAL_PRECIPITATION", "TOTAL_RAIN", "TOTAL_SNOW", "SNOW_ON_GROUND":
		return true
	default:
		return false
	}
}

func climateGustDirection(props map[string]any) any {
	raw, ok := numberValue(climateNumber(props, "DIRECTION_MAX_GUST"))
	if !ok {
		return nil
	}
	if raw > 0 && raw <= 36 {
		raw *= 10
	}
	return degreesToCardinal(raw, true)
}

func parseClimateDate(raw string) (time.Time, error) {
	raw = strings.TrimSpace(raw)
	for _, layout := range []string{"2006-01-02 15:04:05", "2006-01-02"} {
		parsed, err := time.Parse(layout, raw)
		if err == nil {
			return parsed, nil
		}
	}
	return time.Time{}, fmt.Errorf("invalid climate date %q", raw)
}

func aqhiPeriods(props map[string]any) []map[string]any {
	group := mapAt(props, "forecast_period")
	keys := []string{"period_1", "period1", "period_2", "period2", "period_3", "period3", "period_4", "period4", "period_5", "period5", "period_6", "period6"}
	out := []map[string]any{}
	seen := map[string]struct{}{}
	for _, key := range keys {
		period := mapAt(group, key)
		if len(period) == 0 {
			continue
		}
		normalizedKey := strings.ReplaceAll(key, "_", "")
		if _, ok := seen[normalizedKey]; ok {
			continue
		}
		seen[normalizedKey] = struct{}{}
		out = append(out, map[string]any{
			"period":       map[string]any{"en": textAt(period, "forecast_period_en"), "fr": textAt(period, "forecast_period_fr")},
			"aqhi":         period["aqhi"],
			"aqhi_insmoke": period["aqhi_insmoke"],
		})
	}
	return out
}

type geoFeatureCollection struct {
	Features       []geoFeature `json:"features"`
	NumberMatched  int          `json:"numberMatched"`
	NumberReturned int          `json:"numberReturned"`
	TimeStamp      string       `json:"timeStamp"`
}

type geoFeature struct {
	ID         string         `json:"id"`
	Geometry   map[string]any `json:"geometry"`
	Properties map[string]any `json:"properties"`
}

func fetchFeedSpecialtyProducts(ctx context.Context, client *http.Client, publisher events.Publisher, store datastore.Store, feed feedXML, ecccCache map[string]map[string]any) {
	fetchers := []struct {
		kind string
		fn   func(context.Context, *http.Client, feedXML, map[string]map[string]any) (map[string]any, error)
	}{
		{"thunderstorm_outlook", fetchThunderstormOutlookProduct},
		{"coastal_flood", fetchCoastalFloodProduct},
		{"hurricane_tracks", fetchHurricaneTracksProduct},
		{"hydrometric", fetchHydrometricProduct},
		{"precipitation_analysis", fetchPrecipitationAnalysisProduct},
	}
	for _, fetcher := range fetchers {
		payload, err := fetcher.fn(ctx, client, feed, ecccCache)
		if err != nil {
			log.Printf("%s fetch failed for feed %s: %v", fetcher.kind, feed.ID, err)
			continue
		}
		if len(payload) == 0 {
			continue
		}
		if err := store.StoreProductPayload(ctx, datastore.ProductPayloadRecord{
			Kind:    fetcher.kind,
			Source:  "eccc",
			ID:      feed.ID,
			Payload: payload,
		}); err != nil {
			log.Printf("%s store failed for feed %s: %v", fetcher.kind, feed.ID, err)
			continue
		}
		publishDataReady(publisher, feed.ID, fetcher.kind, feed.ID)
	}
}

func fetchThunderstormOutlookProduct(ctx context.Context, client *http.Client, feed feedXML, ecccCache map[string]map[string]any) (map[string]any, error) {
	features, timestamp, err := fetchCollectionFeatures(ctx, client, "thunderstorm_outlook", "limit=1000&sortby=-validity_datetime")
	if err != nil {
		return nil, err
	}
	if ecccCache == nil {
		ecccCache = map[string]map[string]any{}
	}
	refLon, refLat, _, hasReference := feedReferencePoint(ctx, client, feed, ecccCache)
	subtypes := feedSpecialtySubtypes(feed)
	now := time.Now().UTC()
	items := []map[string]any{}
	for _, feature := range features {
		props := feature.Properties
		if !featureCurrent(props, now) || !featureSubtypeMatchesFeed(props, subtypes) {
			continue
		}
		distanceKM := 0.0
		if hasReference {
			var near bool
			distanceKM, near = geoFeatureDistanceToPointKM(feature, refLon, refLat)
			if !near || distanceKM > thunderstormOutlookNearbyKM {
				continue
			}
		}
		item := map[string]any{
			"id":           firstNonBlank(feature.ID, textValue(props["id"])),
			"area":         textValue(props["product_sub_type"]),
			"thunderstorm": textValue(props["metobject.thunderstorm.value"]),
			"risk":         props["metobject.risk_swo.value"],
			"confidence":   props["metobject.confidence.value"],
			"impact":       props["metobject.impact.value"],
			"tornado":      props["metobject.tornado_risk.value"],
			"hail_cm":      props["metobject.hail.value"],
			"hail_unit":    props["metobject.hail.unit"],
			"gust_kmh":     props["metobject.gust.value"],
			"gust_unit":    props["metobject.gust.unit"],
			"rain_mm":      props["metobject.rain.value"],
			"rain_unit":    props["metobject.rain.unit"],
			"published_at": textValue(props["publication_datetime"]),
			"valid_at":     textValue(props["validity_datetime"]),
			"expires_at":   textValue(props["expiration_datetime"]),
		}
		if hasReference {
			item["distance_km"] = math.Round(distanceKM)
			if distanceKM > 0.5 {
				if direction, ok := geoFeatureDirectionFromPoint(feature, refLon, refLat); ok {
					item["direction"] = direction
				}
			}
		}
		items = append(items, item)
	}
	sort.SliceStable(items, func(i, j int) bool {
		left, _ := numberValue(items[i]["risk"])
		right, _ := numberValue(items[j]["risk"])
		return left > right
	})
	return specialtyPayload("thunderstorm_outlook", "Thunderstorm Outlook", timestamp, items), nil
}

func fetchCoastalFloodProduct(ctx context.Context, client *http.Client, feed feedXML, _ map[string]map[string]any) (map[string]any, error) {
	features, timestamp, err := fetchCollectionFeatures(ctx, client, "coastal_flood_risk_index", "limit=1000&sortby=-validity_datetime")
	if err != nil {
		return nil, err
	}
	subtypes := feedSpecialtySubtypes(feed)
	now := time.Now().UTC()
	items := []map[string]any{}
	for _, feature := range features {
		props := feature.Properties
		if !featureCurrent(props, now) || !featureSubtypeMatchesFeed(props, subtypes) {
			continue
		}
		items = append(items, map[string]any{
			"id":           firstNonBlank(feature.ID, textValue(props["id"])),
			"area":         textValue(props["product_sub_type"]),
			"risk":         props["metobject.risk.value"],
			"likelihood":   props["metobject.likelihood.value"],
			"impact":       props["metobject.impact.value"],
			"storm_surge":  props["metobject.storm_surge.value"],
			"tide":         props["metobject.tide.value"],
			"waves":        props["metobject.waves.value"],
			"published_at": textValue(props["publication_datetime"]),
			"valid_at":     textValue(props["validity_datetime"]),
			"expires_at":   textValue(props["expiration_datetime"]),
		})
	}
	return specialtyPayload("coastal_flood", "Coastal Flooding Risk", timestamp, items), nil
}

func fetchHurricaneTracksProduct(ctx context.Context, client *http.Client, feed feedXML, _ map[string]map[string]any) (map[string]any, error) {
	if !feedSupportsHurricane(feed) {
		return nil, nil
	}
	features, timestamp, err := fetchCollectionFeatures(ctx, client, "hurricanes-cyclone-realtime", "limit=100&active=true&latest_publication=true&sortby=-publication_datetime")
	if err != nil {
		return nil, err
	}
	now := time.Now().UTC()
	items := []map[string]any{}
	for _, feature := range features {
		props := feature.Properties
		if !featureCurrent(props, now) {
			continue
		}
		items = append(items, map[string]any{
			"id":             firstNonBlank(feature.ID, textValue(props["id"])),
			"storm_name":     textValue(props["storm_name"]),
			"classification": textValue(props["metobject.classification"]),
			"sub_type":       textValue(props["metobject.sub_type"]),
			"max_wind_kt":    props["metobject.max_wind.value"],
			"gust_kt":        props["metobject.wind_gust.value"],
			"pressure_mb":    props["metobject.pressure.value"],
			"published_at":   textValue(props["publication_datetime"]),
			"valid_at":       textValue(props["validity_datetime"]),
			"forecast_at":    textValue(props["forecast_datetime"]),
		})
	}
	return specialtyPayload("hurricane_tracks", "Hurricane Tracks", timestamp, items), nil
}

func fetchHydrometricProduct(ctx context.Context, client *http.Client, feed feedXML, _ map[string]map[string]any) (map[string]any, error) {
	items := []map[string]any{}
	for _, loc := range feed.Locations.HydrometricLocations.Locations {
		if sourceKind(loc.Source) != "eccc" || strings.TrimSpace(loc.ID) == "" {
			continue
		}
		query := "limit=1&sortby=-DATETIME&STATION_NUMBER=" + url.QueryEscape(strings.TrimSpace(loc.ID))
		features, _, err := fetchCollectionFeatures(ctx, client, "hydrometric-realtime", query)
		if err != nil {
			log.Printf("hydrometric fetch failed for %s: %v", loc.ID, err)
			continue
		}
		if len(features) == 0 {
			continue
		}
		props := features[0].Properties
		items = append(items, map[string]any{
			"id":           firstNonBlank(features[0].ID, textValue(props["IDENTIFIER"])),
			"station_id":   firstNonBlank(textValue(props["STATION_NUMBER"]), loc.ID),
			"station":      firstNonBlank(loc.NameOverride, titleText(textValue(props["STATION_NAME"]))),
			"observed_at":  firstNonBlank(textValue(props["DATETIME_LST"]), textValue(props["DATETIME"])),
			"level_m":      props["LEVEL"],
			"discharge":    props["DISCHARGE"],
			"level_note":   textValue(props["LEVEL_SYMBOL_EN"]),
			"flow_note":    textValue(props["DISCHARGE_SYMBOL_EN"]),
			"published_at": firstNonBlank(textValue(props["DATETIME_LST"]), textValue(props["DATETIME"])),
		})
	}
	return specialtyPayload("hydrometric", "River Conditions", time.Now().UTC().Format(time.RFC3339), items), nil
}

func fetchPrecipitationAnalysisProduct(ctx context.Context, client *http.Client, feed feedXML, ecccCache map[string]map[string]any) (map[string]any, error) {
	lon, lat, location, ok := feedReferencePoint(ctx, client, feed, ecccCache)
	if !ok {
		return nil, nil
	}
	const radius = 0.5
	query := fmt.Sprintf("f=json&bbox=%.4f,%.4f,%.4f,%.4f", lon-radius, lat-radius, lon+radius, lat+radius)
	var raw map[string]any
	if err := fetchJSON(ctx, client, "https://api.weather.gc.ca/collections/weather%3Ardpa%3A10km%3A24f/coverage?"+query, &raw); err != nil {
		return nil, err
	}
	stats, ok := rdpaStats(raw)
	if !ok {
		return nil, nil
	}
	stats["location"] = location
	stats["longitude"] = lon
	stats["latitude"] = lat
	stats["radius_degrees"] = radius
	stats["published_at"] = time.Now().UTC().Format(time.RFC3339)
	return specialtyPayload("precipitation_analysis", "Precipitation Analysis", textValue(stats["published_at"]), []map[string]any{stats}), nil
}

func fetchCollectionFeatures(ctx context.Context, client *http.Client, collection string, query string) ([]geoFeature, string, error) {
	base := "https://api.weather.gc.ca/collections/" + collection + "/items?f=json"
	if strings.TrimSpace(query) != "" {
		base += "&" + strings.TrimLeft(query, "&?")
	}
	var raw geoFeatureCollection
	if err := fetchJSON(ctx, client, base, &raw); err != nil {
		return nil, "", err
	}
	return raw.Features, raw.TimeStamp, nil
}

func specialtyPayload(collection string, title string, timestamp string, items []map[string]any) map[string]any {
	if len(items) == 0 {
		return nil
	}
	return map[string]any{
		"source":     "eccc",
		"collection": collection,
		"title":      title,
		"updated_at": firstNonBlank(timestamp, time.Now().UTC().Format(time.RFC3339)),
		"items":      items,
	}
}

func featureCurrent(props map[string]any, now time.Time) bool {
	if props == nil {
		return false
	}
	for _, key := range []string{"expiration_datetime", "end_datetime"} {
		raw := textValue(props[key])
		if raw == "" {
			continue
		}
		if expires, err := parseAPITime(raw); err == nil && expires.Before(now.Add(-5*time.Minute)) {
			return false
		}
	}
	status := strings.ToLower(textValue(props["status"]))
	return status == "" || status == "final" || status == "updated" || status == "actual"
}

func parseAPITime(raw string) (time.Time, error) {
	raw = strings.TrimSpace(raw)
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05Z", "2006-01-02T15:04:05-0700"} {
		if parsed, err := time.Parse(layout, raw); err == nil {
			return parsed.UTC(), nil
		}
	}
	return time.Time{}, fmt.Errorf("unsupported API time %q", raw)
}

func featureSubtypeMatchesFeed(props map[string]any, subtypes map[string]struct{}) bool {
	if len(subtypes) == 0 {
		return true
	}
	subtype := strings.ToUpper(strings.TrimSpace(textValue(props["product_sub_type"])))
	if subtype == "" {
		return true
	}
	_, ok := subtypes[subtype]
	return ok
}

func feedSpecialtySubtypes(feed feedXML) map[string]struct{} {
	out := map[string]struct{}{}
	for province := range feedProvinceCodes(feed) {
		switch province {
		case "AB", "MB", "SK":
			out["PRAIRIES"] = struct{}{}
		case "BC", "YT":
			out["BC-YT"] = struct{}{}
		case "NB", "NL", "NS", "PE":
			out["ATLANTIC"] = struct{}{}
		case "ON", "QC":
			out[province] = struct{}{}
		}
	}
	return out
}

func feedProvinceCodes(feed feedXML) map[string]struct{} {
	out := map[string]struct{}{}
	addFromID := func(raw string) {
		value := strings.ToUpper(strings.TrimSpace(raw))
		if before, _, ok := strings.Cut(value, "-"); ok && len(before) == 2 && before[0] >= 'A' && before[0] <= 'Z' && before[1] >= 'A' && before[1] <= 'Z' {
			out[before] = struct{}{}
		}
	}
	for _, loc := range feed.Locations.ObservationLocations.Locations {
		addFromID(loc.ID)
	}
	for _, loc := range feed.Locations.AirQualityLocations.Locations {
		addFromID(loc.ID)
	}
	for _, region := range feed.Locations.Coverage.Regions {
		addFromID(region.DeriveForecast)
		addFromID(region.ID)
	}
	return out
}

func feedSupportsHurricane(feed feedXML) bool {
	for province := range feedProvinceCodes(feed) {
		switch province {
		case "NB", "NL", "NS", "PE", "QC":
			return true
		}
	}
	return false
}

func feedReferencePoint(ctx context.Context, client *http.Client, feed feedXML, ecccCache map[string]map[string]any) (float64, float64, string, bool) {
	for _, loc := range feed.Locations.ObservationLocations.Locations {
		if sourceKind(loc.Source) != "eccc" {
			continue
		}
		if lon, lat, ok := configuredPoint(loc); ok {
			return lon, lat, firstNonBlank(loc.NameOverride, loc.ID), true
		}
		raw := ecccCache[loc.ID]
		if raw == nil {
			fetched, err := fetchECCCCitypage(ctx, client, loc.ID)
			if err != nil {
				continue
			}
			raw = fetched
			ecccCache[loc.ID] = fetched
		}
		if lon, lat, name, ok := citypagePoint(raw); ok {
			return lon, lat, firstNonBlank(loc.NameOverride, name, loc.ID), true
		}
	}
	for _, region := range feed.Locations.Coverage.Regions {
		forecastID := fallbackText(region.DeriveForecast, region.ID)
		raw := ecccCache[forecastID]
		if raw == nil && strings.TrimSpace(forecastID) != "" {
			fetched, err := fetchECCCCitypage(ctx, client, forecastID)
			if err != nil {
				continue
			}
			raw = fetched
			ecccCache[forecastID] = fetched
		}
		if lon, lat, name, ok := citypagePoint(raw); ok {
			return lon, lat, firstNonBlank(region.Name, name, forecastID), true
		}
	}
	return 0, 0, "", false
}

func configuredPoint(loc locationXML) (float64, float64, bool) {
	lat, latErr := strconv.ParseFloat(strings.TrimSpace(loc.Latitude), 64)
	lon, lonErr := strconv.ParseFloat(strings.TrimSpace(loc.Longitude), 64)
	if latErr != nil || lonErr != nil {
		return 0, 0, false
	}
	return lon, lat, true
}

func citypagePoint(raw map[string]any) (float64, float64, string, bool) {
	geometry := mapAt(raw, "geometry")
	coordinates, _ := geometry["coordinates"].([]any)
	if len(coordinates) < 2 {
		return 0, 0, "", false
	}
	lon, lonOK := numberValue(coordinates[0])
	lat, latOK := numberValue(coordinates[1])
	if !lonOK || !latOK {
		return 0, 0, "", false
	}
	props := mapAt(raw, "properties")
	name := localizedText(props["name"], "en")
	return lon, lat, name, true
}

func rdpaStats(raw map[string]any) (map[string]any, bool) {
	apcp := mapAt(mapAt(raw, "ranges"), "APCP")
	values, _ := apcp["values"].([]any)
	if len(values) == 0 {
		return nil, false
	}
	count := 0
	sum := 0.0
	minValue := math.Inf(1)
	maxValue := math.Inf(-1)
	for _, value := range values {
		numeric, ok := numberValue(value)
		if !ok {
			continue
		}
		count++
		sum += numeric
		if numeric < minValue {
			minValue = numeric
		}
		if numeric > maxValue {
			maxValue = numeric
		}
	}
	if count == 0 {
		return nil, false
	}
	return map[string]any{
		"count":   count,
		"min_mm":  math.Round(minValue*10) / 10,
		"max_mm":  math.Round(maxValue*10) / 10,
		"mean_mm": math.Round((sum/float64(count))*10) / 10,
	}, true
}

func numberValue(value any) (float64, bool) {
	switch typed := value.(type) {
	case nil:
		return 0, false
	case float64:
		return typed, true
	case int:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	case string:
		parsed, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		return parsed, err == nil
	default:
		return 0, false
	}
}

func intValue(value any) int {
	parsed, ok := numberValue(value)
	if !ok {
		return 0
	}
	return int(math.Round(parsed))
}

func titleText(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	words := strings.Fields(strings.ToLower(value))
	for i, word := range words {
		if word == "" {
			continue
		}
		words[i] = strings.ToUpper(word[:1]) + word[1:]
	}
	return strings.Join(words, " ")
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

func persistClimateSummary(ctx context.Context, store datastore.Store, loc locationXML, payload map[string]any) error {
	if store == nil {
		return datastore.ErrNotConfigured
	}
	source := sourceKind(loc.Source)
	name := firstNonBlank(loc.NameOverride, localizedText(payload["name"], "en"), loc.ID)
	if err := store.UpsertLocation(ctx, datastore.LocationRecord{
		Source:     source,
		LocationID: loc.ID,
		Kind:       "climate_location",
		NameEN:     name,
		NameFR:     firstNonBlank(localizedText(payload["name"], "fr"), name),
		Metadata: map[string]any{
			"configured_source": loc.Source,
			"name_override":     loc.NameOverride,
			"normal_id":         loc.NormalID,
		},
	}); err != nil {
		return err
	}
	return store.StoreProductPayload(ctx, datastore.ProductPayloadRecord{
		Kind:    "climate_summary",
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
