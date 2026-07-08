package dataingest

import (
	"net/http"
	"testing"
	"time"
)

func TestDataIngestHTTPClientUsesReusableTransport(t *testing.T) {
	client := dataIngestHTTPClient(7 * time.Second)
	if client.Timeout != 7*time.Second {
		t.Fatalf("timeout = %s", client.Timeout)
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport = %T", client.Transport)
	}
	if transport.MaxIdleConns < 128 || transport.MaxIdleConnsPerHost < 16 || !transport.ForceAttemptHTTP2 {
		t.Fatalf("transport not tuned: %#v", transport)
	}
}

func TestDataIngestCycleTimeoutIsBounded(t *testing.T) {
	tests := []struct {
		name           string
		interval       time.Duration
		requestTimeout time.Duration
		want           time.Duration
	}{
		{
			name:           "caps long default interval",
			interval:       45 * time.Minute,
			requestTimeout: 20 * time.Second,
			want:           maxDataIngestCycleTimeout,
		},
		{
			name:           "allows short intervals enough request room",
			interval:       time.Minute,
			requestTimeout: 45 * time.Second,
			want:           135 * time.Second,
		},
		{
			name:           "keeps normal midrange interval",
			interval:       5 * time.Minute,
			requestTimeout: 20 * time.Second,
			want:           5 * time.Minute,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := dataIngestCycleTimeout(test.interval, test.requestTimeout); got != test.want {
				t.Fatalf("timeout = %s, want %s", got, test.want)
			}
		})
	}
}

func TestBuildECCCForecastUsesForecastLocationCSVNames(t *testing.T) {
	raw := map[string]any{
		"properties": map[string]any{
			"forecastGroup": map[string]any{
				"forecasts": []any{
					map[string]any{
						"period":      map[string]any{"textForecastName": map[string]any{"en": "Tonight", "fr": "Ce soir"}},
						"cloudPrecip": map[string]any{"en": "Cloudy.", "fr": "Nuageux."},
					},
				},
			},
		},
	}
	payload, ok := buildECCCForecast(raw, coverageRegionXML{
		ID:             "065500",
		Source:         "eccc",
		Name:           "Outlook. Watrous. Hanley. Imperial. Dinsmore Area",
		DeriveForecast: "sk-37",
	}, map[string]forecastRegionName{
		"065500": {
			English: "Outlook - Watrous - Hanley - Imperial - Dinsmore",
			French:  "Outlook - Watrous - Hanley - Imperial - Dinsmore",
		},
	})
	if !ok {
		t.Fatal("forecast was not built")
	}
	name, ok := payload["name"].(map[string]any)
	if !ok {
		t.Fatalf("name block = %#v", payload["name"])
	}
	if got := name["en"]; got != "Outlook, Watrous, Hanley, Imperial, and Dinsmore region" {
		t.Fatalf("english name = %q", got)
	}
}

func TestBuildECCCForecastIncludesIssueAndUpdateTimes(t *testing.T) {
	raw := map[string]any{
		"properties": map[string]any{
			"lastUpdated": "2026-06-18T04:07:12Z",
			"forecastGroup": map[string]any{
				"timestamp": map[string]any{"en": "2026-06-18T03:00:00Z", "fr": "2026-06-18T03:00:00Z"},
				"forecasts": []any{
					map[string]any{
						"period":      map[string]any{"textForecastName": map[string]any{"en": "Tonight", "fr": "Ce soir"}},
						"cloudPrecip": map[string]any{"en": "Cloudy.", "fr": "Nuageux."},
					},
				},
			},
		},
	}

	payload, ok := buildECCCForecast(raw, coverageRegionXML{
		ID:             "065500",
		Source:         "eccc",
		Name:           "Outlook. Watrous. Hanley. Imperial. Dinsmore Area",
		DeriveForecast: "sk-37",
	}, map[string]forecastRegionName{})
	if !ok {
		t.Fatal("forecast was not built")
	}
	if got := payload["issued_at"]; got != "2026-06-18T03:00:00Z" {
		t.Fatalf("issued_at = %#v", got)
	}
	if got := payload["updated_at"]; got != "2026-06-18T04:07:12Z" {
		t.Fatalf("updated_at = %#v", got)
	}
}

func TestForecastRegionFetchIDSkipsWildcardCoverage(t *testing.T) {
	region := coverageRegionXML{ID: "06*", Source: "eccc"}
	if got := forecastRegionFetchID(region); got != "" {
		t.Fatalf("forecast id = %q, want empty", got)
	}
}

func TestForecastRegionFetchIDAllowsDerivedForecastForWildcardCoverage(t *testing.T) {
	region := coverageRegionXML{ID: "06*", Source: "eccc", DeriveForecast: "sk-37"}
	if got := forecastRegionFetchID(region); got != "sk-37" {
		t.Fatalf("forecast id = %q, want sk-37", got)
	}
}

func TestBuildTWCConditionsUsesWeatherComLocationName(t *testing.T) {
	payload := buildTWCConditions(map[string]any{
		"_twc_location_name": "Lake Lenore",
		"validTimeLocal":     "2026-06-17T00:00:00-06:00",
		"temperature":        18,
		"wxPhraseLong":       "Clear",
	}, locationXML{ID: "CPUN", Source: "twc"})

	station, ok := payload["station"].(map[string]any)
	if !ok {
		t.Fatalf("station block = %#v", payload["station"])
	}
	if got := station["en"]; got != "Lake Lenore" {
		t.Fatalf("station name = %q", got)
	}
}

func TestBuildTWCConditionsUsesWeatherComStationNameBeforeLocationName(t *testing.T) {
	payload := buildTWCConditions(map[string]any{
		"obsName":            "Pilger",
		"_twc_location_name": "Lake Lenore",
	}, locationXML{ID: "CPUN", Source: "twc"})

	station := payload["station"].(map[string]any)
	if got := station["en"]; got != "Pilger" {
		t.Fatalf("station name = %q", got)
	}
}

func TestBuildTWCConditionsNameOverrideWins(t *testing.T) {
	payload := buildTWCConditions(map[string]any{
		"obsName":            "Pilger",
		"_twc_location_name": "Lake Lenore",
	}, locationXML{ID: "CPUN", Source: "twc", NameOverride: "Custom Pilger"})

	station := payload["station"].(map[string]any)
	if got := station["en"]; got != "Custom Pilger" {
		t.Fatalf("station name = %q", got)
	}
}

func TestAQHIPeriodsAcceptsUnderscoreKeys(t *testing.T) {
	periods := aqhiPeriods(map[string]any{
		"forecast_period": map[string]any{
			"period_2": map[string]any{
				"forecast_period_en": "Tonight",
				"forecast_period_fr": "Ce soir",
				"aqhi":               2,
				"aqhi_insmoke":       5,
			},
			"period_3": map[string]any{
				"forecast_period_en": "Tomorrow",
				"forecast_period_fr": "Demain",
				"aqhi":               3,
			},
		},
	})
	if len(periods) != 2 {
		t.Fatalf("period count = %d", len(periods))
	}
	first := periods[0]
	name := first["period"].(map[string]any)
	if name["en"] != "Tonight" || first["aqhi"] != 2 || first["aqhi_insmoke"] != 5 {
		t.Fatalf("first period = %#v", first)
	}
}

func TestClimateDailyUsableRejectsMissingFlaggedValues(t *testing.T) {
	props := map[string]any{
		"LOCAL_DATE":               "2026-06-17 00:00:00",
		"MAX_TEMPERATURE":          20.3,
		"MAX_TEMPERATURE_FLAG":     "M",
		"TOTAL_PRECIPITATION":      0,
		"TOTAL_PRECIPITATION_FLAG": "M",
		"SPEED_MAX_GUST":           47,
		"SPEED_MAX_GUST_FLAG":      "M",
	}
	if climateDailyUsable(props, mustTime(t, "2026-06-18T00:00:00Z")) {
		t.Fatalf("missing-flagged climate row should not be usable: %#v", props)
	}
	props["TOTAL_PRECIPITATION_FLAG"] = "T"
	if !climateDailyUsable(props, mustTime(t, "2026-06-18T00:00:00Z")) {
		t.Fatalf("trace precipitation should make climate row usable: %#v", props)
	}
	if !climateTrace(props, "TOTAL_PRECIPITATION") {
		t.Fatalf("trace precipitation flag was not detected")
	}
}

func TestClimateGustDirectionUsesTensOfDegrees(t *testing.T) {
	props := map[string]any{
		"DIRECTION_MAX_GUST":      34,
		"DIRECTION_MAX_GUST_FLAG": "",
	}
	if got := climateGustDirection(props); got != "NNW" {
		t.Fatalf("gust direction = %#v", got)
	}
}

func TestClimateNormalQualityGate(t *testing.T) {
	if climateNormalUsable(map[string]any{
		"PERIOD":                   "NORM",
		"CURRENT_FLAG":             "Y",
		"YEAR_COUNT_NORMAL_PERIOD": 14,
		"PERCENT_OF_POSSIBLE_OBS":  100,
	}) {
		t.Fatal("normal with fewer than 15 years should be rejected")
	}
	if !climateNormalUsable(map[string]any{
		"PERIOD":                   "NORM",
		"CURRENT_FLAG":             "Y",
		"YEAR_COUNT_NORMAL_PERIOD": 27,
		"PERCENT_OF_POSSIBLE_OBS":  98.8,
	}) {
		t.Fatal("good normal should be accepted")
	}
}

func TestClimateRecordExtractionRequiresUsefulValuesAndYears(t *testing.T) {
	records := map[string]any{}
	addClimateRecord(records, "high_temperature", map[string]any{
		"RECORD_HIGH_MAX_TEMP":    34.6,
		"RECORD_HIGH_MAX_TEMP_YR": 1988,
	}, "RECORD_HIGH_MAX_TEMP", "RECORD_HIGH_MAX_TEMP_YR", true)
	addClimateRecord(records, "snowfall", map[string]any{
		"RECORD_SNOWFALL":    0,
		"RECORD_SNOWFALL_YR": 2009,
	}, "RECORD_SNOWFALL", "RECORD_SNOWFALL_YR", false)
	addClimateRecord(records, "precipitation", map[string]any{
		"RECORD_PRECIPITATION": 86.4,
	}, "RECORD_PRECIPITATION", "RECORD_PRECIPITATION_YR", false)

	high := records["high_temperature"].(map[string]any)
	if high["value"] != 34.6 || high["year"] != 1988 {
		t.Fatalf("high record = %#v", high)
	}
	if _, ok := records["snowfall"]; ok {
		t.Fatalf("zero snowfall record should be omitted: %#v", records["snowfall"])
	}
	if _, ok := records["precipitation"]; ok {
		t.Fatalf("record without year should be omitted: %#v", records["precipitation"])
	}
}

func TestClimateSunriseSunsetAlignsToFeedLocalDate(t *testing.T) {
	events, ok := climateSunriseSunset(time.Date(2026, 6, 17, 0, 0, 0, 0, time.UTC), -106.718889, 52.173611, "America/Regina")
	if !ok {
		t.Fatal("sunrise and sunset were not computed")
	}
	loc, err := time.LoadLocation("America/Regina")
	if err != nil {
		t.Fatal(err)
	}
	sunrise, err := time.Parse(time.RFC3339, events["sunrise"].(string))
	if err != nil {
		t.Fatal(err)
	}
	sunset, err := time.Parse(time.RFC3339, events["sunset"].(string))
	if err != nil {
		t.Fatal(err)
	}
	riseLocal := sunrise.In(loc)
	setLocal := sunset.In(loc)
	if riseLocal.Format("2006-01-02") != "2026-06-17" || setLocal.Format("2006-01-02") != "2026-06-17" {
		t.Fatalf("events not aligned to local date: sunrise=%s sunset=%s", riseLocal, setLocal)
	}
	if riseLocal.Hour() < 4 || riseLocal.Hour() > 5 || setLocal.Hour() != 21 {
		t.Fatalf("unexpected local solar times: sunrise=%s sunset=%s", riseLocal.Format(time.Kitchen), setLocal.Format(time.Kitchen))
	}
}

func TestFeedSpecialtySubtypesUsesECCCCitypagePrefix(t *testing.T) {
	feed := feedXML{}
	feed.Locations.Coverage.Regions = []coverageRegionXML{{ID: "065500", Source: "eccc", DeriveForecast: "sk-37"}}
	subtypes := feedSpecialtySubtypes(feed)
	if _, ok := subtypes["PRAIRIES"]; !ok {
		t.Fatalf("subtypes = %#v", subtypes)
	}
}

func TestSpecialtyPayloadPreservesEmptyItems(t *testing.T) {
	payload := specialtyPayload("thunderstorm_outlook", "Thunderstorm Outlook", "2026-07-07T12:00:00Z", nil)
	if payload == nil {
		t.Fatal("empty specialty payload should still be stored to clear stale data")
	}
	items, ok := payload["items"].([]map[string]any)
	if !ok {
		t.Fatalf("items type = %T", payload["items"])
	}
	if len(items) != 0 {
		t.Fatalf("items length = %d", len(items))
	}
}

func TestThunderstormOutlookGeometryOnlyAcceptsContainingPolygons(t *testing.T) {
	const saskatoonLon = -106.6700
	const saskatoonLat = 52.1332
	tests := []struct {
		name        string
		geometry    map[string]any
		wantCovered bool
		maxDistance float64
	}{
		{
			name: "contains saskatoon",
			geometry: polygonGeometry([][]float64{
				{-107.0, 51.9},
				{-106.2, 51.9},
				{-106.2, 52.4},
				{-107.0, 52.4},
				{-107.0, 51.9},
			}),
			wantCovered: true,
			maxDistance: 0,
		},
		{
			name: "near but not covering saskatoon",
			geometry: polygonGeometry([][]float64{
				{-106.8, 53.1},
				{-106.2, 53.1},
				{-106.2, 53.5},
				{-106.8, 53.5},
				{-106.8, 53.1},
			}),
			wantCovered: false,
			maxDistance: thunderstormOutlookCoverageToleranceKM,
		},
		{
			name: "northern alberta near bc border",
			geometry: polygonGeometry([][]float64{
				{-118.5, 58.2},
				{-116.5, 58.2},
				{-116.5, 59.4},
				{-118.5, 59.4},
				{-118.5, 58.2},
			}),
			wantCovered: false,
			maxDistance: thunderstormOutlookCoverageToleranceKM,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			feature := geoFeature{Geometry: test.geometry}
			distance, ok := geoFeatureDistanceToPointKM(feature, saskatoonLon, saskatoonLat)
			if !ok {
				t.Fatal("geometry distance was not computed")
			}
			gotCovered := distance <= test.maxDistance
			if gotCovered != test.wantCovered {
				t.Fatalf("distance = %.1f km, covered = %v, want %v", distance, gotCovered, test.wantCovered)
			}
			if test.name == "northern alberta near bc border" {
				direction, ok := geoFeatureDirectionFromPoint(feature, saskatoonLon, saskatoonLat)
				if !ok || direction != "north west" {
					t.Fatalf("direction = %q, ok = %v", direction, ok)
				}
			}
		})
	}
}

func polygonGeometry(ring [][]float64) map[string]any {
	points := make([]any, 0, len(ring))
	for _, point := range ring {
		points = append(points, []any{point[0], point[1]})
	}
	return map[string]any{
		"type":        "Polygon",
		"coordinates": []any{points},
	}
}

func mustTime(t *testing.T, raw string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		t.Fatal(err)
	}
	return parsed
}
