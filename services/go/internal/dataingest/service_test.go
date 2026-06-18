package dataingest

import "testing"

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
