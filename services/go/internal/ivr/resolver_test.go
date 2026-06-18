package ivr

import "testing"

func TestResolverMapsHelloWeatherCodeToCoveredFeed(t *testing.T) {
	cfg := loadedConfig{
		IVR: Config{DefaultLanguage: "en-CA"},
		ForecastLocations: map[string]locationRecord{
			"000040": {Code: "000040", Source: "eccc_forecast", Name: "Saskatoon", Province: "SK", Forecast: "sk-40"},
		},
		Feeds: []feedXML{
			{
				ID:         "sk-0001",
				EnabledRaw: "true",
				Timezone:   "America/Regina",
				Locations: struct {
					Coverage struct {
						Regions []coverageRegionXML "xml:\"region\""
					} "xml:\"coverage\""
					ObservationLocations struct {
						Locations []feedLocationXML "xml:\"location\""
					} "xml:\"observationLocations\""
				}{
					Coverage: struct {
						Regions []coverageRegionXML "xml:\"region\""
					}{
						Regions: []coverageRegionXML{{ID: "065100", Source: "eccc", DeriveForecast: "sk-40"}},
					},
				},
			},
		},
	}

	location, err := NewResolver(cfg).Resolve("06040")
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if location.FeedID != "sk-0001" {
		t.Fatalf("FeedID = %q", location.FeedID)
	}
	if location.Forecast != "sk-40" {
		t.Fatalf("Forecast = %q", location.Forecast)
	}
	if location.Name != "Saskatoon" {
		t.Fatalf("Name = %q", location.Name)
	}
}

func TestResolverRejectsKnownButUncoveredLocation(t *testing.T) {
	cfg := loadedConfig{
		IVR: Config{DefaultLanguage: "en-CA"},
		CLCs: map[string]locationRecord{
			"065522": {Code: "065522", Source: "clc", Name: "Saskatoon", Province: "SK"},
		},
	}

	_, err := NewResolver(cfg).Resolve("065522")
	if err == nil {
		t.Fatal("Resolve succeeded for uncovered location")
	}
}

func TestT9StationLookup(t *testing.T) {
	cfg := loadedConfig{
		IVR: Config{DefaultLanguage: "en-CA"},
		Feeds: []feedXML{
			{
				ID:         "sk-0001",
				EnabledRaw: "true",
				Locations: struct {
					Coverage struct {
						Regions []coverageRegionXML "xml:\"region\""
					} "xml:\"coverage\""
					ObservationLocations struct {
						Locations []feedLocationXML "xml:\"location\""
					} "xml:\"observationLocations\""
				}{
					ObservationLocations: struct {
						Locations []feedLocationXML "xml:\"location\""
					}{
						Locations: []feedLocationXML{{ID: "CYXE", Source: "eccc", NameOverride: "Saskatoon Airport"}},
					},
				},
			},
		},
	}

	location, err := NewResolver(cfg).Resolve("2993")
	if err != nil {
		t.Fatalf("Resolve returned error: %v", err)
	}
	if location.StationID != "CYXE" {
		t.Fatalf("StationID = %q", location.StationID)
	}
}
