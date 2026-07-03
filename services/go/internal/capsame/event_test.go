package capsame

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
)

func TestResolveEventPrefersExplicitCAPSAMECode(t *testing.T) {
	dir := mappingFixture(t, `{"naadsToEas":{"tornado":"TOR"}}`)
	info := capmodel.AlertInfo{
		Event: "tornado",
		EventCodes: []capmodel.NameValue{
			{Name: "SAME", Value: "TOA"},
			{Name: "profile:CAP-CP:Event:0.4", Value: "tornado"},
		},
		Parameters: []capmodel.NameValue{
			{Name: "layer:EC-MSC-SMC:1.0:Alert_Type", Value: "watch"},
			{Name: "layer:EC-MSC-SMC:1.0:Alert_Name", Value: "yellow watch - tornado"},
		},
	}

	resolution := ResolveEvent(capmodel.Alert{}, info, dir)

	if resolution.Event != "TOA" {
		t.Fatalf("event = %q, want TOA (%#v)", resolution.Event, resolution)
	}
	if resolution.Source != "cap_event_code" || resolution.Confidence != "high" {
		t.Fatalf("resolution metadata = %#v, want explicit high-confidence CAP event code", resolution)
	}
}

func TestResolveEventUsesWatchMetadataBeforeRawNAADSMapping(t *testing.T) {
	dir := mappingFixture(t, `{"naadsToEas":{"tornado":"TOR"}}`)
	info := capmodel.AlertInfo{
		Event:      "tornado",
		Headline:   "yellow watch - tornado - in effect",
		EventCodes: []capmodel.NameValue{{Name: "profile:CAP-CP:Event:0.4", Value: "tornado"}},
		Parameters: []capmodel.NameValue{
			{Name: "layer:EC-MSC-SMC:1.1:Alert_Type", Value: "watch"},
			{Name: "layer:EC-MSC-SMC:1.1:Alert_Name", Value: "yellow watch - tornado"},
		},
	}

	resolution := ResolveEvent(capmodel.Alert{}, info, dir)

	if resolution.Event != "TOA" {
		t.Fatalf("event = %q, want TOA (%#v)", resolution.Event, resolution)
	}
	if resolution.AlertClass != "watch" || resolution.Phenomenon != "tornado" {
		t.Fatalf("classification = %#v, want watch/tornado", resolution)
	}
}

func TestResolveEventDoesNotEscalateStatementToWarning(t *testing.T) {
	dir := mappingFixture(t, `{"naadsToEas":{"tornado":"TOR"}}`)
	info := capmodel.AlertInfo{
		Event:    "tornado",
		Headline: "special weather statement - tornado potential - in effect",
		Parameters: []capmodel.NameValue{
			{Name: "layer:EC-MSC-SMC:1.0:Alert_Type", Value: "statement"},
			{Name: "layer:EC-MSC-SMC:1.0:Alert_Name", Value: "special weather statement"},
		},
	}

	resolution := ResolveEvent(capmodel.Alert{}, info, dir)

	if resolution.Event != "SPS" {
		t.Fatalf("event = %q, want SPS (%#v)", resolution.Event, resolution)
	}
	if resolution.AlertClass != "statement" {
		t.Fatalf("alert class = %q, want statement", resolution.AlertClass)
	}
}

func mappingFixture(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	managed := filepath.Join(dir, "managed")
	if err := os.MkdirAll(managed, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(managed, "sameMapping.json"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
