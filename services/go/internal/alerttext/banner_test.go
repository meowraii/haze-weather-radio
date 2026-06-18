package alerttext

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capingest"
)

func TestBuildSAMETranslationMatchesBannerStyleLead(t *testing.T) {
	text := BuildSAMETranslation(SAMERequest{
		Originator: "WXR",
		Event:      "RWT",
		EventName:  "Required Weekly Test",
		AreaNames:  []string{"Saskatoon"},
		Callsign:   "XLF322",
		SentAt:     time.Date(2026, 6, 17, 1, 0, 0, 0, time.UTC),
		ExpiresAt:  time.Date(2026, 6, 17, 1, 15, 0, 0, time.UTC),
		MimicENDEC: "SAGE",
	})

	if !strings.Contains(text, "Environment Canada has issued a Required Weekly Test for Saskatoon") {
		t.Fatalf("text = %q", text)
	}
	if !strings.Contains(text, "(XLF322)") {
		t.Fatalf("text = %q", text)
	}
}

func TestLoadEventAndAreaLabels(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "managed", "sameMapping.json"), `{"eas":{"SVR":"Severe Thunderstorm Warning"}}`)
	mustWrite(t, filepath.Join(dir, "managed", "csv", "FORECAST_LOCATIONS.csv"), "skip\nCODE,NAME,NOM\n065522,Saskatoon,Saskatoon\n")
	configPath := filepath.Join(dir, "config.yaml")

	if name := EventName(configPath, "SVR"); name != "Severe Thunderstorm Warning" {
		t.Fatalf("event name = %q", name)
	}
	areas := ResolveAreaNames(configPath, nil, []string{"065522"})
	if len(areas) != 1 || areas[0] != "Saskatoon" {
		t.Fatalf("areas = %#v", areas)
	}
}

func TestBuildCAPAlertTextUsesSharedWeatherSpeech(t *testing.T) {
	now := time.Date(2026, 6, 17, 3, 0, 0, 0, time.UTC)
	alert := capingest.Alert{
		Identifier:  "cap-1",
		Sender:      "cap-pac@canada.ca",
		Sent:        "2026-06-17T02:30:00Z",
		MessageType: "Alert",
		Infos: []capingest.AlertInfo{{
			Event:       "Severe Thunderstorm Warning",
			Headline:    "Severe Thunderstorm Warning - in effect",
			SenderName:  "Environment Canada",
			Effective:   "2026-06-17T02:30:00Z",
			Expires:     "2026-06-17T04:00:00Z",
			Description: "Nickel size hail is possible.",
			Instruction: "Take shelter if threatening weather approaches.",
		}},
	}
	text := BuildCAPAlertText(CAPMessageRequest{
		Alert:     alert,
		Info:      alert.Infos[0],
		AreaText:  "City of Saskatoon",
		Timezone:  "America/Regina",
		Now:       now,
		EventName: AlertSubject(alert.Infos[0]),
	})

	if !strings.Contains(text, "Environment Canada has issued a Severe Thunderstorm Warning") {
		t.Fatalf("text = %q", text)
	}
	if !strings.Contains(text, "City of Saskatoon") || !strings.Contains(text, "Nickel size hail is possible.") {
		t.Fatalf("text = %q", text)
	}
}

func mustWrite(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}
