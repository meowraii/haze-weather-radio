package productrender

import (
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capingest"
)

func TestCAPSAMEPayloadSuppressesCancellations(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg, err := loadConfig(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	feed, ok := cfg.feedByID("sk-0001")
	if !ok {
		t.Fatal("fixture feed not found")
	}
	alert := parseTestAlert(t, testCAP("urn:test:cancel", "Cancel", "ended", "2099-06-15T21:30:00-06:00", true))

	payload := capSAMEPayload(alert, feed, cfg.BaseDir, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC))

	if payload["include_same"] != false {
		t.Fatalf("include_same = %#v, want false", payload["include_same"])
	}
	if payload["same_suppressed_reason"] != "cancellation" {
		t.Fatalf("same_suppressed_reason = %#v", payload["same_suppressed_reason"])
	}
}

func TestCAPSAMEPayloadDerivesWeatherOriginatorBeforeFeedDefault(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg, err := loadConfig(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	feed, ok := cfg.feedByID("sk-0001")
	if !ok {
		t.Fatal("fixture feed not found")
	}
	feed.Playout.SAMEOriginator = "EAS"
	alert := parseTestAlert(t, testCAP("urn:test:eccc-originator", "Alert", "active", "2099-06-15T21:30:00-06:00", false))

	payload := capSAMEPayload(alert, feed, cfg.BaseDir, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC))

	if payload["same_originator"] != "WXR" {
		t.Fatalf("same_originator = %#v, want WXR", payload["same_originator"])
	}
	if payload["same_weather_service"] != "Environment Canada" {
		t.Fatalf("same_weather_service = %#v, want Environment Canada", payload["same_weather_service"])
	}
	if payload["same_originator_name"] != "Environment Canada" {
		t.Fatalf("same_originator_name = %#v, want Environment Canada", payload["same_originator_name"])
	}
	if payload["same_event_name"] != "Yellow Warning - Severe Thunderstorm" {
		t.Fatalf("same_event_name = %#v, want Yellow Warning - Severe Thunderstorm", payload["same_event_name"])
	}
}

func TestSameOriginatorForCAPDerivesNWSAndCivilAuthorities(t *testing.T) {
	tests := []struct {
		name           string
		alert          capingest.Alert
		wantOriginator string
		wantService    string
	}{
		{
			name: "nws",
			alert: capingest.Alert{
				Sender: "alerts.weather.gov",
				Infos: []capingest.AlertInfo{{
					SenderName: "National Weather Service",
					Event:      "Severe Thunderstorm Warning",
				}},
			},
			wantOriginator: "WXR",
			wantService:    "The National Weather Service",
		},
		{
			name: "civil authority",
			alert: capingest.Alert{
				Sender: "county-emergency-management@example.gov",
				Infos: []capingest.AlertInfo{{
					SenderName: "County Emergency Management",
					Event:      "Civil Emergency Message",
				}},
			},
			wantOriginator: "CIV",
			wantService:    "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if len(test.alert.Infos) == 0 {
				t.Fatal("test alert missing info")
			}
			if got := sameOriginatorForCAP(test.alert, test.alert.Infos[0]); got != test.wantOriginator {
				t.Fatalf("same originator = %q, want %q", got, test.wantOriginator)
			}
			if got := sameWeatherServiceForCAP(test.alert); got != test.wantService {
				t.Fatalf("same weather service = %q, want %q", got, test.wantService)
			}
		})
	}
}

func TestCAPPriorityBroadcastUsesFreshnessWindow(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg, err := loadConfig(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	alert := parseTestAlert(t, testCAP("urn:test:freshness", "Alert", "active", "2099-06-15T21:30:00-06:00", false))
	service := &Service{cfg: cfg}

	freshUpdates, err := service.recordCAPAlert(alert, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(freshUpdates) != 1 || !freshUpdates[0].Broadcast {
		t.Fatalf("fresh updates = %#v", freshUpdates)
	}

	staleUpdates, err := service.recordCAPAlert(alert, time.Date(2026, 6, 15, 23, 1, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(staleUpdates) != 1 {
		t.Fatalf("stale updates = %#v", staleUpdates)
	}
	if staleUpdates[0].Broadcast {
		t.Fatalf("stale alert should not priority broadcast: %#v", staleUpdates[0])
	}
	if !staleUpdates[0].Renderable {
		t.Fatalf("stale active alert should remain routine-renderable until ended/expired: %#v", staleUpdates[0])
	}
}

func TestCAPPriorityBroadcastSuppressesEndedAlerts(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg, err := loadConfig(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	alert := parseTestAlert(t, testCAP("urn:test:ended", "Update", "ended", "2099-06-15T21:30:00-06:00", true))
	service := &Service{cfg: cfg}

	updates, err := service.recordCAPAlert(alert, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(updates) != 1 {
		t.Fatalf("updates = %#v", updates)
	}
	if updates[0].Broadcast {
		t.Fatalf("ended alert should not priority broadcast: %#v", updates[0])
	}
	if !updates[0].Cancelled {
		t.Fatalf("ended alert should emit cancellation cleanup metadata: %#v", updates[0])
	}
}

func TestSameAlertFreshForToneUsesShortWindowForSVRAndTOR(t *testing.T) {
	alert := parseTestAlert(t, testCAP("urn:test:svr", "Alert", "active", "2099-06-15T21:30:00-06:00", false))
	info := chooseAlertInfo(alert, "en-CA")
	if info == nil {
		t.Fatal("alert info not found")
	}
	sentPlus29 := time.Date(2026, 6, 15, 22, 27, 0, 0, time.UTC)
	sentPlus31 := time.Date(2026, 6, 15, 22, 29, 0, 0, time.UTC)

	if !sameAlertFreshForTone(alert, *info, "SVR", sentPlus29) {
		t.Fatal("SVR should allow SAME tones before 30 minutes")
	}
	if sameAlertFreshForTone(alert, *info, "SVR", sentPlus31) {
		t.Fatal("SVR should suppress SAME tones after 30 minutes")
	}
	if !sameAlertFreshForTone(alert, *info, "TOR", sentPlus29) {
		t.Fatal("TOR should allow SAME tones before 30 minutes")
	}
	if sameAlertFreshForTone(alert, *info, "TOR", sentPlus31) {
		t.Fatal("TOR should suppress SAME tones after 30 minutes")
	}
}

func TestSameAlertFreshForToneUsesLongWindowForOtherAlerts(t *testing.T) {
	alert := parseTestAlert(t, testCAP("urn:test:watch", "Alert", "active", "2099-06-15T21:30:00-06:00", false))
	info := chooseAlertInfo(alert, "en-CA")
	if info == nil {
		t.Fatal("alert info not found")
	}
	sentPlus59 := time.Date(2026, 6, 15, 22, 57, 0, 0, time.UTC)
	sentPlus61 := time.Date(2026, 6, 15, 22, 59, 0, 0, time.UTC)

	if !sameAlertFreshForTone(alert, *info, "SVA", sentPlus59) {
		t.Fatal("SVA should allow SAME tones before 60 minutes")
	}
	if sameAlertFreshForTone(alert, *info, "SVA", sentPlus61) {
		t.Fatal("SVA should suppress SAME tones after 60 minutes")
	}
	if !sameAlertFreshForTone(alert, *info, "ADR", sentPlus59) {
		t.Fatal("other SAME events should allow tones before 60 minutes")
	}
	if sameAlertFreshForTone(alert, *info, "ADR", sentPlus61) {
		t.Fatal("other SAME events should suppress tones after 60 minutes")
	}
}

func TestCAPFeedFilterAllowsModerateAndUp(t *testing.T) {
	var feed feedXML
	feed.Alerts.CapCP.EnabledRaw = "true"
	feed.Alerts.CapCP.Filter.Allowlist.Severities = []string{"Moderate", "Severe", "Extreme"}
	alert := parseTestAlert(t, testCAP("urn:test:filter-moderate", "Alert", "active", "2099-06-15T21:30:00-06:00", false))

	if !feedAllowsCAPAlert(feed, alert) {
		t.Fatal("moderate alert should pass moderate-and-up feed filter")
	}

	alert.Infos[0].Severity = "Minor"
	if feedAllowsCAPAlert(feed, alert) {
		t.Fatal("minor alert should be rejected by moderate-and-up feed filter")
	}
}

func TestCAPFeedFilterCanDisableCoverageMatching(t *testing.T) {
	var feed feedXML
	feed.Alerts.CapCP.EnabledRaw = "true"
	feed.Alerts.CapCP.Filter.UseFeedLocations = "false"
	alert := parseTestAlert(t, testCAP("urn:test:all-locations", "Alert", "active", "2099-06-15T21:30:00-06:00", false))

	if !alertMatchesFeed(alert, feed, t.TempDir()) {
		t.Fatal("feed with use_feed_locations=false should match every alert location")
	}
}

func TestSameLocationsForCAPUsesNWSFIPSAndZones(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "managed", "csv", "NWS_ZONE_COUNTY_CORRELATION.csv"), `STATE|ZONE_CODE|CWA_ID|ZONE_NAME|STATE+ZONE|COUNTY_NAME|FIPS/SAME|TIMEZONE|FE_AREA|LAT|LON
GA|033|FFC|North Fulton|GA033|Fulton|13121|E|nc|33.9350|-84.3557
`)
	var feed feedXML
	info := capingest.AlertInfo{
		Areas: []capingest.AlertArea{{
			Geocodes: []capingest.NameValue{{Name: "UGC", Value: "GAZ033"}},
		}},
	}

	got := sameLocationsForCAP(info, feed, dir)
	want := []string{"013121"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("same locations = %#v, want %#v", got, want)
	}
}

func TestExpandedNWSCoverageMatchesFIPSAndSAMECodes(t *testing.T) {
	db := alertGeoDB{
		NWS: map[string]nwsZone{
			"GA033": {Code: "GA033", Name: "North Fulton", CountyName: "Fulton", FIPS: "13121"},
		},
		FIPS: map[string]nwsZone{
			"13121":  {Code: "GA033", Name: "North Fulton", CountyName: "Fulton", FIPS: "13121"},
			"013121": {Code: "GA033", Name: "North Fulton", CountyName: "Fulton", FIPS: "13121"},
		},
	}
	coverage := map[string]struct{}{}
	for _, code := range expandNWSRegion(db, "GAZ033") {
		addCoverageCode(coverage, code)
	}

	for _, code := range []string{"GAZ033", "GA033", "13121", "013121"} {
		if !coverageCodeMatches(coverage, code) {
			t.Fatalf("coverage should match %q with expanded NWS zone/FIPS map: %#v", code, coverage)
		}
	}
}
