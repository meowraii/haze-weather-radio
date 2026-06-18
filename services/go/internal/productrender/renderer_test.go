package productrender

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capingest"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

func TestCurrentConditionsProductUsesOpenerPackageAndRepeatSegments(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	storeObservationJSON(t, cfg.Store, "CYXE", `{
  "source": "eccc",
  "observed_at": "2026-06-15T20:00:00-06:00",
  "station": {"en": "Saskatoon Diefenbaker Int'l Airport"},
  "station_id": "CYXE",
  "properties": {
    "condition": {"en": "Mostly Cloudy"},
    "temp": 24,
    "dewpoint": 8,
    "humidity": 67,
    "wind": {"direction": "NW", "speed": 38, "gust": 42},
    "visibility": 24,
    "pressure": {"value": 101.4, "tendency": {"en": "falling"}}
  }
}`)
	storeObservationJSON(t, cfg.Store, "OUTLOOK", `{
  "source": "eccc",
  "observed_at": "2026-06-15T20:00:00-06:00",
  "station": {"en": "Outlook"},
  "station_id": "OUTLOOK",
  "properties": {
    "temp": 24,
    "wind": {"direction": "NW", "speed": 17}
  }
}`)

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "current_conditions"})
	if err != nil {
		t.Fatal(err)
	}

	if product.Title != "Current Conditions" {
		t.Fatalf("title = %q", product.Title)
	}
	for _, wanted := range []string{
		"The current weather conditions. Issued by Environment and Climate Change Canada at 8:00 PM Central Standard Time",
		"The weather at Saskatoon Diefenbaker Int'l Airport was Mostly Cloudy",
		"Outlook, 24 degrees, winds were north west at 17 kilometres per hour",
		"Again, at Saskatoon Diefenbaker Int'l Airport",
	} {
		if !strings.Contains(product.Text, wanted) {
			t.Fatalf("product text missing %q:\n%s", wanted, product.Text)
		}
	}
	if len(product.Segments) < 4 {
		t.Fatalf("segments = %#v", product.Segments)
	}
}

func TestForecastRegionTitleDoesNotDoubleRegion(t *testing.T) {
	if got := normalizeRegionTitle("Outlook, Watrous, Hanley, Imperial, and Dinsmore region"); got != "Outlook, Watrous, Hanley, Imperial, and Dinsmore region" {
		t.Fatalf("region = %q", got)
	}
	if got := normalizeRegionTitle("Outlook, Watrous Area"); got != "Outlook, Watrous" {
		t.Fatalf("area = %q", got)
	}
	if got := normalizeRegionTitle("Outlook - Watrous - Hanley - Imperial - Dinsmore"); got != "Outlook, Watrous, Hanley, Imperial, and Dinsmore region" {
		t.Fatalf("hyphenated region = %q", got)
	}
	if got := normalizeRegionTitle("City of Saskatoon"); got != "City of Saskatoon" {
		t.Fatalf("city = %q", got)
	}
}

func TestStationIDAndDateTimeUseLegacyShape(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)

	stationID, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "station_id"})
	if err != nil {
		t.Fatal(err)
	}
	for _, wanted := range []string{
		"You are listening to Canada RadioMET.",
		"Callsign X L F 3 2 2.",
		"Broadcasting from Saskatoon on a frequency of 162.550 megahertz.",
	} {
		if !strings.Contains(stationID.Text, wanted) {
			t.Fatalf("station ID missing %q:\n%s", wanted, stationID.Text)
		}
	}

	fixedTime := time.Date(2026, 6, 15, 18, 5, 0, 0, time.FixedZone("CST", -6*60*60))
	if got := dateTimeAnnouncement(fixedTime, "en-CA"); got != "Good evening. The current time is six oh five P.M., Central Standard Time." {
		t.Fatalf("date/time announcement = %q", got)
	}
}

func TestForecastProductSaysRegionOncePerRegion(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	storeForecastJSON(t, cfg.Store, "065500", "065500", `{
  "issued_at": "2026-06-15T20:00:00-06:00",
  "name": {"en": "Outlook - Watrous - Hanley - Imperial and Dinsmore region"},
  "forecast": [
    {"period": {"en": "Tonight"}, "textSummary": {"en": "Cloudy with showers."}},
    {"period": {"en": "Tuesday"}, "textSummary": {"en": "Sunny."}}
  ]
}`)

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "forecast"})
	if err != nil {
		t.Fatal(err)
	}

	region := "For the Outlook, Watrous, Hanley, Imperial, and Dinsmore region."
	if strings.Count(product.Text, region) != 1 {
		t.Fatalf("region opener count mismatch:\n%s", product.Text)
	}
	if strings.Contains(product.Text, "Outlook, Watrous, Hanley, Imperial, and Dinsmore region, Tuesday") {
		t.Fatalf("period line repeated region name:\n%s", product.Text)
	}
	if !strings.Contains(product.Text, "Tonight. Cloudy with showers.") || !strings.Contains(product.Text, "Tuesday. Sunny.") {
		t.Fatalf("period text missing:\n%s", product.Text)
	}
}

func TestAirQualityProductUsesLegacyAQHINarrative(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	storeProductPayloadJSON(t, cfg.Store, "air_quality", "eccc", "HAHJJ", `{
  "source": "eccc",
  "location": {"en": "Saskatoon", "fr": "Saskatoon"},
  "observed_at": "2026-06-15T21:00:00Z",
  "aqhi": 4,
  "special_notes": {"en": "Smoke may cause fluctuating air quality conditions.", "fr": ""},
  "forecast": {
    "published_at": "2026-06-15T22:00:00Z",
    "periods": [
      {"period": {"en": "Tonight"}, "aqhi": 4, "aqhi_insmoke": 7},
      {"period": {"en": "Tuesday"}, "aqhi": 6},
      {"period": {"en": "Tuesday night"}, "aqhi": 3},
      {"period": {"en": "Wednesday"}, "aqhi": 2}
    ]
  }
}`)

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "air_quality"})
	if err != nil {
		t.Fatal(err)
	}

	for _, wanted := range []string{
		"The air quality health index was observed at Saskatoon and reported a value of 4 at 3:00 PM Central Standard Time.",
		"This is acceptable air quality for outdoor activities for most people.",
		"Smoke may cause fluctuating air quality conditions.",
		"The air quality health index forecast for Saskatoon is 4 for Tonight and is considered Moderate.",
		"The air quality health index is expected to be 7 in smoke.",
		"For Tuesday, the maximum air quality health index is forecast to be 6, or Moderate.",
		"3 on Tuesday night, and lastly, 2 on Wednesday.",
	} {
		if !strings.Contains(product.Text, wanted) {
			t.Fatalf("air quality product missing %q:\n%s", wanted, product.Text)
		}
	}
	if strings.Contains(product.Text, "Air quality for Saskatoon.") {
		t.Fatalf("generic air quality opener leaked into legacy narrative:\n%s", product.Text)
	}
}

func TestMissingProductDataReturnsSkippableError(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)

	_, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "forecast"})
	if err == nil || !strings.Contains(err.Error(), "forecast information is unavailable") {
		t.Fatalf("error = %v", err)
	}
}

func TestGeophysicalAlertRemovesWWVBoilerplate(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	storeTextProduct(t, cfg.Store, "nws", "wwv", `:Product: Geophysical Alert Message wwv.txt
:Issued: 2026 Jun 16 0305 UTC
# Prepared by the US Dept. of Commerce, NOAA, Space Weather Prediction Center
#
#          Geophysical Alert Message
#
Solar-terrestrial indices for 15 June follow.
The planetary A index was 5.`)

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "geophysical_alert"})
	if err != nil {
		t.Fatal(err)
	}

	for _, unwanted := range []string{":Product:", ":Issued:", "Prepared by", "Geophysical Alert Message"} {
		if strings.Contains(product.Text, unwanted) {
			t.Fatalf("WWV boilerplate was not removed:\n%s", product.Text)
		}
	}
	if !strings.Contains(product.Text, "Solar-terrestrial indices") {
		t.Fatalf("WWV body missing:\n%s", product.Text)
	}
}

func TestDiscussionProductFiltersToConfiguredSKMentions(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	storeTextProduct(t, cfg.Store, "eccc", "focn45.cwwg", `FOCN45 CWWG 161900 SIGNIFICANT WEATHER DISCUSSION ISSUED BY THE PRAIRIE AND ARCTIC STORM PREDICTION CENTRE OF ENVIRONMENT CANADA AT 2:00 PM CDT TUESDAY JUNE 16 2026.
ALERTS IN EFFECT...SEVERE THUNDERSTORM WATCHES FOR SOUTHERN ALBERTA AND PORTIONS OF SOUTHERN SASKATCHEWAN.
OVERVIEW...A LARGE UPPER LOW COMPLEX OVER HUDSON BAY.
DISCUSSION... ALBERTA...LOW AND ASSOCIATED FRONTS MOVING THROUGH ALBERTA TODAY WILL TRIGGER WIDESPREAD THUNDERSTORMS. THE PRIMARY THREAT AREA IS FROM NORTHWEST OF CALGARY THROUGH THE SOUTHEAST AND INTO SOUTHERN SASKATCHEWAN.
SOUTHERN SK...LOW PRESSURE MOVING ACROSS SOUTHERN ALBERTA WILL TRACK INTO EASTERN MONTANA TODAY. SCATTERED THUNDERSTORMS, SEVERAL OF WHICH COULD BECOME SEVERE.
SOUTHERN MB...NIL SIG WX. END/FULTON`)

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "eccc_discussion"})
	if err != nil {
		t.Fatal(err)
	}

	for _, wanted := range []string{
		"SOUTHERN SASKATCHEWAN",
		"Southern Saskatchewan. LOW PRESSURE",
		"SCATTERED THUNDERSTORMS",
	} {
		if !strings.Contains(product.Text, wanted) {
			t.Fatalf("discussion product missing %q:\n%s", wanted, product.Text)
		}
	}
	for _, unwanted := range []string{
		"ALERTS IN EFFECT",
		"SEVERE THUNDERSTORM WATCHES",
		"SOUTHERN MB",
		"NIL SIG WX",
		"LOW AND ASSOCIATED FRONTS MOVING THROUGH ALBERTA TODAY WILL TRIGGER WIDESPREAD THUNDERSTORMS",
	} {
		if strings.Contains(product.Text, unwanted) {
			t.Fatalf("discussion product leaked %q:\n%s", unwanted, product.Text)
		}
	}
}

func TestDiscussionProductAddsOpenerAndNormalizesBulletinShorthand(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	storeTextProduct(t, cfg.Store, "eccc", "focn45.cwwg", `FOCN45 CWWG 161900 SIGNIFICANT WEATHER DISCUSSION.
OVERVIEW...A 500 MB LOW WILL BRING 20-40 MM QPF AND 70 KM/H WINDS INTO SRN SK.
SOUTHERN SK...NIL SIG WX EARLY, THEN TSTMS OVER SRN SK WITH 8 C TEMPS. END/FULTON`)

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "eccc_discussion"})
	if err != nil {
		t.Fatal(err)
	}
	if len(product.Segments) < 2 || product.Segments[0].Kind != "opener" {
		t.Fatalf("discussion opener segment missing: %#v", product.Segments)
	}
	for _, wanted := range []string{
		"Here is the latest significant weather discussion",
		"500 millibars",
		"20 to 40 millimetres",
		"quantitative precipitation forecast",
		"70 kilometres per hour",
		"southern Saskatchewan",
		"no significant weather",
		"thunderstorms",
		"8 degrees Celsius",
	} {
		if !strings.Contains(product.Text, wanted) {
			t.Fatalf("discussion product missing %q:\n%s", wanted, product.Text)
		}
	}
	for _, unwanted := range []string{"QPF", "KM/H", "NIL SIG WX", "TSTMS", "SRN SK", "8 C"} {
		if strings.Contains(product.Text, unwanted) {
			t.Fatalf("discussion product leaked shorthand %q:\n%s", unwanted, product.Text)
		}
	}
}

func TestAlertsProductUsesNativeCAPRegistry(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	alert, err := capingest.ParseCAP([]byte(testCAP("urn:test:alert:1", "Update", "active", "2099-06-15T21:30:00-06:00", false)))
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{cfg: cfg}
	if _, err := service.recordCAPAlert(alert, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "alerts"})
	if err != nil {
		t.Fatal(err)
	}

	for _, wanted := range []string{
		"Environment Canada has updated a Yellow Warning - Severe Thunderstorm",
		"for City of Saskatoon; R.M. of Corman Park",
		"Forecast confidence is high with moderate impact expected",
		"capable of producing very strong wind gusts",
	} {
		if !strings.Contains(product.Text, wanted) {
			t.Fatalf("alert product missing %q:\n%s", wanted, product.Text)
		}
	}
	rows, err := cfg.Store.ListCAPArchives(context.Background(), "accepted", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || !strings.Contains(rows[0].RawXML, `<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">`) {
		t.Fatalf("archive is not storing native CAP payloads: %#v", rows)
	}
}

func TestCAPEndedUpdateRemovesReferencedActiveAlert(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	service := &Service{cfg: cfg}
	now := time.Now().UTC()
	active, err := capingest.ParseCAP([]byte(testCAP("urn:test:alert:original", "Alert", "active", "2099-06-15T21:30:00-06:00", false)))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.recordCAPAlert(active, now); err != nil {
		t.Fatal(err)
	}
	endedRaw := capWithReferences(
		testCAP("urn:test:alert:ended-update", "Update", "ended", "2099-06-15T21:30:00-06:00", true),
		"cap-pac@canada.ca,urn:test:alert:original,2026-06-15T15:58:00-06:00",
	)
	ended, err := capingest.ParseCAP([]byte(endedRaw))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := service.recordCAPAlert(ended, now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}

	rows, err := cfg.Store.ListCAPArchives(context.Background(), "accepted", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		if row.AlertID == active.Identifier {
			t.Fatalf("referenced active alert remained in accepted archive: %#v", rows)
		}
	}
	if len(rows) != 0 {
		t.Fatalf("accepted archive rows = %#v", rows)
	}
	expired, err := cfg.Store.ListCAPArchives(context.Background(), "expired", time.Now().UTC().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if len(expired) == 0 || expired[0].AlertID != ended.Identifier {
		t.Fatalf("ended update was not archived as expired: %#v", expired)
	}
}

func TestAlertBroadcastAudioPrefersBroadcastMP3Resource(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "testdata", "cap", "example_civilEmerg_AlertReady_2026_04_10T19_08_12_03_00IA99FB9E1_6FB6_4951_B234_863F1341C4C1.xml"))
	if err != nil {
		t.Fatal(err)
	}
	alert, err := capingest.ParseCAP(raw)
	if err != nil {
		t.Fatal(err)
	}

	audio := alertBroadcastAudio(alert, "en-CA")

	if audio.URL == "" || !strings.Contains(audio.URL, "naadstts.s3.amazonaws.com") {
		t.Fatalf("broadcast audio URL = %#v", audio)
	}
	if audio.MimeType != "audio/mpeg" {
		t.Fatalf("mime type = %q", audio.MimeType)
	}
}

func TestEndedAlertsArePrunedAfterGrace(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	alert, err := capingest.ParseCAP([]byte(testCAP("urn:test:alert:ended", "Update", "ended", "2099-06-15T21:30:00-06:00", true)))
	if err != nil {
		t.Fatal(err)
	}
	storeCAPArchiveRecord(cfg.Store, "accepted", capArchiveRecord{ID: alert.Identifier, FeedID: "sk-0001", UpdatedAt: time.Now().UTC().Add(-11 * time.Minute), Alert: alert, RawXML: alert.RawXML})

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "alerts"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(product.Text, "There are no weather alerts currently in effect") {
		t.Fatalf("ended alert should have been pruned:\n%s", product.Text)
	}
	accepted, err := cfg.Store.ListCAPArchives(context.Background(), "accepted", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if len(accepted) != 0 {
		t.Fatalf("ended alert remained accepted after grace: %#v", accepted)
	}
	expired, err := cfg.Store.ListCAPArchives(context.Background(), "expired", time.Now().UTC().Add(-time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	if len(expired) != 1 || expired[0].AlertID != alert.Identifier {
		t.Fatalf("ended alert was not archived as expired: %#v", expired)
	}

	_ = cfg.Store.DeleteCAPArchive(context.Background(), alert.Identifier, "sk-0001")
	recentRaw := strings.Replace(
		testCAP("urn:test:alert:recent-ended", "Update", "ended", "2099-06-15T21:30:00-06:00", true),
		"2026-06-15T15:58:00-06:00",
		time.Now().UTC().Format(time.RFC3339),
		1,
	)
	recentEnded, err := capingest.ParseCAP([]byte(recentRaw))
	if err != nil {
		t.Fatal(err)
	}
	storeCAPArchiveRecord(cfg.Store, "accepted", capArchiveRecord{ID: recentEnded.Identifier, FeedID: "sk-0001", UpdatedAt: time.Now().UTC(), Alert: recentEnded, RawXML: recentEnded.RawXML})
	product, err = newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "alerts"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(product.Text, "Environment Canada has ended a Yellow Warning - Severe Thunderstorm") {
		t.Fatalf("recent ended alert was not rendered:\n%s", product.Text)
	}
}

func TestExpiredAlertsArePrunedAfterExpiryGrace(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	expires := time.Now().UTC().Add(-11 * time.Minute).Format(time.RFC3339)
	alert, err := capingest.ParseCAP([]byte(testCAP("urn:test:alert:expired", "Alert", "active", expires, false)))
	if err != nil {
		t.Fatal(err)
	}
	storeCAPArchiveRecord(cfg.Store, "accepted", capArchiveRecord{ID: alert.Identifier, FeedID: "sk-0001", UpdatedAt: time.Now().UTC(), Alert: alert, RawXML: alert.RawXML})

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "alerts"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(product.Text, "There are no weather alerts currently in effect") {
		t.Fatalf("expired alert should have been pruned:\n%s", product.Text)
	}
	accepted, err := cfg.Store.ListCAPArchives(context.Background(), "accepted", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if len(accepted) != 0 {
		t.Fatalf("expired alert remained accepted after expiry grace: %#v", accepted)
	}
}

func TestBroadAlertsUseConfiguredCoverageRegionNames(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	alert, err := capingest.ParseCAP([]byte(testBroadCAP()))
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{cfg: cfg}
	if _, err := service.recordCAPAlert(alert, time.Now().UTC()); err != nil {
		t.Fatal(err)
	}

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "alerts"})
	if err != nil {
		t.Fatal(err)
	}

	wanted := "for areas within the Martensville, Warman, Rosthern, Delisle, and Wakaw region; and the Outlook, Watrous, Hanley, Imperial, and Dinsmore region"
	if !strings.Contains(product.Text, wanted) {
		t.Fatalf("broad alert region text missing %q:\n%s", wanted, product.Text)
	}
	if strings.Contains(product.Text, "R.M. of Corman Park") || strings.Contains(product.Text, "R.M. of Rudy") {
		t.Fatalf("broad alert should not list subregion area names:\n%s", product.Text)
	}
}

func TestCAPAlertMatchesExpandedCoverageRegion(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	alert, err := capingest.ParseCAP([]byte(testOutlookCAP()))
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{cfg: cfg}
	updates, err := service.recordCAPAlert(alert, time.Now().UTC())
	if err != nil {
		t.Fatal(err)
	}
	if len(updates) != 1 || !updates[0].Renderable {
		t.Fatalf("updates = %#v", updates)
	}

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: "sk-0001", PackageID: "alerts"})
	if err != nil {
		t.Fatal(err)
	}
	for _, wanted := range []string{
		"Environment Canada has updated a Yellow Warning - Severe Thunderstorm",
		"R.M. of Fertile Valley including Conquest Macrorie and Bounty",
		"Outlook",
	} {
		if !strings.Contains(product.Text, wanted) {
			t.Fatalf("alert product missing %q:\n%s", wanted, product.Text)
		}
	}
}

func writeFixture(t *testing.T, dir string) {
	t.Helper()
	mustWrite(t, filepath.Join(dir, "config.yaml"), `version: test
feeds_file: managed/configs/feeds.xml
operator:
  on_air_name:
    - text: Canada RadioMET
services:
  go:
    product_render:
      enabled: true
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "feeds.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<feeds>
  <feed id="sk-0001" enabled="true" timezone="America/Regina">
    <playout routine="true"/>
    <languages><lang code="en-CA"/></languages>
    <description><lang code="en-CA" text="Automated weather radio." suffix=""/></description>
    <alerts><cap_cp enabled="true"/></alerts>
    <locations>
      <observationLocations>
        <location id="CYXE" source="eccc"/>
        <location id="OUTLOOK" source="eccc" name_override="Outlook"/>
      </observationLocations>
      <coverage>
        <region id="065400" source="eccc" name="Martensville - Warman - Rosthern - Delisle and Wakaw region">
          <subregion id="065435"/>
        </region>
        <region id="065500" source="eccc" name="Outlook - Watrous - Hanley - Imperial and Dinsmore region">
          <subregion id="065522"/>
        </region>
      </coverage>
      <airQualityLocations>
        <location id="HAHJJ" source="eccc"/>
      </airQualityLocations>
    </locations>
    <transmitter_metadata>
      <transmitter>
        <site_name>Saskatoon</site_name>
        <callsign>XLF322</callsign>
        <frequency_mhz>162.550</frequency_mhz>
      </transmitter>
    </transmitter_metadata>
  </feed>
</feeds>
`)
	mustWrite(t, filepath.Join(dir, "managed", "csv", "FORECAST_LOCATIONS.csv"), `List of all the Forecast Locations by Program / Liste de tous les emplacements de prévisions par programme,,,,,,
CODE,NAME,NOM,PROGRAMS,PROGRAMMES,PROVINCE/WATERBODY 2 PROVINCE/PLAN D'EAU 2,PROVINCE/WATERBODY MOD. PROVINCE/PLAN D'EAU MOD.
065400,Martensville - Warman - Rosthern - Delisle - Wakaw,Martensville - Warman - Rosthern - Delisle - Wakaw,Public,Public,SK,SK
065500,Outlook - Watrous - Hanley - Imperial - Dinsmore,Outlook - Watrous - Hanley - Imperial - Dinsmore,Public,Public,SK,SK
`)
	mustWrite(t, filepath.Join(dir, "managed", "csv", "CLC_Base_Zone.csv"), `CLC,UUID,English,French
065435,fixture,R.M. of Corman Park northeast of the Yellowhead Highway incl. Martensville Warman and Langham,m.r. de Corman Park
065514,fixture,R.M. of Fertile Valley including Conquest Macrorie and Bounty,m.r. de Fertile Valley incluant Conquest Macrorie et Bounty
065522,fixture,R.M. of Rudy including Outlook and Glenside,m.r. de Rudy incluant Outlook et Glenside
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "packages.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<Packages>
  <defaults><enabled>true</enabled><reader_id>00</reader_id></defaults>
  <package id="current_conditions" enabled="true"/>
  <package id="forecast" enabled="true"/>
  <package id="air_quality" enabled="true"/>
  <package id="eccc_discussion" enabled="true"><locations stateProv="SK"/></package>
  <package id="alerts" enabled="true"/>
  <package id="geophysical_alert" enabled="true"/>
</Packages>
`)
}

func testCAP(identifier string, msgType string, locationStatus string, expires string, ended bool) string {
	response := "Monitor"
	headlineState := "in effect"
	description := "Environment Canada meteorologists are tracking a severe thunderstorm capable of producing very strong wind gusts, nickel size hail and heavy rain."
	if ended {
		response = "AllClear"
		headlineState = "ended"
		description = "The severe thunderstorm warning has ended for the area."
	}
	return `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>` + identifier + `</identifier>
  <sender>cap-pac@canada.ca</sender>
  <sent>2026-06-15T15:58:00-06:00</sent>
  <status>Actual</status>
  <msgType>` + msgType + `</msgType>
  <scope>Public</scope>
  <references/>
  <info>
    <language>en-CA</language>
    <category>Met</category>
    <event>thunderstorm</event>
    <responseType>` + response + `</responseType>
    <urgency>Immediate</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2026-06-15T15:58:00-06:00</effective>
    <onset>2026-06-15T16:00:00-06:00</onset>
    <expires>` + expires + `</expires>
    <senderName>Environment Canada</senderName>
    <headline>yellow warning - severe thunderstorm - ` + headlineState + `</headline>
    <description>` + description + `</description>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Location_Status</valueName><value>` + locationStatus + `</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Name</valueName><value>yellow warning - severe thunderstorm</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Coverage</valueName><value>Saskatchewan</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.1:MSC_Impact</valueName><value>moderate</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.1:MSC_Confidence</valueName><value>high</value></parameter>
    <area>
      <areaDesc>City of Saskatoon</areaDesc>
      <geocode><valueName>profile:CAP-CP:Location:0.3</valueName><value>065522</value></geocode>
    </area>
    <area>
      <areaDesc>R.M of Corman Park</areaDesc>
      <geocode><valueName>profile:CAP-CP:Location:0.3</valueName><value>065522</value></geocode>
    </area>
  </info>
</alert>`
}

func capWithReferences(raw string, references string) string {
	return strings.Replace(raw, "<references/>", "<references>"+references+"</references>", 1)
}

func testBroadCAP() string {
	return `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>urn:test:broad-alert</identifier>
  <sender>cap-pac@canada.ca</sender>
  <sent>2026-06-15T15:58:00-06:00</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <references/>
  <info>
    <language>en-CA</language>
    <category>Met</category>
    <event>snowfall</event>
    <responseType>Monitor</responseType>
    <urgency>Future</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2026-06-15T15:58:00-06:00</effective>
    <onset>2026-06-16T06:00:00-06:00</onset>
    <expires>2099-06-15T21:30:00-06:00</expires>
    <senderName>Environment Canada</senderName>
    <headline>yellow warning - snowfall - in effect</headline>
    <description>Snowfall with total amounts of 15 to 25 centimetres is expected.</description>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Location_Status</valueName><value>active</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Name</valueName><value>yellow warning - snowfall</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.1:Newly_Active_Areas</valueName><value>065435,065522</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.1:MSC_Impact</valueName><value>moderate</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.1:MSC_Confidence</valueName><value>high</value></parameter>
    <area>
      <areaDesc>R.M. of Corman Park northeast of the Yellowhead Highway incl. Martensville Warman and Langham</areaDesc>
      <geocode><valueName>layer:EC-MSC-SMC:1.0:CLC</valueName><value>065435</value></geocode>
    </area>
    <area>
      <areaDesc>R.M. of Rudy including Outlook and Glenside</areaDesc>
      <geocode><valueName>layer:EC-MSC-SMC:1.0:CLC</valueName><value>065522</value></geocode>
    </area>
  </info>
</alert>`
}

func testOutlookCAP() string {
	return `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>urn:test:outlook-alert</identifier>
  <sender>cap-pac@canada.ca</sender>
  <sent>2026-06-16T15:49:00-06:00</sent>
  <status>Actual</status>
  <msgType>Update</msgType>
  <scope>Public</scope>
  <info>
    <language>en-CA</language>
    <category>Met</category>
    <event>thunderstorm</event>
    <responseType>Monitor</responseType>
    <urgency>Immediate</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2026-06-16T15:49:00-06:00</effective>
    <expires>2099-06-16T18:48:40-06:00</expires>
    <senderName>Environment Canada</senderName>
    <headline>yellow warning - severe thunderstorm - in effect</headline>
    <description>Thunderstorm Location: Near Conquest. Locations in the Path: Conquest Outlook Macrorie.</description>
    <eventCode><valueName>profile:CAP-CP:Event:0.4</valueName><value>thunderstorm</value></eventCode>
    <eventCode><valueName>SAME</valueName><value>SVR</value></eventCode>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Location_Status</valueName><value>active</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Name</valueName><value>yellow warning - severe thunderstorm</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.0:Alert_Coverage</valueName><value>Saskatchewan</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.1:MSC_Impact</valueName><value>moderate</value></parameter>
    <parameter><valueName>layer:EC-MSC-SMC:1.1:MSC_Confidence</valueName><value>high</value></parameter>
    <area>
      <areaDesc>R.M. of Fertile Valley including Conquest Macrorie and Bounty</areaDesc>
      <geocode><valueName>layer:EC-MSC-SMC:1.0:CLC</valueName><value>065514</value></geocode>
      <geocode><valueName>profile:CAP-CP:Location:0.3</valueName><value>4711027</value></geocode>
      <geocode><valueName>profile:CAP-CP:Location:0.3</valueName><value>4712019</value></geocode>
      <geocode><valueName>profile:CAP-CP:Location:0.3</valueName><value>4712022</value></geocode>
    </area>
  </info>
</alert>`
}

func mustWrite(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func loadFixtureConfig(t *testing.T, dir string) loadedConfig {
	t.Helper()
	cfg, err := loadConfig(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	store, err := datastore.OpenSQLite(context.Background(), datastore.SQLiteConfig{}, dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(store.Close)
	cfg.Store = store
	return cfg
}

func storeObservationJSON(t *testing.T, store datastore.Store, id string, raw string) {
	t.Helper()
	payload := mustJSONMap(t, raw)
	observedAt, _ := payload["observed_at"].(string)
	if err := store.UpsertObservation(context.Background(), datastore.ObservationRecord{
		Source:        "eccc",
		LocationID:    id,
		StationID:     id,
		ObservedAtRaw: observedAt,
		Payload:       payload,
	}); err != nil {
		t.Fatal(err)
	}
}

func storeForecastJSON(t *testing.T, store datastore.Store, id string, regionID string, raw string) {
	t.Helper()
	payload := mustJSONMap(t, raw)
	issuedAt, _ := payload["issued_at"].(string)
	if err := store.UpsertForecast(context.Background(), datastore.ForecastRecord{
		Source:      "eccc",
		ForecastID:  id,
		RegionID:    regionID,
		IssuedAtRaw: issuedAt,
		Payload:     payload,
	}); err != nil {
		t.Fatal(err)
	}
}

func storeProductPayloadJSON(t *testing.T, store datastore.Store, kind string, source string, id string, raw string) {
	t.Helper()
	if err := store.StoreProductPayload(context.Background(), datastore.ProductPayloadRecord{
		Kind:    kind,
		Source:  source,
		ID:      id,
		Payload: mustJSONMap(t, raw),
	}); err != nil {
		t.Fatal(err)
	}
}

func storeTextProduct(t *testing.T, store datastore.Store, source string, id string, text string) {
	t.Helper()
	if err := store.StoreTextProduct(context.Background(), datastore.TextProductRecord{
		Source: source,
		ID:     id,
		Text:   text,
	}); err != nil {
		t.Fatal(err)
	}
}

func mustJSONMap(t *testing.T, raw string) map[string]any {
	t.Helper()
	var payload map[string]any
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		t.Fatal(err)
	}
	return payload
}
