package webgateway

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

func TestBannerPayloadSerializesAcceptedActiveAlerts(t *testing.T) {
	dir := t.TempDir()
	configPath := testBannerConfig(t, dir)
	rawCAP := `<alert>
  <identifier>urn:test:banner:svr</identifier>
  <sender>cap-pac@canada.ca</sender>
  <sent>2099-06-17T01:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-CA</language>
    <event>Severe Thunderstorm Warning</event>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Likely</certainty>
    <effective>2099-06-17T01:00:00Z</effective>
    <expires>2099-06-17T03:00:00Z</expires>
    <senderName>Environment Canada</senderName>
    <headline>Severe Thunderstorm Warning - in effect</headline>
    <description>Nickel size hail is possible.</description>
    <instruction>Take shelter if threatening weather approaches.</instruction>
    <area>
      <areaDesc>City of Saskatoon</areaDesc>
      <geocode><valueName>SAME</valueName><value>065100</value></geocode>
    </area>
  </info>
</alert>`
	storeBannerCAP(t, configPath, rawCAP, "sk-0001", "accepted")

	payload := bannerPayloadForTest(configPath, "sk-0001", time.Date(2026, 6, 17, 2, 0, 0, 0, time.UTC))
	if !payload.Active {
		t.Fatal("banner should be active")
	}
	if payload.Signature == "" {
		t.Fatal("missing signature")
	}
	if len(payload.Alerts) != 1 {
		t.Fatalf("alerts = %#v", payload.Alerts)
	}
	alert := payload.Alerts[0]
	if alert.Headline != "Severe Thunderstorm Warning" {
		t.Fatalf("headline = %q", alert.Headline)
	}
	if alert.BackgroundColor != "#931102" {
		t.Fatalf("background = %q", alert.BackgroundColor)
	}
	if !strings.Contains(alert.Message, "Environment Canada has issued") || !strings.Contains(alert.Message, "City of Saskatoon") {
		t.Fatalf("message = %q", alert.Message)
	}
}

func TestBannerPayloadFiltersExpiredAndFeed(t *testing.T) {
	dir := t.TempDir()
	configPath := testBannerConfig(t, dir)
	rawCAP := `<alert>
  <identifier>urn:test:banner:expired</identifier>
  <sender>cap-pac@canada.ca</sender>
  <sent>2026-06-17T00:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-CA</language>
    <event>Thunderstorm Watch</event>
    <severity>Moderate</severity>
    <effective>2026-06-17T00:00:00Z</effective>
    <expires>2026-06-17T01:00:00Z</expires>
    <headline>Thunderstorm Watch - in effect</headline>
  </info>
</alert>`
	storeBannerCAP(t, configPath, rawCAP, "sk-0002", "accepted")

	payload := bannerPayloadForTest(configPath, "sk-0001", time.Date(2026, 6, 17, 2, 0, 0, 0, time.UTC))
	if payload.Active || len(payload.Alerts) != 0 {
		t.Fatalf("payload should be idle: %#v", payload)
	}
}

func TestBannerHubActivatesFromPlayoutEvent(t *testing.T) {
	dir := t.TempDir()
	configPath := testBannerConfig(t, dir)
	rawCAP := `<alert>
  <identifier>urn:test:banner:onair</identifier>
  <sender>cap-pac@canada.ca</sender>
  <sent>2099-06-17T01:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-CA</language>
    <event>Severe Thunderstorm Warning</event>
    <severity>Severe</severity>
    <effective>2099-06-17T01:00:00Z</effective>
    <expires>2099-06-17T03:00:00Z</expires>
    <senderName>Environment Canada</senderName>
    <headline>Severe Thunderstorm Warning - in effect</headline>
    <area><areaDesc>City of Saskatoon</areaDesc></area>
  </info>
</alert>`
	storeBannerCAP(t, configPath, rawCAP, "sk-0001", "accepted")
	queueID := "000_sk-0001_urn_test_banner_onair_same_header"
	queue := sameQueueItem{
		ID:        queueID,
		AlertID:   "urn:test:banner:onair",
		Type:      "same_header",
		Status:    "claimed",
		FeedIDs:   []string{"sk-0001"},
		Header:    "ZCZC-WXR-SVR-065100+0030-1680100-XLF322  -",
		Event:     "SVR",
		CreatedAt: time.Date(2026, 6, 17, 1, 0, 0, 0, time.UTC),
	}
	rawQueue, err := json.Marshal(queue)
	if err != nil {
		t.Fatal(err)
	}
	mustWrite(t, dir+"/runtime/queues/alerts/"+queueID+".json", string(rawQueue))

	hub := NewBannerHub(configPath, "")
	hub.handleEvent([]byte(`{"type":"alert.playout.started","feed_ids":["sk-0001"],"queue_id":"`+queueID+`","event":"SVR"}`), time.Now().UTC())
	payload := buildBannerPayload(configPath, "sk-0001", hub)

	if !payload.Active || len(payload.Alerts) != 1 {
		t.Fatalf("payload = %#v", payload)
	}
	if payload.Alerts[0].Identifier != "urn:test:banner:onair" {
		t.Fatalf("alert = %#v", payload.Alerts[0])
	}
}

func testBannerConfig(t *testing.T, dir string) string {
	t.Helper()
	configPath := dir + "/config.yaml"
	mustWrite(t, configPath, "storage:\n  sqlite:\n    enabled: true\n    path: runtime/state/haze.db\n")
	return configPath
}

func storeBannerCAP(t *testing.T, configPath string, rawCAP string, feedID string, bucket string) {
	t.Helper()
	cfg, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	store, err := datastore.Open(context.Background(), cfg.Storage, dirOf(configPath))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err := store.StoreCAPArchive(context.Background(), datastore.CAPArchiveRecord{
		AlertID: "test-" + feedID + "-" + bucket,
		FeedID:  feedID,
		Bucket:  bucket,
		Status:  bucket,
		Event:   "SVR",
		RawXML:  rawCAP,
	}); err != nil {
		t.Fatal(err)
	}
}

func dirOf(path string) string {
	index := strings.LastIndexAny(path, `/\`)
	if index < 0 {
		return "."
	}
	return path[:index]
}
