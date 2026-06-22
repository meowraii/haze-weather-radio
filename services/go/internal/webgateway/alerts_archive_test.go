package webgateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capingest"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

func TestWithArchiveStoreUsesSQLiteDefault(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(`storage:
  sqlite:
    enabled: true
    path: runtime/state/haze.db
`), 0o644); err != nil {
		t.Fatal(err)
	}
	called := false

	err := withArchiveStore(configPath, func(context.Context, datastore.Store) error {
		called = true
		return nil
	})

	if err != nil {
		t.Fatalf("withArchiveStore returned %v", err)
	}
	if !called {
		t.Fatal("archive callback was not called")
	}
}

func TestArchiveSAMEAllowedSuppressesCancellationsAndStaleWarnings(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	cancel := parseArchiveTestAlert(t, archiveTestCAP("urn:test:cancel", "Cancel", "yellow warning - severe thunderstorm - ended", "2099-06-15T21:30:00-06:00", true))
	if archiveSAMEAllowed(configPath, cancel, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC)) {
		t.Fatal("cancellation should not be eligible for SAME rebroadcast")
	}

	warning := parseArchiveTestAlert(t, archiveTestCAP("urn:test:svr", "Alert", "yellow warning - severe thunderstorm - in effect", "2099-06-15T21:30:00-06:00", false))
	if !archiveSAMEAllowed(configPath, warning, time.Date(2026, 6, 15, 22, 27, 0, 0, time.UTC)) {
		t.Fatal("fresh severe thunderstorm warning should be eligible for SAME")
	}
	if archiveSAMEAllowed(configPath, warning, time.Date(2026, 6, 15, 22, 29, 0, 0, time.UTC)) {
		t.Fatal("stale severe thunderstorm warning should not be eligible for SAME")
	}
}

func TestArchiveRecordPayloadIncludesCAPXMLURLAndSAMEPreviewFlag(t *testing.T) {
	baseDir := t.TempDir()
	alert := parseArchiveTestAlert(t, archiveTestCAP("urn:test:svr", "Alert", "yellow warning - severe thunderstorm - in effect", "2099-06-15T21:30:00-06:00", false))
	payload := archiveRecordPayload(archiveCAPRecord{
		ID:     "urn:test:svr",
		FeedID: "sk-0001",
		Status: "accepted",
		Alert:  alert,
	}, "accepted", baseDir)

	if got := strings.TrimSpace(payload["cap_xml_url"].(string)); !strings.Contains(got, "/api/v1/alerts/archive/cap.xml?") || !strings.Contains(got, "feed_id=sk-0001") {
		t.Fatalf("cap_xml_url was not populated correctly: %q", got)
	}
	if available, _ := payload["same_preview_available"].(bool); !available {
		t.Fatal("SAME preview should be available for a mappable warning")
	}
}

func TestAlertsArchiveCAPXMLRequiresAuthAndServesStoredXML(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte(`webpanel:
  authentication:
    enabled: true
storage:
  sqlite:
    enabled: true
    path: runtime/state/haze.db
`), 0o644); err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(Config{}, configPath, dir)
	authEnabled := true
	authConfig := Config{}
	authConfig.Webpanel.Authentication.Enabled = &authEnabled
	server.auth = NewAuthManager(authConfig)
	server.auth.password = []byte("secret")
	token, err := server.auth.Login("secret")
	if err != nil {
		t.Fatal(err)
	}
	rawCAP := archiveTestCAP("urn:test:xml", "Alert", "yellow warning - severe thunderstorm - in effect", "2099-06-15T21:30:00-06:00", false)
	if err := withArchiveStore(configPath, func(ctx context.Context, store datastore.Store) error {
		return store.StoreCAPArchive(ctx, datastore.CAPArchiveRecord{
			AlertID: "urn:test:xml",
			FeedID:  "sk-0001",
			Status:  "accepted",
			RawXML:  rawCAP,
		})
	}); err != nil {
		t.Fatal(err)
	}

	unauthorized := httptest.NewRecorder()
	server.alertsArchiveCAPXML(unauthorized, httptest.NewRequest(http.MethodGet, "/api/v1/alerts/archive/cap.xml?id=urn:test:xml&feed_id=sk-0001", nil))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("expected unauthorized status, got %d", unauthorized.Code)
	}

	request := httptest.NewRequest(http.MethodGet, "/api/v1/alerts/archive/cap.xml?id=urn:test:xml&feed_id=sk-0001&token="+token, nil)
	response := httptest.NewRecorder()
	server.alertsArchiveCAPXML(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Content-Type"); !strings.Contains(got, "application/cap+xml") {
		t.Fatalf("unexpected content type %q", got)
	}
	if !strings.Contains(response.Body.String(), "<identifier>urn:test:xml</identifier>") {
		t.Fatal("served CAP XML did not contain the archived alert")
	}
}

func parseArchiveTestAlert(t *testing.T, raw string) capingest.Alert {
	t.Helper()
	alert, err := capingest.ParseCAP([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return alert
}

func archiveTestCAP(identifier string, msgType string, headline string, expires string, ended bool) string {
	response := "Monitor"
	if ended {
		response = "AllClear"
	}
	return `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>` + identifier + `</identifier>
  <sender>cap-pac@canada.ca</sender>
  <sent>2026-06-15T15:58:00-06:00</sent>
  <status>Actual</status>
  <msgType>` + msgType + `</msgType>
  <scope>Public</scope>
  <info>
    <language>en-CA</language>
    <category>Met</category>
    <event>thunderstorm</event>
    <responseType>` + response + `</responseType>
    <urgency>Immediate</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <effective>2026-06-15T15:58:00-06:00</effective>
    <expires>` + expires + `</expires>
    <headline>` + headline + `</headline>
    <area>
      <areaDesc>R.M of Rudy including Outlook and Glenside</areaDesc>
      <geocode><valueName>profile:CAP-CP:Location:0.3</valueName><value>065522</value></geocode>
    </area>
  </info>
</alert>`
}
