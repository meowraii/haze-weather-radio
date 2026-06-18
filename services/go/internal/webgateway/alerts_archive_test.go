package webgateway

import (
	"context"
	"os"
	"path/filepath"
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
	cancel := parseArchiveTestAlert(t, archiveTestCAP("urn:test:cancel", "Cancel", "yellow warning - severe thunderstorm - ended", "2099-06-15T21:30:00-06:00", true))
	if archiveSAMEAllowed(cancel, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC)) {
		t.Fatal("cancellation should not be eligible for SAME rebroadcast")
	}

	warning := parseArchiveTestAlert(t, archiveTestCAP("urn:test:svr", "Alert", "yellow warning - severe thunderstorm - in effect", "2099-06-15T21:30:00-06:00", false))
	if !archiveSAMEAllowed(warning, time.Date(2026, 6, 15, 22, 27, 0, 0, time.UTC)) {
		t.Fatal("fresh severe thunderstorm warning should be eligible for SAME")
	}
	if archiveSAMEAllowed(warning, time.Date(2026, 6, 15, 22, 29, 0, 0, time.UTC)) {
		t.Fatal("stale severe thunderstorm warning should not be eligible for SAME")
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
