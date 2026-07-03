package productrender

import (
	"context"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

func TestRunMaintenanceCleanupPrunesExpiredCAPArchiveRows(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 6, 16, 9, 0, 0, 0, time.UTC)
	expired := parseTestAlert(t, testCAP("urn:test:expired", "Alert", "", now.Add(-30*time.Minute).Format(time.RFC3339), false))
	current := parseTestAlert(t, testCAP("urn:test:current", "Alert", "", now.Add(2*time.Hour).Format(time.RFC3339), false))
	store, err := datastore.OpenSQLite(context.Background(), datastore.SQLiteConfig{}, dir)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	storeCAPArchiveRecord(store, "accepted", capArchiveRecord{ID: expired.Identifier, FeedID: "sk-0001", UpdatedAt: now.Add(-1 * time.Hour), Alert: expired, RawXML: expired.RawXML})
	storeCAPArchiveRecord(store, "accepted", capArchiveRecord{ID: current.Identifier, FeedID: "sk-0001", UpdatedAt: now, Alert: current, RawXML: current.RawXML})

	result := runMaintenanceCleanup(dir, now, store)

	if result.PurgedAlerts != 1 || result.ArchivedExpiredAlerts != 1 {
		t.Fatalf("cleanup result = %#v", result)
	}
	active, err := store.ListCAPArchives(context.Background(), "accepted", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if len(active) != 1 || active[0].AlertID != current.Identifier {
		t.Fatalf("active registry = %#v", active)
	}
	archive, err := store.ListCAPArchives(context.Background(), "expired", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if len(archive) != 1 || archive[0].AlertID != expired.Identifier {
		t.Fatalf("expired archive = %#v", archive)
	}
}

func TestNextCleanupDelayUsesNextDailyWindow(t *testing.T) {
	now := time.Date(2026, 6, 16, 2, 30, 0, 0, time.Local)
	hour := 3
	minute := 0
	defaultDelay, ok := nextCleanupDelay(cleanupConfig{}, now)
	if !ok || defaultDelay != 30*time.Minute {
		t.Fatalf("default delay = %s ok=%v", defaultDelay, ok)
	}
	delay, ok := nextCleanupDelay(cleanupConfig{Hour: &hour, Minute: &minute}, now)
	if !ok || delay != 30*time.Minute {
		t.Fatalf("delay before window = %s ok=%v", delay, ok)
	}
	delay, ok = nextCleanupDelay(cleanupConfig{Hour: &hour, Minute: &minute}, now.Add(45*time.Minute))
	if !ok || delay != 23*time.Hour+45*time.Minute {
		t.Fatalf("delay after window = %s ok=%v", delay, ok)
	}
	disabled := false
	if _, ok := nextCleanupDelay(cleanupConfig{Enabled: &disabled}, now); ok {
		t.Fatalf("disabled cleanup returned ok")
	}
}

func parseTestAlert(t *testing.T, raw string) capmodel.Alert {
	t.Helper()
	alert, err := capmodel.ParseCAP([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	return alert
}
