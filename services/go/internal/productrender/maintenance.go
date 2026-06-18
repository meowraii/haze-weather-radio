package productrender

import (
	"context"
	"log"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

type maintenanceCleanupResult struct {
	PurgedAlerts          int
	ProcessedRegistries   int
	ArchivedExpiredAlerts int
	Errors                []string
}

func (s *Service) newCleanupTimer(now time.Time) *time.Timer {
	delay, ok := nextCleanupDelay(s.cfg.Root.Services.Go.ProductRender.Cleanup, now)
	if !ok {
		return nil
	}
	return time.NewTimer(delay)
}

func (s *Service) resetCleanupTimer(timer *time.Timer, now time.Time) *time.Timer {
	stopTimer(timer)
	return s.newCleanupTimer(now)
}

func (s *Service) runScheduledCleanup(now time.Time) {
	s.refreshConfigIfNeeded()
	if !cleanupEnabled(s.cfg.Root.Services.Go.ProductRender.Cleanup) {
		return
	}
	result := runMaintenanceCleanup(s.cfg.BaseDir, now.UTC(), s.cfg.Store)
	if len(result.Errors) > 0 {
		for _, detail := range result.Errors {
			log.Printf("maintenance cleanup warning: %s", detail)
		}
	}
	log.Printf(
		"maintenance cleanup purged %d expired alert(s)",
		result.PurgedAlerts,
	)
	_ = s.bridge.Publish(map[string]any{
		"type":    "maintenance.cleanup.completed",
		"source":  serviceID,
		"subject": "nightly-cleanup",
		"data": map[string]any{
			"purged_alerts":           result.PurgedAlerts,
			"processed_registries":    result.ProcessedRegistries,
			"archived_expired_alerts": result.ArchivedExpiredAlerts,
			"errors":                  result.Errors,
		},
	})
}

func cleanupEnabled(cfg cleanupConfig) bool {
	return cfg.Enabled == nil || *cfg.Enabled
}

func nextCleanupDelay(cfg cleanupConfig, now time.Time) (time.Duration, bool) {
	if !cleanupEnabled(cfg) {
		return 0, false
	}
	hour := 3
	if cfg.Hour != nil {
		hour = *cfg.Hour
	}
	if hour < 0 || hour > 23 {
		hour = 3
	}
	minute := 0
	if cfg.Minute != nil {
		minute = *cfg.Minute
	}
	if minute < 0 || minute > 59 {
		minute = 0
	}
	next := time.Date(now.Year(), now.Month(), now.Day(), hour, minute, 0, 0, now.Location())
	if !next.After(now) {
		next = next.Add(24 * time.Hour)
	}
	return next.Sub(now), true
}

func runMaintenanceCleanup(baseDir string, now time.Time, store datastore.Store) maintenanceCleanupResult {
	result := maintenanceCleanupResult{}
	purged, archived, registries, err := purgeExpiredCAPRegistries(baseDir, now, store)
	result.PurgedAlerts = purged
	result.ArchivedExpiredAlerts = archived
	result.ProcessedRegistries = registries
	if err != nil {
		result.Errors = append(result.Errors, err.Error())
	}
	return result
}

func purgeExpiredCAPRegistries(_ string, now time.Time, store datastore.Store) (int, int, int, error) {
	if store == nil {
		return 0, 0, 0, datastore.ErrNotConfigured
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	rows, err := store.ListCAPArchives(ctx, "accepted", time.Time{})
	if err != nil {
		return 0, 0, 0, err
	}
	purged := 0
	for _, row := range rows {
		alert, ok := storedCAPArchiveAlert(row)
		if !ok || isRenderableCAPEntry(capRegistryEntry{UpdatedAt: firstNonZeroTime(row.UpdatedAt, row.StoredAt), Alert: alert}, now) {
			continue
		}
		record := capArchiveRecord{
			ID:        row.AlertID,
			FeedID:    row.FeedID,
			Status:    "expired",
			Reason:    "expired or ended outside relay grace",
			UpdatedAt: now,
			Alert:     alert,
			RawXML:    row.RawXML,
		}
		storeCAPArchiveRecord(store, "expired", record)
		if err := store.DeleteCAPArchiveBucketItem(ctx, row.AlertID, row.FeedID, "accepted"); err != nil {
			return purged, purged, len(rows), err
		}
		purged++
	}
	return purged, purged, len(rows), nil
}

func timerChannel(timer *time.Timer) <-chan time.Time {
	if timer == nil {
		return nil
	}
	return timer.C
}

func stopTimer(timer *time.Timer) {
	if timer == nil {
		return
	}
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
}
