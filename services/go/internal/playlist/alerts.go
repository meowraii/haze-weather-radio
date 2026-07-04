package playlist

import (
	"context"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

const alertRegistryGrace = 10 * time.Minute

type capRegistryEntry struct {
	UpdatedAt time.Time
	Alert     capmodel.Alert
}

func (p *feedPlanner) hasRoutineAlerts(now time.Time) bool {
	if p.cfg.Store == nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	rows, err := p.cfg.Store.ListCAPArchives(ctx, "accepted", time.Time{})
	if err != nil {
		return false
	}
	for _, row := range rows {
		if strings.TrimSpace(row.FeedID) != p.feed.ID {
			continue
		}
		alert, err := capmodel.ParseCAP([]byte(strings.TrimSpace(row.RawXML)))
		if err != nil || alert.Identifier == "" {
			_ = p.cfg.Store.DeleteCAPArchiveBucketItem(ctx, row.AlertID, row.FeedID, "accepted")
			continue
		}
		updated := row.UpdatedAt
		if updated.IsZero() {
			updated = row.StoredAt
		}
		if isRenderableCAPEntry(capRegistryEntry{UpdatedAt: updated, Alert: alert}, now.UTC()) {
			return true
		}
		archiveExpiredCAPRow(ctx, p.cfg.Store, row, alert, now.UTC())
	}
	return false
}

func archiveExpiredCAPRow(ctx context.Context, store datastore.Store, row datastore.StoredCAPArchive, alert capmodel.Alert, now time.Time) {
	if store == nil || strings.TrimSpace(row.AlertID) == "" {
		return
	}
	info := firstCAPInfo(alert)
	_ = store.StoreCAPArchive(ctx, datastore.CAPArchiveRecord{
		AlertID:      row.AlertID,
		FeedID:       row.FeedID,
		Bucket:       "expired",
		Status:       "expired",
		Reason:       "expired or ended outside relay grace",
		Sender:       alert.Sender,
		Source:       row.Source,
		SentAtRaw:    alert.Sent,
		UpdatedAtRaw: now.Format(time.RFC3339Nano),
		ExpiresAtRaw: info.Expires,
		Event:        info.Event,
		Headline:     firstNonBlankText(info.Headline, info.Event),
		RawXML:       row.RawXML,
		Metadata: map[string]any{
			"message_type": alert.MessageType,
			"scope":        alert.Scope,
		},
	})
	_ = store.DeleteCAPArchiveBucketItem(ctx, row.AlertID, row.FeedID, "accepted")
}

func firstCAPInfo(alert capmodel.Alert) capmodel.AlertInfo {
	for _, info := range alert.Infos {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(info.Language)), "en") {
			return info
		}
	}
	if len(alert.Infos) > 0 {
		return alert.Infos[0]
	}
	return capmodel.AlertInfo{}
}

func firstNonBlankText(values ...string) string {
	for _, value := range values {
		if text := strings.TrimSpace(value); text != "" {
			return text
		}
	}
	return ""
}

func isRenderableCAPEntry(entry capRegistryEntry, now time.Time) bool {
	alert := entry.Alert
	if alert.Identifier == "" || strings.EqualFold(alert.MessageType, "Cancel") {
		return false
	}
	if isExplicitCAPEnd(alert) {
		anchor := capRegistryAnchor(alert, entry.UpdatedAt)
		return anchor.IsZero() || now.Before(anchor.Add(alertRegistryGrace))
	}
	if expires := alertExpiresAt(alert); !expires.IsZero() {
		return now.Before(expires.Add(alertRegistryGrace))
	}
	return true
}

func capRegistryAnchor(alert capmodel.Alert, fallback time.Time) time.Time {
	if isExplicitCAPEnd(alert) {
		if anchor := firstCAPTime(alert.Sent, alert.Infos); !anchor.IsZero() {
			return anchor
		}
	}
	return fallback
}

func isExplicitCAPEnd(alert capmodel.Alert) bool {
	if strings.EqualFold(alert.MessageType, "Cancel") {
		return true
	}
	sawEnd := false
	for _, info := range alert.Infos {
		if capInfoExplicitEnd(info) {
			sawEnd = true
			continue
		}
		return false
	}
	return sawEnd
}

func capInfoExplicitEnd(info capmodel.AlertInfo) bool {
	for _, response := range info.Response {
		if strings.EqualFold(response, "AllClear") {
			return true
		}
	}
	status := strings.ToLower(alertParam(info, "layer:EC-MSC-SMC:1.0:Alert_Location_Status"))
	if status == "" {
		status = strings.ToLower(alertParam(info, "layer:EC-MSC-SMC:1.1:Alert_Location_Status"))
	}
	return status == "ended" || strings.Contains(strings.ToLower(info.Headline), "ended")
}

func isCAPEnded(alert capmodel.Alert, now time.Time) bool {
	if isExplicitCAPEnd(alert) {
		return true
	}
	if expires := alertExpiresAt(alert); !expires.IsZero() && now.After(expires) {
		return true
	}
	return false
}

func alertExpiresAt(alert capmodel.Alert) time.Time {
	var latest time.Time
	for _, info := range capActiveOrAllInfos(alert.Infos) {
		if parsed := parseTime(info.Expires); !parsed.IsZero() && parsed.After(latest) {
			latest = parsed
		}
	}
	return latest
}

func capActiveOrAllInfos(infos []capmodel.AlertInfo) []capmodel.AlertInfo {
	active := make([]capmodel.AlertInfo, 0, len(infos))
	sawEnd := false
	for _, info := range infos {
		if capInfoExplicitEnd(info) {
			sawEnd = true
			continue
		}
		active = append(active, info)
	}
	if sawEnd && len(active) > 0 {
		return active
	}
	return infos
}

func firstCAPTime(sent string, infos []capmodel.AlertInfo) time.Time {
	if parsed := parseTime(sent); !parsed.IsZero() {
		return parsed
	}
	for _, info := range infos {
		for _, raw := range []string{info.Effective, info.Onset, info.Expires} {
			if parsed := parseTime(raw); !parsed.IsZero() {
				return parsed
			}
		}
	}
	return time.Time{}
}

func alertParam(info capmodel.AlertInfo, name string) string {
	for _, param := range info.Parameters {
		if strings.EqualFold(strings.TrimSpace(param.Name), name) {
			return strings.TrimSpace(param.Value)
		}
	}
	return ""
}
