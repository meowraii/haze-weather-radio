package productrender

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alerttext"
	"github.com/meowraii/haze-weather-radio/services/go/internal/capingest"
	"github.com/meowraii/haze-weather-radio/services/go/internal/capsame"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

const alertRegistryGrace = 10 * time.Minute

type capRegistryEntry struct {
	ID        string
	UpdatedAt time.Time
	Alert     capingest.Alert
	RawXML    string
}

type capArchiveRecord struct {
	ID        string          `json:"id"`
	FeedID    string          `json:"feed_id,omitempty"`
	Status    string          `json:"status"`
	Reason    string          `json:"reason,omitempty"`
	UpdatedAt time.Time       `json:"updated_at"`
	Alert     capingest.Alert `json:"alert"`
	RawXML    string          `json:"raw_xml,omitempty"`
}

func (s *Service) handleCAPAlert(event map[string]any) {
	alert, ok := capAlertFromEvent(event)
	if !ok {
		return
	}
	s.refreshConfigIfNeeded()
	updated, err := s.recordCAPAlert(alert, time.Now().UTC())
	if err != nil {
		return
	}
	for _, item := range updated {
		_ = s.bridge.Publish(map[string]any{
			"type":    "cap.alert.registry.updated",
			"source":  serviceID,
			"feed_id": item.FeedID,
			"subject": alert.Identifier,
			"data": map[string]any{
				"feed_id":    item.FeedID,
				"alert_id":   alert.Identifier,
				"path":       item.Path,
				"renderable": item.Renderable,
			},
		})
		if item.Cancelled {
			_ = s.bridge.Publish(map[string]any{
				"type":    "cap.alert.cancelled",
				"source":  serviceID,
				"feed_id": item.FeedID,
				"subject": alert.Identifier,
				"data": map[string]any{
					"feed_id":   item.FeedID,
					"alert_id":  alert.Identifier,
					"alert_ids": item.CancelledIDs,
				},
			})
		}
		if item.Broadcast {
			data := map[string]any{
				"feed_id":       item.FeedID,
				"alert_id":      alert.Identifier,
				"alert_sent_at": alert.Sent,
				"message_type":  alert.MessageType,
				"path":          item.Path,
				"package_id":    "alerts",
				"title":         alertBroadcastTitle(alert),
			}
			if item.Headline != "" {
				data["headline"] = item.Headline
				data["title"] = item.Headline
			}
			if item.Event != "" {
				data["event"] = item.Event
			}
			if item.Severity != "" {
				data["severity"] = item.Severity
			}
			if item.Urgency != "" {
				data["urgency"] = item.Urgency
			}
			if item.Certainty != "" {
				data["certainty"] = item.Certainty
			}
			if item.Description != "" {
				data["description"] = item.Description
			}
			if item.Instruction != "" {
				data["instruction"] = item.Instruction
			}
			if item.BackgroundColor != "" {
				data["background_color"] = item.BackgroundColor
			}
			if item.BroadcastImmediate {
				data["broadcast_immediate"] = true
			}
			if expires := alertExpiresAt(alert); !expires.IsZero() {
				data["alert_expires_at"] = expires.Format(time.RFC3339Nano)
			}
			if item.AlertText != "" {
				data["alert_text"] = item.AlertText
			}
			for key, value := range item.SAME {
				data[key] = value
			}
			if item.AudioURL != "" {
				data["audio_url"] = item.AudioURL
				data["audio_mime_type"] = item.AudioMimeType
				data["audio_language"] = item.AudioLanguage
				data["audio_description"] = item.AudioDescription
				data["audio_authoritative"] = true
			}
			_ = s.bridge.Publish(map[string]any{
				"type":    "cap.alert.broadcast.requested",
				"source":  serviceID,
				"feed_id": item.FeedID,
				"subject": alert.Identifier,
				"data":    data,
			})
		}
	}
}

func capSAMEPayload(alert capingest.Alert, feed feedXML, baseDir string, now time.Time) map[string]any {
	payload := map[string]any{
		"include_same": xmlBool(feed.Playout.SAME, true),
		"cap_source":   detectCAPSource(alert),
	}
	if !xmlBool(feed.Playout.SAME, true) {
		payload["same_suppressed_reason"] = "feed SAME disabled"
		return payload
	}
	info := chooseAlertInfo(alert, feedLanguage(feed))
	if info == nil {
		payload["include_same"] = false
		payload["same_suppressed_reason"] = "no alert info"
		return payload
	}
	if strings.EqualFold(alert.MessageType, "Cancel") {
		payload["include_same"] = false
		payload["same_suppressed_reason"] = "cancellation"
		return payload
	}
	if isCAPEnded(alert, now) {
		payload["include_same"] = false
		payload["same_suppressed_reason"] = "ended"
		return payload
	}
	resolution := sameEventResolutionForCAP(alert, *info, baseDir)
	event := resolution.Event
	locations := sameLocationsForCAP(*info, feed, baseDir)
	if event == "" {
		event = "ADR"
	}
	if len(locations) == 0 {
		locations = []string{"000000"}
	}
	payload["same_event"] = event
	payload["same_event_source"] = resolution.Source
	payload["same_event_reason"] = resolution.Reason
	payload["same_event_confidence"] = resolution.Confidence
	if resolution.AlertClass != "" {
		payload["same_alert_class"] = resolution.AlertClass
	}
	if resolution.Phenomenon != "" {
		payload["same_event_phenomenon"] = resolution.Phenomenon
	}
	if len(resolution.Evidence) > 0 {
		payload["same_event_evidence"] = resolution.Evidence
	}
	originator := sameOriginatorForCAP(alert, *info)
	payload["same_originator"] = originator
	payload["same_originator_name"] = sameOriginatorNameForCAP(alert, *info)
	payload["same_event_name"] = alertSubject(*info)
	if originator == "WXR" {
		payload["same_weather_service"] = sameWeatherServiceForCAP(alert)
	}
	payload["same_locations"] = locations
	payload["same_duration"] = sameDurationForCAP(alert, *info)
	if sent := firstCAPTime(alert.Sent, []capingest.AlertInfo{*info}); !sent.IsZero() {
		payload["same_sent_at"] = sent.Format(time.RFC3339Nano)
	}
	if begins := sameBeginsForCAP(alert, *info); !begins.IsZero() {
		payload["same_begins_at"] = begins.Format(time.RFC3339Nano)
	}
	if expires := parseCAPTime(info.Expires); !expires.IsZero() {
		payload["same_expires_at"] = expires.Format(time.RFC3339Nano)
	}
	payload["same_tone"] = sameToneForCAP(*info, feed)
	return payload
}

func sameAlertFreshForTone(alert capingest.Alert, info capingest.AlertInfo, event string, now time.Time) bool {
	anchor := firstCAPTime(alert.Sent, []capingest.AlertInfo{info})
	if anchor.IsZero() || now.IsZero() || now.Before(anchor) {
		return true
	}
	limit := time.Hour
	switch strings.ToUpper(strings.TrimSpace(event)) {
	case "SVR", "TOR":
		limit = 30 * time.Minute
	}
	return now.Sub(anchor) <= limit
}

func sameEventForCAP(alert capingest.Alert, info capingest.AlertInfo, baseDir string) string {
	return sameEventResolutionForCAP(alert, info, baseDir).Event
}

func sameEventResolutionForCAP(alert capingest.Alert, info capingest.AlertInfo, baseDir string) capsame.EventResolution {
	return capsame.ResolveEvent(alert, info, baseDir)
}

func sameLocationsForCAP(info capingest.AlertInfo, feed feedXML, baseDir string) []string {
	db := loadAlertGeoDB(baseDir)
	coverage := feedCoverageModel(baseDir, feed, nil)
	alertCodes := alertInfoCoverageCodes(info)
	out := []string{}
	addCodes := func(codes []string) {
		for _, clean := range codes {
			if clean != "" {
				out = append(out, clean)
			}
		}
	}

	if len(coverage.Codes) == 0 {
		out = append(out, "000000")
		for code := range alertCodes {
			addCodes(sameLocationCodesForAlertCode(db, code))
		}
	} else {
		for code := range alertCodes {
			locations := sameLocationCodesForAlertCode(db, code)
			if coverageMatchesAlertCode(db, coverage.Codes, code) {
				addCodes(locations)
			}
		}
	}
	if len(out) == 0 && len(coverage.Codes) > 0 {
		for code := range coverage.Codes {
			addCodes(sameLocationCodesForAlertCode(db, code))
		}
	}
	out = uniqueStrings(out)
	sort.Strings(out)
	if len(out) == 0 {
		out = []string{"000000"}
	}
	if len(out) > 31 {
		out = out[:31]
	}
	return out
}

func sameLocationCodesForAlertCode(db alertGeoDB, raw string) []string {
	if clean := sameLocationCode(raw); clean != "" {
		return []string{clean}
	}
	if clean := capCPToCLC(db, raw); clean != "" {
		return []string{clean}
	}
	code := normalizeNWSCode(raw)
	if code == "" {
		return nil
	}
	if item, ok := db.NWS[code]; ok {
		if clean := sameLocationCode(item.FIPS); clean != "" {
			return []string{clean}
		}
	}
	if item, ok := db.FIPS[code]; ok {
		if clean := sameLocationCode(item.FIPS); clean != "" {
			return []string{clean}
		}
	}
	return nil
}

func sameLocationCode(raw string) string {
	value := digitsOnly(raw)
	if len(value) == 5 {
		return "0" + value
	}
	if len(value) != 6 {
		return ""
	}
	return value
}

func capCPToCLC(db alertGeoDB, raw string) string {
	for _, code := range uniqueStrings([]string{strings.TrimSpace(raw), digitsOnly(raw)}) {
		item, ok := db.CAPCP[code]
		if !ok || !validGeoPoint(item.Lat, item.Lon) {
			continue
		}
		if nearest := nearestCLCForCAPCP(db, item, true); nearest != "" {
			return nearest
		}
		if nearest := nearestCLCForCAPCP(db, item, false); nearest != "" {
			return nearest
		}
	}
	return ""
}

func nearestCLCForCAPCP(db alertGeoDB, item capCPGeocode, sameProvinceOnly bool) string {
	province := strings.ToUpper(strings.TrimSpace(item.Province))
	bestCode := ""
	bestDistance := math.MaxFloat64
	for _, zone := range db.CLC {
		if !validGeoPoint(zone.Lat, zone.Lon) {
			continue
		}
		if sameProvinceOnly && province != "" && strings.ToUpper(strings.TrimSpace(zone.Province)) != province {
			continue
		}
		distance := geoDistanceKM(item.Lat, item.Lon, zone.Lat, zone.Lon)
		if distance < bestDistance {
			bestDistance = distance
			bestCode = zone.Code
		}
	}
	return sameLocationCode(bestCode)
}

func geoDistanceKM(lat1 float64, lon1 float64, lat2 float64, lon2 float64) float64 {
	const earthRadiusKM = 6371.0
	rad := func(value float64) float64 { return value * math.Pi / 180 }
	dLat := rad(lat2 - lat1)
	dLon := rad(lon2 - lon1)
	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(rad(lat1))*math.Cos(rad(lat2))*math.Sin(dLon/2)*math.Sin(dLon/2)
	return earthRadiusKM * 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
}

func validGeoPoint(lat float64, lon float64) bool {
	return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180 && (lat != 0 || lon != 0)
}

func digitsOnly(raw string) string {
	digits := strings.Builder{}
	for _, ch := range raw {
		if ch >= '0' && ch <= '9' {
			digits.WriteRune(ch)
		}
	}
	return digits.String()
}

func uniqueStrings(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]struct{}{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func sameOriginatorForCAP(alert capingest.Alert, info capingest.AlertInfo) string {
	if originator := strings.ToUpper(strings.TrimSpace(alertParam(info, "eas-org"))); len(originator) == 3 {
		return originator
	}
	switch detectCAPSource(alert) {
	case "eccc", "nws":
		return "WXR"
	default:
		return "CIV"
	}
}

func sameWeatherServiceForCAP(alert capingest.Alert) string {
	switch detectCAPSource(alert) {
	case "eccc":
		return "Environment Canada"
	case "nws":
		return "The National Weather Service"
	default:
		return ""
	}
}

func sameOriginatorNameForCAP(alert capingest.Alert, info capingest.AlertInfo) string {
	if detectCAPSource(alert) == "nws" {
		return ""
	}
	if name := strings.TrimSpace(info.SenderName); name != "" {
		return name
	}
	return sameWeatherServiceForCAP(alert)
}

func sameBeginsForCAP(alert capingest.Alert, info capingest.AlertInfo) time.Time {
	for _, raw := range []string{info.Onset, info.Effective, alert.Sent} {
		if parsed := parseCAPTime(raw); !parsed.IsZero() {
			return parsed
		}
	}
	return time.Time{}
}

func sameDurationForCAP(alert capingest.Alert, info capingest.AlertInfo) string {
	start := firstCAPTime(alert.Sent, []capingest.AlertInfo{info})
	expires := parseCAPTime(info.Expires)
	if expires.IsZero() {
		return "0100"
	}
	minutes := int(expires.Sub(start).Minutes())
	if start.IsZero() {
		minutes = int(time.Until(expires).Minutes())
	}
	if minutes < 15 {
		minutes = 15
	}
	if minutes > 6*60 {
		minutes = 6 * 60
	}
	return fmt.Sprintf("%02d%02d", minutes/60, minutes%60)
}

func sameToneForCAP(info capingest.AlertInfo, feed feedXML) string {
	if isBroadcastImmediateInfo(info) {
		return "NPAS"
	}
	if tone := strings.ToUpper(strings.TrimSpace(feed.Playout.SAMEAttentionTone)); tone != "" {
		return tone
	}
	return "WXR"
}

type capRegistryUpdate struct {
	FeedID             string
	Path               string
	Renderable         bool
	Broadcast          bool
	Cancelled          bool
	CancelledIDs       []string
	BroadcastImmediate bool
	AlertText          string
	Headline           string
	Event              string
	Severity           string
	Urgency            string
	Certainty          string
	Description        string
	Instruction        string
	BackgroundColor    string
	AudioURL           string
	AudioMimeType      string
	AudioLanguage      string
	AudioDescription   string
	SAME               map[string]any
}

func (s *Service) recordCAPAlert(alert capingest.Alert, now time.Time) ([]capRegistryUpdate, error) {
	alert = normalizeRawCAPAlert(alert)
	if alert.Identifier == "" || strings.TrimSpace(alert.RawXML) == "" {
		return nil, nil
	}

	updates := []capRegistryUpdate{}
	for _, feed := range s.cfg.Feeds {
		if strings.TrimSpace(feed.ID) == "" || !xmlBool(feed.EnabledRaw, true) {
			continue
		}
		if !feedAcceptsCAPSource(feed, alert) {
			record := capArchiveRecord{
				ID:        alert.Identifier,
				FeedID:    feed.ID,
				Status:    "rejected",
				Reason:    "source disabled for feed",
				UpdatedAt: now,
				Alert:     alert,
				RawXML:    alert.RawXML,
			}
			storeCAPArchiveRecord(s.cfg.Store, "rejected", record)
			continue
		}
		if !routineCAPAlertAllowed(alert) {
			record := capArchiveRecord{
				ID:        alert.Identifier,
				FeedID:    feed.ID,
				Status:    "rejected",
				Reason:    "test alert",
				UpdatedAt: now,
				Alert:     alert,
				RawXML:    alert.RawXML,
			}
			storeCAPArchiveRecord(s.cfg.Store, "rejected", record)
			continue
		}
		if !alertMatchesFeed(alert, feed, s.cfg.BaseDir) {
			record := capArchiveRecord{
				ID:        alert.Identifier,
				FeedID:    feed.ID,
				Status:    "rejected",
				Reason:    "outside feed coverage",
				UpdatedAt: now,
				Alert:     alert,
				RawXML:    alert.RawXML,
			}
			storeCAPArchiveRecord(s.cfg.Store, "rejected", record)
			continue
		}
		entries := loadActiveCAPEntries(s.cfg.Store, feed.ID, now)
		archiveExpiredCAPEntries("", feed.ID, entries, now, s.cfg.Store)
		entries = pruneCAPEntries(entries, now)
		entries = removeCAPReferences(entries, alert.References)
		deleteCAPReferences(s.cfg.Store, feed.ID, parseCAPReferences(alert.References))

		if strings.EqualFold(alert.MessageType, "Cancel") {
			deleteCAPReferences(s.cfg.Store, feed.ID, cancelledAlertIDs(alert))
			updates = append(updates, capRegistryUpdate{
				FeedID:       feed.ID,
				Cancelled:    true,
				CancelledIDs: cancelledAlertIDs(alert),
			})
			continue
		}

		nextEntry := capRegistryEntry{
			ID:        alert.Identifier,
			UpdatedAt: capRegistryAnchor(alert, now),
			Alert:     alert,
			RawXML:    alert.RawXML,
		}
		entries = upsertCAPEntry(entries, nextEntry)
		archiveExpiredCAPEntries("", feed.ID, entries, now, s.cfg.Store)
		entries = pruneCAPEntries(entries, now)
		if !isRenderableCAPEntry(nextEntry, now) {
			storeCAPArchiveRecord(s.cfg.Store, "expired", capArchiveRecord{
				ID:        alert.Identifier,
				FeedID:    feed.ID,
				Status:    "expired",
				Reason:    "expired or ended outside relay grace",
				UpdatedAt: now,
				Alert:     alert,
				RawXML:    alert.RawXML,
			})
			deleteCAPReferences(s.cfg.Store, feed.ID, []string{alert.Identifier})
			updates = append(updates, capRegistryUpdate{
				FeedID:       feed.ID,
				Renderable:   hasRenderableCAPEntries(entries, now),
				Broadcast:    false,
				Cancelled:    isCAPEnded(alert, now),
				CancelledIDs: cancelledAlertIDs(alert),
				SAME:         capSAMEPayload(alert, feed, s.cfg.BaseDir, now),
			})
			continue
		}
		storeCAPArchiveRecord(s.cfg.Store, "accepted", capArchiveRecord{
			ID:        alert.Identifier,
			FeedID:    feed.ID,
			Status:    "accepted",
			UpdatedAt: nextEntry.UpdatedAt,
			Alert:     alert,
			RawXML:    alert.RawXML,
		})
		info := chooseAlertInfo(alert, feedLanguage(feed))
		alertText := ""
		headline := ""
		eventName := ""
		severity := ""
		urgency := ""
		certainty := ""
		description := ""
		instruction := ""
		backgroundColor := ""
		if info != nil {
			alertText = renderCAPAlertSentence(capRegistryEntry{
				ID:        alert.Identifier,
				UpdatedAt: now,
				Alert:     alert,
				RawXML:    alert.RawXML,
			}, *info, feed, s.cfg.BaseDir, s.cfg.ForecastNames, now)
			headline = alerttext.NormalizeHeadline(firstNonBlank(info.Headline, info.Event, "Weather Alert"))
			eventName = firstNonBlank(info.Event, alertSubject(*info), "CAP")
			severity = info.Severity
			urgency = info.Urgency
			certainty = info.Certainty
			description = info.Description
			instruction = info.Instruction
			backgroundColor = alerttext.PickBannerColor([]alerttext.AlertVisualInput{{
				Severity:           info.Severity,
				Event:              strings.Join([]string{sameEventForCAP(alert, *info, s.cfg.BaseDir), alertSubject(*info), info.Event, info.Headline}, " "),
				BroadcastImmediate: isBroadcastImmediateInfo(*info),
			}})
		}
		audio := alertBroadcastAudio(alert, feedLanguage(feed))
		broadcastImmediate := info != nil && isBroadcastImmediateInfo(*info)
		updates = append(updates, capRegistryUpdate{
			FeedID:             feed.ID,
			Renderable:         hasRenderableCAPEntries(entries, now),
			Broadcast:          capPriorityBroadcastAllowed(alert, feed, s.cfg.BaseDir, now),
			Cancelled:          isCAPEnded(alert, now),
			CancelledIDs:       cancelledAlertIDs(alert),
			BroadcastImmediate: broadcastImmediate,
			AlertText:          alertText,
			Headline:           headline,
			Event:              eventName,
			Severity:           severity,
			Urgency:            urgency,
			Certainty:          certainty,
			Description:        description,
			Instruction:        instruction,
			BackgroundColor:    backgroundColor,
			AudioURL:           audio.URL,
			AudioMimeType:      audio.MimeType,
			AudioLanguage:      audio.Language,
			AudioDescription:   audio.Description,
			SAME:               capSAMEPayload(alert, feed, s.cfg.BaseDir, now),
		})
	}
	return updates, nil
}

func routineCAPAlertAllowed(alert capingest.Alert) bool {
	switch strings.ToLower(strings.TrimSpace(alert.Status)) {
	case "test", "exercise", "draft":
		return false
	}
	info := chooseAlertInfo(alert, "en-CA")
	if info == nil {
		return true
	}
	text := strings.ToLower(strings.Join([]string{
		info.Event,
		info.Headline,
		alertParam(*info, "layer:EC-MSC-SMC:1.0:Alert_Name"),
	}, " "))
	for _, marker := range []string{
		"test message",
		"practice demo",
		"practice/demo",
		"required weekly test",
		"required monthly test",
	} {
		if strings.Contains(text, marker) {
			return false
		}
	}
	return true
}

func capPriorityBroadcastAllowed(alert capingest.Alert, feed feedXML, baseDir string, now time.Time) bool {
	if isCAPEnded(alert, now) {
		return false
	}
	if !feedAllowsCAPAlert(feed, alert) {
		return false
	}
	info := chooseAlertInfo(alert, feedLanguage(feed))
	if info == nil {
		return false
	}
	if !alertMatchesFeed(alert, feed, baseDir) {
		return false
	}
	if isBroadcastImmediateInfo(*info) {
		return true
	}
	return sameAlertFreshForTone(alert, *info, sameEventForCAP(alert, *info, baseDir), now)
}

type capBroadcastAudio struct {
	URL         string
	MimeType    string
	Language    string
	Description string
}

func alertBroadcastAudio(alert capingest.Alert, preferredLanguage string) capBroadcastAudio {
	for _, preferred := range []bool{true, false} {
		for _, info := range preferredCAPInfos(alert.Infos, preferredLanguage) {
			if preferred && !isBroadcastImmediateInfo(info) && !hasBroadcastAudioResource(info) {
				continue
			}
			for _, resource := range info.Resources {
				if !isAudioResource(resource) {
					continue
				}
				url := strings.TrimSpace(resource.URI)
				if url == "" {
					url = strings.TrimSpace(resource.DerefURI)
				}
				if url == "" {
					continue
				}
				return capBroadcastAudio{
					URL:         url,
					MimeType:    strings.TrimSpace(resource.MimeType),
					Language:    strings.TrimSpace(info.Language),
					Description: strings.TrimSpace(resource.Description),
				}
			}
		}
	}
	return capBroadcastAudio{}
}

func hasBroadcastAudioResource(info capingest.AlertInfo) bool {
	for _, resource := range info.Resources {
		if isAudioResource(resource) {
			return true
		}
	}
	return false
}

func isAudioResource(resource capingest.Resource) bool {
	mimeType := strings.ToLower(strings.TrimSpace(resource.MimeType))
	description := strings.ToLower(strings.TrimSpace(resource.Description))
	uri := strings.ToLower(strings.TrimSpace(resource.URI))
	return strings.HasPrefix(mimeType, "audio/") ||
		strings.Contains(description, "broadcast audio") ||
		strings.Contains(description, "audio") ||
		strings.Contains(uri, ".mp3") ||
		strings.Contains(uri, ".wav") ||
		strings.Contains(uri, ".m4a")
}

func isBroadcastImmediateInfo(info capingest.AlertInfo) bool {
	return alerttext.IsBroadcastImmediateInfo(info)
}

func preferredCAPInfos(infos []capingest.AlertInfo, language string) []capingest.AlertInfo {
	language = strings.ToLower(strings.TrimSpace(language))
	short := language
	if index := strings.Index(short, "-"); index > 0 {
		short = short[:index]
	}
	var exact []capingest.AlertInfo
	var prefix []capingest.AlertInfo
	var rest []capingest.AlertInfo
	for _, info := range infos {
		infoLang := strings.ToLower(strings.TrimSpace(info.Language))
		switch {
		case language != "" && infoLang == language:
			exact = append(exact, info)
		case short != "" && strings.HasPrefix(infoLang, short):
			prefix = append(prefix, info)
		default:
			rest = append(rest, info)
		}
	}
	return append(append(exact, prefix...), rest...)
}

func alertBroadcastTitle(alert capingest.Alert) string {
	for _, info := range alert.Infos {
		if text := firstNonBlank(info.Headline, info.Event); text != "" {
			return titleText(text)
		}
	}
	return "Weather Alert"
}

func capAlertFromEvent(event map[string]any) (capingest.Alert, bool) {
	data := mapAt(event, "data")
	value, ok := data["alert"]
	if !ok {
		value = event["alert"]
	}
	if value == nil {
		return capingest.Alert{}, false
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return capingest.Alert{}, false
	}
	var alert capingest.Alert
	if err := json.Unmarshal(raw, &alert); err != nil {
		return capingest.Alert{}, false
	}
	alert = normalizeRawCAPAlert(alert)
	return alert, alert.Identifier != "" && strings.TrimSpace(alert.RawXML) != ""
}

func normalizeRawCAPAlert(alert capingest.Alert) capingest.Alert {
	raw := strings.TrimSpace(alert.RawXML)
	if raw == "" {
		return alert
	}
	parsed, err := capingest.ParseCAP([]byte(raw))
	if err != nil {
		return alert
	}
	if parsed.RawXML == "" {
		parsed.RawXML = raw
	}
	return parsed
}

func loadActiveCAPEntries(store datastore.Store, feedID string, now time.Time) []capRegistryEntry {
	if store == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	rows, err := store.ListCAPArchives(ctx, "accepted", time.Time{})
	if err != nil {
		return nil
	}
	entries := make([]capRegistryEntry, 0, len(rows))
	for _, row := range rows {
		if strings.TrimSpace(feedID) != "" && row.FeedID != feedID {
			continue
		}
		alert, ok := storedCAPArchiveAlert(row)
		if !ok {
			_ = store.DeleteCAPArchiveBucketItem(ctx, row.AlertID, row.FeedID, "accepted")
			continue
		}
		entry := capRegistryEntry{
			ID:        fallbackText(row.AlertID, alert.Identifier),
			UpdatedAt: firstNonZeroTime(row.UpdatedAt, row.StoredAt, now),
			Alert:     alert,
			RawXML:    row.RawXML,
		}
		if isRenderableCAPEntry(entry, now) {
			entries = append(entries, entry)
			continue
		}
		storeCAPArchiveRecord(store, "expired", capArchiveRecord{
			ID:        fallbackText(row.AlertID, alert.Identifier),
			FeedID:    row.FeedID,
			Status:    "expired",
			Reason:    "expired or ended outside relay grace",
			UpdatedAt: now,
			Alert:     alert,
			RawXML:    row.RawXML,
		})
		_ = store.DeleteCAPArchiveBucketItem(ctx, row.AlertID, row.FeedID, "accepted")
	}
	sort.SliceStable(entries, func(i, j int) bool {
		return entries[i].UpdatedAt.Before(entries[j].UpdatedAt)
	})
	return entries
}

func storedCAPArchiveAlert(row datastore.StoredCAPArchive) (capingest.Alert, bool) {
	rawXML := strings.TrimSpace(row.RawXML)
	if rawXML == "" {
		return capingest.Alert{}, false
	}
	alert, err := capingest.ParseCAP([]byte(rawXML))
	if err != nil || alert.Identifier == "" {
		return capingest.Alert{}, false
	}
	if alert.RawXML == "" {
		alert.RawXML = rawXML
	}
	return alert, true
}

func firstNonZeroTime(values ...time.Time) time.Time {
	for _, value := range values {
		if !value.IsZero() {
			return value
		}
	}
	return time.Time{}
}

func deleteCAPReferences(store datastore.Store, feedID string, ids []string) {
	if store == nil || len(ids) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	for _, id := range ids {
		_ = store.DeleteCAPArchiveBucketItem(ctx, id, feedID, "accepted")
	}
}

func storeCAPArchiveRecord(store datastore.Store, bucket string, record capArchiveRecord) {
	if store == nil {
		return
	}
	rawXML := fallbackText(record.RawXML, record.Alert.RawXML)
	if strings.TrimSpace(rawXML) == "" {
		return
	}
	info := chooseAlertInfo(record.Alert, "en-CA")
	event := ""
	headline := ""
	expires := ""
	if info != nil {
		event = info.Event
		headline = fallbackText(info.Headline, info.Event)
		expires = info.Expires
	}
	updated := ""
	if !record.UpdatedAt.IsZero() {
		updated = record.UpdatedAt.UTC().Format(time.RFC3339Nano)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = store.StoreCAPArchive(ctx, datastore.CAPArchiveRecord{
		AlertID:      fallbackText(record.ID, record.Alert.Identifier),
		FeedID:       record.FeedID,
		Bucket:       bucket,
		Status:       fallbackText(record.Status, bucket),
		Reason:       record.Reason,
		Sender:       record.Alert.Sender,
		Source:       detectCAPSource(record.Alert),
		SentAtRaw:    record.Alert.Sent,
		UpdatedAtRaw: updated,
		ExpiresAtRaw: expires,
		Event:        event,
		Headline:     headline,
		RawXML:       rawXML,
		Metadata: map[string]any{
			"message_type": record.Alert.MessageType,
			"scope":        record.Alert.Scope,
		},
	})
}

func archiveExpiredCAPEntries(_ string, feedID string, entries []capRegistryEntry, now time.Time, store datastore.Store) {
	for _, entry := range entries {
		if isRenderableCAPEntry(entry, now) {
			continue
		}
		if entry.Alert.Identifier == "" {
			continue
		}
		record := capArchiveRecord{
			ID:        fallbackText(entry.ID, entry.Alert.Identifier),
			FeedID:    feedID,
			Status:    "expired",
			Reason:    "expired or ended outside relay grace",
			UpdatedAt: now,
			Alert:     entry.Alert,
			RawXML:    fallbackText(entry.RawXML, entry.Alert.RawXML),
		}
		storeCAPArchiveRecord(store, "expired", record)
	}
}

func upsertCAPEntry(entries []capRegistryEntry, next capRegistryEntry) []capRegistryEntry {
	for i := range entries {
		if entries[i].ID == next.ID || strings.EqualFold(entries[i].Alert.Identifier, next.Alert.Identifier) {
			if !entries[i].UpdatedAt.IsZero() {
				next.UpdatedAt = entries[i].UpdatedAt
			}
			entries[i] = next
			return entries
		}
	}
	return append(entries, next)
}

func removeCAPReferences(entries []capRegistryEntry, references string) []capRegistryEntry {
	ids := parseCAPReferences(references)
	if len(ids) == 0 {
		return entries
	}
	remove := map[string]struct{}{}
	for _, id := range ids {
		remove[id] = struct{}{}
	}
	out := entries[:0]
	for _, entry := range entries {
		if _, ok := remove[entry.ID]; ok {
			continue
		}
		if _, ok := remove[entry.Alert.Identifier]; ok {
			continue
		}
		out = append(out, entry)
	}
	return out
}

func parseCAPReferences(references string) []string {
	fields := strings.Fields(strings.TrimSpace(references))
	ids := make([]string, 0, len(fields))
	for _, field := range fields {
		parts := strings.Split(field, ",")
		if len(parts) >= 2 {
			if id := strings.TrimSpace(parts[1]); id != "" {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

func cancelledAlertIDs(alert capingest.Alert) []string {
	ids := append([]string{}, parseCAPReferences(alert.References)...)
	if strings.TrimSpace(alert.Identifier) != "" {
		ids = append(ids, strings.TrimSpace(alert.Identifier))
	}
	return uniqueStrings(ids)
}

func pruneCAPEntries(entries []capRegistryEntry, now time.Time) []capRegistryEntry {
	out := entries[:0]
	for _, entry := range entries {
		if isRenderableCAPEntry(entry, now) {
			out = append(out, entry)
		}
	}
	return out
}

func hasRenderableCAPEntries(entries []capRegistryEntry, now time.Time) bool {
	for _, entry := range entries {
		if isRenderableCAPEntry(entry, now) {
			return true
		}
	}
	return false
}

func isRenderableCAPEntry(entry capRegistryEntry, now time.Time) bool {
	alert := entry.Alert
	if alert.Identifier == "" || strings.EqualFold(alert.MessageType, "Cancel") {
		return false
	}
	if isExplicitCAPEnd(alert) {
		anchor := capRegistryAnchor(alert, entry.UpdatedAt)
		if anchor.IsZero() {
			return true
		}
		return now.Before(anchor.Add(alertRegistryGrace))
	}
	if expires := alertExpiresAt(alert); !expires.IsZero() {
		return now.Before(expires.Add(alertRegistryGrace))
	}
	return true
}

func capRegistryAnchor(alert capingest.Alert, fallback time.Time) time.Time {
	if isExplicitCAPEnd(alert) {
		if anchor := firstCAPTime(alert.Sent, alert.Infos); !anchor.IsZero() {
			return anchor
		}
	}
	return fallback
}

func isExplicitCAPEnd(alert capingest.Alert) bool {
	if strings.EqualFold(alert.MessageType, "Cancel") {
		return true
	}
	for _, info := range alert.Infos {
		for _, response := range info.Response {
			if strings.EqualFold(response, "AllClear") {
				return true
			}
		}
		status := strings.ToLower(alertParam(info, "layer:EC-MSC-SMC:1.0:Alert_Location_Status"))
		if status == "" {
			status = strings.ToLower(alertParam(info, "layer:EC-MSC-SMC:1.1:Alert_Location_Status"))
		}
		if status == "ended" || strings.Contains(strings.ToLower(info.Headline), "ended") {
			return true
		}
	}
	return false
}

func feedAcceptsCAPSource(feed feedXML, alert capingest.Alert) bool {
	source, sourceConfig := feedCAPSourceConfig(feed, alert)
	if source == "generic" {
		return xmlBool(feed.Alerts.CapCP.EnabledRaw, true) || xmlBool(feed.Alerts.NWSCAP.EnabledRaw, false)
	}
	return xmlBool(sourceConfig.EnabledRaw, source == "eccc")
}

func feedAllowsCAPAlert(feed feedXML, alert capingest.Alert) bool {
	_, sourceConfig := feedCAPSourceConfig(feed, alert)
	return alertFilterAllows(sourceConfig.Filter, alert, feedLanguage(feed))
}

func feedUsesAlertCoverage(feed feedXML, alert capingest.Alert) bool {
	_, sourceConfig := feedCAPSourceConfig(feed, alert)
	return xmlBool(sourceConfig.Filter.UseFeedLocations, true)
}

func feedCAPSourceConfig(feed feedXML, alert capingest.Alert) (string, feedAlertSourceXML) {
	source := detectCAPSource(alert)
	switch source {
	case "nws":
		return source, feed.Alerts.NWSCAP
	default:
		return source, feed.Alerts.CapCP
	}
}

func alertFilterAllows(filter alertFilterXML, alert capingest.Alert, language string) bool {
	info := chooseAlertInfo(alert, language)
	if info == nil {
		return true
	}
	if alertFilterListMatches(filter.Blocklist, alert, *info) {
		return false
	}
	if alertFilterListEmpty(filter.Allowlist) {
		return true
	}
	return alertFilterListAllows(filter.Allowlist, alert, *info)
}

func alertFilterListAllows(list alertFilterListXML, alert capingest.Alert, info capingest.AlertInfo) bool {
	if len(list.Severities) > 0 && !textInList(info.Severity, list.Severities) {
		return false
	}
	if len(list.Urgencies) > 0 && !textInList(info.Urgency, list.Urgencies) {
		return false
	}
	if len(list.Certainties) > 0 && !textInList(info.Certainty, list.Certainties) {
		return false
	}
	if len(list.MessageTypes) > 0 && !textInList(alert.MessageType, list.MessageTypes) {
		return false
	}
	if len(list.Events) > 0 && !alertEventMatches(info, list.Events) {
		return false
	}
	if len(list.NAADSEvents) > 0 && !alertEventMatches(info, list.NAADSEvents) {
		return false
	}
	for _, other := range list.Others {
		if !alertOtherFilterMatches(info, other) {
			return false
		}
	}
	return true
}

func alertFilterListMatches(list alertFilterListXML, alert capingest.Alert, info capingest.AlertInfo) bool {
	return (len(list.Severities) > 0 && textInList(info.Severity, list.Severities)) ||
		(len(list.Urgencies) > 0 && textInList(info.Urgency, list.Urgencies)) ||
		(len(list.Certainties) > 0 && textInList(info.Certainty, list.Certainties)) ||
		(len(list.MessageTypes) > 0 && textInList(alert.MessageType, list.MessageTypes)) ||
		(len(list.Events) > 0 && alertEventMatches(info, list.Events)) ||
		(len(list.NAADSEvents) > 0 && alertEventMatches(info, list.NAADSEvents)) ||
		alertOtherFiltersMatch(info, list.Others)
}

func alertFilterListEmpty(list alertFilterListXML) bool {
	return len(list.Severities) == 0 &&
		len(list.Urgencies) == 0 &&
		len(list.Certainties) == 0 &&
		len(list.MessageTypes) == 0 &&
		len(list.Events) == 0 &&
		len(list.NAADSEvents) == 0 &&
		len(list.Others) == 0
}

func alertEventMatches(info capingest.AlertInfo, wanted []string) bool {
	if textInList(info.Event, wanted) || textInList(info.Headline, wanted) {
		return true
	}
	for _, code := range info.EventCodes {
		if textInList(code.Value, wanted) {
			return true
		}
	}
	return false
}

func alertOtherFiltersMatch(info capingest.AlertInfo, filters []alertFilterOtherXML) bool {
	for _, filter := range filters {
		if alertOtherFilterMatches(info, filter) {
			return true
		}
	}
	return false
}

func alertOtherFilterMatches(info capingest.AlertInfo, filter alertFilterOtherXML) bool {
	name := strings.TrimSpace(filter.ValueName)
	value := strings.TrimSpace(filter.Value)
	if name == "" || value == "" {
		return false
	}
	for _, param := range info.Parameters {
		if strings.EqualFold(strings.TrimSpace(param.Name), name) && strings.EqualFold(strings.TrimSpace(param.Value), value) {
			return true
		}
	}
	return false
}

func textInList(value string, list []string) bool {
	value = strings.TrimSpace(value)
	for _, item := range list {
		if strings.EqualFold(value, strings.TrimSpace(item)) {
			return true
		}
	}
	return false
}

func detectCAPSource(alert capingest.Alert) string {
	for _, info := range alert.Infos {
		for _, param := range info.Parameters {
			if strings.HasPrefix(param.Name, "layer:EC-MSC-SMC") || strings.Contains(strings.ToLower(param.Name), "cap-cp") {
				return "eccc"
			}
		}
	}
	sender := strings.ToLower(alert.Sender)
	if strings.Contains(sender, "canada") || strings.Contains(sender, "cap-pac") {
		return "eccc"
	}
	if strings.Contains(sender, "weather.gov") || strings.Contains(sender, "nws") || strings.Contains(sender, "noaa") {
		return "nws"
	}
	return "generic"
}

type coverageModel struct {
	Codes   map[string]struct{}
	Regions []coverageRegion
}

type coverageRegion struct {
	ID                 string
	Source             string
	Name               string
	Subregions         map[string]struct{}
	RequiredSubregions map[string]struct{}
}

type alertGeoDB struct {
	CLC   map[string]clcBaseZone
	CAPCP map[string]capCPGeocode
	NWS   map[string]nwsZone
	FIPS  map[string]nwsZone
}

type clcBaseZone struct {
	Code     string
	En       string
	Fr       string
	Lat      float64
	Lon      float64
	Province string
}

type capCPGeocode struct {
	Code     string
	En       string
	Fr       string
	Lat      float64
	Lon      float64
	Province string
}

type nwsZone struct {
	Code       string
	Name       string
	CountyName string
	FIPS       string
}

var alertGeoCache = struct {
	sync.Mutex
	byBase map[string]alertGeoDB
}{byBase: map[string]alertGeoDB{}}

func alertMatchesFeed(alert capingest.Alert, feed feedXML, baseDir string) bool {
	if !feedUsesAlertCoverage(feed, alert) {
		return true
	}
	coverage := feedCoverageModel(baseDir, feed, nil)
	if len(coverage.Codes) == 0 {
		return true
	}
	db := loadAlertGeoDB(baseDir)
	for _, code := range alertCoverageCodes(alert) {
		if coverageMatchesAlertCode(db, coverage.Codes, code) {
			return true
		}
	}
	return false
}

func feedCoverageModel(baseDir string, feed feedXML, forecastNames map[string]forecastRegionName) coverageModel {
	db := loadAlertGeoDB(baseDir)
	model := coverageModel{Codes: map[string]struct{}{}}
	for _, region := range feed.Locations.Coverage.Regions {
		regionID := strings.TrimSpace(region.ID)
		if regionID == "" {
			continue
		}
		source := strings.ToLower(strings.TrimSpace(region.Source))
		item := coverageRegion{
			ID:                 regionID,
			Source:             source,
			Name:               coverageRegionDisplayName(region, forecastNames, feedLanguage(feed), fallbackText(region.Name, alertRegionName(db, regionID, feedLanguage(feed)))),
			Subregions:         map[string]struct{}{},
			RequiredSubregions: map[string]struct{}{},
		}
		addCoverageCode(model.Codes, regionID)
		addCoverageCode(item.Subregions, regionID)
		for _, subregion := range expandAlertRegion(db, regionID, source) {
			addCoverageCode(model.Codes, subregion)
			addCoverageCode(item.Subregions, subregion)
			addCoverageCode(item.RequiredSubregions, subregion)
		}
		for _, subregion := range region.Subregions {
			addCoverageCode(model.Codes, subregion.ID)
			addCoverageCode(item.Subregions, subregion.ID)
			addCoverageCode(item.RequiredSubregions, subregion.ID)
		}
		if strings.HasSuffix(regionID, "00") && len(regionID) >= 4 {
			addCoverageCode(model.Codes, regionID[:4]+"*")
			addCoverageCode(item.Subregions, regionID[:4]+"*")
		}
		model.Regions = append(model.Regions, item)
	}
	return model
}

func coverageRegionDisplayName(region coverageRegionXML, forecastNames map[string]forecastRegionName, lang string, fallback string) string {
	if strings.EqualFold(strings.TrimSpace(region.Source), "eccc") && forecastNames != nil {
		if names, ok := forecastNames[forecastRegionBaseCode(region.ID)]; ok {
			langShort := strings.ToLower(strings.TrimSpace(lang))
			if len(langShort) > 2 {
				langShort = langShort[:2]
			}
			if langShort == "fr" && names.French != "" {
				return pauseForecastRegionName(names.French, "fr")
			}
			if names.English != "" {
				return pauseForecastRegionName(names.English, "en")
			}
		}
	}
	return fallback
}

func loadAlertGeoDB(baseDir string) alertGeoDB {
	key := filepath.Clean(baseDir)
	alertGeoCache.Lock()
	if db, ok := alertGeoCache.byBase[key]; ok {
		alertGeoCache.Unlock()
		return db
	}
	alertGeoCache.Unlock()

	db := alertGeoDB{
		CLC:   loadCLCBaseZones(filepath.Join(baseDir, "managed", "csv", "CLC_Base_Zone.csv")),
		CAPCP: loadCAPCPGeocodes(filepath.Join(baseDir, "managed", "csv", "CAP-CP_Geocodes.csv")),
		NWS:   loadNWSZones(filepath.Join(baseDir, "managed", "csv", "NWS_ZONE_COUNTY_CORRELATION.csv")),
		FIPS:  loadNWSFIPS(filepath.Join(baseDir, "managed", "csv", "NWS_ZONE_COUNTY_CORRELATION.csv")),
	}
	alertGeoCache.Lock()
	alertGeoCache.byBase[key] = db
	alertGeoCache.Unlock()
	return db
}

func loadCLCBaseZones(path string) map[string]clcBaseZone {
	rows := readCSVRows(path, ',')
	out := map[string]clcBaseZone{}
	for _, row := range rows {
		if len(row) < 4 || strings.EqualFold(row[0], "CLC") {
			continue
		}
		code := strings.TrimSpace(row[0])
		if code == "" {
			continue
		}
		if _, exists := out[code]; exists {
			continue
		}
		out[code] = clcBaseZone{
			Code:     code,
			En:       strings.TrimSpace(row[2]),
			Fr:       strings.TrimSpace(row[3]),
			Lat:      csvFloat(row, 6),
			Lon:      csvFloat(row, 7),
			Province: csvString(row, 11),
		}
	}
	return out
}

func loadCAPCPGeocodes(path string) map[string]capCPGeocode {
	rows := readCSVRows(path, ',')
	out := map[string]capCPGeocode{}
	for _, row := range rows {
		if len(row) < 4 || strings.EqualFold(row[2], "CAPCPGCODE") {
			continue
		}
		code := strings.TrimSpace(row[2])
		if code == "" {
			continue
		}
		if _, exists := out[code]; exists {
			continue
		}
		out[code] = capCPGeocode{
			Code:     code,
			En:       strings.TrimSpace(row[0]),
			Fr:       strings.TrimSpace(row[1]),
			Lat:      csvFloat(row, 3),
			Lon:      csvFloat(row, 4),
			Province: csvString(row, 6),
		}
	}
	return out
}

func loadNWSZones(path string) map[string]nwsZone {
	rows := readCSVRows(path, '|')
	out := map[string]nwsZone{}
	for _, row := range rows {
		if len(row) < 7 || strings.EqualFold(row[0], "STATE") {
			continue
		}
		stateZone := strings.TrimSpace(row[4])
		if stateZone == "" {
			stateZone = strings.TrimSpace(row[0]) + strings.TrimSpace(row[1])
		}
		item := nwsZone{
			Code:       strings.ToUpper(stateZone),
			Name:       strings.TrimSpace(row[3]),
			CountyName: strings.TrimSpace(row[5]),
			FIPS:       strings.TrimSpace(row[6]),
		}
		out[item.Code] = item
		if len(item.Code) == 5 {
			out[item.Code[:2]+"Z"+item.Code[2:]] = item
			out[item.Code[:2]+"C"+item.Code[2:]] = item
		}
	}
	return out
}

func loadNWSFIPS(path string) map[string]nwsZone {
	rows := readCSVRows(path, '|')
	out := map[string]nwsZone{}
	for _, row := range rows {
		if len(row) < 7 || strings.EqualFold(row[0], "STATE") {
			continue
		}
		fips := strings.TrimSpace(row[6])
		if fips == "" {
			continue
		}
		if _, exists := out[fips]; exists {
			continue
		}
		item := nwsZone{
			Code:       strings.ToUpper(strings.TrimSpace(row[4])),
			Name:       strings.TrimSpace(row[3]),
			CountyName: strings.TrimSpace(row[5]),
			FIPS:       fips,
		}
		out[fips] = item
		if clean := sameLocationCode(fips); clean != "" {
			out[clean] = item
		}
	}
	return out
}

func readCSVRows(path string, comma rune) [][]string {
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.Comma = comma
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil {
		return nil
	}
	return rows
}

func csvString(row []string, index int) string {
	if index < 0 || index >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[index])
}

func csvFloat(row []string, index int) float64 {
	value := csvString(row, index)
	if value == "" {
		return 0
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0
	}
	return parsed
}

func expandAlertRegion(db alertGeoDB, regionID string, source string) []string {
	if strings.EqualFold(source, "nws") {
		return expandNWSRegion(db, regionID)
	}
	if len(regionID) < 4 || !strings.HasSuffix(regionID, "00") {
		return nil
	}
	prefix := regionID[:4]
	subregions := []string{}
	for code := range db.CLC {
		if code != regionID && strings.HasPrefix(code, prefix) {
			subregions = append(subregions, code)
		}
	}
	sort.Strings(subregions)
	return subregions
}

func expandNWSRegion(db alertGeoDB, regionID string) []string {
	code := normalizeNWSCode(regionID)
	if code == "" {
		return nil
	}
	if item, ok := db.NWS[code]; ok {
		return uniqueStrings([]string{item.Code, item.FIPS, sameLocationCode(item.FIPS)})
	}
	if item, ok := db.FIPS[code]; ok {
		return uniqueStrings([]string{item.Code, item.FIPS, sameLocationCode(item.FIPS)})
	}
	return nil
}

func normalizeNWSCode(raw string) string {
	code := strings.ToUpper(strings.TrimSpace(raw))
	if len(code) == 6 && (code[2] == 'Z' || code[2] == 'C') {
		return code[:2] + code[3:]
	}
	return code
}

func alertRegionName(db alertGeoDB, code string, lang string) string {
	code = strings.TrimSpace(code)
	if item, ok := db.CLC[code]; ok {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(lang)), "fr") && strings.TrimSpace(item.Fr) != "" {
			return strings.TrimSpace(item.Fr)
		}
		return strings.TrimSpace(item.En)
	}
	if item, ok := db.CAPCP[code]; ok {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(lang)), "fr") && strings.TrimSpace(item.Fr) != "" {
			return strings.TrimSpace(item.Fr)
		}
		return strings.TrimSpace(item.En)
	}
	if item, ok := db.NWS[strings.ToUpper(code)]; ok {
		return fallbackText(item.Name, item.CountyName)
	}
	if item, ok := db.FIPS[code]; ok {
		return fallbackText(item.CountyName, item.Name)
	}
	return ""
}

func alertCodeName(db alertGeoDB, code string, lang string) string {
	name := alertRegionName(db, code, lang)
	if name != "" {
		return name
	}
	return strings.TrimSpace(code)
}

func alertAreaNameFromGeocodes(db alertGeoDB, area capingest.AlertArea, lang string) string {
	if desc := cleanAreaName(area.Description); desc != "" {
		return desc
	}
	for _, geocode := range area.Geocodes {
		if name := alertCodeName(db, geocode.Value, lang); name != "" {
			return cleanAreaName(name)
		}
	}
	return ""
}

func alertCoverageCodes(alert capingest.Alert) []string {
	seen := map[string]struct{}{}
	add := func(raw string) {
		for _, part := range strings.Split(raw, ",") {
			value := strings.TrimSpace(part)
			if value == "" {
				continue
			}
			seen[value] = struct{}{}
		}
	}
	for _, info := range alert.Infos {
		for _, param := range info.Parameters {
			name := strings.ToLower(param.Name)
			if strings.Contains(name, "newly_active_areas") || strings.Contains(name, "clc") || strings.Contains(name, "location") {
				add(param.Value)
			}
		}
		for _, area := range info.Areas {
			for _, geocode := range area.Geocodes {
				add(geocode.Value)
			}
		}
	}
	out := make([]string, 0, len(seen))
	for value := range seen {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func coverageCodeMatches(coverage map[string]struct{}, raw string) bool {
	code := strings.TrimSpace(raw)
	if code == "" {
		return false
	}
	candidates := uniqueStrings([]string{code, strings.ToUpper(code), normalizeNWSCode(code), sameLocationCode(code)})
	for _, candidate := range candidates {
		if _, ok := coverage[candidate]; ok {
			return true
		}
	}
	for feedCode := range coverage {
		for _, candidate := range candidates {
			if strings.HasSuffix(feedCode, "*") && strings.HasPrefix(candidate, strings.TrimSuffix(feedCode, "*")) {
				return true
			}
		}
	}
	return false
}

func coverageMatchesAlertCode(db alertGeoDB, coverage map[string]struct{}, raw string) bool {
	if coverageCodeMatches(coverage, raw) {
		return true
	}
	for _, code := range sameLocationCodesForAlertCode(db, raw) {
		if coverageCodeMatches(coverage, code) {
			return true
		}
	}
	return false
}

func (r renderer) alertsProduct(base Product, feed feedXML) (Product, error) {
	now := time.Now().UTC()
	entries := loadActiveCAPEntries(r.cfg.Store, feed.ID, now)
	pruned := pruneCAPEntries(entries, now)

	base.Title = "Weather Alerts"
	base.Inputs = append(base.Inputs, InputRef{Type: "store", ID: "archive.cap_xml/accepted/" + feed.ID})
	for _, entry := range pruned {
		if !routineCAPAlertAllowed(entry.Alert) || !alertMatchesFeed(entry.Alert, feed, r.cfg.BaseDir) {
			continue
		}
		info := chooseAlertInfo(entry.Alert, feedLanguage(feed))
		if info == nil {
			continue
		}
		text := renderCAPAlertSentence(entry, *info, feed, r.cfg.BaseDir, r.cfg.ForecastNames, now)
		if text == "" {
			continue
		}
		base.Segments = append(base.Segments, Segment{
			Kind:  "package",
			Label: alertSubject(*info),
			Text:  text,
		})
	}
	if len(base.Segments) == 0 {
		base.Segments = []Segment{{Kind: "package", Label: "clear", Text: "There are no weather alerts currently in effect."}}
	}
	return base, nil
}

func renderCAPAlertSentence(entry capRegistryEntry, info capingest.AlertInfo, feed feedXML, baseDir string, forecastNames map[string]forecastRegionName, now time.Time) string {
	alert := entry.Alert
	sender := fallbackText(info.SenderName, alertSenderName(alert))
	subject := alertSubject(info)
	source := detectCAPSource(alert)
	areas := alertAreas(info, feed, baseDir, forecastNames)
	if areas == "" {
		areas = fallbackText(alertParam(info, "layer:EC-MSC-SMC:1.0:Alert_Coverage"), "the listening area")
	}
	if areas == "the listening area" && source == "nws" {
		for _, area := range info.Areas {
			if strings.TrimSpace(area.Description) != "" {
				areas = strings.TrimSpace(area.Description)
				break
			}
		}
	}
	return alerttext.BuildCAPAlertText(alerttext.CAPMessageRequest{
		Alert:     alert,
		Info:      info,
		AreaText:  areas,
		Sender:    sender,
		EventName: subject,
		Timezone:  feed.Timezone,
		Now:       now,
		UpdatedAt: entry.UpdatedAt,
	})
}

func chooseAlertInfo(alert capingest.Alert, lang string) *capingest.AlertInfo {
	if len(alert.Infos) == 0 {
		return nil
	}
	lang = strings.ToLower(strings.TrimSpace(lang))
	short := lang
	if idx := strings.Index(short, "-"); idx >= 0 {
		short = short[:idx]
	}
	for i := range alert.Infos {
		infoLang := strings.ToLower(strings.TrimSpace(alert.Infos[i].Language))
		if infoLang == lang || infoLang == short || (short == "en" && strings.HasPrefix(infoLang, "en")) {
			return &alert.Infos[i]
		}
	}
	for i := range alert.Infos {
		infoLang := strings.ToLower(strings.TrimSpace(alert.Infos[i].Language))
		if strings.HasPrefix(infoLang, "en") {
			return &alert.Infos[i]
		}
	}
	return &alert.Infos[0]
}

func alertSenderName(alert capingest.Alert) string {
	if detectCAPSource(alert) == "eccc" {
		return "Environment Canada"
	}
	return fallbackText(alert.Sender, "The alerting authority")
}

func alertSubject(info capingest.AlertInfo) string {
	name := alertParam(info, "layer:EC-MSC-SMC:1.0:Alert_Name")
	if name == "" {
		name = stripAlertHeadlineState(info.Headline)
	}
	if name == "" {
		name = info.Event
	}
	name = strings.ReplaceAll(name, "_", " ")
	parts := strings.Split(name, " - ")
	for i := range parts {
		parts[i] = titleWords(parts[i])
	}
	return strings.Join(parts, " - ")
}

func stripAlertHeadlineState(headline string) string {
	value := strings.TrimSpace(headline)
	for _, suffix := range []string{" - in effect", " - ended", " - updated", " - cancelled", " - canceled"} {
		if strings.HasSuffix(strings.ToLower(value), suffix) {
			return strings.TrimSpace(value[:len(value)-len(suffix)])
		}
	}
	return value
}

func alertAreas(info capingest.AlertInfo, feed feedXML, baseDir string, forecastNames map[string]forecastRegionName) string {
	coverage := feedCoverageModel(baseDir, feed, forecastNames)
	db := loadAlertGeoDB(baseDir)
	if useBroadAlertRegions(info) {
		if broad := broadAlertAreaPhrase(info, coverage, db); broad != "" {
			return broad
		}
	}
	areas := []string{}
	seen := map[string]struct{}{}
	for _, area := range info.Areas {
		if len(coverage.Codes) > 0 {
			matched := false
			for _, geocode := range area.Geocodes {
				if coverageMatchesAlertCode(db, coverage.Codes, geocode.Value) {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		desc := alertAreaNameFromGeocodes(db, area, feedLanguage(feed))
		if desc == "" {
			continue
		}
		if _, ok := seen[desc]; ok {
			continue
		}
		seen[desc] = struct{}{}
		areas = append(areas, desc)
	}
	if len(areas) == 0 {
		return ""
	}
	return strings.Join(areas, "; ")
}

func useBroadAlertRegions(info capingest.AlertInfo) bool {
	haystack := strings.ToLower(strings.Join([]string{
		alertSubject(info),
		info.Event,
		info.Headline,
		alertParam(info, "layer:EC-MSC-SMC:1.0:Alert_Type"),
	}, " "))
	if strings.Contains(haystack, "watch") {
		return true
	}
	for _, hyperlocal := range []string{"tornado", "severe thunderstorm"} {
		if strings.Contains(haystack, hyperlocal) {
			return false
		}
	}
	for _, broad := range []string{
		"advisory",
		"statement",
		"snowfall",
		"winter",
		"blizzard",
		"freezing rain",
		"snow squall",
		"rainfall",
		"wind",
		"extreme cold",
		"heat",
		"frost",
		"fog",
	} {
		if strings.Contains(haystack, broad) {
			return true
		}
	}
	return false
}

func broadAlertAreaPhrase(info capingest.AlertInfo, coverage coverageModel, db alertGeoDB) string {
	if len(coverage.Regions) == 0 {
		return ""
	}
	alertCodes := alertInfoCoverageCodes(info)
	if len(alertCodes) == 0 {
		return ""
	}
	names := []string{}
	seen := map[string]struct{}{}
	for _, region := range coverage.Regions {
		if !coverageRegionMatchesAlert(region, alertCodes, db) {
			continue
		}
		name := broadRegionDisplayName(region.Name)
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}
	if len(names) == 0 {
		return ""
	}
	return "areas within " + joinBroadRegionNames(names)
}

func alertInfoCoverageCodes(info capingest.AlertInfo) map[string]struct{} {
	codes := map[string]struct{}{}
	add := func(raw string) {
		for _, part := range strings.Split(raw, ",") {
			value := strings.TrimSpace(part)
			if value != "" {
				codes[value] = struct{}{}
			}
		}
	}
	for _, param := range info.Parameters {
		name := strings.ToLower(param.Name)
		if strings.Contains(name, "newly_active_areas") || strings.Contains(name, "clc") || strings.Contains(name, "location") {
			add(param.Value)
		}
	}
	for _, area := range info.Areas {
		for _, geocode := range area.Geocodes {
			add(geocode.Value)
		}
	}
	return codes
}

func coverageRegionMatchesAlert(region coverageRegion, alertCodes map[string]struct{}, db alertGeoDB) bool {
	if len(alertCodes) == 0 {
		return false
	}
	parent := map[string]struct{}{}
	addCoverageCode(parent, region.ID)
	for code := range alertCodes {
		if coverageMatchesAlertCode(db, parent, code) {
			return true
		}
	}
	required := region.RequiredSubregions
	if len(required) == 0 {
		return false
	}
	for requiredCode := range required {
		if !alertCodesCoverRegionCode(alertCodes, requiredCode, db) {
			return false
		}
	}
	return true
}

func alertCodesCoverRegionCode(alertCodes map[string]struct{}, regionCode string, db alertGeoDB) bool {
	required := map[string]struct{}{}
	addCoverageCode(required, regionCode)
	for code := range alertCodes {
		if coverageMatchesAlertCode(db, required, code) {
			return true
		}
	}
	return false
}

func broadRegionDisplayName(raw string) string {
	name := strings.TrimSpace(raw)
	if name == "" {
		return ""
	}
	if strings.Contains(strings.ToLower(name), "region") || strings.HasPrefix(strings.ToLower(name), "city of ") {
		return name
	}
	parts := strings.Split(name, " - ")
	if len(parts) > 1 {
		return strings.Join(parts[:len(parts)-1], " - ") + " and " + parts[len(parts)-1] + " region"
	}
	return name + " region"
}

func joinBroadRegionNames(names []string) string {
	if len(names) == 0 {
		return ""
	}
	if len(names) == 1 {
		return "the " + names[0]
	}
	parts := make([]string, 0, len(names))
	for i, name := range names {
		prefix := "the "
		if i == len(names)-1 {
			prefix = "and the "
		}
		parts = append(parts, prefix+name)
	}
	return strings.Join(parts, "; ")
}

func cleanAreaName(raw string) string {
	value := strings.TrimSpace(raw)
	value = strings.ReplaceAll(value, "R.M ", "R.M. ")
	value = strings.ReplaceAll(value, "R.M of", "R.M. of")
	value = strings.ReplaceAll(value, "R.M. of ", "R.M. of ")
	return value
}

func alertParam(info capingest.AlertInfo, name string) string {
	for _, param := range info.Parameters {
		if strings.EqualFold(strings.TrimSpace(param.Name), name) {
			return strings.TrimSpace(param.Value)
		}
	}
	return ""
}

func isCAPEnded(alert capingest.Alert, now time.Time) bool {
	if isExplicitCAPEnd(alert) {
		return true
	}
	if expires := alertExpiresAt(alert); !expires.IsZero() && now.After(expires) {
		return true
	}
	return false
}

func alertExpiresAt(alert capingest.Alert) time.Time {
	for _, info := range alert.Infos {
		if expires := parseCAPTime(info.Expires); !expires.IsZero() {
			return expires
		}
	}
	return time.Time{}
}

func firstCAPTime(sent string, infos []capingest.AlertInfo) time.Time {
	for _, raw := range []string{sent} {
		if parsed := parseCAPTime(raw); !parsed.IsZero() {
			return parsed
		}
	}
	for _, info := range infos {
		for _, raw := range []string{info.Effective, info.Onset, info.Expires} {
			if parsed := parseCAPTime(raw); !parsed.IsZero() {
				return parsed
			}
		}
	}
	return time.Time{}
}

func parseCAPTime(raw string) time.Time {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}
	}
	layouts := []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05.000Z", "2006-01-02T15:04:05Z07:00"}
	for _, layout := range layouts {
		if parsed, err := time.Parse(layout, raw); err == nil {
			return parsed.UTC()
		}
	}
	return time.Time{}
}

func cleanAlertText(raw string) string {
	value := strings.TrimSpace(raw)
	if value == "" {
		return ""
	}
	value = strings.ReplaceAll(value, "\r\n", "\n")
	value = strings.ReplaceAll(value, "\r", "\n")
	value = strings.ReplaceAll(value, "###", " ")
	cutMarkers := []string{
		"Please continue to monitor alerts and forecasts issued by Environment Canada.",
		"To report severe weather",
	}
	for _, marker := range cutMarkers {
		if idx := strings.Index(value, marker); idx >= 0 {
			value = strings.TrimSpace(value[:idx])
		}
	}
	fields := strings.Fields(value)
	if len(fields) > 120 {
		fields = fields[:120]
	}
	return sentence(strings.Join(fields, " "))
}

func titleWords(raw string) string {
	words := strings.Fields(strings.ToLower(strings.TrimSpace(raw)))
	for i := range words {
		words[i] = titleWord(words[i])
	}
	return strings.Join(words, " ")
}

func titleWord(raw string) string {
	if raw == "" {
		return ""
	}
	runes := []rune(raw)
	for i, ch := range runes {
		if unicode.IsLetter(ch) {
			runes[i] = unicode.ToUpper(ch)
			break
		}
	}
	return string(runes)
}
