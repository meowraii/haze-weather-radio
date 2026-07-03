package productrender

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alertmodel"
	"github.com/meowraii/haze-weather-radio/services/go/internal/alerttext"
	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
	"github.com/meowraii/haze-weather-radio/services/go/internal/capsame"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
	"github.com/meowraii/haze-weather-radio/services/go/internal/locationdb"
)

const alertRegistryGrace = 10 * time.Minute

type capRegistryEntry struct {
	ID        string
	UpdatedAt time.Time
	Alert     capmodel.Alert
	RawXML    string
}

type capArchiveRecord struct {
	ID        string         `json:"id"`
	FeedID    string         `json:"feed_id,omitempty"`
	Status    string         `json:"status"`
	Reason    string         `json:"reason,omitempty"`
	UpdatedAt time.Time      `json:"updated_at"`
	Alert     capmodel.Alert `json:"alert"`
	RawXML    string         `json:"raw_xml,omitempty"`
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
			extra := map[string]any{
				"path":       item.Path,
				"package_id": "alerts",
			}
			if item.AlertText != "" {
				extra["alert_text"] = item.AlertText
			}
			data := alertmodel.WithLegacyFields(item.AlertPacket, extra)
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

func capSAMEPayload(alert capmodel.Alert, feed feedXML, baseDir string, now time.Time) map[string]any {
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
	payload["same_event_name"] = sameEventNameForCAP(alert, *info, event, baseDir)
	if originator == "WXR" {
		payload["same_weather_service"] = sameWeatherServiceForCAP(alert)
	}
	if callsign := sameCallsignForCAP(feed); callsign != "" {
		payload["same_callsign"] = callsign
	}
	payload["same_locations"] = locations
	payload["same_duration"] = sameDurationForCAP(alert, *info)
	if sent := firstCAPTime(alert.Sent, []capmodel.AlertInfo{*info}); !sent.IsZero() {
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

func sameAlertFreshForTone(alert capmodel.Alert, info capmodel.AlertInfo, event string, now time.Time) bool {
	anchor := firstCAPTime(alert.Sent, []capmodel.AlertInfo{info})
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

func sameEventForCAP(alert capmodel.Alert, info capmodel.AlertInfo, baseDir string) string {
	return sameEventResolutionForCAP(alert, info, baseDir).Event
}

func sameEventResolutionForCAP(alert capmodel.Alert, info capmodel.AlertInfo, baseDir string) capsame.EventResolution {
	return capsame.ResolveEvent(alert, info, baseDir)
}

func sameLocationsForCAP(info capmodel.AlertInfo, feed feedXML, baseDir string) []string {
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

func sameEventNameForCAP(alert capmodel.Alert, info capmodel.AlertInfo, event string, baseDir string) string {
	if detectCAPSource(alert) == "nws" {
		if name := alerttext.EventName(filepath.Join(baseDir, "config.yaml"), event); strings.TrimSpace(name) != "" {
			return name
		}
	}
	return alertSubject(info)
}

func sameCallsignForCAP(feed feedXML) string {
	if strings.EqualFold(strings.TrimSpace(feed.ID), "CAP-IT-ALL") {
		return "CAP-IT-ALL"
	}
	return strings.TrimSpace(stationTransmitter(feed).Callsign)
}

func sameLocationCodesForAlertCode(db alertGeoDB, raw string) []string {
	if clean := sameLocationCode(raw); clean != "" {
		return []string{clean}
	}
	if codes := capCPToCLCs(db, raw); len(codes) > 0 {
		return codes
	}
	code := normalizeNWSCode(raw)
	if code == "" {
		return nil
	}
	if codes := db.NWSZoneSAME[code]; len(codes) > 0 {
		return append([]string(nil), codes...)
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
	if item, ok := db.Marine[code]; ok {
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

func capCPToCLCs(db alertGeoDB, raw string) []string {
	out := []string{}
	for _, code := range uniqueStrings([]string{strings.TrimSpace(raw), digitsOnly(raw)}) {
		lookup := strings.ToUpper(strings.TrimSpace(code))
		for _, linked := range db.CAPCPToCLC[lookup] {
			if clean := sameLocationCode(linked); clean != "" {
				out = append(out, clean)
			}
		}
		item, ok := db.CAPCP[code]
		if !ok || !validGeoPoint(item.Lat, item.Lon) {
			continue
		}
		if nearest := nearestCLCForCAPCP(db, item, true); nearest != "" {
			out = append(out, nearest)
		}
		if nearest := nearestCLCForCAPCP(db, item, false); nearest != "" {
			out = append(out, nearest)
		}
	}
	return uniqueStrings(out)
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

func sameOriginatorForCAP(alert capmodel.Alert, info capmodel.AlertInfo) string {
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

func sameWeatherServiceForCAP(alert capmodel.Alert) string {
	switch detectCAPSource(alert) {
	case "eccc":
		return "Environment Canada"
	case "nws":
		return "The National Weather Service"
	default:
		return ""
	}
}

func sameOriginatorNameForCAP(alert capmodel.Alert, info capmodel.AlertInfo) string {
	if detectCAPSource(alert) == "nws" {
		return ""
	}
	if name := strings.TrimSpace(info.SenderName); name != "" {
		return name
	}
	return sameWeatherServiceForCAP(alert)
}

func sameBeginsForCAP(alert capmodel.Alert, info capmodel.AlertInfo) time.Time {
	for _, raw := range []string{info.Onset, info.Effective, alert.Sent} {
		if parsed := parseCAPTime(raw); !parsed.IsZero() {
			return parsed
		}
	}
	return time.Time{}
}

func sameDurationForCAP(alert capmodel.Alert, info capmodel.AlertInfo) string {
	start := firstCAPTime(alert.Sent, []capmodel.AlertInfo{info})
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

func sameToneForCAP(info capmodel.AlertInfo, feed feedXML) string {
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
	AlertPacket        alertmodel.Packet
}

func (s *Service) recordCAPAlert(alert capmodel.Alert, now time.Time) ([]capRegistryUpdate, error) {
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
		entries := loadActiveCAPEntries(s.cfg.Store, feed.ID, now)
		archiveExpiredCAPEntries("", feed.ID, entries, now, s.cfg.Store)
		entries = pruneCAPEntries(entries, now)
		priorEntries := append([]capRegistryEntry(nil), entries...)
		hadAcceptedArchive := capAcceptedArchiveContains(s.cfg.Store, feed.ID, alert.Identifier)
		hadPriorityQueue := capPriorityQueueContains(s.cfg.BaseDir, feed.ID, alert.Identifier)
		if hadAcceptedArchive && hadPriorityQueue && !capEntryExists(priorEntries, alert.Identifier) {
			priorEntries = append(priorEntries, capRegistryEntry{ID: alert.Identifier})
		}
		if isExplicitCAPEnd(alert) {
			cancelledIDs := cancelledAlertIDsWithOverrides(priorEntries, alert, s.cfg.BaseDir)
			deleteCAPReferences(s.cfg.Store, feed.ID, cancelledIDs)
			storeCAPArchiveRecord(s.cfg.Store, "expired", capArchiveRecord{
				ID:        alert.Identifier,
				FeedID:    feed.ID,
				Status:    "expired",
				Reason:    "cancelled or ended by alerting authority",
				UpdatedAt: now,
				Alert:     alert,
				RawXML:    alert.RawXML,
			})
			updates = append(updates, capRegistryUpdate{
				FeedID:       feed.ID,
				Renderable:   hasRenderableCAPEntries(removeCAPIDs(entries, cancelledIDs), now),
				Broadcast:    false,
				Cancelled:    true,
				CancelledIDs: cancelledIDs,
				SAME:         capSAMEPayload(alert, feed, s.cfg.BaseDir, now),
			})
			continue
		}
		if capOperatorSuppressedSameVersion(s.cfg.Store, feed.ID, alert, now) {
			if hadAcceptedArchive {
				deleteCAPReferences(s.cfg.Store, feed.ID, []string{alert.Identifier})
				updates = append(updates, capRegistryUpdate{
					FeedID:     feed.ID,
					Renderable: hasRenderableCAPEntries(removeCAPIDs(entries, []string{alert.Identifier}), now),
					Broadcast:  false,
					Cancelled:  false,
				})
			}
			continue
		}
		if !feedAllowsRoutineOnlyCAPAlert(feed, alert) {
			record := capArchiveRecord{
				ID:        alert.Identifier,
				FeedID:    feed.ID,
				Status:    "rejected",
				Reason:    "below feed alert threshold",
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
		entries = removeCAPReferences(entries, alert.References)
		deleteCAPReferences(s.cfg.Store, feed.ID, parseCAPReferences(alert.References))

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
		broadcastPriorEntries := priorEntries
		if !hadPriorityQueue {
			broadcastPriorEntries = removeCAPIDs(broadcastPriorEntries, []string{alert.Identifier})
		}
		broadcast := capPriorityBroadcastAllowedWithPrior(alert, feed, s.cfg.BaseDir, now, broadcastPriorEntries)
		if broadcast && !s.claimCAPPriorityBroadcast(feed.ID, alert.Identifier) {
			broadcast = false
		}
		samePayload := capSAMEPayload(alert, feed, s.cfg.BaseDir, now)
		packet := capAlertPacket(alert, feed, headline, eventName, severity, urgency, certainty, description, instruction, backgroundColor, broadcastImmediate, alertText, audio, samePayload)
		updates = append(updates, capRegistryUpdate{
			FeedID:             feed.ID,
			Renderable:         hasRenderableCAPEntries(entries, now),
			Broadcast:          broadcast,
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
			SAME:               samePayload,
			AlertPacket:        packet,
		})
	}
	return updates, nil
}

func capAlertPacket(alert capmodel.Alert, feed feedXML, headline string, eventName string, severity string, urgency string, certainty string, description string, instruction string, backgroundColor string, broadcastImmediate bool, alertText string, audio capBroadcastAudio, samePayload map[string]any) alertmodel.Packet {
	data := map[string]any{
		"feed_id":             feed.ID,
		"alert_id":            alert.Identifier,
		"alert_sent_at":       alert.Sent,
		"message_type":        alert.MessageType,
		"title":               firstNonBlank(headline, alertBroadcastTitle(alert)),
		"headline":            headline,
		"event":               eventName,
		"severity":            severity,
		"urgency":             urgency,
		"certainty":           certainty,
		"description":         description,
		"instruction":         instruction,
		"background_color":    backgroundColor,
		"broadcast_immediate": broadcastImmediate,
	}
	if expires := alertExpiresAt(alert); !expires.IsZero() {
		data["alert_expires_at"] = expires.Format(time.RFC3339Nano)
	}
	for key, value := range samePayload {
		data[key] = value
	}
	if audio.URL != "" {
		data["audio_url"] = audio.URL
		data["audio_mime_type"] = audio.MimeType
		data["audio_language"] = audio.Language
		data["audio_description"] = audio.Description
		data["audio_authoritative"] = true
	}
	packet, _ := alertmodel.FromMap(data)
	packet.Presentation.SpeechText = strings.TrimSpace(alertText)
	return packet
}

func (s *Service) claimCAPPriorityBroadcast(feedID string, alertID string) bool {
	feedID = strings.TrimSpace(feedID)
	alertID = strings.TrimSpace(alertID)
	if feedID == "" || alertID == "" {
		return true
	}
	key := feedID + "\x00" + alertID
	s.tonedMu.Lock()
	defer s.tonedMu.Unlock()
	if s.tonedCAP == nil {
		s.tonedCAP = map[string]struct{}{}
	}
	if _, exists := s.tonedCAP[key]; exists {
		return false
	}
	s.tonedCAP[key] = struct{}{}
	return true
}

func routineCAPAlertAllowed(alert capmodel.Alert) bool {
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

func capPriorityBroadcastAllowed(alert capmodel.Alert, feed feedXML, baseDir string, now time.Time) bool {
	return capPriorityBroadcastAllowedWithPrior(alert, feed, baseDir, now, nil)
}

func capPriorityBroadcastAllowedWithPrior(alert capmodel.Alert, feed feedXML, baseDir string, now time.Time, priorEntries []capRegistryEntry) bool {
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
	if capEntryExists(priorEntries, alert.Identifier) {
		return false
	}
	if isBroadcastImmediateInfo(*info) {
		return true
	}
	if capMessageTypeIsUpdate(alert) && !capUpdateAddsFeedLocations(alert, *info, feed, baseDir) {
		if feedUsesAlertCoverage(feed, alert) {
			return false
		}
	}
	return sameAlertFreshForTone(alert, *info, sameEventForCAP(alert, *info, baseDir), now)
}

type capPriorityQueueManifest struct {
	ID      string   `json:"id"`
	AlertID string   `json:"alert_id"`
	FeedID  string   `json:"feed_id"`
	FeedIDs []string `json:"feed_ids"`
	Status  string   `json:"status"`
	QueueID string   `json:"queue_id"`
	Subject string   `json:"subject"`
}

func capPriorityQueueContains(baseDir string, feedID string, alertID string) bool {
	alertID = strings.TrimSpace(alertID)
	if alertID == "" || strings.TrimSpace(baseDir) == "" {
		return false
	}
	queuePath := filepath.Join(baseDir, "runtime", "queues", "alerts")
	entries, err := os.ReadDir(queuePath)
	if err != nil {
		return false
	}
	safeAlertID := safeID(alertID)
	safeFeedID := safeID(feedID)
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".json") {
			continue
		}
		manifestPath := filepath.Join(queuePath, entry.Name())
		raw, err := os.ReadFile(manifestPath)
		if err != nil {
			continue
		}
		var manifest capPriorityQueueManifest
		if err := json.Unmarshal(raw, &manifest); err != nil {
			continue
		}
		if strings.TrimSpace(feedID) != "" && !capQueueTargetsFeed(manifest, feedID) {
			continue
		}
		if capQueueMatchesAlert(manifest, entry.Name(), alertID, safeAlertID, safeFeedID) {
			return true
		}
	}
	return false
}

func capQueueTargetsFeed(manifest capPriorityQueueManifest, feedID string) bool {
	if strings.TrimSpace(manifest.FeedID) == feedID {
		return true
	}
	for _, id := range manifest.FeedIDs {
		if strings.TrimSpace(id) == feedID {
			return true
		}
	}
	return false
}

func capQueueMatchesAlert(manifest capPriorityQueueManifest, filename string, alertID string, safeAlertID string, safeFeedID string) bool {
	for _, value := range []string{manifest.AlertID, manifest.Subject} {
		if strings.TrimSpace(value) == alertID {
			return true
		}
	}
	name := strings.TrimSuffix(filename, filepath.Ext(filename))
	if safeAlertID == "" {
		return false
	}
	for _, value := range []string{manifest.ID, manifest.QueueID, name} {
		value = strings.TrimSpace(value)
		if value == "" || !strings.Contains(value, safeAlertID) {
			continue
		}
		if safeFeedID == "" || strings.Contains(value, safeFeedID) {
			return true
		}
	}
	return false
}

func capEntryExists(entries []capRegistryEntry, id string) bool {
	id = strings.TrimSpace(id)
	if id == "" {
		return false
	}
	for _, entry := range entries {
		if entry.ID == id || entry.Alert.Identifier == id {
			return true
		}
	}
	return false
}

func capMessageTypeIsUpdate(alert capmodel.Alert) bool {
	return strings.EqualFold(strings.TrimSpace(alert.MessageType), "Update")
}

func capUpdateAddsFeedLocations(alert capmodel.Alert, info capmodel.AlertInfo, feed feedXML, baseDir string) bool {
	newCodes := capNewlyActiveCodes(info)
	if len(newCodes) == 0 {
		return false
	}
	if !feedUsesAlertCoverage(feed, alert) {
		return true
	}
	coverage := feedCoverageModel(baseDir, feed, nil)
	if len(coverage.Codes) == 0 {
		return true
	}
	db := loadAlertGeoDB(baseDir)
	for _, code := range newCodes {
		if coverageMatchesAlertCode(db, coverage.Codes, code) {
			return true
		}
	}
	return false
}

func capNewlyActiveCodes(info capmodel.AlertInfo) []string {
	seen := map[string]struct{}{}
	add := func(raw string) {
		for _, part := range strings.FieldsFunc(raw, func(ch rune) bool {
			return ch == ',' || ch == ';' || ch == '|' || ch == '\n' || ch == '\r' || ch == '\t'
		}) {
			value := strings.TrimSpace(part)
			if value != "" {
				seen[value] = struct{}{}
			}
		}
	}
	for _, param := range info.Parameters {
		name := strings.ToLower(strings.TrimSpace(param.Name))
		if strings.Contains(name, "newly_active_areas") || strings.Contains(name, "newly active") {
			add(param.Value)
		}
	}
	out := make([]string, 0, len(seen))
	for value := range seen {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

type capBroadcastAudio struct {
	URL         string
	MimeType    string
	Language    string
	Description string
}

func alertBroadcastAudio(alert capmodel.Alert, preferredLanguage string) capBroadcastAudio {
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

func hasBroadcastAudioResource(info capmodel.AlertInfo) bool {
	for _, resource := range info.Resources {
		if isAudioResource(resource) {
			return true
		}
	}
	return false
}

func isAudioResource(resource capmodel.Resource) bool {
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

func isBroadcastImmediateInfo(info capmodel.AlertInfo) bool {
	return alerttext.IsBroadcastImmediateInfo(info)
}

func preferredCAPInfos(infos []capmodel.AlertInfo, language string) []capmodel.AlertInfo {
	language = strings.ToLower(strings.TrimSpace(language))
	short := language
	if index := strings.Index(short, "-"); index > 0 {
		short = short[:index]
	}
	var exact []capmodel.AlertInfo
	var prefix []capmodel.AlertInfo
	var rest []capmodel.AlertInfo
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

func alertBroadcastTitle(alert capmodel.Alert) string {
	for _, info := range alert.Infos {
		if text := firstNonBlank(info.Headline, info.Event); text != "" {
			return titleText(text)
		}
	}
	return "Weather Alert"
}

func capAlertFromEvent(event map[string]any) (capmodel.Alert, bool) {
	data := mapAt(event, "data")
	value, ok := data["alert"]
	if !ok {
		value = event["alert"]
	}
	if value == nil {
		return capmodel.Alert{}, false
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return capmodel.Alert{}, false
	}
	var alert capmodel.Alert
	if err := json.Unmarshal(raw, &alert); err != nil {
		return capmodel.Alert{}, false
	}
	alert = normalizeRawCAPAlert(alert)
	return alert, alert.Identifier != "" && strings.TrimSpace(alert.RawXML) != ""
}

func normalizeRawCAPAlert(alert capmodel.Alert) capmodel.Alert {
	raw := strings.TrimSpace(alert.RawXML)
	if raw == "" {
		return alert
	}
	parsed, err := capmodel.ParseCAP([]byte(raw))
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

func capAcceptedArchiveContains(store datastore.Store, feedID string, alertID string) bool {
	if store == nil {
		return false
	}
	alertID = strings.TrimSpace(alertID)
	if alertID == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	rows, err := store.ListCAPArchives(ctx, "accepted", time.Time{})
	if err != nil {
		return false
	}
	for _, row := range rows {
		if strings.TrimSpace(feedID) != "" && row.FeedID != feedID {
			continue
		}
		if row.AlertID == alertID {
			return true
		}
	}
	return false
}

func storedCAPArchiveAlert(row datastore.StoredCAPArchive) (capmodel.Alert, bool) {
	rawXML := strings.TrimSpace(row.RawXML)
	if rawXML == "" {
		return capmodel.Alert{}, false
	}
	alert, err := capmodel.ParseCAP([]byte(rawXML))
	if err != nil || alert.Identifier == "" {
		return capmodel.Alert{}, false
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

func capOperatorSuppressedSameVersion(store datastore.Store, feedID string, alert capmodel.Alert, now time.Time) bool {
	if store == nil || strings.TrimSpace(alert.Identifier) == "" || strings.TrimSpace(alert.RawXML) == "" {
		return false
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	rows, err := store.ListCAPArchives(ctx, "expired", now.UTC().Add(-30*24*time.Hour))
	if err != nil {
		return false
	}
	alertID := strings.TrimSpace(alert.Identifier)
	feedID = strings.TrimSpace(feedID)
	rawXML := strings.TrimSpace(alert.RawXML)
	for _, row := range rows {
		if strings.TrimSpace(row.AlertID) != alertID || strings.TrimSpace(row.FeedID) != feedID {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(row.Reason), "expired by operator") {
			continue
		}
		if strings.TrimSpace(row.RawXML) == rawXML {
			return true
		}
	}
	return false
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
	archiveRecord := datastore.CAPArchiveRecord{
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
	}
	if err := storeCAPArchiveRecordWithRetry(store, archiveRecord); err != nil {
		log.Printf("CAP archive store failed bucket=%s feed_id=%s alert_id=%s status=%s: %v",
			archiveRecord.Bucket,
			archiveRecord.FeedID,
			archiveRecord.AlertID,
			archiveRecord.Status,
			err,
		)
	}
}

func storeCAPArchiveRecordWithRetry(store datastore.Store, record datastore.CAPArchiveRecord) error {
	var err error
	for attempt := 1; attempt <= 3; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
		err = store.StoreCAPArchive(ctx, record)
		cancel()
		if err == nil {
			return nil
		}
		time.Sleep(time.Duration(attempt) * 75 * time.Millisecond)
	}
	return err
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

func cancelledAlertIDs(alert capmodel.Alert) []string {
	ids := append([]string{}, parseCAPReferences(alert.References)...)
	if strings.TrimSpace(alert.Identifier) != "" {
		ids = append(ids, strings.TrimSpace(alert.Identifier))
	}
	return uniqueStrings(ids)
}

func cancelledAlertIDsWithOverrides(entries []capRegistryEntry, alert capmodel.Alert, baseDir string) []string {
	referenced := map[string]struct{}{}
	for _, id := range parseCAPReferences(alert.References) {
		referenced[id] = struct{}{}
	}
	ids := []string{}
	seen := map[string]struct{}{}
	add := func(id string) {
		id = strings.TrimSpace(id)
		if id == "" {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		ids = append(ids, id)
		seen[id] = struct{}{}
	}
	for _, entry := range entries {
		id := fallbackText(entry.ID, entry.Alert.Identifier)
		if id == "" {
			continue
		}
		if strings.EqualFold(id, strings.TrimSpace(alert.Identifier)) || strings.EqualFold(entry.Alert.Identifier, strings.TrimSpace(alert.Identifier)) {
			add(id)
			continue
		}
		if _, ok := referenced[id]; ok && capAlertCancelsEntry(alert, entry.Alert, baseDir) {
			add(id)
			continue
		}
		if _, ok := referenced[entry.Alert.Identifier]; ok && capAlertCancelsEntry(alert, entry.Alert, baseDir) {
			add(id)
			continue
		}
		if capAlertOverrides(alert, entry.Alert, baseDir) {
			add(id)
		}
	}
	return uniqueStrings(ids)
}

func capAlertCancelsEntry(next capmodel.Alert, previous capmodel.Alert, baseDir string) bool {
	if !isExplicitCAPEnd(next) {
		return false
	}
	if detectCAPSource(next) != detectCAPSource(previous) {
		return false
	}
	if next.Sender != "" && previous.Sender != "" && !strings.EqualFold(next.Sender, previous.Sender) {
		return false
	}
	if !stringSetOverlaps(capLifecycleEventSet(next), capLifecycleEventSet(previous)) {
		return false
	}
	return stringSetOverlaps(capLifecycleLocationSet(next, baseDir), capLifecycleLocationSet(previous, baseDir))
}

func capAlertOverrides(next capmodel.Alert, previous capmodel.Alert, baseDir string) bool {
	if next.Identifier == "" || previous.Identifier == "" || next.Identifier == previous.Identifier {
		return false
	}
	return capAlertCancelsEntry(next, previous, baseDir)
}

func capLifecycleEventSet(alert capmodel.Alert) map[string]struct{} {
	specific := map[string]struct{}{}
	broad := map[string]struct{}{}
	add := func(out map[string]struct{}, raw string) {
		key := normalizeLifecycleKey(raw)
		if key != "" {
			out[key] = struct{}{}
		}
	}
	addSpecific := func(raw string) {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			return
		}
		add(specific, raw)
		if canonical := canonicalLifecycleEventKey(raw); canonical != "" {
			add(specific, canonical)
		}
	}
	for _, info := range alert.Infos {
		alertType := lifecycleAlertParam(info, "alert_type")
		alertName := lifecycleAlertParam(info, "alert_name")
		addSpecific(alertName)
		addSpecific(alertSubject(info))
		addSpecific(stripAlertHeadlineState(info.Headline))
		if alertType != "" {
			addSpecific(joinLifecycleParts(alertType, info.Event))
			addSpecific(joinLifecycleParts(alertType, alertName))
		}
		add(broad, info.Event)
		for _, code := range info.EventCodes {
			name := strings.ToLower(strings.TrimSpace(code.Name))
			if strings.Contains(name, "same") || strings.Contains(name, "event") {
				addSpecific(code.Value)
			}
		}
	}
	if len(specific) > 0 {
		return specific
	}
	return broad
}

func lifecycleAlertParam(info capmodel.AlertInfo, token string) string {
	token = strings.ToLower(strings.TrimSpace(token))
	for _, param := range info.Parameters {
		if strings.Contains(strings.ToLower(strings.TrimSpace(param.Name)), token) {
			return strings.TrimSpace(param.Value)
		}
	}
	return ""
}

func joinLifecycleParts(values ...string) string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return strings.Join(out, " ")
}

func canonicalLifecycleEventKey(raw string) string {
	key := normalizeLifecycleKey(raw)
	if key == "" {
		return ""
	}
	class := lifecycleAlertClass(key)
	if class == "" {
		return ""
	}
	words := strings.Fields(key)
	hazard := make([]string, 0, len(words))
	for _, word := range words {
		switch word {
		case class, "yellow", "orange", "red", "green", "blue", "grey", "gray", "alert", "in", "effect", "ended", "updated", "cancelled", "canceled":
			continue
		default:
			hazard = append(hazard, word)
		}
	}
	if len(hazard) == 0 {
		return class
	}
	return class + " " + strings.Join(hazard, " ")
}

func lifecycleAlertClass(key string) string {
	for _, class := range []string{"warning", "watch", "advisory", "statement"} {
		for _, word := range strings.Fields(key) {
			if word == class {
				return class
			}
		}
	}
	switch strings.ToUpper(strings.TrimSpace(key)) {
	case "SVA", "TOA", "FFA", "FLA", "HUA", "HWA", "TRA", "TSA", "WSA":
		return "watch"
	case "SVR", "TOR", "FFW", "FLW", "HUW", "HWW", "SQW", "WSW":
		return "warning"
	case "SPS", "SPSA", "DMO", "RWT", "RMT":
		return "advisory"
	default:
		return ""
	}
}

func capLifecycleLocationSet(alert capmodel.Alert, baseDir string) map[string]struct{} {
	db := loadAlertGeoDB(baseDir)
	out := map[string]struct{}{}
	add := func(raw string) {
		value := strings.TrimSpace(raw)
		if value == "" {
			return
		}
		out[strings.ToUpper(value)] = struct{}{}
		if digits := digitsOnly(value); digits != "" {
			out[digits] = struct{}{}
			if same := sameLocationCode(digits); same != "" {
				out[same] = struct{}{}
			}
		}
		if normalized := normalizeNWSCode(value); normalized != "" {
			out[normalized] = struct{}{}
		}
		for _, same := range sameLocationCodesForAlertCode(db, value) {
			if same != "" {
				out[same] = struct{}{}
			}
		}
	}
	for _, code := range alertCoverageCodes(alert) {
		add(code)
	}
	return out
}

func normalizeLifecycleKey(raw string) string {
	raw = strings.ToLower(strings.TrimSpace(raw))
	raw = strings.ReplaceAll(raw, "_", " ")
	raw = strings.ReplaceAll(raw, "-", " ")
	return strings.Join(strings.Fields(raw), " ")
}

func stringSetOverlaps(left map[string]struct{}, right map[string]struct{}) bool {
	if len(left) == 0 || len(right) == 0 {
		return false
	}
	if len(left) > len(right) {
		left, right = right, left
	}
	for value := range left {
		if _, ok := right[value]; ok {
			return true
		}
	}
	return false
}

func removeCAPIDs(entries []capRegistryEntry, ids []string) []capRegistryEntry {
	if len(entries) == 0 || len(ids) == 0 {
		return entries
	}
	remove := map[string]struct{}{}
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id != "" {
			remove[id] = struct{}{}
		}
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

func feedAcceptsCAPSource(feed feedXML, alert capmodel.Alert) bool {
	source, sourceConfig := feedCAPSourceConfig(feed, alert)
	if source == "generic" {
		return xmlBool(feed.Alerts.CapCP.EnabledRaw, true) || xmlBool(feed.Alerts.NWSCAP.EnabledRaw, false)
	}
	return xmlBool(sourceConfig.EnabledRaw, source == "eccc")
}

func feedAllowsCAPAlert(feed feedXML, alert capmodel.Alert) bool {
	_, sourceConfig := feedCAPSourceConfig(feed, alert)
	return alertFilterAllows(sourceConfig.Filter, alert, feedLanguage(feed))
}

func feedAllowsRoutineOnlyCAPAlert(feed feedXML, alert capmodel.Alert) bool {
	if xmlBool(feed.Playout.Routine, true) {
		return true
	}
	if feedAllowsCAPAlert(feed, alert) {
		return true
	}
	_, sourceConfig := feedCAPSourceConfig(feed, alert)
	return alertFilterAllowsActiveRetention(sourceConfig.Filter, alert, feedLanguage(feed))
}

func feedUsesAlertCoverage(feed feedXML, alert capmodel.Alert) bool {
	_, sourceConfig := feedCAPSourceConfig(feed, alert)
	return xmlBool(sourceConfig.Filter.UseFeedLocations, true)
}

func feedCAPSourceConfig(feed feedXML, alert capmodel.Alert) (string, feedAlertSourceXML) {
	source := detectCAPSource(alert)
	switch source {
	case "nws":
		return source, feed.Alerts.NWSCAP
	default:
		return source, feed.Alerts.CapCP
	}
}

func alertFilterAllows(filter alertFilterXML, alert capmodel.Alert, language string) bool {
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

func alertFilterAllowsActiveRetention(filter alertFilterXML, alert capmodel.Alert, language string) bool {
	info := chooseAlertInfo(alert, language)
	if info == nil {
		return true
	}
	if alertFilterListMatches(filter.Blocklist, alert, *info) {
		return false
	}
	if len(filter.Allowlist.Severities) > 0 {
		return textInList(info.Severity, filter.Allowlist.Severities)
	}
	if alertFilterListEmpty(filter.Allowlist) {
		return true
	}
	return alertFilterListAllows(filter.Allowlist, alert, *info)
}

func alertFilterListAllows(list alertFilterListXML, alert capmodel.Alert, info capmodel.AlertInfo) bool {
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

func alertFilterListMatches(list alertFilterListXML, alert capmodel.Alert, info capmodel.AlertInfo) bool {
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

func alertEventMatches(info capmodel.AlertInfo, wanted []string) bool {
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

func alertOtherFiltersMatch(info capmodel.AlertInfo, filters []alertFilterOtherXML) bool {
	for _, filter := range filters {
		if alertOtherFilterMatches(info, filter) {
			return true
		}
	}
	return false
}

func alertOtherFilterMatches(info capmodel.AlertInfo, filter alertFilterOtherXML) bool {
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

func detectCAPSource(alert capmodel.Alert) string {
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
	CLC         map[string]clcBaseZone
	CAPCP       map[string]capCPGeocode
	CAPCPToCLC  map[string][]string
	NWS         map[string]nwsZone
	FIPS        map[string]nwsZone
	Marine      map[string]nwsZone
	NWSZoneSAME map[string][]string
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

func alertMatchesFeed(alert capmodel.Alert, feed feedXML, baseDir string) bool {
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

	db, ok := loadAlertGeoDBFromSQLite(baseDir)
	if !ok {
		db = alertGeoDB{
			CLC:    loadCLCBaseZones(filepath.Join(baseDir, "managed", "csv", "CLC_Base_Zone.csv")),
			CAPCP:  loadCAPCPGeocodes(filepath.Join(baseDir, "managed", "csv", "CAP-CP_Geocodes.csv")),
			NWS:    loadNWSZones(filepath.Join(baseDir, "managed", "csv", "NWS_ZONE_COUNTY_CORRELATION.csv")),
			FIPS:   loadNWSFIPS(filepath.Join(baseDir, "managed", "csv", "NWS_ZONE_COUNTY_CORRELATION.csv")),
			Marine: loadNWSMarineZones(filepath.Join(baseDir, "managed", "csv", "NWS_MARINE_ZONES.csv")),
		}
	}
	alertGeoCache.Lock()
	alertGeoCache.byBase[key] = db
	alertGeoCache.Unlock()
	return db
}

func loadAlertGeoDBFromSQLite(baseDir string) (alertGeoDB, bool) {
	snap, ok := locationdb.Load(baseDir)
	if !ok {
		return alertGeoDB{}, false
	}
	db := alertGeoDB{
		CLC:         map[string]clcBaseZone{},
		CAPCP:       map[string]capCPGeocode{},
		CAPCPToCLC:  map[string][]string{},
		NWS:         map[string]nwsZone{},
		FIPS:        map[string]nwsZone{},
		Marine:      map[string]nwsZone{},
		NWSZoneSAME: map[string][]string{},
	}
	for _, place := range snap.PlacesBySource("clc") {
		db.CLC[place.Code] = clcBaseZone{Code: place.Code, En: place.Name, Fr: place.NameFR, Lat: place.Lat, Lon: place.Lon, Province: place.Region}
	}
	for _, place := range snap.PlacesBySource("sgc") {
		db.CAPCP[place.Code] = capCPGeocode{Code: place.Code, En: place.Name, Fr: place.NameFR, Lat: place.Lat, Lon: place.Lon, Province: place.Region}
	}
	for _, place := range snap.PlacesBySource("nws_same") {
		item := nwsZone{Code: place.Code, Name: place.Name, CountyName: place.Name, FIPS: place.Code}
		db.FIPS[place.Code] = item
	}
	for _, place := range snap.PlacesBySource("nws_zone") {
		item := nwsZone{Code: place.Code, Name: place.Name}
		db.NWS[normalizeNWSCode(place.Code)] = item
		db.NWS[place.Code] = item
	}
	for _, place := range snap.PlacesBySource("nws_marine_same") {
		item := nwsZone{Code: place.Code, Name: place.Name, FIPS: place.Code}
		db.Marine[place.Code] = item
		db.FIPS[place.Code] = item
	}
	for _, place := range snap.PlacesBySource("nws_marine_zone") {
		item := nwsZone{Code: place.Code, Name: place.Name}
		db.Marine[place.Code] = item
		db.Marine[normalizeNWSCode(place.Code)] = item
	}
	for _, link := range snap.Links {
		switch link.Type {
		case "sgc_to_clc":
			from := strings.ToUpper(strings.TrimSpace(link.FromCode))
			to := sameLocationCode(link.ToCode)
			if from == "" || to == "" || strings.EqualFold(strings.TrimSpace(link.Confidence), "low") {
				continue
			}
			if strings.EqualFold(link.FromSource, "sgc") && strings.EqualFold(link.ToSource, "clc") {
				db.CAPCPToCLC[from] = append(db.CAPCPToCLC[from], to)
			}
		case "nws_same_to_zone":
			same := sameLocationCode(link.FromCode)
			zone := normalizeNWSCode(link.ToCode)
			if same == "" || zone == "" {
				continue
			}
			db.NWSZoneSAME[zone] = append(db.NWSZoneSAME[zone], same)
			if item, ok := db.NWS[zone]; ok && strings.TrimSpace(item.FIPS) == "" {
				item.FIPS = same
				db.NWS[zone] = item
				db.NWS[link.ToCode] = item
			}
		case "nws_marine_same_to_zone":
			same := sameLocationCode(link.FromCode)
			zone := normalizeNWSCode(link.ToCode)
			if same == "" || zone == "" {
				continue
			}
			db.NWSZoneSAME[zone] = append(db.NWSZoneSAME[zone], same)
			if item, ok := db.Marine[zone]; ok {
				item.FIPS = same
				db.Marine[zone] = item
				db.Marine[link.ToCode] = item
				db.Marine[same] = item
			}
		}
	}
	for zone, codes := range db.NWSZoneSAME {
		db.NWSZoneSAME[zone] = uniqueStrings(codes)
	}
	for code, links := range db.CAPCPToCLC {
		db.CAPCPToCLC[code] = uniqueStrings(links)
	}
	return db, true
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

func loadNWSMarineZones(path string) map[string]nwsZone {
	rows := readCSVRows(path, ',')
	out := map[string]nwsZone{}
	if len(rows) == 0 {
		return out
	}
	header := map[string]int{}
	for i, value := range rows[0] {
		header[strings.ToLower(strings.TrimSpace(value))] = i
	}
	zoneIndex := csvHeaderLookup(header, "zone_ugc", "zone", "ugc")
	sameIndex := csvHeaderLookup(header, "same_code", "same", "ssnum")
	nameIndex := csvHeaderLookup(header, "name", "zonename", "zone_name")
	if zoneIndex < 0 || sameIndex < 0 || nameIndex < 0 {
		return out
	}
	for _, row := range rows[1:] {
		zone := strings.ToUpper(csvString(row, zoneIndex))
		same := sameLocationCode(csvString(row, sameIndex))
		name := csvString(row, nameIndex)
		if zone == "" || same == "" || name == "" {
			continue
		}
		item := nwsZone{
			Code: zone,
			Name: name,
			FIPS: same,
		}
		out[zone] = item
		out[normalizeNWSCode(zone)] = item
		out[same] = item
	}
	return out
}

func csvHeaderLookup(header map[string]int, keys ...string) int {
	for _, key := range keys {
		if index, ok := header[strings.ToLower(strings.TrimSpace(key))]; ok {
			return index
		}
	}
	return -1
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
	if item, ok := db.Marine[code]; ok {
		return uniqueStrings([]string{item.Code, item.FIPS, sameLocationCode(item.FIPS)})
	}
	if codes := db.NWSZoneSAME[code]; len(codes) > 0 {
		return append([]string(nil), codes...)
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
	if item, ok := db.Marine[strings.ToUpper(code)]; ok {
		return strings.TrimSpace(item.Name)
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

func alertAreaNameFromGeocodes(db alertGeoDB, area capmodel.AlertArea, lang string) string {
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

func alertCoverageCodes(alert capmodel.Alert) []string {
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

func renderCAPAlertSentence(entry capRegistryEntry, info capmodel.AlertInfo, feed feedXML, baseDir string, forecastNames map[string]forecastRegionName, now time.Time) string {
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

func chooseAlertInfo(alert capmodel.Alert, lang string) *capmodel.AlertInfo {
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

func alertSenderName(alert capmodel.Alert) string {
	if detectCAPSource(alert) == "eccc" {
		return "Environment Canada"
	}
	return fallbackText(alert.Sender, "The alerting authority")
}

func alertSubject(info capmodel.AlertInfo) string {
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

func alertAreas(info capmodel.AlertInfo, feed feedXML, baseDir string, forecastNames map[string]forecastRegionName) string {
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

func useBroadAlertRegions(info capmodel.AlertInfo) bool {
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

func broadAlertAreaPhrase(info capmodel.AlertInfo, coverage coverageModel, db alertGeoDB) string {
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

func alertInfoCoverageCodes(info capmodel.AlertInfo) map[string]struct{} {
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

func alertParam(info capmodel.AlertInfo, name string) string {
	for _, param := range info.Parameters {
		if strings.EqualFold(strings.TrimSpace(param.Name), name) {
			return strings.TrimSpace(param.Value)
		}
	}
	return ""
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
	for _, info := range alert.Infos {
		if expires := parseCAPTime(info.Expires); !expires.IsZero() {
			return expires
		}
	}
	return time.Time{}
}

func firstCAPTime(sent string, infos []capmodel.AlertInfo) time.Time {
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
