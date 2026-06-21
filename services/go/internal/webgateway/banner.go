package webgateway

import (
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/meowraii/haze-weather-radio/services/go/internal/alerttext"
	"github.com/meowraii/haze-weather-radio/services/go/internal/capingest"
)

type bannerPayload struct {
	Active          bool                        `json:"active"`
	Signature       string                      `json:"signature"`
	FeedID          string                      `json:"feed_id,omitempty"`
	FeedName        string                      `json:"feed_name,omitempty"`
	GeneratedAt     time.Time                   `json:"generated_at"`
	PrimaryColor    string                      `json:"primary_color"`
	PrimaryGradient []string                    `json:"primary_gradient"`
	Alerts          []alerttext.SerializedAlert `json:"alerts"`
}

func (s *Server) bannerStream(writer http.ResponseWriter, request *http.Request) {
	if !s.auth.Authenticated(request) {
		http.Error(writer, "unauthorized", http.StatusUnauthorized)
		return
	}
	flusher, ok := writer.(http.Flusher)
	if !ok {
		http.Error(writer, "streaming is not supported", http.StatusInternalServerError)
		return
	}
	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-store")
	writer.Header().Set("Connection", "keep-alive")
	writer.Header().Set("X-Accel-Buffering", "no")

	feedID := strings.TrimSpace(request.URL.Query().Get("feed"))
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	lastSignature := ""
	for {
		payload := buildBannerPayload(s.configPath, feedID, s.bannerHub)
		if payload.Signature != lastSignature {
			if err := writeBannerEvent(writer, payload); err != nil {
				return
			}
			flusher.Flush()
			lastSignature = payload.Signature
		}

		select {
		case <-request.Context().Done():
			return
		case <-ticker.C:
		}
	}
}

func (s *Server) bannerAudio(writer http.ResponseWriter, request *http.Request) {
	if !s.auth.Authenticated(request) {
		http.Error(writer, "unauthorized", http.StatusUnauthorized)
		return
	}
	connection, err := websocket.Accept(writer, request, &websocket.AcceptOptions{
		OriginPatterns: sameOriginPatterns(request),
	})
	if err != nil {
		return
	}
	defer connection.CloseNow()

	for {
		if _, _, err := connection.Read(request.Context()); err != nil {
			return
		}
	}
}

func (s *Server) bannerWebRTCOffer(writer http.ResponseWriter, request *http.Request) {
	if !s.auth.Authenticated(request) {
		http.Error(writer, "unauthorized", http.StatusUnauthorized)
		return
	}
	if request.Method != http.MethodPost {
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if s.media == nil || !s.media.Available() {
		http.Error(writer, "media bridge is not available", http.StatusServiceUnavailable)
		return
	}
	var payload struct {
		FeedID         string `json:"feed_id"`
		SDP            string `json:"sdp"`
		DisableG722    bool   `json:"disable_g722"`
		RequireOpus    bool   `json:"require_opus"`
		Codec          string `json:"codec"`
		PreferredCodec string `json:"preferred_codec"`
	}
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		http.Error(writer, "invalid JSON", http.StatusBadRequest)
		return
	}
	feedID := strings.TrimSpace(payload.FeedID)
	if feedID == "" {
		feedID = strings.TrimSpace(request.URL.Query().Get("feed"))
	}
	if feedID == "" {
		http.Error(writer, "feed_id is required", http.StatusBadRequest)
		return
	}
	if !s.feedWebRTCEnabled(feedID) {
		http.Error(writer, "feed WebRTC output is not enabled", http.StatusForbidden)
		return
	}
	answer, err := s.media.AnswerWithOptions(request.Context(), feedID, payload.SDP, WebRTCAnswerOptions{
		DisableG722:    payload.DisableG722,
		RequireOpus:    payload.RequireOpus,
		PreferredCodec: firstNonBlank(payload.Codec, payload.PreferredCodec),
	})
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(writer, map[string]any{
		"feed_id":      feedID,
		"sdp":          answer.SDP,
		"sdp_type":     "answer",
		"media_recent": answer.MediaRecent,
		"codec":        answer.Codec.String(),
		"payload_type": answer.PayloadType,
	})
}

func (s *Server) feedWebRTCEnabled(feedID string) bool {
	feeds, err := loadFeedSummaries(s.configPath)
	if err != nil {
		return false
	}
	for _, feed := range feeds {
		if stringValue(feed, "id") != feedID {
			continue
		}
		enabled, _ := feed["enabled"].(bool)
		webrtc, _ := feed["webrtc_enabled"].(bool)
		return enabled && webrtc
	}
	return false
}

func writeBannerEvent(writer http.ResponseWriter, payload bannerPayload) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal banner payload: %w", err)
	}
	if _, err := fmt.Fprintf(writer, "event: banner\ndata: %s\n\n", raw); err != nil {
		return fmt.Errorf("write banner event: %w", err)
	}
	return nil
}

func buildBannerPayload(configPath string, feedID string, hub *BannerHub) bannerPayload {
	now := time.Now().UTC()
	records := onAirBannerRecords(configPath, feedID, hub, now)
	alerts := make([]alerttext.SerializedAlert, 0, len(records))
	visuals := make([]alerttext.AlertVisualInput, 0, len(records))
	for _, record := range records {
		info := chooseArchiveInfo(record.Alert)
		areas := archiveAreaNames(info)
		alerts = append(alerts, alerttext.SerializeCAPAlert(
			record.Alert,
			info,
			record.FeedID,
			areas,
			bannerFeedTimezone(configPath, record.FeedID),
			"cap",
			now,
		))
		visuals = append(visuals, alerttext.AlertVisualInput{
			Severity: info.Severity,
			Event:    fallbackString(info.Event, info.Headline),
		})
	}
	gradient := alerttext.PickBannerGradient(visuals)
	if len(alerts) == 0 {
		gradient = alerttext.PickBannerGradient(nil)
	}
	payload := bannerPayload{
		Active:          len(alerts) > 0,
		FeedID:          strings.TrimSpace(feedID),
		FeedName:        bannerFeedName(configPath, feedID),
		GeneratedAt:     now,
		PrimaryColor:    gradient[0],
		PrimaryGradient: gradient,
		Alerts:          alerts,
	}
	payload.Signature = bannerSignature(payload)
	return payload
}

func activeBannerRecords(configPath string, feedID string, now time.Time) []archiveCAPRecord {
	rows := archiveStoreRecords(configPath, "accepted", time.Time{})
	out := make([]archiveCAPRecord, 0, len(rows))
	for _, record := range rows {
		if feedID != "" && record.FeedID != feedID {
			continue
		}
		if archiveAlertExpired(record.Alert, now) {
			continue
		}
		out = append(out, record)
	}
	sort.SliceStable(out, func(i, j int) bool {
		return bannerSortKey(out[i].Alert).less(bannerSortKey(out[j].Alert))
	})
	return out
}

func onAirBannerRecords(configPath string, feedID string, hub *BannerHub, now time.Time) []archiveCAPRecord {
	active := hub.Active(feedID, now)
	if len(active) == 0 {
		return nil
	}
	out := make([]archiveCAPRecord, 0, len(active))
	for _, item := range active {
		record, ok := findArchiveAlert(configPath, item.AlertID, item.FeedID)
		if !ok {
			record, ok = findArchiveAlert(configPath, item.AlertID, "")
			if ok {
				record.FeedID = item.FeedID
			}
		}
		if !ok {
			record, ok = findArchiveAlertByQueueHint(configPath, item)
		}
		if !ok {
			record = bannerRecordFromOnAirAlert(item, now)
		}
		if archiveAlertExpired(record.Alert, now) {
			continue
		}
		out = append(out, record)
	}
	sort.SliceStable(out, func(i, j int) bool {
		return bannerSortKey(out[i].Alert).less(bannerSortKey(out[j].Alert))
	})
	return out
}

func findArchiveAlertByQueueHint(configPath string, item bannerOnAirAlert) (archiveCAPRecord, bool) {
	hint := strings.ToLower(strings.Join([]string{item.AlertID, item.Event, item.Header, item.QueueID}, " "))
	for _, record := range archiveStoreRecords(configPath, "accepted", time.Time{}) {
		if item.FeedID != "" && record.FeedID != item.FeedID {
			continue
		}
		info := chooseArchiveInfo(record.Alert)
		haystack := strings.ToLower(strings.Join([]string{
			record.ID,
			record.Alert.Identifier,
			info.Event,
			info.Headline,
			sameEventFromCAP(info),
		}, " "))
		sameEvent := strings.ToLower(strings.TrimSpace(sameEventFromCAP(info)))
		if haystack != "" && hint != "" && sameEvent != "" && strings.Contains(hint, sameEvent) {
			return record, true
		}
		if haystack != "" && hint != "" && strings.Contains(hint, strings.ToLower(safeID(record.ID))) {
			return record, true
		}
	}
	return archiveCAPRecord{}, false
}

func bannerRecordFromOnAirAlert(item bannerOnAirAlert, now time.Time) archiveCAPRecord {
	if now.IsZero() {
		now = time.Now().UTC()
	}
	updated := item.UpdatedAt
	if updated.IsZero() {
		updated = now
	}
	expires := item.ExpiresAt
	if expires.IsZero() {
		expires = now.Add(30 * time.Minute)
	}
	alertID := fallbackString(item.AlertID, item.QueueID, fmt.Sprintf("on-air-%d", updated.UnixNano()))
	event := fallbackString(item.Event, "Alert")
	headline := fallbackString(item.Header, event, "Weather Alert")
	sent := updated.UTC().Format(time.RFC3339Nano)
	return archiveCAPRecord{
		ID:        alertID,
		FeedID:    item.FeedID,
		Status:    "on_air",
		UpdatedAt: updated,
		Alert: capingest.Alert{
			Identifier:  alertID,
			Sent:        sent,
			Status:      "Actual",
			MessageType: "Alert",
			Scope:       "Public",
			Infos: []capingest.AlertInfo{{
				Language:    "en-CA",
				Event:       event,
				Severity:    "Unknown",
				Urgency:     "Unknown",
				Certainty:   "Unknown",
				Effective:   sent,
				Expires:     expires.UTC().Format(time.RFC3339Nano),
				SenderName:  "Haze Weather Radio",
				Headline:    headline,
				Description: headline,
			}},
		},
	}
}

type bannerSort struct {
	severityRank int
	timestamp    time.Time
	identifier   string
}

func bannerSortKey(alert capingest.Alert) bannerSort {
	info := chooseArchiveInfo(alert)
	severity := strings.Title(strings.ToLower(strings.TrimSpace(info.Severity)))
	if severity == "" {
		severity = "Unknown"
	}
	rank := len([]string{"Extreme", "Severe", "Moderate", "Minor", "Unknown"})
	for i, value := range []string{"Extreme", "Severe", "Moderate", "Minor", "Unknown"} {
		if severity == value {
			rank = i
			break
		}
	}
	timestamp := time.Time{}
	for _, raw := range []string{info.Effective, alert.Sent, info.Onset} {
		if parsed := alerttext.ParseCAPTime(raw); !parsed.IsZero() {
			timestamp = parsed
			break
		}
	}
	return bannerSort{
		severityRank: rank,
		timestamp:    timestamp,
		identifier:   strings.TrimSpace(alert.Identifier),
	}
}

func (key bannerSort) less(other bannerSort) bool {
	if key.severityRank != other.severityRank {
		return key.severityRank < other.severityRank
	}
	if !key.timestamp.Equal(other.timestamp) {
		return key.timestamp.After(other.timestamp)
	}
	return key.identifier < other.identifier
}

func bannerFeedName(configPath string, feedID string) string {
	if feedID == "" {
		return ""
	}
	feeds, err := loadFeedSummaries(configPath)
	if err != nil {
		return feedID
	}
	for _, feed := range feeds {
		if stringValue(feed, "id") == feedID {
			return fallbackString(stringValue(feed, "name"), feedID)
		}
	}
	return feedID
}

func bannerFeedTimezone(configPath string, feedID string) string {
	if feedID == "" {
		return "Local"
	}
	feeds, err := loadFeedSummaries(configPath)
	if err != nil {
		return "Local"
	}
	for _, feed := range feeds {
		if stringValue(feed, "id") == feedID {
			return fallbackString(stringValue(feed, "timezone"), "Local")
		}
	}
	return "Local"
}

func bannerSignature(payload bannerPayload) string {
	hash := sha1.New()
	_, _ = fmt.Fprintf(hash, "%t|%s|%s|", payload.Active, payload.FeedID, payload.FeedName)
	for _, alert := range payload.Alerts {
		_, _ = fmt.Fprintf(hash, "%s|%s|%s|%s|%s|", alert.Identifier, alert.FeedID, alert.Headline, alert.ExpiresAt, alert.Message)
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func bannerPayloadForTest(configPath string, feedID string, now time.Time) bannerPayload {
	records := activeBannerRecords(configPath, feedID, now)
	alerts := make([]alerttext.SerializedAlert, 0, len(records))
	for _, record := range records {
		info := chooseArchiveInfo(record.Alert)
		alerts = append(alerts, alerttext.SerializeCAPAlert(record.Alert, info, record.FeedID, archiveAreaNames(info), "UTC", "cap", now))
	}
	payload := bannerPayload{Active: len(alerts) > 0, FeedID: feedID, GeneratedAt: now, PrimaryGradient: []string{"#019310", "#0b3810"}, PrimaryColor: "#019310", Alerts: alerts}
	payload.Signature = bannerSignature(payload)
	return payload
}
