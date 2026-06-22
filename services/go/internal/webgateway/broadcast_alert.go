package webgateway

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

func (s *wsSession) broadcastAlert(payload map[string]any) (map[string]any, error) {
	targets, err := alertTargetFeedIDs(s.configPath, payload)
	if err != nil {
		return nil, err
	}
	includeSame := boolPayload(payload, "include_same", true)
	alertID := safeID(firstNonBlank(
		stringPayload(payload, "alert_id", ""),
		fmt.Sprintf("manual-%d", time.Now().UTC().UnixNano()),
	))
	scheduleAt := parseOptionalTime(stringPayload(payload, "schedule_at", ""))
	data := s.broadcastAlertData(payload, targets, alertID, includeSame)

	if !scheduleAt.IsZero() && scheduleAt.After(time.Now()) {
		delay := time.Until(scheduleAt)
		configPath := s.configPath
		go func() {
			timer := time.NewTimer(delay)
			defer timer.Stop()
			<-timer.C
			_ = publishAlertBroadcast(configPath, targets, data)
		}()
		return map[string]any{
			"scheduled":    true,
			"schedule_at":  scheduleAt.UTC().Format(time.RFC3339Nano),
			"alert_id":     alertID,
			"feed_ids":     targets,
			"include_same": includeSame,
			"intro":        data["same_intro"],
			"message":      "Alert scheduled",
		}, nil
	}

	if err := publishAlertBroadcast(s.configPath, targets, data); err != nil {
		return nil, err
	}
	return map[string]any{
		"queued":       true,
		"alert_id":     alertID,
		"feed_ids":     targets,
		"include_same": includeSame,
		"intro":        data["same_intro"],
		"message":      "Alert broadcast requested",
	}, nil
}

func (s *wsSession) broadcastAlertData(payload map[string]any, targets []string, alertID string, includeSame bool) map[string]any {
	primaryFeed := ""
	if len(targets) > 0 {
		primaryFeed = targets[0]
	}
	sameLocations := expandSameLocationsForFeeds(s.configPath, targets, stringSlicePayload(payload, "locations"))
	introPayload := withFeedFallback(payload, primaryFeed)
	introPayload["locations"] = sameLocations
	introRequest := alertIntroRequestFromPayload(s.configPath, introPayload)
	intro := buildSAMEToTextIntro(introRequest)
	event := strings.ToUpper(firstNonBlank(stringPayload(payload, "same_event", ""), stringPayload(payload, "event", "ADR")))
	title := strings.TrimSpace(firstNonBlank(
		stringPayload(payload, "title", ""),
		fmt.Sprintf("%s - %s", event, introRequest.EventName),
	))
	customText := strings.TrimSpace(firstNonBlank(
		stringPayload(payload, "alert_text", ""),
		stringPayload(payload, "voice_message", ""),
		stringPayload(payload, "text", ""),
	))
	prependIntro := boolPayload(payload, "prepend_same_translation", false)
	baseSpeech := manualAlertSpeechText(customText, stringPayload(payload, "description", ""), stringPayload(payload, "instruction", ""), title, introRequest.EventName)
	alertText := baseSpeech
	if prependIntro {
		alertText = strings.TrimSpace(strings.Join(nonEmptyManualAlertParts(intro, baseSpeech), " "))
	}
	bannerText := bannerTextFromManualAlert(intro, customText, stringPayload(payload, "description", ""), stringPayload(payload, "instruction", ""))
	data := map[string]any{
		"feed_ids":                 targets,
		"alert_id":                 alertID,
		"message_type":             "Alert",
		"title":                    title,
		"event":                    event,
		"alert_text":               alertText,
		"banner_text":              bannerText,
		"description":              stringPayload(payload, "description", ""),
		"instruction":              stringPayload(payload, "instruction", ""),
		"include_same":             includeSame,
		"same_intro":               intro,
		"same_translation":         intro,
		"same_event":               event,
		"same_originator":          strings.ToUpper(stringPayload(payload, "originator", "EAS")),
		"same_locations":           sameLocations,
		"same_duration":            sameDuration(payload),
		"same_tone":                strings.ToUpper(stringPayload(payload, "tone_type", "WXR")),
		"same_callsign":            sameCallsignFromConfig(s.configPath, primaryFeed),
		"alert_sent_at":            time.Now().UTC().Format(time.RFC3339Nano),
		"source":                   "webpanel",
		"prepend_same_translation": prependIntro,
	}
	if scheduleAt := parseOptionalTime(stringPayload(payload, "schedule_at", "")); !scheduleAt.IsZero() {
		data["scheduled_for"] = scheduleAt.UTC().Format(time.RFC3339Nano)
	}
	return data
}

func manualAlertSpeechText(customText string, description string, instruction string, fallbackValues ...string) string {
	if text := strings.TrimSpace(customText); text != "" {
		return text
	}
	parts := []string{}
	for _, value := range []string{description, instruction} {
		if clean := strings.TrimSpace(value); clean != "" {
			parts = append(parts, clean)
		}
	}
	if len(parts) > 0 {
		return strings.TrimSpace(strings.Join(parts, " "))
	}
	for _, value := range fallbackValues {
		if clean := strings.TrimSpace(value); clean != "" {
			return clean
		}
	}
	return ""
}

func nonEmptyManualAlertParts(values ...string) []string {
	parts := make([]string, 0, len(values))
	for _, value := range values {
		if clean := strings.TrimSpace(value); clean != "" {
			parts = append(parts, clean)
		}
	}
	return parts
}

func bannerTextFromManualAlert(intro string, customText string, description string, instruction string) string {
	parts := []string{}
	for _, value := range []string{intro, customText, description, instruction} {
		clean := strings.TrimSpace(value)
		if clean != "" {
			parts = append(parts, clean)
		}
	}
	return strings.TrimSpace(strings.Join(parts, " "))
}

func withFeedFallback(payload map[string]any, feedID string) map[string]any {
	out := map[string]any{}
	for key, value := range payload {
		out[key] = value
	}
	if strings.TrimSpace(fmt.Sprint(out["feed_id"])) == "" {
		out["feed_id"] = feedID
	}
	return out
}

func publishAlertBroadcast(configPath string, targets []string, data map[string]any) error {
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return fmt.Errorf("event bridge is not available")
	}
	for _, feedID := range targets {
		eventData := cloneBroadcastMap(data)
		eventData["feed_id"] = feedID
		delete(eventData, "feed_ids")
		publisher := events.NewHostBridgePublisher(bridgeAddr)
		err := publisher.Publish(events.Event{
			Type:    "cap.alert.broadcast.requested",
			Source:  "haze-web",
			Subject: strings.TrimSpace(fmt.Sprint(data["alert_id"])),
			Data:    eventData,
		})
		_ = publisher.Close()
		if err != nil {
			return err
		}
	}
	_ = configPath
	return nil
}

func cloneBroadcastMap(source map[string]any) map[string]any {
	out := make(map[string]any, len(source))
	for key, value := range source {
		out[key] = value
	}
	return out
}
