package webgateway

import (
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alerttext"
)

type alertIntroRequest = alerttext.SAMERequest

func sameIntroPayload(configPath string, payload map[string]any) (map[string]any, error) {
	request := alertIntroRequestFromPayload(configPath, payload)
	return map[string]any{
		"intro":       alerttext.BuildSAMETranslation(request),
		"event":       request.Event,
		"event_name":  request.EventName,
		"originator":  request.Originator,
		"area_names":  request.AreaNames,
		"callsign":    request.Callsign,
		"expires_at":  optionalTimeString(request.ExpiresAt),
		"begins_at":   optionalTimeString(request.BeginsAt),
		"mimic_endec": request.MimicENDEC,
	}, nil
}

func alertIntroRequestFromPayload(configPath string, payload map[string]any) alertIntroRequest {
	originator := strings.ToUpper(strings.TrimSpace(stringPayload(payload, "originator", "EAS")))
	if len(originator) > 3 {
		originator = originator[:3]
	}
	event := strings.ToUpper(strings.TrimSpace(firstNonBlank(
		stringPayload(payload, "same_event", ""),
		stringPayload(payload, "event", "ADR"),
	)))
	if len(event) > 3 {
		event = event[:3]
	}
	callsign := strings.TrimSpace(firstNonBlank(
		stringPayload(payload, "callsign", ""),
		sameCallsignFromConfig(configPath, stringPayload(payload, "feed_id", "")),
	))
	sentAt := parseOptionalTime(stringPayload(payload, "sent_at", ""))
	if sentAt.IsZero() {
		sentAt = time.Now()
	}
	beginsAt := parseOptionalTime(stringPayload(payload, "schedule_at", ""))
	if beginsAt.IsZero() {
		beginsAt = parseOptionalTime(stringPayload(payload, "begins_at", ""))
	}
	expiresAt := parseOptionalTime(stringPayload(payload, "expires_at", ""))
	if expiresAt.IsZero() {
		expiresAt = sentAt.Add(durationFromPayload(payload))
	}
	return alerttext.SAMERequest{
		Originator:  fallbackText(originator, "EAS"),
		Event:       fallbackText(event, "ADR"),
		EventName:   alerttext.EventName(configPath, event),
		Locations:   stringSlicePayload(payload, "locations"),
		AreaNames:   areaNamesForPayload(configPath, payload),
		Callsign:    fallbackText(callsign, "HAZE"),
		SentAt:      sentAt,
		ExpiresAt:   expiresAt,
		BeginsAt:    beginsAt,
		MimicENDEC:  alerttext.ResolveENDECMode(stringPayload(payload, "mimic_endec", "SAGE")),
		Description: alerttext.CleanFragment(stringPayload(payload, "description", "")),
		Instruction: alerttext.CleanFragment(stringPayload(payload, "instruction", "")),
	}
}

func buildSAMEToTextIntro(request alertIntroRequest) string {
	return alerttext.BuildSAMETranslation(request)
}

func areaNamesForPayload(configPath string, payload map[string]any) []string {
	return alerttext.ResolveAreaNames(
		configPath,
		stringListAny(payload["area_names"]),
		stringSlicePayload(payload, "locations"),
	)
}

func durationFromPayload(payload map[string]any) time.Duration {
	duration := sameDuration(payload)
	if len(duration) != 4 {
		return time.Hour
	}
	hours := intPayload(map[string]any{"v": duration[:2]}, "v", 1)
	minutes := intPayload(map[string]any{"v": duration[2:]}, "v", 0)
	if hours == 0 && minutes == 0 {
		minutes = 15
	}
	return time.Duration(hours)*time.Hour + time.Duration(minutes)*time.Minute
}

func parseOptionalTime(raw string) time.Time {
	text := strings.TrimSpace(raw)
	if text == "" {
		return time.Time{}
	}
	if parsed, err := time.Parse(time.RFC3339Nano, text); err == nil {
		return parsed
	}
	if parsed, err := time.Parse("2006-01-02T15:04", text); err == nil {
		return parsed
	}
	return time.Time{}
}

func optionalTimeString(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	return value.UTC().Format(time.RFC3339Nano)
}
