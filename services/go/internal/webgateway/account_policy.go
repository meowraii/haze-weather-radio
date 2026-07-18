package webgateway

import (
	"net/http"
	"strings"
)

func nonAdminCommandAllowed(command string, payload map[string]any, account Account) bool {
	switch command {
	case "state", "health", "same.event_codes", "same.location_names", "same.templates.get",
		"same.intro", "same.generate", "same.test", "same.air", "alert.broadcast", "alert.preview",
		"profile.get", "profile.password.change", "profile.sessions", "profile.session.revoke":
		return true
	case "alerts.archive.get", "logs.list", "logs.tail":
		return account.CanViewLogs
	case "alerts.archive.action":
		action := strings.ToLower(strings.TrimSpace(stringValue(payload, "action")))
		return account.CanViewLogs && (archiveActionBroadcasts(payload) || action == "preview_same")
	default:
		return false
	}
}

func isOriginationExecution(command string, payload map[string]any) bool {
	switch command {
	case "same.test", "same.air", "alert.broadcast":
		return true
	case "playlist.insert":
		return strings.EqualFold(stringValue(payload, "kind"), "same")
	case "alerts.archive.action":
		return archiveActionBroadcasts(payload)
	default:
		return false
	}
}

func usesOriginationPolicy(command string, payload map[string]any) bool {
	if isOriginationExecution(command, payload) || command == "same.generate" || command == "alert.preview" {
		return true
	}
	return command == "alerts.archive.action" && strings.EqualFold(stringValue(payload, "action"), "preview_same")
}

func archiveActionBroadcasts(payload map[string]any) bool {
	switch strings.ToLower(strings.TrimSpace(stringValue(payload, "action"))) {
	case "rebroadcast", "force_broadcast", "rebroadcast_without_same", "force_broadcast_without_same":
		return true
	default:
		return false
	}
}

func prepareAccountOriginationPolicyPayload(configPath string, command string, payload map[string]any) (map[string]any, error) {
	out := cloneMap(payload)
	if command == "same.test" {
		if template, ok := out["template"].(map[string]any); ok {
			same := mapFromAny(template["same"])
			if strings.TrimSpace(stringValue(out, "originator")) == "" {
				out["originator"] = strings.ToUpper(firstNonBlank(stringFromMap(same, "originator"), "WXR"))
			}
			if strings.TrimSpace(firstNonBlank(stringValue(out, "same_event"), stringValue(out, "event"), stringValue(out, "event_code"))) == "" {
				eventCode := strings.ToUpper(firstNonBlank(stringFromMap(same, "event"), stringValue(template, "sameEvent")))
				if eventCode != "" {
					out["event"] = eventCode
					out["same_event"] = eventCode
					out["event_code"] = eventCode
				}
			}
		}
	}
	if command != "alerts.archive.action" {
		return out, nil
	}
	action := strings.ToLower(strings.TrimSpace(stringValue(out, "action")))
	if !archiveActionBroadcasts(out) && action != "preview_same" {
		return out, nil
	}
	record, ok := findArchiveAlert(configPath, strings.TrimSpace(stringValue(out, "id")), strings.TrimSpace(stringValue(out, "feed_id")))
	if !ok {
		return nil, &AuthError{Code: "archive_alert_not_found", Detail: "The archived alert was not found.", HTTPStatus: http.StatusNotFound}
	}
	originator := archiveRecordOriginator(record)
	out["originator"] = originator
	out["same_originator"] = originator
	if eventCode := archiveRecordEvent(configPath, record); eventCode != "" {
		out["event"] = eventCode
		out["same_event"] = eventCode
		out["event_code"] = eventCode
	}
	return out, nil
}

func applyAccountOriginationPolicy(payload map[string]any, identity Identity) (map[string]any, error) {
	out := cloneMap(payload)
	defaultOriginator := "WXR"
	if _, has := out["same_event"]; has || stringValue(out, "audio_mode") != "" {
		defaultOriginator = "EAS"
	}
	originator := strings.ToUpper(strings.TrimSpace(firstNonBlank(
		stringValue(out, "originator"),
		stringValue(out, "same_originator"),
		defaultOriginator,
	)))
	allowed := false
	for _, code := range identity.Account.AllowedOriginators {
		if code == originator {
			allowed = true
			break
		}
	}
	if !allowed {
		return nil, &AuthError{
			Code:       "originator_forbidden",
			Detail:     "Originator " + originator + " is not permitted for this account.",
			HTTPStatus: http.StatusForbidden,
		}
	}
	eventCode := strings.ToUpper(strings.TrimSpace(firstNonBlank(
		stringValue(out, "same_event"),
		stringValue(out, "event"),
		stringValue(out, "event_code"),
	)))
	for _, blocked := range identity.Account.BlockedEventCodes {
		if eventCode != "" && strings.EqualFold(blocked, eventCode) {
			return nil, &AuthError{
				Code:       "event_code_forbidden",
				Detail:     "SAME event code " + eventCode + " is blocked for this account.",
				HTTPStatus: http.StatusForbidden,
			}
		}
	}
	out["originator"] = originator
	out["same_originator"] = originator
	if identity.Account.ForceSenderID {
		out["sender_id"] = identity.Account.SenderID
		out["same_callsign"] = identity.Account.SenderID
	}
	if identity.Account.ForceOriginatorName {
		name := identity.Account.OriginatorNameText
		if identity.Account.IncludeIPInBrackets {
			name += " [" + identity.Session.IP + "]"
		}
		out["originator_name"] = name
		out["same_originator_name"] = name
	}
	out["originated_by_user_id"] = identity.Account.ID
	out["originated_by_username"] = identity.Account.Username
	out["originated_by_session_id"] = identity.Session.ID
	out["originated_from_ip"] = identity.Session.IP
	return out, nil
}
