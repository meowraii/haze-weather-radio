package datastore

import (
	"encoding/json"
	"strings"
)

func mergeForecastPayloadTimes(raw []byte, issuedAt string, updatedAt string, target any) error {
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil || payload == nil {
		return json.Unmarshal(raw, target)
	}
	if strings.TrimSpace(issuedAt) != "" && blankJSONText(payload["issued_at"]) {
		payload["issued_at"] = strings.TrimSpace(issuedAt)
	}
	if strings.TrimSpace(updatedAt) != "" && blankJSONText(payload["updated_at"]) {
		payload["updated_at"] = strings.TrimSpace(updatedAt)
	}
	merged, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return json.Unmarshal(merged, target)
}

func blankJSONText(value any) bool {
	switch v := value.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(v) == ""
	default:
		return false
	}
}
