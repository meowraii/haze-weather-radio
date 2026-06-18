package webgateway

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

func stateSignature(state map[string]any) string {
	normalized := cloneMap(state)
	removePath(normalized, "summary", "uptime_seconds")
	removePath(normalized, "last_connected", "at")
	if events, ok := normalized["events"].([]any); ok {
		for _, event := range events {
			if eventMap, ok := event.(map[string]any); ok {
				delete(eventMap, "timestamp")
			}
		}
	}
	raw, err := json.Marshal(normalized)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func removePath(root map[string]any, path ...string) {
	if len(path) == 0 {
		return
	}
	cursor := root
	for _, part := range path[:len(path)-1] {
		next, ok := cursor[part].(map[string]any)
		if !ok {
			return
		}
		cursor = next
	}
	delete(cursor, path[len(path)-1])
}
