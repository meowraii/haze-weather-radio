package capsame

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
)

// EventResolution explains how a CAP alert was mapped to a SAME/EAS event code.
type EventResolution struct {
	Event      string
	Source     string
	Reason     string
	Confidence string
	AlertClass string
	Phenomenon string
	Evidence   []string
}

type namedText struct {
	Name  string
	Value string
}

type hazardRule struct {
	Phenomenon string
	Terms      []string
	Watch      string
	Warning    string
	Statement  string
	Advisory   string
	Default    string
}

var hazardRules = []hazardRule{
	{
		Phenomenon: "tornado",
		Terms:      []string{"tornado"},
		Watch:      "TOA",
		Warning:    "TOR",
		Default:    "TOR",
	},
	{
		Phenomenon: "severe thunderstorm",
		Terms:      []string{"severe thunderstorm"},
		Watch:      "SVA",
		Warning:    "SVR",
		Statement:  "SVS",
		Default:    "SVR",
	},
	{
		Phenomenon: "thunderstorm",
		Terms:      []string{"thunderstorm"},
		Watch:      "SVA",
		Warning:    "SVR",
		Statement:  "SVS",
		Default:    "SVR",
	},
	{
		Phenomenon: "flash flood",
		Terms:      []string{"flash flood"},
		Watch:      "FFA",
		Warning:    "FFW",
		Statement:  "FFS",
		Default:    "FFW",
	},
	{
		Phenomenon: "flood",
		Terms:      []string{"flood"},
		Watch:      "FLA",
		Warning:    "FLW",
		Statement:  "FLS",
		Default:    "FLW",
	},
	{
		Phenomenon: "snow squall",
		Terms:      []string{"snow squall", "squall"},
		Warning:    "SQW",
		Default:    "SQW",
	},
	{
		Phenomenon: "blizzard",
		Terms:      []string{"blizzard"},
		Warning:    "BZW",
		Default:    "BZW",
	},
	{
		Phenomenon: "winter storm",
		Terms:      []string{"winter storm"},
		Watch:      "WSA",
		Warning:    "WSW",
		Default:    "WSW",
	},
	{
		Phenomenon: "hurricane",
		Terms:      []string{"hurricane"},
		Watch:      "HUA",
		Warning:    "HUW",
		Statement:  "HLS",
		Default:    "HUW",
	},
	{
		Phenomenon: "tropical storm",
		Terms:      []string{"tropical storm"},
		Watch:      "TRA",
		Warning:    "TRW",
		Statement:  "HLS",
		Default:    "TRW",
	},
	{
		Phenomenon: "storm surge",
		Terms:      []string{"storm surge"},
		Watch:      "SSA",
		Warning:    "SSW",
		Default:    "SSW",
	},
	{
		Phenomenon: "wildfire",
		Terms:      []string{"wildfire", "wild fire", "forest fire"},
		Watch:      "WFA",
		Warning:    "WFW",
		Default:    "WFW",
	},
}

// ResolveEvent maps a CAP alert info block to the best SAME/EAS event code.
func ResolveEvent(alert capmodel.Alert, info capmodel.AlertInfo, baseDir string) EventResolution {
	_ = alert
	if resolution := explicitCAPEvent(info); resolution.Event != "" {
		return resolution
	}

	alertClass, classEvidence := detectAlertClass(info)
	if resolution := classifiedEvent(info, alertClass, classEvidence); resolution.Event != "" {
		return resolution
	}

	mapping := loadNAADSToEASMapping(baseDir)
	for _, code := range info.EventCodes {
		if mapped := mapping[normalizeEventKey(code.Value)]; mapped != "" {
			return EventResolution{
				Event:      mapped,
				Source:     "cap_event_code_mapping",
				Reason:     fmt.Sprintf("mapped CAP eventCode %s=%q through sameMapping.json", strings.TrimSpace(code.Name), strings.TrimSpace(code.Value)),
				Confidence: "medium",
				AlertClass: alertClass,
				Evidence:   compactEvidence(append(classEvidence, evidenceText("eventCode "+code.Name, code.Value))),
			}
		}
	}
	if mapped := mapping[normalizeEventKey(info.Event)]; mapped != "" {
		return EventResolution{
			Event:      mapped,
			Source:     "cap_event_mapping",
			Reason:     fmt.Sprintf("mapped CAP event %q through sameMapping.json", strings.TrimSpace(info.Event)),
			Confidence: "medium",
			AlertClass: alertClass,
			Evidence:   compactEvidence(append(classEvidence, evidenceText("event", info.Event))),
		}
	}

	if resolution := classifiedEvent(info, "", nil); resolution.Event != "" {
		resolution.Source = "cap_hazard_fallback"
		resolution.Confidence = "low"
		return resolution
	}

	return EventResolution{
		Event:      "ADR",
		Source:     "default",
		Reason:     "no explicit SAME code, CAP alert class, known hazard phrase, or NAADS mapping matched",
		Confidence: "low",
		AlertClass: alertClass,
		Evidence:   compactEvidence(capTextEvidence(info)),
	}
}

func explicitCAPEvent(info capmodel.AlertInfo) EventResolution {
	for _, code := range info.EventCodes {
		if event := explicitEventCode(code.Name, code.Value); event != "" {
			return EventResolution{
				Event:      event,
				Source:     "cap_event_code",
				Reason:     fmt.Sprintf("CAP eventCode %s supplied explicit SAME/EAS event %s", strings.TrimSpace(code.Name), event),
				Confidence: "high",
				Evidence:   []string{evidenceText("eventCode "+code.Name, code.Value)},
			}
		}
	}
	for _, param := range info.Parameters {
		if event := explicitEventCode(param.Name, param.Value); event != "" {
			return EventResolution{
				Event:      event,
				Source:     "cap_parameter",
				Reason:     fmt.Sprintf("CAP parameter %s supplied explicit SAME/EAS event %s", strings.TrimSpace(param.Name), event),
				Confidence: "high",
				Evidence:   []string{evidenceText("parameter "+param.Name, param.Value)},
			}
		}
	}
	return EventResolution{}
}

func explicitEventCode(name string, raw string) string {
	label := strings.ToLower(strings.TrimSpace(name))
	if !strings.Contains(label, "same") && !strings.Contains(label, "eas") {
		return ""
	}
	for _, token := range eventCodeTokens(raw) {
		if validSAMEEventCode(token) {
			return token
		}
	}
	return ""
}

func eventCodeTokens(raw string) []string {
	parts := strings.FieldsFunc(raw, func(ch rune) bool {
		return ch == ',' || ch == ';' || ch == '|' || ch == '/' || ch == '\\' || ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'
	})
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.ToUpper(strings.TrimSpace(part))
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func validSAMEEventCode(value string) bool {
	if len(value) != 3 || originatorCode(value) {
		return false
	}
	for _, ch := range value {
		if ch < 'A' || ch > 'Z' {
			return false
		}
	}
	return true
}

func originatorCode(value string) bool {
	switch strings.ToUpper(strings.TrimSpace(value)) {
	case "CIV", "EAS", "PEP", "WXR":
		return true
	default:
		return false
	}
}

func classifiedEvent(info capmodel.AlertInfo, alertClass string, classEvidence []string) EventResolution {
	if alertClass == "" {
		alertClass, classEvidence = detectAlertClass(info)
	}
	for _, field := range capTextFields(info) {
		text := strings.ToLower(field.Value)
		if text == "" {
			continue
		}
		for _, rule := range hazardRules {
			if !containsAnyPhrase(text, rule.Terms) {
				continue
			}
			event := rule.eventForClass(alertClass)
			if event == "" {
				continue
			}
			source := "cap_alert_class"
			confidence := "high"
			if alertClass == "" {
				source = "cap_hazard"
				confidence = "medium"
			}
			reason := fmt.Sprintf("CAP %s %q matched %s", field.Name, strings.TrimSpace(field.Value), rule.Phenomenon)
			if alertClass != "" {
				reason = fmt.Sprintf("%s with %s metadata", reason, alertClass)
			}
			return EventResolution{
				Event:      event,
				Source:     source,
				Reason:     reason,
				Confidence: confidence,
				AlertClass: alertClass,
				Phenomenon: rule.Phenomenon,
				Evidence:   compactEvidence(append(classEvidence, evidenceText(field.Name, field.Value))),
			}
		}
	}
	if alertClass == "statement" {
		return EventResolution{
			Event:      "SPS",
			Source:     "cap_alert_class",
			Reason:     "CAP metadata identified a weather statement without a more specific SAME event",
			Confidence: "medium",
			AlertClass: alertClass,
			Evidence:   compactEvidence(append(classEvidence, capTextEvidence(info)...)),
		}
	}
	return EventResolution{}
}

func (rule hazardRule) eventForClass(alertClass string) string {
	switch alertClass {
	case "watch":
		return firstNonBlank(rule.Watch, rule.Default)
	case "warning":
		return firstNonBlank(rule.Warning, rule.Default)
	case "statement":
		return firstNonBlank(rule.Statement, "SPS")
	case "advisory":
		return firstNonBlank(rule.Advisory, rule.Statement, "SPS")
	default:
		return rule.Default
	}
}

func detectAlertClass(info capmodel.AlertInfo) (string, []string) {
	fields := []namedText{
		{Name: "Alert_Type", Value: alertParamContaining(info, "alert_type")},
		{Name: "Alert_Name", Value: alertParamContaining(info, "alert_name")},
		{Name: "headline", Value: info.Headline},
		{Name: "event", Value: info.Event},
	}
	for _, class := range []string{"watch", "warning", "advisory", "statement"} {
		for _, field := range fields {
			if containsWord(field.Value, class) {
				return class, []string{evidenceText(field.Name, field.Value)}
			}
		}
	}
	return "", nil
}

func capTextFields(info capmodel.AlertInfo) []namedText {
	fields := []namedText{
		{Name: "Alert_Name", Value: alertParamContaining(info, "alert_name")},
		{Name: "headline", Value: info.Headline},
		{Name: "event", Value: info.Event},
		{Name: "Alert_Type", Value: alertParamContaining(info, "alert_type")},
	}
	for _, code := range info.EventCodes {
		fields = append(fields, namedText{Name: "eventCode " + code.Name, Value: code.Value})
	}
	return fields
}

func capTextEvidence(info capmodel.AlertInfo) []string {
	out := []string{}
	for _, field := range capTextFields(info) {
		out = append(out, evidenceText(field.Name, field.Value))
	}
	return out
}

func alertParamContaining(info capmodel.AlertInfo, token string) string {
	token = strings.ToLower(strings.TrimSpace(token))
	for _, param := range info.Parameters {
		if strings.Contains(strings.ToLower(strings.TrimSpace(param.Name)), token) {
			return strings.TrimSpace(param.Value)
		}
	}
	return ""
}

func containsAnyPhrase(text string, phrases []string) bool {
	for _, phrase := range phrases {
		if strings.Contains(text, strings.ToLower(phrase)) {
			return true
		}
	}
	return false
}

func containsWord(raw string, word string) bool {
	word = strings.ToLower(strings.TrimSpace(word))
	for _, part := range strings.FieldsFunc(strings.ToLower(raw), func(ch rune) bool {
		return (ch < 'a' || ch > 'z') && (ch < '0' || ch > '9')
	}) {
		if part == word {
			return true
		}
	}
	return false
}

func loadNAADSToEASMapping(baseDir string) map[string]string {
	path := filepath.Join(baseDir, "managed", "sameMapping.json")
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil
	}
	var payload struct {
		NAADSToEAS map[string]string `json:"naadsToEas"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil
	}
	out := make(map[string]string, len(payload.NAADSToEAS))
	for key, value := range payload.NAADSToEAS {
		value = strings.ToUpper(strings.TrimSpace(value))
		if validSAMEEventCode(value) {
			out[normalizeEventKey(key)] = value
		}
	}
	return out
}

func normalizeEventKey(raw string) string {
	var builder strings.Builder
	for _, ch := range strings.ToLower(strings.TrimSpace(raw)) {
		if ch >= 'a' && ch <= 'z' || ch >= '0' && ch <= '9' {
			builder.WriteRune(ch)
		}
	}
	return builder.String()
}

func evidenceText(name string, value string) string {
	name = strings.TrimSpace(name)
	value = strings.TrimSpace(value)
	if name == "" || value == "" {
		return ""
	}
	return name + "=" + value
}

func compactEvidence(values []string) []string {
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

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
