package ivr

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alerttext"
	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
)

const maxIVRAlertMenuOptions = 9

type ivrActiveAlert struct {
	ID        string
	FeedID    string
	UpdatedAt time.Time
	Alert     capmodel.Alert
	Info      capmodel.AlertInfo
	Title     string
	Score     ivrAlertScore
}

type ivrAlertScore struct {
	Event     int
	Impact    int
	Severity  int
	Urgency   int
	Certainty int
	UpdatedAt time.Time
}

func (s *Service) activeIVRAlerts(ctx context.Context, location ResolvedLocation) []ivrActiveAlert {
	if s == nil || s.store == nil {
		return nil
	}
	feedID := strings.TrimSpace(location.FeedID)
	if feedID == "" {
		return nil
	}
	queryCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	rows, err := s.store.ListCAPArchives(queryCtx, "accepted", time.Time{})
	if err != nil {
		log.Printf("IVR active alert lookup failed for feed %s: %v", feedID, err)
		return nil
	}
	now := time.Now().UTC()
	alerts := make([]ivrActiveAlert, 0, len(rows))
	for _, row := range rows {
		if !strings.EqualFold(strings.TrimSpace(row.FeedID), feedID) {
			continue
		}
		alert, err := capmodel.ParseCAP([]byte(row.RawXML))
		if err != nil || strings.TrimSpace(alert.Identifier) == "" {
			continue
		}
		if alert.RawXML == "" {
			alert.RawXML = row.RawXML
		}
		if alerttext.IsCAPEnded(alert, now) {
			continue
		}
		info, ok := chooseIVRAlertInfo(alert, location.Language)
		if !ok {
			continue
		}
		updated := row.UpdatedAt
		if updated.IsZero() {
			updated = row.StoredAt
		}
		title := alerttext.NormalizeHeadline(firstNonBlank(alerttext.AlertSubject(info), info.Headline, info.Event, "Alert"))
		alerts = append(alerts, ivrActiveAlert{
			ID:        firstNonBlank(row.AlertID, alert.Identifier),
			FeedID:    row.FeedID,
			UpdatedAt: updated,
			Alert:     alert,
			Info:      info,
			Title:     title,
			Score:     ivrAlertPriority(alert, info, updated),
		})
	}
	sortIVRAlerts(alerts)
	return alerts
}

func chooseIVRAlertInfo(alert capmodel.Alert, language string) (capmodel.AlertInfo, bool) {
	if len(alert.Infos) == 0 {
		return capmodel.AlertInfo{}, false
	}
	want := strings.ToLower(strings.TrimSpace(language))
	if want != "" {
		for _, info := range alert.Infos {
			if strings.EqualFold(strings.TrimSpace(info.Language), want) {
				return info, true
			}
		}
		wantBase := strings.SplitN(want, "-", 2)[0]
		for _, info := range alert.Infos {
			haveBase := strings.SplitN(strings.ToLower(strings.TrimSpace(info.Language)), "-", 2)[0]
			if wantBase != "" && haveBase == wantBase {
				return info, true
			}
		}
	}
	for _, info := range alert.Infos {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(info.Language)), "en") {
			return info, true
		}
	}
	return alert.Infos[0], true
}

func sortIVRAlerts(alerts []ivrActiveAlert) {
	sort.SliceStable(alerts, func(i int, j int) bool {
		left := alerts[i].Score
		right := alerts[j].Score
		for _, cmp := range []struct {
			left  int
			right int
		}{
			{left.Event, right.Event},
			{left.Impact, right.Impact},
			{left.Severity, right.Severity},
			{left.Urgency, right.Urgency},
			{left.Certainty, right.Certainty},
		} {
			if cmp.left != cmp.right {
				return cmp.left > cmp.right
			}
		}
		if !left.UpdatedAt.Equal(right.UpdatedAt) {
			return left.UpdatedAt.After(right.UpdatedAt)
		}
		if !strings.EqualFold(alerts[i].Title, alerts[j].Title) {
			return strings.ToLower(alerts[i].Title) < strings.ToLower(alerts[j].Title)
		}
		return alerts[i].ID < alerts[j].ID
	})
}

func ivrAlertPriority(alert capmodel.Alert, info capmodel.AlertInfo, updated time.Time) ivrAlertScore {
	return ivrAlertScore{
		Event:     ivrEventPriority(alert, info),
		Impact:    ivrImpactPriority(info),
		Severity:  ivrSeverityPriority(info.Severity),
		Urgency:   ivrUrgencyPriority(info.Urgency),
		Certainty: ivrCertaintyPriority(info.Certainty),
		UpdatedAt: updated,
	}
}

func ivrEventPriority(alert capmodel.Alert, info capmodel.AlertInfo) int {
	text := strings.ToLower(strings.Join([]string{
		info.Event,
		info.Headline,
		alerttext.AlertSubject(info),
		strings.Join(alert.Code, " "),
	}, " "))
	switch {
	case strings.Contains(text, "tornado") && strings.Contains(text, "warning"):
		return 1000
	case strings.Contains(text, "tornado") && strings.Contains(text, "watch"):
		return 950
	case strings.Contains(text, "severe thunderstorm") && strings.Contains(text, "warning"):
		return 900
	case strings.Contains(text, "flash flood") && strings.Contains(text, "warning"):
		return 880
	case strings.Contains(text, "severe thunderstorm") && strings.Contains(text, "watch"):
		return 850
	case strings.Contains(text, "warning"):
		return 800
	case strings.Contains(text, "watch"):
		return 700
	case strings.Contains(text, "advisory"):
		return 500
	case strings.Contains(text, "statement"):
		return 350
	default:
		return 100
	}
}

func ivrImpactPriority(info capmodel.AlertInfo) int {
	impact := strings.ToLower(firstNonBlank(
		alerttext.CAPParam(info, "layer:EC-MSC-SMC:1.1:MSC_Impact"),
		alerttext.CAPParam(info, "layer:EC-MSC-SMC:1.0:MSC_Impact"),
		alerttext.CAPParam(info, "impact"),
	))
	switch {
	case strings.Contains(impact, "extreme"), strings.Contains(impact, "severe"), strings.Contains(impact, "high"):
		return 400
	case strings.Contains(impact, "moderate"):
		return 300
	case strings.Contains(impact, "minor"), strings.Contains(impact, "low"):
		return 200
	default:
		return 0
	}
}

func ivrSeverityPriority(value string) int {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "extreme":
		return 400
	case "severe":
		return 300
	case "moderate":
		return 200
	case "minor":
		return 100
	default:
		return 0
	}
}

func ivrUrgencyPriority(value string) int {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "immediate":
		return 300
	case "expected":
		return 200
	case "future":
		return 100
	default:
		return 0
	}
}

func ivrCertaintyPriority(value string) int {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "observed":
		return 300
	case "likely":
		return 200
	case "possible":
		return 100
	default:
		return 0
	}
}

func (s *Service) locationMenuAlertText(location ResolvedLocation, alertCount int) string {
	values := s.promptValues(map[string]string{
		"location":           spokenLocationName(location),
		"feed_id":            location.FeedID,
		"alert_count_phrase": ivrAlertCountPhrase(alertCount),
	})
	if lineKey := s.locationMenuAlertLine(location); lineKey != "" {
		if text := s.cfg.Prompts.MenuLine("location_menu", lineKey, values); text != "" {
			return text
		}
	}
	normal := s.cfg.Prompts.MenuLine("location_menu", s.locationMenuMainLine(location), values)
	options := ivrLocationMenuOptions(normal)
	if options == "" {
		options = "1 for regional observations, 2 for your 7 day outlook, 3 for air quality indices, 4 for the climate summary, 5 for the thunderstorm outlook, or 6 for specialty products."
	}
	return fmt.Sprintf("%s has %s in effect. Please press the asterisk key, or %s", spokenLocationName(location), ivrAlertCountPhrase(alertCount), options)
}

func (s *Service) locationMenuAlertLine(location ResolvedLocation) string {
	if s.locationBroadcastAvailable(location) {
		if _, ok := s.cfg.Prompts.Line("location_menu", "main_alerts"); ok {
			return "main_alerts"
		}
		return ""
	}
	if _, ok := s.cfg.Prompts.Line("location_menu", "main_no_broadcast_alerts"); ok {
		return "main_no_broadcast_alerts"
	}
	if _, ok := s.cfg.Prompts.Line("location_menu", "main_alerts"); ok {
		return "main_alerts"
	}
	return ""
}

func ivrLocationMenuOptions(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	lower := strings.ToLower(text)
	if index := strings.Index(lower, "1 for "); index >= 0 {
		return strings.TrimSpace(text[index:])
	}
	if index := strings.Index(lower, "press 1 "); index >= 0 {
		return strings.TrimSpace(text[index+len("press "):])
	}
	return ""
}

func ivrAlertCountPhrase(count int) string {
	if count == 1 {
		return "1 alert"
	}
	return fmt.Sprintf("%d alerts", count)
}

func (s *Service) alertMenuText(location ResolvedLocation, alerts []ivrActiveAlert) string {
	if len(alerts) == 0 {
		return fmt.Sprintf("%s has no active alerts in effect.", spokenLocationName(location))
	}
	limit := minInt(len(alerts), maxIVRAlertMenuOptions)
	parts := []string{fmt.Sprintf("%s alert menu. The most critical alerts are listed first.", spokenLocationName(location))}
	for index := 0; index < limit; index++ {
		parts = append(parts, fmt.Sprintf("Press %d for %s.", index+1, ivrAlertMenuLabel(alerts[index])))
	}
	if len(alerts) > limit {
		parts = append(parts, fmt.Sprintf("%d additional lower priority alerts are not listed in this phone menu.", len(alerts)-limit))
	}
	parts = append(parts, "Press pound to return to the location menu.")
	return strings.Join(parts, " ")
}

func ivrAlertMenuLabel(alert ivrActiveAlert) string {
	label := firstNonBlank(alert.Title, alert.Info.Headline, alert.Info.Event, "Alert")
	impact := strings.ToLower(firstNonBlank(
		alerttext.CAPParam(alert.Info, "layer:EC-MSC-SMC:1.1:MSC_Impact"),
		alerttext.CAPParam(alert.Info, "layer:EC-MSC-SMC:1.0:MSC_Impact"),
		alert.Info.Severity,
	))
	if impact != "" && impact != "unknown" {
		return label + ", " + impact + " impact"
	}
	return label
}

func (s *Service) alertReadoutText(location ResolvedLocation, alert ivrActiveAlert) string {
	info := alert.Info
	areas := ivrAlertAreaNames(info)
	areaText := alerttext.JoinParts(areas)
	if len(areas) > 6 || len(areaText) > 360 {
		areaText = spokenLocationName(location) + " area"
	}
	text := alerttext.BuildCAPAlertText(alerttext.CAPMessageRequest{
		Alert:     alert.Alert,
		Info:      info,
		AreaText:  areaText,
		Sender:    firstNonBlank(info.SenderName, alert.Alert.Sender),
		EventName: firstNonBlank(alerttext.AlertSubject(info), info.Event, alert.Title),
		Timezone:  firstNonBlank(location.Timezone, "UTC"),
		Now:       time.Now().UTC(),
		UpdatedAt: alert.UpdatedAt,
	})
	if strings.TrimSpace(text) == "" {
		return firstNonBlank(info.Description, info.Headline, alert.Title, "Alert details are not available.")
	}
	return text
}

func ivrAlertAreaNames(info capmodel.AlertInfo) []string {
	out := []string{}
	seen := map[string]struct{}{}
	add := func(value string) {
		value = alerttext.CleanFragment(value)
		if value == "" {
			return
		}
		key := strings.ToLower(value)
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		out = append(out, value)
	}
	for _, area := range info.Areas {
		add(area.Description)
	}
	return out
}

func ivrAlertByDigit(alerts []ivrActiveAlert, digit string) (ivrActiveAlert, bool) {
	index, err := strconv.Atoi(strings.TrimSpace(digit))
	if err != nil || index < 1 || index > len(alerts) || index > maxIVRAlertMenuOptions {
		return ivrActiveAlert{}, false
	}
	return alerts[index-1], true
}

func (s *Service) textPromptAudio(ctx context.Context, lineKey string, text string) (CachedAudio, error) {
	if s == nil || s.cache == nil {
		return CachedAudio{}, fmt.Errorf("IVR prompt cache is unavailable")
	}
	return s.cache.GetTextPromptWithPolicy(ctx, "dynamic", safeID(lineKey), text, s.staticPromptPolicy(), false)
}
