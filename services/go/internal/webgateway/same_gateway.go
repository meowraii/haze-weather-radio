package webgateway

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alertmodel"
)

const alertQueueDir = "runtime/queues/alerts"
const alertAudioDir = "runtime/audio/alerts"

func (s *wsSession) generateSame(payload map[string]any) (map[string]any, error) {
	request, err := buildSameRequest(s.configPath, payload)
	if err != nil {
		return nil, err
	}
	return runSameGenerator(s.configPath, request)
}

func (s *wsSession) generateSameTest(payload map[string]any) (map[string]any, error) {
	if template, ok := payload["template"].(map[string]any); ok {
		request, err := buildTemplateSameRequest(s.configPath, template, payload)
		if err != nil {
			return nil, err
		}
		targets, err := templateTargetFeedIDs(s.configPath, payload)
		if err != nil {
			return nil, err
		}
		result, err := runSameGenerator(s.configPath, request)
		if err != nil {
			return nil, err
		}
		item, err := persistSameQueueItem(s.configPath, request, targets, result, bannerTextFromTemplate(template))
		if err != nil {
			return nil, err
		}
		result["feed_id"] = targets[0]
		result["queued"] = true
		result["queue_id"] = item.ID
		result["header"] = item.Header
		result["audio_path"] = item.AudioPath
		result["manifest_path"] = item.ManifestPath
		result["message"] = "SAME template queued for priority playout"
		return result, nil
	}
	event := strings.ToUpper(strings.TrimSpace(stringPayload(payload, "event_code", "RWT")))
	feeds, err := loadFeedSummaries(s.configPath)
	if err != nil {
		return nil, err
	}
	locations := []string{}
	feedID := ""
	for _, feed := range feeds {
		if enabled, _ := feed["enabled"].(bool); !enabled {
			continue
		}
		if feedID == "" {
			feedID, _ = feed["id"].(string)
		}
		codes := anySlice(feed["same_locations"])
		if len(codes) == 0 {
			codes = anySlice(feed["clc_codes"])
		}
		for _, code := range codes {
			locations = append(locations, strings.TrimSpace(fmt.Sprint(code)))
		}
		if len(locations) > 0 {
			break
		}
	}
	if len(locations) == 0 {
		return nil, fmt.Errorf("no configured feed locations are available for SAME test generation")
	}
	request := sameGenerateRequest{
		Originator: "WXR",
		Event:      event,
		Locations:  uniqueStrings(locations),
		Duration:   "0015",
		Callsign:   s.sameCallsign(feedID),
		Tone:       "WXR",
	}
	result, err := runSameGenerator(s.configPath, request)
	if err != nil {
		return nil, err
	}
	item, err := persistSameQueueItem(s.configPath, request, []string{feedID}, result, "")
	if err != nil {
		return nil, err
	}
	result["feed_id"] = feedID
	result["queued"] = true
	result["queue_id"] = item.ID
	result["header"] = item.Header
	result["audio_path"] = item.AudioPath
	result["manifest_path"] = item.ManifestPath
	result["message"] = "SAME test queued for playout"
	return result, nil
}

func (s *wsSession) airSame(payload map[string]any) (map[string]any, error) {
	request, err := buildSameRequest(s.configPath, payload)
	if err != nil {
		return nil, err
	}
	targets, err := alertTargetFeedIDs(s.configPath, payload)
	if err != nil {
		return nil, err
	}
	result, err := runSameGenerator(s.configPath, request)
	if err != nil {
		return nil, err
	}
	item, err := persistSameQueueItem(s.configPath, request, targets, result, bannerTextFromManualAlert("", stringPayload(payload, "alert_text", ""), stringPayload(payload, "description", ""), stringPayload(payload, "instruction", "")))
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"queued":        true,
		"queue_id":      item.ID,
		"feed_id":       targets[0],
		"feeds_aired":   targets,
		"header":        item.Header,
		"event":         item.Event,
		"locations":     item.Locations,
		"audio_path":    item.AudioPath,
		"manifest_path": item.ManifestPath,
		"message":       "SAME alert queued for playout",
	}, nil
}

type sameGenerateRequest struct {
	Originator string
	Event      string
	Locations  []string
	Duration   string
	Callsign   string
	Tone       string
	Sequence   string
}

type sameQueueItem struct {
	ID                 string             `json:"id"`
	AlertID            string             `json:"alert_id,omitempty"`
	AlertPacket        *alertmodel.Packet `json:"alert_packet,omitempty"`
	Type               string             `json:"type"`
	Status             string             `json:"status"`
	CreatedAt          time.Time          `json:"created_at"`
	FeedID             string             `json:"feed_id,omitempty"`
	FeedIDs            []string           `json:"feed_ids"`
	Header             string             `json:"header"`
	Originator         string             `json:"originator"`
	Event              string             `json:"event"`
	AlertText          string             `json:"alert_text,omitempty"`
	BannerText         string             `json:"banner_text,omitempty"`
	BroadcastImmediate bool               `json:"broadcast_immediate,omitempty"`
	Locations          []string           `json:"locations"`
	Duration           string             `json:"duration"`
	Callsign           string             `json:"callsign"`
	Tone               string             `json:"tone"`
	AudioPath          string             `json:"audio_path"`
	ManifestPath       string             `json:"manifest_path"`
	Format             string             `json:"format"`
	SampleRate         int                `json:"sample_rate"`
	Channels           int                `json:"channels"`
	AudioBytes         int                `json:"audio_bytes"`
	Source             string             `json:"source"`
	Priority           string             `json:"priority"`
	Outputs            []sameOutputTarget `json:"outputs"`
	ClaimedAt          string             `json:"claimed_at,omitempty"`
	PlayedAt           string             `json:"played_at,omitempty"`
	FailedAt           string             `json:"failed_at,omitempty"`
	LastError          string             `json:"last_error,omitempty"`
}

type sameOutputTarget struct {
	FeedID  string `json:"feed_id"`
	Type    string `json:"type"`
	Address string `json:"address,omitempty"`
	Format  string `json:"format,omitempty"`
	Acodec  string `json:"acodec,omitempty"`
	URL     string `json:"url,omitempty"`
}

func buildTemplateSameRequest(configPath string, template map[string]any, payload map[string]any) (sameGenerateRequest, error) {
	same := mapFromAny(template["same"])
	event := strings.ToUpper(firstNonBlank(
		stringPayload(payload, "event_code", ""),
		stringFromMap(template, "sameEvent"),
		stringFromMap(same, "event"),
	))
	if event == "" {
		return sameGenerateRequest{}, fmt.Errorf("template SAME event is required")
	}
	locations := templateLocationIDs(same["locations"])
	if len(locations) == 0 {
		locations = stringSlicePayload(payload, "locations")
	}
	if len(locations) == 0 {
		return sameGenerateRequest{}, fmt.Errorf("template has no SAME locations")
	}
	locations = expandSameLocationsForFeeds(configPath, sameFeedIDsFromPayload(payload), locations)
	duration := stringFromMap(template, "sameExpire")
	if duration == "" {
		duration = durationFromTemplate(same["duration"])
	}
	if duration == "" {
		duration = "0015"
	}
	content := mapFromAny(same["content"])
	tone := strings.ToUpper(firstNonBlank(stringPayload(payload, "tone_type", ""), stringFromMap(content, "attention_tone"), "WXR"))
	senderID := stringFromMap(same, "sender_id")
	if senderID == "" {
		senderID = sameCallsignFromConfig(configPath, stringPayload(payload, "feed_id", ""))
	}
	return sameGenerateRequest{
		Originator: strings.ToUpper(strings.TrimSpace(stringPayload(payload, "originator", "WXR"))),
		Event:      event,
		Locations:  uniqueStrings(locations),
		Duration:   normalizeSAMEDuration(duration),
		Callsign:   senderID,
		Tone:       tone,
	}, nil
}

func bannerTextFromTemplate(template map[string]any) string {
	if text := strings.TrimSpace(stringFromMap(template, "alert_text")); text != "" {
		return text
	}
	msg := mapFromAny(template["msg"])
	if text := strings.TrimSpace(fmt.Sprint(msg["en"])); text != "" && text != "<nil>" {
		return text
	}
	same := mapFromAny(template["same"])
	content := mapFromAny(same["content"])
	langs := mapFromAny(content["lang"])
	if text := strings.TrimSpace(fmt.Sprint(langs["en"])); text != "" && text != "<nil>" {
		return text
	}
	return ""
}

func buildSameRequest(configPath string, payload map[string]any) (sameGenerateRequest, error) {
	feedID := stringPayload(payload, "feed_id", "")
	originator := strings.ToUpper(strings.TrimSpace(stringPayload(payload, "originator", "WXR")))
	event := strings.ToUpper(strings.TrimSpace(stringPayload(payload, "event", "RWT")))
	locations := stringSlicePayload(payload, "locations")
	if len(locations) == 0 {
		return sameGenerateRequest{}, fmt.Errorf("at least one SAME location is required")
	}
	locations = expandSameLocationsForFeeds(configPath, sameFeedIDsFromPayload(payload), locations)
	request := sameGenerateRequest{
		Originator: originator,
		Event:      event,
		Locations:  locations,
		Duration:   sameDuration(payload),
		Callsign:   sameCallsignFromConfig(configPath, feedID),
		Tone:       strings.ToUpper(strings.TrimSpace(stringPayload(payload, "tone_type", "WXR"))),
	}
	if request.Tone == "" {
		request.Tone = "WXR"
	}
	return request, nil
}

func sameFeedIDsFromPayload(payload map[string]any) []string {
	feedIDs := stringListAny(payload["feed_ids"])
	if feedID := strings.TrimSpace(stringPayload(payload, "feed_id", "")); feedID != "" {
		feedIDs = append(feedIDs, feedID)
	}
	return uniqueStrings(feedIDs)
}

func expandSameLocationsForFeeds(configPath string, feedIDs []string, locations []string) []string {
	selected := map[string]struct{}{}
	out := []string{}
	add := func(raw string) {
		code := cleanLocationCode(raw)
		if code == "" {
			return
		}
		if code == "000000" {
			selected = map[string]struct{}{"000000": {}}
			out = []string{"000000"}
			return
		}
		if _, national := selected["000000"]; national {
			return
		}
		if _, ok := selected[code]; ok {
			return
		}
		selected[code] = struct{}{}
		out = append(out, code)
	}
	for _, location := range locations {
		add(location)
	}
	if len(selected) == 0 {
		return nil
	}

	clcNames := loadCLCNames(resolveConfigPath(configPath, "managed/csv/CLC_Base_Zone.csv"))
	for _, location := range locations {
		for _, subregion := range alertWildcardSubregions(location, clcNames) {
			add(subregion)
		}
	}

	root, err := loadYAMLMap(configPath)
	if err != nil {
		sort.Strings(out)
		return out
	}
	feeds, err := loadFeedsXML(configPath, root)
	if err != nil {
		sort.Strings(out)
		return out
	}
	wantedFeeds := map[string]bool{}
	for _, feedID := range feedIDs {
		if id := strings.TrimSpace(feedID); id != "" {
			wantedFeeds[id] = true
		}
	}
	for _, feed := range feeds.Feeds {
		feedID := strings.TrimSpace(feed.ID)
		if len(wantedFeeds) > 0 && !wantedFeeds[feedID] {
			continue
		}
		for _, region := range feed.Locations.Coverage.Regions {
			parent := cleanLocationCode(region.ID)
			if parent == "" {
				continue
			}
			if _, ok := selected[parent]; !ok {
				continue
			}
			for _, subregion := range alertWildcardSubregions(parent, clcNames) {
				add(subregion)
			}
			for _, subregion := range region.Subregions {
				add(subregion.ID)
			}
		}
	}
	sort.Strings(out)
	return out
}

func templateTargetFeedIDs(configPath string, payload map[string]any) ([]string, error) {
	targets, err := targetFeedIDs(configPath, payload)
	if err == nil && len(targets) > 0 {
		return targets, nil
	}
	feeds, err := loadFeedSummaries(configPath)
	if err != nil {
		return nil, err
	}
	for _, feed := range feeds {
		if enabled, _ := feed["enabled"].(bool); !enabled {
			continue
		}
		if id := strings.TrimSpace(fmt.Sprint(feed["id"])); id != "" {
			return []string{id}, nil
		}
	}
	return nil, fmt.Errorf("no enabled feed is available for SAME template")
}

func templateLocationIDs(value any) []string {
	switch typed := value.(type) {
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			switch loc := item.(type) {
			case string:
				if id := strings.TrimSpace(loc); id != "" {
					out = append(out, id)
				}
			case map[string]any:
				if id := strings.TrimSpace(fmt.Sprint(loc["id"])); id != "" && id != "<nil>" {
					out = append(out, id)
				}
			}
		}
		return out
	case []map[string]string:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if id := strings.TrimSpace(item["id"]); id != "" {
				out = append(out, id)
			}
		}
		return out
	default:
		return nil
	}
}

func durationFromTemplate(value any) string {
	source := mapFromAny(value)
	if len(source) == 0 {
		return ""
	}
	hours := intFromMap(source, "hr", 0)
	minutes := intFromMap(source, "min", 15)
	return fmt.Sprintf("%02d%02d", clampInt(hours, 0, 99), clampInt(minutes, 0, 59))
}

func normalizeSAMEDuration(raw string) string {
	digits := strings.Builder{}
	for _, ch := range raw {
		if ch >= '0' && ch <= '9' {
			digits.WriteRune(ch)
		}
	}
	value := digits.String()
	if len(value) >= 4 {
		return value[:4]
	}
	return strings.Repeat("0", 4-len(value)) + value
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if text := strings.TrimSpace(value); text != "" {
			return text
		}
	}
	return ""
}

func (s *wsSession) sameCallsign(feedID string) string {
	return sameCallsignFromConfig(s.configPath, feedID)
}

func sameCallsignFromConfig(configPath string, feedID string) string {
	if value := strings.TrimSpace(os.Getenv("SAME_ID")); value != "" {
		return value
	}
	feeds, err := loadFeedSummaries(configPath)
	if err == nil {
		for _, feed := range feeds {
			id, _ := feed["id"].(string)
			if feedID != "" && id != feedID {
				continue
			}
			transmitter, _ := feed["transmitter"].(map[string]any)
			if callsign := strings.TrimSpace(fmt.Sprint(transmitter["callsign"])); callsign != "" {
				return callsign
			}
			if feedID == "" {
				break
			}
		}
	}
	return "HAZE"
}

func runSameGenerator(configPath string, request sameGenerateRequest) (map[string]any, error) {
	executable, err := findHostExecutable(configPath)
	if err != nil {
		return nil, err
	}
	args := []string{
		"same", "generate", "--json",
		"--originator", request.Originator,
		"--event", request.Event,
		"--locations", strings.Join(request.Locations, ","),
		"--duration", request.Duration,
		"--callsign", request.Callsign,
		"--tone", request.Tone,
	}
	if request.Sequence != "" {
		args = append(args, "--sequence", request.Sequence)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, executable, args...)
	cmd.Dir = filepath.Dir(filepath.Clean(configPath))
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	raw, err := cmd.Output()
	if ctx.Err() == context.DeadlineExceeded {
		return nil, fmt.Errorf("SAME generator timed out")
	}
	if err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail == "" {
			detail = err.Error()
		}
		return nil, fmt.Errorf("SAME generator failed: %s", detail)
	}
	var result map[string]any
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("SAME generator returned invalid JSON: %w", err)
	}
	result["path"] = ""
	result["download_url"] = ""
	return result, nil
}

func persistSameQueueItem(configPath string, request sameGenerateRequest, feedIDs []string, result map[string]any, bannerText string) (sameQueueItem, error) {
	return persistSameQueueItemWithID(configPath, "", request, feedIDs, result, bannerText)
}

func persistSameQueueItemWithID(configPath string, forcedID string, request sameGenerateRequest, feedIDs []string, result map[string]any, bannerText string) (sameQueueItem, error) {
	audioBase64, _ := result["audio_base64"].(string)
	if audioBase64 == "" {
		return sameQueueItem{}, fmt.Errorf("SAME generator returned no audio payload")
	}
	audio, err := base64.StdEncoding.DecodeString(audioBase64)
	if err != nil {
		return sameQueueItem{}, fmt.Errorf("decode SAME audio payload: %w", err)
	}
	header, _ := result["header"].(string)
	if header == "" {
		return sameQueueItem{}, fmt.Errorf("SAME generator returned no header")
	}
	id := safeID(forcedID)
	if id == "" {
		generated, err := queueItemID(request.Event)
		if err != nil {
			return sameQueueItem{}, err
		}
		id = generated
	}
	audioRel := filepath.ToSlash(filepath.Join(alertAudioDir, id+".pcm16le"))
	manifestRel := filepath.ToSlash(filepath.Join(alertQueueDir, id+".json"))
	audioPath := resolveConfigPath(configPath, audioRel)
	manifestPath := resolveConfigPath(configPath, manifestRel)
	if err := os.MkdirAll(filepath.Dir(audioPath), 0o755); err != nil {
		return sameQueueItem{}, err
	}
	if err := os.MkdirAll(filepath.Dir(manifestPath), 0o755); err != nil {
		return sameQueueItem{}, err
	}
	if err := writeFileAtomic(audioPath, audio, 0o600); err != nil {
		return sameQueueItem{}, err
	}
	outputs, err := outputTargetsForFeeds(configPath, feedIDs)
	if err != nil {
		return sameQueueItem{}, err
	}
	item := sameQueueItem{
		ID: id,
		AlertPacket: &alertmodel.Packet{
			ID:      id,
			Source:  "webpanel",
			FeedIDs: feedIDs,
			Content: alertmodel.Content{
				Event: request.Event,
			},
			SAME: &alertmodel.SAME{
				Include:    true,
				Event:      request.Event,
				Originator: request.Originator,
				Locations:  request.Locations,
				Duration:   request.Duration,
				Callsign:   request.Callsign,
				Tone:       request.Tone,
				Header:     header,
			},
			Audio: &alertmodel.Audio{
				Path:       audioRel,
				Format:     stringPayload(result, "format", "raw"),
				SampleRate: intPayload(result, "sample_rate", 48000),
				Channels:   intPayload(result, "channels", 1),
				Bytes:      len(audio),
				Source:     "webpanel",
			},
			Presentation: alertmodel.Presentation{
				BannerText: strings.TrimSpace(bannerText),
			},
		},
		Type:         "same_alert",
		Status:       "pending",
		CreatedAt:    time.Now().UTC(),
		FeedIDs:      feedIDs,
		Header:       header,
		Originator:   request.Originator,
		Event:        request.Event,
		BannerText:   strings.TrimSpace(bannerText),
		Locations:    request.Locations,
		Duration:     request.Duration,
		Callsign:     request.Callsign,
		Tone:         request.Tone,
		AudioPath:    audioRel,
		ManifestPath: manifestRel,
		Format:       stringPayload(result, "format", "raw"),
		SampleRate:   intPayload(result, "sample_rate", 48000),
		Channels:     intPayload(result, "channels", 1),
		AudioBytes:   len(audio),
		Source:       "webpanel",
		Priority:     "alert",
		Outputs:      outputs,
	}
	raw, err := json.MarshalIndent(item, "", "  ")
	if err != nil {
		return sameQueueItem{}, err
	}
	if err := writeFileAtomic(manifestPath, append(raw, '\n'), 0o600); err != nil {
		return sameQueueItem{}, err
	}
	return item, nil
}

func loadAlertQueueItems(configPath string) ([]sameQueueItem, error) {
	queuePath := resolveConfigPath(configPath, alertQueueDir)
	entries, err := os.ReadDir(queuePath)
	if err != nil {
		if os.IsNotExist(err) {
			return []sameQueueItem{}, nil
		}
		return nil, err
	}
	items := []sameQueueItem{}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".json") {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(queuePath, entry.Name()))
		if err != nil {
			continue
		}
		var item sameQueueItem
		if err := json.Unmarshal(raw, &item); err != nil {
			continue
		}
		items = append(items, item)
	}
	return items, nil
}

func alertQueueState(items []sameQueueItem, feedID string) (int, []map[string]any, *sameQueueItem) {
	matching := make([]sameQueueItem, 0, len(items))
	for _, item := range items {
		if itemTargetsFeed(item, feedID) {
			matching = append(matching, item)
		}
	}
	sortSameQueueItems(matching)
	activeDepth := 0
	for _, item := range matching {
		if activeQueueStatus(item.Status) {
			activeDepth++
		}
	}
	recent := make([]map[string]any, 0, minInt(len(matching), 5))
	for i := 0; i < len(matching) && i < 5; i++ {
		item := matching[i]
		recent = append(recent, map[string]any{
			"id":         item.ID,
			"event":      item.Event,
			"header":     item.Header,
			"created_at": item.CreatedAt,
			"status":     item.Status,
			"last_error": item.LastError,
		})
	}
	var latest *sameQueueItem
	if len(matching) > 0 {
		latest = &matching[0]
	}
	return activeDepth, recent, latest
}

func activeQueueStatus(status string) bool {
	switch strings.ToLower(strings.TrimSpace(status)) {
	case "", "pending", "queued", "playing":
		return true
	default:
		return false
	}
}

func itemTargetsFeed(item sameQueueItem, feedID string) bool {
	for _, id := range queueItemFeedIDs(item) {
		if id == feedID {
			return true
		}
	}
	return false
}

func queueItemFeedIDs(item sameQueueItem) []string {
	out := []string{}
	seen := map[string]struct{}{}
	add := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		if _, ok := seen[value]; ok {
			return
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	add(item.FeedID)
	for _, id := range item.FeedIDs {
		add(id)
	}
	return out
}

func sortSameQueueItems(items []sameQueueItem) {
	for i := 1; i < len(items); i++ {
		for j := i; j > 0 && items[j].CreatedAt.After(items[j-1].CreatedAt); j-- {
			items[j], items[j-1] = items[j-1], items[j]
		}
	}
}

func minInt(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

func alertTargetFeedIDs(configPath string, payload map[string]any) ([]string, error) {
	targets, err := targetFeedIDs(configPath, payload)
	if err != nil {
		return nil, err
	}
	return includeAllLocationAlertFeeds(configPath, targets)
}

func targetFeedIDs(configPath string, payload map[string]any) ([]string, error) {
	feeds, err := loadFeedSummaries(configPath)
	if err != nil {
		return nil, err
	}
	enabled := map[string]bool{}
	order := []string{}
	for _, feed := range feeds {
		id, _ := feed["id"].(string)
		if id == "" {
			continue
		}
		isEnabled, _ := feed["enabled"].(bool)
		if isEnabled {
			enabled[id] = true
			order = append(order, id)
		}
	}
	if len(order) == 0 {
		return nil, fmt.Errorf("no enabled feeds are configured")
	}
	requested := []string{}
	if boolPayload(payload, "air_on_all_feeds", false) {
		requested = append(requested, order...)
	} else {
		for _, feedID := range stringListAny(payload["feed_ids"]) {
			requested = append(requested, feedID)
		}
		if feedID := strings.TrimSpace(stringPayload(payload, "feed_id", "")); feedID != "" {
			requested = append(requested, feedID)
		}
	}
	if len(requested) == 0 {
		requested = append(requested, order...)
	}
	targets := []string{}
	for _, feedID := range uniqueStrings(requested) {
		if !enabled[feedID] {
			return nil, fmt.Errorf("feed %q is not enabled or does not exist", feedID)
		}
		targets = append(targets, feedID)
	}
	if len(targets) == 0 {
		return nil, fmt.Errorf("no target feeds selected")
	}
	return targets, nil
}

func includeAllLocationAlertFeeds(configPath string, targets []string) ([]string, error) {
	feeds, err := loadFeedSummaries(configPath)
	if err != nil {
		return nil, err
	}
	out := append([]string{}, targets...)
	seen := map[string]struct{}{}
	for _, feedID := range out {
		seen[feedID] = struct{}{}
	}
	for _, feed := range feeds {
		id := strings.TrimSpace(fmt.Sprint(feed["id"]))
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		enabled, _ := feed["enabled"].(bool)
		allLocations, _ := feed["same_all_locations"].(bool)
		if enabled && allLocations {
			out = append(out, id)
			seen[id] = struct{}{}
		}
	}
	return out, nil
}

func outputTargetsForFeeds(configPath string, feedIDs []string) ([]sameOutputTarget, error) {
	root, err := loadYAMLMap(configPath)
	if err != nil {
		return nil, err
	}
	parsed, err := loadFeedsXML(configPath, root)
	if err != nil {
		return nil, err
	}
	outputs, err := loadOutputsXML(configPath, root)
	if err != nil {
		return nil, err
	}
	wanted := map[string]bool{}
	for _, feedID := range feedIDs {
		wanted[feedID] = true
	}
	targets := []sameOutputTarget{}
	for _, feed := range parsed.Feeds {
		id := strings.TrimSpace(feed.ID)
		if !wanted[id] || !xmlBool(feed.EnabledRaw, true) {
			continue
		}
		targets = append(targets, outputTargetsForFeed(id, outputs[id])...)
	}
	return targets, nil
}

func outputTargetsForFeed(feedID string, output outputXML) []sameOutputTarget {
	targets := []sameOutputTarget{}
	if xmlBool(output.WebRTC.EnabledRaw, false) {
		targets = append(targets, sameOutputTarget{
			FeedID: feedID,
			Type:   "webrtc",
			Format: "pcm_s16le",
		})
	}
	if xmlBool(output.UDP.EnabledRaw, false) {
		address := ""
		if ip := strings.TrimSpace(output.UDP.IP); ip != "" && strings.TrimSpace(output.UDP.Port) != "" {
			address = net.JoinHostPort(ip, strings.TrimSpace(output.UDP.Port))
		}
		targets = append(targets, sameOutputTarget{
			FeedID:  feedID,
			Type:    "udp",
			Address: address,
			Format:  strings.TrimSpace(output.UDP.Format),
			Acodec:  strings.TrimSpace(output.UDP.Acodec),
		})
	}
	if xmlBool(output.RTP.EnabledRaw, false) {
		address := ""
		if ip := strings.TrimSpace(output.RTP.IP); ip != "" && strings.TrimSpace(output.RTP.Port) != "" {
			address = net.JoinHostPort(ip, strings.TrimSpace(output.RTP.Port))
		}
		targets = append(targets, sameOutputTarget{
			FeedID:  feedID,
			Type:    "rtp",
			Address: address,
			Format:  strings.TrimSpace(output.RTP.Format),
			Acodec:  strings.TrimSpace(output.RTP.Acodec),
		})
	}
	if xmlBool(output.Stream.EnabledRaw, false) {
		targetType := strings.TrimSpace(output.Stream.Type)
		if targetType == "" {
			targetType = "stream"
		}
		targets = append(targets, sameOutputTarget{
			FeedID: feedID,
			Type:   targetType,
			Format: strings.TrimSpace(output.Stream.Format),
			Acodec: strings.TrimSpace(output.Stream.Acodec),
		})
	}
	if xmlBool(output.RTMP.EnabledRaw, false) {
		targets = append(targets, sameOutputTarget{FeedID: feedID, Type: "rtmp", URL: strings.TrimSpace(output.RTMP.URL), Format: strings.TrimSpace(output.RTMP.Format), Acodec: strings.TrimSpace(output.RTMP.Acodec)})
	}
	if xmlBool(output.SRT.EnabledRaw, false) {
		targets = append(targets, sameOutputTarget{FeedID: feedID, Type: "srt", URL: strings.TrimSpace(output.SRT.URL), Format: strings.TrimSpace(output.SRT.Format), Acodec: strings.TrimSpace(output.SRT.Acodec)})
	}
	if xmlBool(output.RTSP.EnabledRaw, false) {
		targets = append(targets, sameOutputTarget{FeedID: feedID, Type: "rtsp", URL: strings.TrimSpace(output.RTSP.URL), Format: strings.TrimSpace(output.RTSP.Format), Acodec: strings.TrimSpace(output.RTSP.Acodec)})
	}
	if xmlBool(output.AudioDevice.EnabledRaw, false) {
		targets = append(targets, sameOutputTarget{FeedID: feedID, Type: "audio_device"})
	}
	if xmlBool(output.File.EnabledRaw, false) {
		targets = append(targets, sameOutputTarget{FeedID: feedID, Type: "file", URL: strings.TrimSpace(output.File.Path), Format: strings.TrimSpace(output.File.Format), Acodec: strings.TrimSpace(output.File.Acodec)})
	}
	return targets
}

func findHostExecutable(configPath string) (string, error) {
	candidates := []string{}
	if configured := strings.TrimSpace(os.Getenv("HAZE_SAME_GENERATOR")); configured != "" {
		candidates = append(candidates, configured)
	}
	if host := strings.TrimSpace(os.Getenv("HAZE_HOST_EXE")); host != "" {
		candidates = append(candidates, host)
	}
	base := filepath.Dir(filepath.Clean(configPath))
	name := "haze"
	if os.PathSeparator == '\\' {
		name = "haze.exe"
	}
	candidates = append(candidates,
		filepath.Join(base, name),
		filepath.Join(base, "dist", "haze", name),
		filepath.Join(base, "..", name),
	)
	for _, candidate := range candidates {
		candidate = filepath.Clean(candidate)
		if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("Haze SAME generator executable was not found")
}

func sameDuration(payload map[string]any) string {
	raw := strings.TrimSpace(stringPayload(payload, "duration", ""))
	if len(raw) == 4 && allDigits(raw) {
		return raw
	}
	hours := intPayload(payload, "duration_hours", 0)
	minutes := intPayload(payload, "duration_minutes", 15)
	if hours < 0 {
		hours = 0
	}
	if hours > 99 {
		hours = 99
	}
	if minutes < 0 {
		minutes = 0
	}
	if minutes > 59 {
		minutes = 59
	}
	return fmt.Sprintf("%02d%02d", hours, minutes)
}

func stringPayload(payload map[string]any, key string, fallback string) string {
	value, ok := payload[key]
	if !ok || value == nil {
		return fallback
	}
	text := strings.TrimSpace(fmt.Sprint(value))
	if text == "" {
		return fallback
	}
	return text
}

func intPayload(payload map[string]any, key string, fallback int) int {
	value, ok := payload[key]
	if !ok || value == nil {
		return fallback
	}
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case string:
		return parseIntText(typed, fallback)
	default:
		return fallback
	}
}

func boolPayload(payload map[string]any, key string, fallback bool) bool {
	value, ok := payload[key]
	if !ok || value == nil {
		return fallback
	}
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		return xmlBool(typed, fallback)
	default:
		return fallback
	}
}

func stringSlicePayload(payload map[string]any, key string) []string {
	values := []string{}
	for _, item := range anySlice(payload[key]) {
		text := cleanLocationCode(fmt.Sprint(item))
		if text != "" {
			values = append(values, text)
		}
	}
	if len(values) == 0 {
		for _, item := range strings.Split(strings.TrimSpace(fmt.Sprint(payload[key])), ",") {
			text := cleanLocationCode(item)
			if text != "" {
				values = append(values, text)
			}
		}
	}
	values = uniqueStrings(values)
	if containsString(values, "000000") {
		return []string{"000000"}
	}
	sort.Strings(values)
	return values
}

func anySlice(value any) []any {
	switch typed := value.(type) {
	case []any:
		return typed
	case []string:
		values := make([]any, 0, len(typed))
		for _, item := range typed {
			values = append(values, item)
		}
		return values
	default:
		return nil
	}
}

func stringListAny(value any) []string {
	values := []string{}
	for _, item := range anySlice(value) {
		text := strings.TrimSpace(fmt.Sprint(item))
		if text != "" {
			values = append(values, text)
		}
	}
	return values
}

func allDigits(value string) bool {
	for _, ch := range value {
		if ch < '0' || ch > '9' {
			return false
		}
	}
	return true
}

func queueItemID(event string) (string, error) {
	var raw [6]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", err
	}
	now := time.Now().UTC().Format("20060102T150405000")
	return fmt.Sprintf("same_%s_%s_%x", now, strings.ToLower(event), raw), nil
}

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
