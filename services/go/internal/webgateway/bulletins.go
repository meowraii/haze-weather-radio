package webgateway

import (
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"
)

const defaultBulletinsFile = "managed/configs/userBulletins.xml"

type bulletinsXML struct {
	XMLName   xml.Name      `xml:"bulletins"`
	Version   string        `xml:"version,attr,omitempty"`
	Bulletins []bulletinXML `xml:"bulletin"`
}

type bulletinXML struct {
	ID        string              `xml:"id,attr"`
	Enabled   string              `xml:"enabled,attr,omitempty"`
	Title     string              `xml:"title"`
	Active    bulletinActiveXML   `xml:"active"`
	Schedule  bulletinScheduleXML `xml:"schedule"`
	Target    bulletinTargetXML   `xml:"target"`
	Content   bulletinContentXML  `xml:"content"`
	UpdatedAt string              `xml:"updated_at,omitempty"`
}

type bulletinActiveXML struct {
	Start  string `xml:"start,attr,omitempty"`
	Expire string `xml:"expire,attr,omitempty"`
}

type bulletinScheduleXML struct {
	Mode    string   `xml:"mode,attr,omitempty"`
	Hours   []string `xml:"hours>hour,omitempty"`
	Days    []string `xml:"days>day,omitempty"`
	EndEach string   `xml:"end_of_cycle,attr,omitempty"`
}

type bulletinTargetXML struct {
	Feeds []bulletinFeedXML `xml:"feed"`
}

type bulletinFeedXML struct {
	ID string `xml:"id,attr"`
}

type bulletinContentXML struct {
	Type  string            `xml:"type,attr,omitempty"`
	Langs []bulletinLangXML `xml:"lang,omitempty"`
	Audio bulletinAudioXML  `xml:"audio,omitempty"`
}

type bulletinLangXML struct {
	Code string `xml:"code,attr"`
	Text string `xml:",chardata"`
}

type bulletinAudioXML struct {
	File string `xml:"file,attr,omitempty"`
	URL  string `xml:"url,attr,omitempty"`
}

var bulletinIDCleaner = regexp.MustCompile(`[^a-zA-Z0-9_.-]+`)

func loadBulletinsPayload(configPath string) (map[string]any, error) {
	path := bulletinsPath(configPath)
	items, err := readBulletinsXML(path)
	if err != nil {
		return nil, err
	}
	return bulletinsPayload(path, items), nil
}

func saveBulletinsPayload(configPath string, payload map[string]any) (map[string]any, error) {
	rawItems, ok := payload["bulletins"].([]any)
	if !ok {
		return nil, fmt.Errorf("bulletins payload is required")
	}
	items := make([]bulletinXML, 0, len(rawItems))
	for _, raw := range rawItems {
		item, err := bulletinFromMap(raw)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	path := bulletinsPath(configPath)
	if err := writeBulletinsXML(path, items); err != nil {
		return nil, err
	}
	return loadBulletinsPayload(configPath)
}

func importBulletinsPayload(configPath string, payload map[string]any) (map[string]any, error) {
	raw := strings.TrimSpace(stringValue(payload, "xml"))
	if raw == "" {
		return nil, fmt.Errorf("bulletin XML is required")
	}
	imported, err := parseBulletinXML(raw)
	if err != nil {
		return nil, err
	}
	if len(imported) == 0 {
		return nil, fmt.Errorf("no bulletins found in XML")
	}
	path := bulletinsPath(configPath)
	existing, err := readBulletinsXML(path)
	if err != nil {
		return nil, err
	}
	byID := map[string]bulletinXML{}
	for _, item := range existing {
		byID[item.ID] = item
	}
	for _, item := range imported {
		byID[item.ID] = item
	}
	merged := make([]bulletinXML, 0, len(byID))
	for _, item := range byID {
		merged = append(merged, item)
	}
	if err := writeBulletinsXML(path, merged); err != nil {
		return nil, err
	}
	return loadBulletinsPayload(configPath)
}

func exportBulletinsPayload(configPath string, payload map[string]any) (map[string]any, error) {
	path := bulletinsPath(configPath)
	items, err := readBulletinsXML(path)
	if err != nil {
		return nil, err
	}
	id := strings.TrimSpace(stringValue(payload, "id"))
	if id != "" {
		filtered := []bulletinXML{}
		for _, item := range items {
			if item.ID == id {
				filtered = append(filtered, item)
				break
			}
		}
		if len(filtered) == 0 {
			return nil, fmt.Errorf("bulletin %q was not found", id)
		}
		items = filtered
	}
	raw, err := marshalBulletinsXML(items)
	if err != nil {
		return nil, err
	}
	return map[string]any{"xml": raw, "filename": fallbackText(id, "userBulletins") + ".xml"}, nil
}

func uploadBulletinAudio(configPath string, payload map[string]any) (map[string]any, error) {
	name := sanitizeBulletinFilename(stringValue(payload, "filename"))
	raw64 := strings.TrimSpace(stringValue(payload, "audio_base64"))
	if name == "" || raw64 == "" {
		return nil, fmt.Errorf("audio filename and content are required")
	}
	data, err := base64.StdEncoding.DecodeString(raw64)
	if err != nil {
		return nil, fmt.Errorf("audio content is not valid base64")
	}
	if len(data) > 20*1024*1024 {
		return nil, fmt.Errorf("audio file is too large; maximum is 20 MB")
	}
	ext := strings.ToLower(filepath.Ext(name))
	switch ext {
	case ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".flac":
	default:
		return nil, fmt.Errorf("unsupported bulletin audio type %q", ext)
	}
	rel := filepath.ToSlash(filepath.Join("managed", "audio", "bulletins", name))
	path := resolveConfigPath(configPath, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return nil, err
	}
	return map[string]any{"path": rel}, nil
}

func bulletinsPath(configPath string) string {
	return resolveConfigPath(configPath, defaultBulletinsFile)
}

func readBulletinsXML(path string) ([]bulletinXML, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return []bulletinXML{}, nil
		}
		return nil, err
	}
	items, err := parseBulletinXML(string(raw))
	if err != nil {
		return nil, fmt.Errorf("parse bulletins XML: %w", err)
	}
	return items, nil
}

func parseBulletinXML(raw string) ([]bulletinXML, error) {
	var all bulletinsXML
	if err := xml.Unmarshal([]byte(raw), &all); err == nil && len(all.Bulletins) > 0 {
		return normalizeBulletins(all.Bulletins)
	}
	var one bulletinXML
	if err := xml.Unmarshal([]byte(raw), &one); err != nil {
		return nil, err
	}
	return normalizeBulletins([]bulletinXML{one})
}

func writeBulletinsXML(path string, items []bulletinXML) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := marshalBulletinsXML(items)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(raw), 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func marshalBulletinsXML(items []bulletinXML) (string, error) {
	items, err := normalizeBulletins(items)
	if err != nil {
		return "", err
	}
	out := bulletinsXML{Version: "1", Bulletins: items}
	raw, err := xml.MarshalIndent(out, "", "  ")
	if err != nil {
		return "", err
	}
	return xml.Header + string(raw) + "\n", nil
}

func normalizeBulletins(items []bulletinXML) ([]bulletinXML, error) {
	out := make([]bulletinXML, 0, len(items))
	seen := map[string]struct{}{}
	for _, item := range items {
		item.ID = cleanBulletinID(item.ID, item.Title)
		if item.ID == "" {
			return nil, fmt.Errorf("bulletin id or title is required")
		}
		if _, ok := seen[item.ID]; ok {
			return nil, fmt.Errorf("duplicate bulletin id %q", item.ID)
		}
		seen[item.ID] = struct{}{}
		item.Title = strings.TrimSpace(item.Title)
		if item.Title == "" {
			item.Title = item.ID
		}
		item.Enabled = boolText(xmlBool(item.Enabled, true))
		item.Active.Start = strings.TrimSpace(item.Active.Start)
		item.Active.Expire = strings.TrimSpace(item.Active.Expire)
		item.Schedule.Mode = normalizeBulletinMode(item.Schedule.Mode)
		item.Schedule.EndEach = boolText(xmlBool(item.Schedule.EndEach, true))
		item.Schedule.Hours = cleanUniqueStrings(item.Schedule.Hours, cleanHour)
		item.Schedule.Days = cleanUniqueStrings(item.Schedule.Days, cleanWeekday)
		item.Target.Feeds = cleanBulletinFeeds(item.Target.Feeds)
		item.Content.Type = normalizeBulletinContentType(item.Content.Type)
		item.Content.Langs = cleanBulletinLangs(item.Content.Langs)
		item.Content.Audio.File = sanitizeRelPath(item.Content.Audio.File)
		item.Content.Audio.URL = sanitizeHTTPURL(item.Content.Audio.URL)
		if item.Content.Type == "tts" && len(item.Content.Langs) == 0 {
			return nil, fmt.Errorf("bulletin %q needs TTS text", item.ID)
		}
		if item.Content.Type == "audio" && item.Content.Audio.File == "" && item.Content.Audio.URL == "" {
			return nil, fmt.Errorf("bulletin %q needs an audio file or URL", item.ID)
		}
		item.UpdatedAt = time.Now().UTC().Format(time.RFC3339)
		out = append(out, item)
	}
	sort.SliceStable(out, func(i, j int) bool { return strings.ToLower(out[i].Title) < strings.ToLower(out[j].Title) })
	return out, nil
}

func bulletinFromMap(raw any) (bulletinXML, error) {
	source, ok := raw.(map[string]any)
	if !ok {
		return bulletinXML{}, fmt.Errorf("bulletin entries must be objects")
	}
	item := bulletinXML{
		ID:      stringFromAny(source["id"]),
		Enabled: boolText(boolFromAny(source["enabled"], true)),
		Title:   stringFromAny(source["title"]),
		Active: bulletinActiveXML{
			Start:  stringFromAny(source["start"]),
			Expire: stringFromAny(source["expire"]),
		},
		Schedule: bulletinScheduleXML{
			Mode:    stringFromAny(source["schedule_mode"]),
			Hours:   stringSliceFromAny(source["hours"]),
			Days:    stringSliceFromAny(source["days"]),
			EndEach: boolText(boolFromAny(source["end_of_cycle"], true)),
		},
		Content: bulletinContentXML{
			Type: stringFromAny(source["content_type"]),
			Audio: bulletinAudioXML{
				File: stringFromAny(source["audio_file"]),
				URL:  stringFromAny(source["audio_url"]),
			},
		},
	}
	for _, code := range []string{"en-CA", "fr-CA"} {
		key := "text_" + strings.ToLower(strings.ReplaceAll(code, "-", "_"))
		if text := stringFromAny(source[key]); strings.TrimSpace(text) != "" {
			item.Content.Langs = append(item.Content.Langs, bulletinLangXML{Code: code, Text: text})
		}
	}
	for _, feed := range stringSliceFromAny(source["feeds"]) {
		item.Target.Feeds = append(item.Target.Feeds, bulletinFeedXML{ID: feed})
	}
	items, err := normalizeBulletins([]bulletinXML{item})
	if err != nil {
		return bulletinXML{}, err
	}
	return items[0], nil
}

func bulletinsPayload(path string, items []bulletinXML) map[string]any {
	rows := make([]map[string]any, 0, len(items))
	for _, item := range items {
		text := map[string]string{}
		for _, lang := range item.Content.Langs {
			text[lang.Code] = lang.Text
		}
		feeds := make([]string, 0, len(item.Target.Feeds))
		for _, feed := range item.Target.Feeds {
			feeds = append(feeds, feed.ID)
		}
		rows = append(rows, map[string]any{
			"id":            item.ID,
			"enabled":       xmlBool(item.Enabled, true),
			"title":         item.Title,
			"start":         item.Active.Start,
			"expire":        item.Active.Expire,
			"schedule_mode": item.Schedule.Mode,
			"hours":         item.Schedule.Hours,
			"days":          item.Schedule.Days,
			"end_of_cycle":  xmlBool(item.Schedule.EndEach, true),
			"content_type":  item.Content.Type,
			"text":          text,
			"text_en_ca":    text["en-CA"],
			"text_fr_ca":    text["fr-CA"],
			"audio_file":    item.Content.Audio.File,
			"audio_url":     item.Content.Audio.URL,
			"feeds":         feeds,
			"updated_at":    item.UpdatedAt,
		})
	}
	return map[string]any{
		"path":      filepath.ToSlash(path),
		"bulletins": rows,
		"summary": map[string]any{
			"count": len(rows),
		},
	}
}

func cleanBulletinID(id string, title string) string {
	id = strings.TrimSpace(id)
	if id == "" {
		id = strings.ToLower(strings.Join(strings.Fields(title), "-"))
	}
	id = bulletinIDCleaner.ReplaceAllString(id, "-")
	id = strings.Trim(id, "-_.")
	if len(id) > 80 {
		id = id[:80]
	}
	return id
}

func normalizeBulletinMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "hours", "hourly":
		return "hours"
	case "days", "weekdays":
		return "days"
	default:
		return "always"
	}
}

func normalizeBulletinContentType(value string) string {
	if strings.EqualFold(strings.TrimSpace(value), "audio") {
		return "audio"
	}
	return "tts"
}

func cleanBulletinLangs(langs []bulletinLangXML) []bulletinLangXML {
	out := []bulletinLangXML{}
	seen := map[string]struct{}{}
	for _, lang := range langs {
		code := strings.TrimSpace(lang.Code)
		if code == "" {
			code = "en-CA"
		}
		text := strings.TrimSpace(lang.Text)
		if text == "" {
			continue
		}
		if len(text) > 4000 {
			text = text[:4000]
		}
		if _, ok := seen[code]; ok {
			continue
		}
		seen[code] = struct{}{}
		out = append(out, bulletinLangXML{Code: code, Text: text})
	}
	return out
}

func cleanBulletinFeeds(feeds []bulletinFeedXML) []bulletinFeedXML {
	out := []bulletinFeedXML{}
	seen := map[string]struct{}{}
	for _, feed := range feeds {
		id := strings.TrimSpace(feed.ID)
		if id == "" {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, bulletinFeedXML{ID: id})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func cleanUniqueStrings(values []string, clean func(string) string) []string {
	out := []string{}
	seen := map[string]struct{}{}
	for _, value := range values {
		value = clean(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func cleanHour(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	hour := parseIntText(value, -1)
	if hour < 0 || hour > 23 {
		return ""
	}
	return fmt.Sprintf("%02d", hour)
}

func cleanWeekday(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "sun", "sunday", "0":
		return "sun"
	case "mon", "monday", "1":
		return "mon"
	case "tue", "tues", "tuesday", "2":
		return "tue"
	case "wed", "wednesday", "3":
		return "wed"
	case "thu", "thur", "thurs", "thursday", "4":
		return "thu"
	case "fri", "friday", "5":
		return "fri"
	case "sat", "saturday", "6":
		return "sat"
	default:
		return ""
	}
}

func sanitizeRelPath(value string) string {
	value = filepath.ToSlash(strings.TrimSpace(value))
	if value == "" || strings.Contains(value, "..") || strings.HasPrefix(value, "/") {
		return ""
	}
	return value
}

func sanitizeHTTPURL(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	parsed, err := url.Parse(value)
	if err != nil {
		return ""
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return ""
	}
	return value
}

func sanitizeBulletinFilename(value string) string {
	value = filepath.Base(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, " ", "_")
	value = bulletinIDCleaner.ReplaceAllString(value, "-")
	return strings.Trim(value, "-_.")
}

func stringFromAny(value any) string {
	if value == nil {
		return ""
	}
	return strings.TrimSpace(fmt.Sprint(value))
}

func boolFromAny(value any, fallback bool) bool {
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		if strings.TrimSpace(typed) == "" {
			return fallback
		}
		return xmlBool(typed, fallback)
	default:
		return fallback
	}
}

func stringSliceFromAny(value any) []string {
	switch typed := value.(type) {
	case []string:
		return typed
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			out = append(out, stringFromAny(item))
		}
		return out
	case string:
		parts := strings.FieldsFunc(typed, func(r rune) bool { return r == ',' || r == '\n' || r == ';' })
		out := make([]string, 0, len(parts))
		for _, part := range parts {
			out = append(out, strings.TrimSpace(part))
		}
		return out
	default:
		return nil
	}
}
