package webgateway

import (
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const defaultCgenFile = "managed/configs/cgen.xml"

type cgenXML struct {
	XMLName xml.Name      `xml:"cgen"`
	Enabled string        `xml:"enabled,attr,omitempty"`
	Feeds   []cgenFeedXML `xml:"feed"`
}

type cgenFeedXML struct {
	ID            string               `xml:"id,attr"`
	Name          string               `xml:"name,attr,omitempty"`
	Enabled       string               `xml:"enabled,attr,omitempty"`
	LegacyInput   cgenEndpointXML      `xml:"input,omitempty"`
	LegacyOutput  cgenEndpointXML      `xml:"output,omitempty"`
	ProgramInput  cgenEndpointXML      `xml:"programInput"`
	PriorityInput cgenPriorityInputXML `xml:"priorityInput"`
	ProgramOutput cgenEndpointXML      `xml:"programOutput"`
	AlertOutput   cgenEndpointXML      `xml:"alertOutput"`
	Video         cgenVideoXML         `xml:"video"`
	Audio         cgenAudioXML         `xml:"audio"`
	Banner        cgenBannerXML        `xml:"banner"`
	Graphics      cgenGraphicsXML      `xml:"graphics"`
	Clock         cgenClockXML         `xml:"clock"`
	Text          cgenTextXML          `xml:"text"`
	State         cgenStateXML         `xml:"state"`
	UpdatedAt     string               `xml:"updated_at,attr,omitempty"`
}

type cgenEndpointXML struct {
	URL              string `xml:"url,attr,omitempty"`
	Format           string `xml:"format,attr,omitempty"`
	VCodec           string `xml:"vcodec,attr,omitempty"`
	ACodec           string `xml:"acodec,attr,omitempty"`
	VideoBitrateKbps string `xml:"video_bitrate_kbps,attr,omitempty"`
	AudioBitrateKbps string `xml:"audio_bitrate_kbps,attr,omitempty"`
}

type cgenPriorityInputXML struct {
	FeedID string `xml:"feed_id,attr,omitempty"`
	URL    string `xml:"url,attr,omitempty"`
	Format string `xml:"format,attr,omitempty"`
}

type cgenVideoXML struct {
	Width      string `xml:"width,attr,omitempty"`
	Height     string `xml:"height,attr,omitempty"`
	FPS        string `xml:"fps,attr,omitempty"`
	Interlaced string `xml:"interlaced,attr,omitempty"`
	FieldOrder string `xml:"field_order,attr,omitempty"`
	Standard   string `xml:"standard,attr,omitempty"`
}

type cgenAudioXML struct {
	Idle      string `xml:"idle,attr,omitempty"`
	AlertMode string `xml:"alert_mode,attr,omitempty"`
	DuckDB    string `xml:"duck_db,attr,omitempty"`
}

type cgenBannerXML struct {
	Mode                    string `xml:"mode,attr,omitempty"`
	TickerHeight            string `xml:"ticker_height,attr,omitempty"`
	Font                    string `xml:"font,attr,omitempty"`
	FontSize                string `xml:"font_size,attr,omitempty"`
	ScrollSpeed             string `xml:"scroll_speed,attr,omitempty"`
	X                       string `xml:"x,attr,omitempty"`
	Y                       string `xml:"y,attr,omitempty"`
	BackgroundColor         string `xml:"background_color,attr,omitempty"`
	BackgroundGradientColor string `xml:"background_gradient_color,attr,omitempty"`
	BackgroundEnabled       string `xml:"background_enabled,attr,omitempty"`
}

type cgenGraphicsXML struct {
	BackgroundColor string `xml:"background_color,attr,omitempty"`
	Font            string `xml:"font,attr,omitempty"`
	FontSize        string `xml:"font_size,attr,omitempty"`
	TextX           string `xml:"text_x,attr,omitempty"`
	TextY           string `xml:"text_y,attr,omitempty"`
	BannerX         string `xml:"banner_x,attr,omitempty"`
	BannerY         string `xml:"banner_y,attr,omitempty"`
	BannerWidth     string `xml:"banner_width,attr,omitempty"`
	BannerHeight    string `xml:"banner_height,attr,omitempty"`
}

type cgenClockXML struct {
	Enabled  string `xml:"enabled,attr,omitempty"`
	Format   string `xml:"format,attr,omitempty"`
	X        string `xml:"x,attr,omitempty"`
	Y        string `xml:"y,attr,omitempty"`
	FontSize string `xml:"font_size,attr,omitempty"`
	Color    string `xml:"color,attr,omitempty"`
}

type cgenTextXML struct {
	Enabled  string `xml:"enabled,attr,omitempty"`
	X        string `xml:"x,attr,omitempty"`
	Y        string `xml:"y,attr,omitempty"`
	FontSize string `xml:"font_size,attr,omitempty"`
	Color    string `xml:"color,attr,omitempty"`
	Content  string `xml:",chardata"`
}

type cgenStateXML struct {
	Mode      string `xml:"mode,attr,omitempty"`
	SMPTEBars string `xml:"smpte_bars,attr,omitempty"`
	UpdatedAt string `xml:"updated_at,attr,omitempty"`
}

func loadCgenPayload(configPath string) (map[string]any, error) {
	path := cgenPath(configPath)
	config, err := readCgenXML(path)
	if err != nil {
		return nil, err
	}
	return cgenPayload(path, config), nil
}

func saveCgenPayload(configPath string, payload map[string]any) (map[string]any, error) {
	rawFeeds, ok := payload["feeds"].([]any)
	if !ok {
		return nil, fmt.Errorf("cgen feeds payload is required")
	}
	config := cgenXML{Enabled: boolText(boolFromAny(payload["enabled"], true))}
	for _, raw := range rawFeeds {
		feed, err := cgenFeedFromMap(raw)
		if err != nil {
			return nil, err
		}
		config.Feeds = append(config.Feeds, feed)
	}
	path := cgenPath(configPath)
	if err := writeCgenXML(path, config); err != nil {
		return nil, err
	}
	return loadCgenPayload(configPath)
}

func cgenActionPayload(configPath string, payload map[string]any) (map[string]any, error) {
	feedID := strings.TrimSpace(stringValue(payload, "feed_id"))
	action := strings.ToLower(strings.TrimSpace(stringValue(payload, "action")))
	if feedID == "" {
		return nil, fmt.Errorf("feed_id is required")
	}
	if action == "" {
		return nil, fmt.Errorf("action is required")
	}
	path := cgenPath(configPath)
	config, err := readCgenXML(path)
	if err != nil {
		return nil, err
	}
	index := -1
	for i, feed := range config.Feeds {
		if feed.ID == feedID {
			index = i
			break
		}
	}
	if index < 0 {
		return nil, fmt.Errorf("cgen feed %q was not found", feedID)
	}
	now := time.Now().UTC().Format(time.RFC3339)
	feed := config.Feeds[index]
	switch action {
	case "release":
		feed.State.Mode = "release"
		feed.State.SMPTEBars = "false"
	case "overlay":
		feed.State.Mode = "overlay"
	case "smpte_bars":
		feed.State.Mode = "overlay"
		feed.State.SMPTEBars = boolText(boolFromAny(payload["enabled"], true))
	case "clock":
		feed.Clock.Enabled = boolText(boolFromAny(payload["enabled"], true))
	case "insert_text":
		feed.State.Mode = "overlay"
		feed.Text.Enabled = "true"
		feed.Text.Content = stringValue(payload, "text")
	case "clear_text":
		feed.Text.Enabled = "false"
		feed.Text.Content = ""
	case "insert_banner_background":
		feed.Banner.BackgroundEnabled = "true"
		if color := strings.TrimSpace(stringValue(payload, "color")); color != "" {
			feed.Banner.BackgroundColor = color
		}
		if color := strings.TrimSpace(stringValue(payload, "gradient_color")); color != "" {
			feed.Banner.BackgroundGradientColor = color
		}
	case "clear_banner_background":
		feed.Banner.BackgroundEnabled = "false"
	default:
		return nil, fmt.Errorf("unsupported cgen action %q", action)
	}
	feed.State.UpdatedAt = now
	feed.UpdatedAt = now
	config.Feeds[index] = feed
	if err := writeCgenXML(path, config); err != nil {
		return nil, err
	}
	return loadCgenPayload(configPath)
}

func cgenPath(configPath string) string {
	return resolveConfigPath(configPath, defaultCgenFile)
}

func readCgenXML(path string) (cgenXML, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return cgenXML{Enabled: "true"}, nil
		}
		return cgenXML{}, err
	}
	var config cgenXML
	if err := xml.Unmarshal(raw, &config); err != nil {
		return cgenXML{}, fmt.Errorf("parse cgen XML: %w", err)
	}
	return normalizeCgen(config)
}

func writeCgenXML(path string, config cgenXML) error {
	config, err := normalizeCgen(config)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := xml.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, []byte(xml.Header+string(raw)+"\n"), 0o600); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func normalizeCgen(config cgenXML) (cgenXML, error) {
	config.Enabled = boolText(xmlBool(config.Enabled, true))
	seen := map[string]struct{}{}
	for i := range config.Feeds {
		feed := &config.Feeds[i]
		feed.ID = cleanCgenID(feed.ID)
		if feed.ID == "" {
			return cgenXML{}, fmt.Errorf("cgen feed id is required")
		}
		if _, ok := seen[feed.ID]; ok {
			return cgenXML{}, fmt.Errorf("duplicate cgen feed id %q", feed.ID)
		}
		seen[feed.ID] = struct{}{}
		feed.Name = strings.TrimSpace(feed.Name)
		if feed.Name == "" {
			feed.Name = feed.ID
		}
		feed.Enabled = boolText(xmlBool(feed.Enabled, false))
		if feed.ProgramInput.URL == "" {
			feed.ProgramInput = feed.LegacyInput
		}
		if feed.ProgramOutput.URL == "" {
			feed.ProgramOutput = feed.LegacyOutput
		}
		feed.LegacyInput = cgenEndpointXML{}
		feed.LegacyOutput = cgenEndpointXML{}
		feed.ProgramInput = cleanCgenEndpoint(feed.ProgramInput)
		feed.ProgramOutput = cleanCgenEndpoint(feed.ProgramOutput)
		feed.AlertOutput = cleanCgenEndpoint(feed.AlertOutput)
		if feed.AlertOutput.URL == "" {
			feed.AlertOutput = feed.ProgramOutput
		}
		feed.PriorityInput.FeedID = fallbackText(cleanCgenID(feed.PriorityInput.FeedID), feed.ID)
		feed.PriorityInput.URL = strings.TrimSpace(feed.PriorityInput.URL)
		feed.PriorityInput.Format = fallbackText(strings.TrimSpace(feed.PriorityInput.Format), "priority-audio")
		feed.Video.Width = cleanPositive(feed.Video.Width, "1280")
		feed.Video.Height = cleanPositive(feed.Video.Height, "720")
		feed.Video.FPS = fallbackText(strings.TrimSpace(feed.Video.FPS), "source")
		feed.Video.Interlaced = boolText(xmlBool(feed.Video.Interlaced, false))
		feed.Video.FieldOrder = normalizeCgenFieldOrder(feed.Video.FieldOrder)
		feed.Video.Standard = normalizeCgenStandard(feed.Video.Standard)
		feed.Audio.Idle = fallbackText(strings.TrimSpace(feed.Audio.Idle), "source")
		feed.Audio.AlertMode = fallbackText(strings.TrimSpace(feed.Audio.AlertMode), "replace")
		feed.Audio.DuckDB = cleanNumber(feed.Audio.DuckDB, "-18")
		feed.Banner.Mode = fallbackText(strings.TrimSpace(feed.Banner.Mode), "auto")
		feed.Banner.TickerHeight = cleanPositive(feed.Banner.TickerHeight, "96")
		feed.Banner.Font = fallbackText(strings.TrimSpace(feed.Banner.Font), "Zalando Sans SemiExpanded")
		feed.Banner.FontSize = cleanPositive(feed.Banner.FontSize, "26")
		feed.Banner.ScrollSpeed = cleanPositive(feed.Banner.ScrollSpeed, "4")
		feed.Banner.X = cleanNumber(feed.Banner.X, "0")
		feed.Banner.Y = cleanNumber(feed.Banner.Y, "0")
		feed.Banner.BackgroundColor = cleanColor(feed.Banner.BackgroundColor, "#b45309")
		feed.Banner.BackgroundGradientColor = cleanColor(feed.Banner.BackgroundGradientColor, "#7f1d1d")
		feed.Banner.BackgroundEnabled = boolText(xmlBool(feed.Banner.BackgroundEnabled, true))
		feed.Graphics.BackgroundColor = cleanColor(feed.Graphics.BackgroundColor, "#000000")
		feed.Graphics.Font = fallbackText(strings.TrimSpace(feed.Graphics.Font), feed.Banner.Font)
		feed.Graphics.FontSize = cleanPositive(feed.Graphics.FontSize, feed.Banner.FontSize)
		feed.Graphics.TextX = cleanNumber(feed.Graphics.TextX, "48")
		feed.Graphics.TextY = cleanNumber(feed.Graphics.TextY, "96")
		feed.Graphics.BannerX = cleanNumber(feed.Graphics.BannerX, "0")
		feed.Graphics.BannerY = cleanNumber(feed.Graphics.BannerY, "0")
		feed.Graphics.BannerWidth = cleanPositive(feed.Graphics.BannerWidth, feed.Video.Width)
		feed.Graphics.BannerHeight = cleanPositive(feed.Graphics.BannerHeight, feed.Banner.TickerHeight)
		feed.Clock.Enabled = boolText(xmlBool(feed.Clock.Enabled, false))
		feed.Clock.Format = fallbackText(strings.TrimSpace(feed.Clock.Format), "Jan 02 15:04:05")
		feed.Clock.X = cleanNumber(feed.Clock.X, "48")
		feed.Clock.Y = cleanNumber(feed.Clock.Y, "48")
		feed.Clock.FontSize = cleanPositive(feed.Clock.FontSize, "30")
		feed.Clock.Color = cleanColor(feed.Clock.Color, "#ffffff")
		feed.Text.Enabled = boolText(xmlBool(feed.Text.Enabled, false))
		feed.Text.X = cleanNumber(feed.Text.X, feed.Graphics.TextX)
		feed.Text.Y = cleanNumber(feed.Text.Y, feed.Graphics.TextY)
		feed.Text.FontSize = cleanPositive(feed.Text.FontSize, feed.Graphics.FontSize)
		feed.Text.Color = cleanColor(feed.Text.Color, "#ffffff")
		feed.Text.Content = strings.TrimSpace(feed.Text.Content)
		feed.State.Mode = normalizeCgenMode(feed.State.Mode)
		feed.State.SMPTEBars = boolText(xmlBool(feed.State.SMPTEBars, false))
		feed.State.UpdatedAt = strings.TrimSpace(feed.State.UpdatedAt)
		feed.UpdatedAt = strings.TrimSpace(feed.UpdatedAt)
	}
	sort.SliceStable(config.Feeds, func(i, j int) bool { return strings.ToLower(config.Feeds[i].ID) < strings.ToLower(config.Feeds[j].ID) })
	return config, nil
}

func cgenFeedFromMap(raw any) (cgenFeedXML, error) {
	source, ok := raw.(map[string]any)
	if !ok {
		return cgenFeedXML{}, fmt.Errorf("cgen feed entries must be objects")
	}
	feed := cgenFeedXML{
		ID:      stringFromAny(source["id"]),
		Name:    stringFromAny(source["name"]),
		Enabled: boolText(boolFromAny(source["enabled"], false)),
		ProgramInput: cgenEndpointXML{
			URL:    stringFromAny(source["program_input_url"]),
			Format: stringFromAny(source["program_input_format"]),
		},
		PriorityInput: cgenPriorityInputXML{
			FeedID: stringFromAny(source["priority_feed_id"]),
			URL:    stringFromAny(source["priority_input_url"]),
			Format: stringFromAny(source["priority_input_format"]),
		},
		ProgramOutput: cgenEndpointXML{
			URL:              stringFromAny(source["program_output_url"]),
			Format:           stringFromAny(source["program_output_format"]),
			VCodec:           stringFromAny(source["vcodec"]),
			ACodec:           stringFromAny(source["acodec"]),
			VideoBitrateKbps: stringFromAny(source["video_bitrate_kbps"]),
			AudioBitrateKbps: stringFromAny(source["audio_bitrate_kbps"]),
		},
		AlertOutput: cgenEndpointXML{
			URL:              stringFromAny(source["alert_output_url"]),
			Format:           stringFromAny(source["alert_output_format"]),
			VCodec:           stringFromAny(source["vcodec"]),
			ACodec:           stringFromAny(source["acodec"]),
			VideoBitrateKbps: stringFromAny(source["video_bitrate_kbps"]),
			AudioBitrateKbps: stringFromAny(source["audio_bitrate_kbps"]),
		},
		Video: cgenVideoXML{
			Width:      stringFromAny(source["width"]),
			Height:     stringFromAny(source["height"]),
			FPS:        stringFromAny(source["fps"]),
			Interlaced: boolText(boolFromAny(source["interlaced"], false)),
			FieldOrder: stringFromAny(source["field_order"]),
			Standard:   stringFromAny(source["standard"]),
		},
		Audio: cgenAudioXML{
			Idle:      stringFromAny(source["audio_idle"]),
			AlertMode: stringFromAny(source["audio_alert_mode"]),
			DuckDB:    stringFromAny(source["duck_db"]),
		},
		Banner: cgenBannerXML{
			Mode:                    stringFromAny(source["banner_mode"]),
			TickerHeight:            stringFromAny(source["ticker_height"]),
			Font:                    stringFromAny(source["font"]),
			FontSize:                stringFromAny(source["font_size"]),
			ScrollSpeed:             stringFromAny(source["scroll_speed"]),
			X:                       stringFromAny(source["banner_x"]),
			Y:                       stringFromAny(source["banner_y"]),
			BackgroundColor:         stringFromAny(source["banner_background_color"]),
			BackgroundGradientColor: stringFromAny(source["banner_background_gradient_color"]),
			BackgroundEnabled:       boolText(boolFromAny(source["banner_background_enabled"], true)),
		},
		Graphics: cgenGraphicsXML{
			BackgroundColor: stringFromAny(source["background_color"]),
			Font:            stringFromAny(source["font"]),
			FontSize:        stringFromAny(source["font_size"]),
			TextX:           stringFromAny(source["text_x"]),
			TextY:           stringFromAny(source["text_y"]),
			BannerX:         stringFromAny(source["banner_x"]),
			BannerY:         stringFromAny(source["banner_y"]),
			BannerWidth:     stringFromAny(source["banner_width"]),
			BannerHeight:    stringFromAny(source["banner_height"]),
		},
		Clock: cgenClockXML{
			Enabled:  boolText(boolFromAny(source["clock_enabled"], false)),
			Format:   stringFromAny(source["clock_format"]),
			X:        stringFromAny(source["clock_x"]),
			Y:        stringFromAny(source["clock_y"]),
			FontSize: stringFromAny(source["clock_font_size"]),
			Color:    stringFromAny(source["clock_color"]),
		},
		Text: cgenTextXML{
			Enabled:  boolText(boolFromAny(source["text_enabled"], false)),
			X:        stringFromAny(source["text_x"]),
			Y:        stringFromAny(source["text_y"]),
			FontSize: stringFromAny(source["text_font_size"]),
			Color:    stringFromAny(source["text_color"]),
			Content:  stringFromAny(source["text"]),
		},
		State: cgenStateXML{
			Mode:      stringFromAny(source["mode"]),
			SMPTEBars: boolText(boolFromAny(source["smpte_bars"], false)),
		},
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	config, err := normalizeCgen(cgenXML{Enabled: "true", Feeds: []cgenFeedXML{feed}})
	if err != nil {
		return cgenFeedXML{}, err
	}
	return config.Feeds[0], nil
}

func cgenPayload(path string, config cgenXML) map[string]any {
	rows := make([]map[string]any, 0, len(config.Feeds))
	for _, feed := range config.Feeds {
		rows = append(rows, map[string]any{
			"id":                               feed.ID,
			"name":                             feed.Name,
			"enabled":                          xmlBool(feed.Enabled, false),
			"mode":                             feed.State.Mode,
			"smpte_bars":                       xmlBool(feed.State.SMPTEBars, false),
			"program_input_url":                feed.ProgramInput.URL,
			"program_input_format":             feed.ProgramInput.Format,
			"priority_feed_id":                 feed.PriorityInput.FeedID,
			"priority_input_url":               feed.PriorityInput.URL,
			"priority_input_format":            feed.PriorityInput.Format,
			"program_output_url":               feed.ProgramOutput.URL,
			"program_output_format":            feed.ProgramOutput.Format,
			"alert_output_url":                 feed.AlertOutput.URL,
			"alert_output_format":              feed.AlertOutput.Format,
			"vcodec":                           feed.ProgramOutput.VCodec,
			"acodec":                           feed.ProgramOutput.ACodec,
			"video_bitrate_kbps":               feed.ProgramOutput.VideoBitrateKbps,
			"audio_bitrate_kbps":               feed.ProgramOutput.AudioBitrateKbps,
			"width":                            feed.Video.Width,
			"height":                           feed.Video.Height,
			"fps":                              feed.Video.FPS,
			"interlaced":                       xmlBool(feed.Video.Interlaced, false),
			"field_order":                      feed.Video.FieldOrder,
			"standard":                         feed.Video.Standard,
			"audio_idle":                       feed.Audio.Idle,
			"audio_alert_mode":                 feed.Audio.AlertMode,
			"duck_db":                          feed.Audio.DuckDB,
			"banner_mode":                      feed.Banner.Mode,
			"ticker_height":                    feed.Banner.TickerHeight,
			"scroll_speed":                     feed.Banner.ScrollSpeed,
			"font":                             feed.Graphics.Font,
			"font_size":                        feed.Graphics.FontSize,
			"background_color":                 feed.Graphics.BackgroundColor,
			"banner_background_color":          feed.Banner.BackgroundColor,
			"banner_background_gradient_color": feed.Banner.BackgroundGradientColor,
			"banner_background_enabled":        xmlBool(feed.Banner.BackgroundEnabled, true),
			"banner_x":                         feed.Graphics.BannerX,
			"banner_y":                         feed.Graphics.BannerY,
			"banner_width":                     feed.Graphics.BannerWidth,
			"banner_height":                    feed.Graphics.BannerHeight,
			"text_enabled":                     xmlBool(feed.Text.Enabled, false),
			"text":                             feed.Text.Content,
			"text_x":                           feed.Text.X,
			"text_y":                           feed.Text.Y,
			"text_font_size":                   feed.Text.FontSize,
			"text_color":                       feed.Text.Color,
			"clock_enabled":                    xmlBool(feed.Clock.Enabled, false),
			"clock_format":                     feed.Clock.Format,
			"clock_x":                          feed.Clock.X,
			"clock_y":                          feed.Clock.Y,
			"clock_font_size":                  feed.Clock.FontSize,
			"clock_color":                      feed.Clock.Color,
			"updated_at":                       fallbackText(feed.State.UpdatedAt, feed.UpdatedAt),
		})
	}
	return map[string]any{
		"path":    filepath.ToSlash(path),
		"enabled": xmlBool(config.Enabled, true),
		"feeds":   rows,
		"summary": map[string]any{
			"count": len(rows),
		},
	}
}

func cleanCgenEndpoint(value cgenEndpointXML) cgenEndpointXML {
	value.URL = strings.TrimSpace(value.URL)
	value.Format = strings.TrimSpace(value.Format)
	value.VCodec = strings.TrimSpace(value.VCodec)
	value.ACodec = strings.TrimSpace(value.ACodec)
	value.VideoBitrateKbps = cleanOptionalPositive(value.VideoBitrateKbps)
	value.AudioBitrateKbps = cleanOptionalPositive(value.AudioBitrateKbps)
	return value
}

func normalizeCgenFieldOrder(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "bff" || value == "bottom" || value == "bottom_first" {
		return "bff"
	}
	return "tff"
}

func normalizeCgenStandard(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "atsc", "pal", "secam":
		return value
	default:
		return ""
	}
}

func cleanCgenID(value string) string {
	value = strings.TrimSpace(value)
	if value == "*" {
		return value
	}
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return -1
	}, value)
}

func normalizeCgenMode(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "overlay", "program", "active":
		return "overlay"
	default:
		return "release"
	}
}

func cleanPositive(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	for _, ch := range value {
		if ch < '0' || ch > '9' {
			return fallback
		}
	}
	return value
}

func cleanOptionalPositive(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return cleanPositive(value, "")
}

func cleanNumber(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	for i, ch := range value {
		if ch >= '0' && ch <= '9' || ch == '.' || (ch == '-' && i == 0) {
			continue
		}
		return fallback
	}
	return value
}

func cleanColor(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if len(value) == 7 && value[0] == '#' {
		for _, ch := range value[1:] {
			if ch >= '0' && ch <= '9' || ch >= 'a' && ch <= 'f' || ch >= 'A' && ch <= 'F' {
				continue
			}
			return fallback
		}
		return value
	}
	return fallback
}
