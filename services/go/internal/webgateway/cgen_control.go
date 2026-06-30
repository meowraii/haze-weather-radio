package webgateway

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
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
	ProgramInput  cgenEndpointXML      `xml:"programInput"`
	PriorityInput cgenPriorityInputXML `xml:"priorityInput"`
	ProgramOutput cgenEndpointXML      `xml:"programOutput"`
	Video         cgenVideoXML         `xml:"video"`
	Audio         cgenAudioXML         `xml:"audio"`
	Ladder        cgenLadderXML        `xml:"ladder"`
	Banner        cgenBannerXML        `xml:"banner"`
	Graphics      cgenGraphicsXML      `xml:"graphics"`
	Clock         cgenClockXML         `xml:"clock"`
	Text          cgenTextXML          `xml:"text"`
	State         cgenStateXML         `xml:"state"`
	Standby       cgenStandbyXML       `xml:"standby"`
	Sync          cgenSyncXML          `xml:"sync"`
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

type cgenLadderXML struct {
	Videos []cgenVideoRenditionXML `xml:"video"`
	Audios []cgenAudioRenditionXML `xml:"audio"`
}

type cgenVideoRenditionXML struct {
	ID          string `xml:"id,attr"`
	Enabled     string `xml:"enabled,attr,omitempty"`
	Width       string `xml:"width,attr,omitempty"`
	Height      string `xml:"height,attr,omitempty"`
	FPS         string `xml:"fps,attr,omitempty"`
	Interlaced  string `xml:"interlaced,attr,omitempty"`
	FieldOrder  string `xml:"field_order,attr,omitempty"`
	Standard    string `xml:"standard,attr,omitempty"`
	VCodec      string `xml:"vcodec,attr,omitempty"`
	BitrateKbps string `xml:"bitrate_kbps,attr,omitempty"`
	Program     string `xml:"program,attr,omitempty"`
	VideoPID    string `xml:"video_pid,attr,omitempty"`
	PMTPID      string `xml:"pmt_pid,attr,omitempty"`
}

type cgenAudioRenditionXML struct {
	ID          string `xml:"id,attr"`
	Enabled     string `xml:"enabled,attr,omitempty"`
	Channels    string `xml:"channels,attr,omitempty"`
	ACodec      string `xml:"acodec,attr,omitempty"`
	BitrateKbps string `xml:"bitrate_kbps,attr,omitempty"`
	Language    string `xml:"language,attr,omitempty"`
}

type cgenPriorityInputXML struct {
	FeedID      string `xml:"feed_id,attr,omitempty"`
	AudioSource string `xml:"audio_source,attr,omitempty"`
	Format      string `xml:"format,attr,omitempty"`
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
	Idle               string `xml:"idle,attr,omitempty"`
	AlertMode          string `xml:"alert_mode,attr,omitempty"`
	MuteStandbyRoutine string `xml:"mute_standby_routine,attr,omitempty"`
}

type cgenBannerXML struct {
	Mode              string `xml:"mode,attr,omitempty"`
	TickerHeight      string `xml:"ticker_height,attr,omitempty"`
	Font              string `xml:"font,attr,omitempty"`
	FontSize          string `xml:"font_size,attr,omitempty"`
	ScrollSpeed       string `xml:"scroll_speed,attr,omitempty"`
	BackgroundEnabled string `xml:"background_enabled,attr,omitempty"`
}

type cgenGraphicsXML struct {
	Font         string `xml:"font,attr,omitempty"`
	FontSize     string `xml:"font_size,attr,omitempty"`
	BannerHeight string `xml:"banner_height,attr,omitempty"`
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
	FontSize string `xml:"font_size,attr,omitempty"`
	Color    string `xml:"color,attr,omitempty"`
	Content  string `xml:",chardata"`
}

type cgenStateXML struct {
	Mode      string `xml:"mode,attr,omitempty"`
	SMPTEBars string `xml:"smpte_bars,attr,omitempty"`
	UpdatedAt string `xml:"updated_at,attr,omitempty"`
}

type cgenStandbyXML struct {
	Mode     string `xml:"mode,attr,omitempty"`
	Text     string `xml:"text,attr,omitempty"`
	FontSize string `xml:"font_size,attr,omitempty"`
	YPercent string `xml:"y_percent,attr,omitempty"`
}

type cgenSyncXML struct {
	HardResetMS            string `xml:"hard_reset_ms,attr,omitempty"`
	MaxAudioFramesPerVideo string `xml:"max_audio_frames_per_video,attr,omitempty"`
	SourceBufferMS         string `xml:"source_buffer_ms,attr,omitempty"`
	ReconnectInitialMS     string `xml:"reconnect_initial_ms,attr,omitempty"`
	ReconnectMaxMS         string `xml:"reconnect_max_ms,attr,omitempty"`
	StatusIntervalMS       string `xml:"status_interval_ms,attr,omitempty"`
}

func loadCgenPayload(configPath string) (map[string]any, error) {
	path := cgenPath(configPath)
	config, err := readCgenXML(path)
	if err != nil {
		return nil, err
	}
	return cgenPayload(configPath, path, config), nil
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
	if err := rejectLegacyCgenXML(raw, path); err != nil {
		return cgenXML{}, err
	}
	var config cgenXML
	if err := xml.Unmarshal(raw, &config); err != nil {
		return cgenXML{}, fmt.Errorf("parse cgen XML: %w", err)
	}
	return normalizeCgen(config)
}

func rejectLegacyCgenXML(raw []byte, path string) error {
	decoder := xml.NewDecoder(bytes.NewReader(raw))
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("scan cgen XML %s: %w", path, err)
		}
		start, ok := token.(xml.StartElement)
		if !ok {
			continue
		}
		name := start.Name.Local
		switch name {
		case "input", "output", "alertOutput":
			return fmt.Errorf("legacy cgen <%s> element is no longer supported in %s; use programInput, priorityInput, and programOutput", name, path)
		}
		for _, attr := range start.Attr {
			if legacyCgenAttribute(name, attr.Name.Local) {
				return fmt.Errorf("legacy cgen %s @%s attribute is no longer supported in %s", name, attr.Name.Local, path)
			}
		}
	}
}

func legacyCgenAttribute(element string, attr string) bool {
	switch element {
	case "cgen":
		return attr == "graphics_backend" || attr == "ffmpeg"
	case "priorityInput":
		return attr == "url"
	case "audio":
		return attr == "duck_db"
	case "banner":
		return attr == "x" || attr == "y"
	case "graphics":
		switch attr {
		case "background_color", "text_x", "text_y", "banner_x", "banner_y", "banner_width":
			return true
		}
	}
	return false
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
		feed.ProgramInput = cleanCgenEndpoint(feed.ProgramInput)
		feed.ProgramOutput = cleanCgenEndpoint(feed.ProgramOutput)
		feed.PriorityInput.FeedID = fallbackText(cleanCgenID(feed.PriorityInput.FeedID), feed.ID)
		feed.PriorityInput.AudioSource = normalizeCgenAudioSource(feed.PriorityInput.AudioSource)
		feed.PriorityInput.Format = fallbackText(strings.TrimSpace(feed.PriorityInput.Format), "priority-audio")
		feed.Video.Width = cleanPositive(feed.Video.Width, "1280")
		feed.Video.Height = cleanPositive(feed.Video.Height, "720")
		feed.Video.FPS = fallbackText(strings.TrimSpace(feed.Video.FPS), "source")
		feed.Video.Interlaced = boolText(xmlBool(feed.Video.Interlaced, false))
		feed.Video.FieldOrder = normalizeCgenFieldOrder(feed.Video.FieldOrder)
		feed.Video.Standard = normalizeCgenStandard(feed.Video.Standard)
		feed.Audio.Idle = fallbackText(strings.TrimSpace(feed.Audio.Idle), "source")
		feed.Audio.AlertMode = fallbackText(strings.TrimSpace(feed.Audio.AlertMode), "replace")
		feed.Audio.MuteStandbyRoutine = boolText(xmlBool(feed.Audio.MuteStandbyRoutine, true))
		normalizeCgenLadder(feed)
		feed.Banner.Mode = fallbackText(strings.TrimSpace(feed.Banner.Mode), "auto")
		feed.Banner.TickerHeight = cleanPositive(feed.Banner.TickerHeight, "128")
		feed.Banner.Font = fallbackText(strings.TrimSpace(feed.Banner.Font), "Arial")
		feed.Banner.FontSize = cleanPositive(feed.Banner.FontSize, "58")
		feed.Banner.ScrollSpeed = cleanPositive(feed.Banner.ScrollSpeed, "8")
		feed.Banner.BackgroundEnabled = boolText(xmlBool(feed.Banner.BackgroundEnabled, true))
		feed.Graphics.Font = fallbackText(strings.TrimSpace(feed.Graphics.Font), feed.Banner.Font)
		feed.Graphics.FontSize = cleanPositive(feed.Graphics.FontSize, feed.Banner.FontSize)
		feed.Graphics.BannerHeight = cleanPositive(feed.Graphics.BannerHeight, feed.Banner.TickerHeight)
		feed.Clock.Enabled = boolText(xmlBool(feed.Clock.Enabled, false))
		feed.Clock.Format = fallbackText(strings.TrimSpace(feed.Clock.Format), "Jan 02 15:04:05")
		feed.Clock.X = cleanNumber(feed.Clock.X, "48")
		feed.Clock.Y = cleanNumber(feed.Clock.Y, "48")
		feed.Clock.FontSize = cleanPositive(feed.Clock.FontSize, "30")
		feed.Clock.Color = cleanColor(feed.Clock.Color, "#ffffff")
		feed.Text.Enabled = boolText(xmlBool(feed.Text.Enabled, false))
		feed.Text.FontSize = cleanPositive(feed.Text.FontSize, feed.Graphics.FontSize)
		feed.Text.Color = cleanColor(feed.Text.Color, "#ffffff")
		feed.Text.Content = strings.TrimSpace(feed.Text.Content)
		feed.State.Mode = normalizeCgenMode(feed.State.Mode)
		feed.State.SMPTEBars = boolText(xmlBool(feed.State.SMPTEBars, false))
		feed.State.UpdatedAt = strings.TrimSpace(feed.State.UpdatedAt)
		feed.Standby.Mode = normalizeCgenStandbyMode(feed.Standby.Mode)
		feed.Standby.Text = fallbackText(strings.TrimSpace(feed.Standby.Text), "EAS Details Channel")
		feed.Standby.FontSize = cleanPositive(feed.Standby.FontSize, feed.Banner.FontSize)
		feed.Standby.YPercent = cleanPercent(feed.Standby.YPercent, "10")
		feed.Sync.HardResetMS = cleanPositive(feed.Sync.HardResetMS, "250")
		feed.Sync.MaxAudioFramesPerVideo = cleanPositive(feed.Sync.MaxAudioFramesPerVideo, "8")
		feed.Sync.SourceBufferMS = cleanPositive(feed.Sync.SourceBufferMS, "240")
		feed.Sync.ReconnectInitialMS = cleanPositive(feed.Sync.ReconnectInitialMS, "500")
		feed.Sync.ReconnectMaxMS = cleanPositive(feed.Sync.ReconnectMaxMS, "10000")
		feed.Sync.StatusIntervalMS = cleanPositive(feed.Sync.StatusIntervalMS, "750")
		feed.UpdatedAt = strings.TrimSpace(feed.UpdatedAt)
	}
	sort.SliceStable(config.Feeds, func(i, j int) bool { return strings.ToLower(config.Feeds[i].ID) < strings.ToLower(config.Feeds[j].ID) })
	return config, nil
}

func normalizeCgenLadder(feed *cgenFeedXML) {
	videos := map[string]cgenVideoRenditionXML{}
	for _, video := range feed.Ladder.Videos {
		id := strings.ToLower(strings.TrimSpace(video.ID))
		if id != "" {
			video.ID = id
			videos[id] = video
		}
	}
	audios := map[string]cgenAudioRenditionXML{}
	for _, audio := range feed.Ladder.Audios {
		id := strings.ToLower(strings.TrimSpace(audio.ID))
		if id != "" {
			audio.ID = id
			audios[id] = audio
		}
	}

	hd := normalizeVideoRendition(videos["hd"], cgenVideoRenditionXML{
		ID:          "hd",
		Enabled:     "auto",
		Width:       feed.Video.Width,
		Height:      feed.Video.Height,
		FPS:         feed.Video.FPS,
		Interlaced:  feed.Video.Interlaced,
		FieldOrder:  feed.Video.FieldOrder,
		Standard:    feed.Video.Standard,
		VCodec:      feed.ProgramOutput.VCodec,
		BitrateKbps: feed.ProgramOutput.VideoBitrateKbps,
		Program:     "1",
		VideoPID:    "256",
		PMTPID:      "4096",
	})
	p720 := normalizeVideoRendition(videos["p720"], cgenVideoRenditionXML{
		ID:          "p720",
		Enabled:     "false",
		Width:       "1280",
		Height:      "720",
		FPS:         fallbackText(feed.Video.FPS, "30000/1001"),
		Interlaced:  "false",
		FieldOrder:  "tff",
		Standard:    "atsc",
		VCodec:      feed.ProgramOutput.VCodec,
		BitrateKbps: "8000",
		Program:     "2",
		VideoPID:    "288",
		PMTPID:      "4097",
	})
	sd := normalizeVideoRendition(videos["sd"], cgenVideoRenditionXML{
		ID:          "sd",
		Enabled:     "false",
		Width:       "720",
		Height:      "480",
		FPS:         fallbackText(feed.Video.FPS, "30000/1001"),
		Interlaced:  "true",
		FieldOrder:  "tff",
		Standard:    "atsc",
		VCodec:      feed.ProgramOutput.VCodec,
		BitrateKbps: "5000",
		Program:     "3",
		VideoPID:    "320",
		PMTPID:      "4098",
	})
	surround := normalizeAudioRendition(audios["surround_51"], cgenAudioRenditionXML{
		ID:          "surround_51",
		Enabled:     "true",
		Channels:    "6",
		ACodec:      feed.ProgramOutput.ACodec,
		BitrateKbps: "384",
		Language:    "eng",
	})
	stereo := normalizeAudioRendition(audios["stereo"], cgenAudioRenditionXML{
		ID:          "stereo",
		Enabled:     "true",
		Channels:    "2",
		ACodec:      feed.ProgramOutput.ACodec,
		BitrateKbps: fallbackText(feed.ProgramOutput.AudioBitrateKbps, "192"),
		Language:    "eng",
	})
	feed.Ladder = cgenLadderXML{
		Videos: []cgenVideoRenditionXML{hd, p720, sd},
		Audios: []cgenAudioRenditionXML{stereo, surround},
	}
	feed.ProgramOutput.VideoBitrateKbps = hd.BitrateKbps
	feed.ProgramOutput.AudioBitrateKbps = stereo.BitrateKbps
}

func normalizeVideoRendition(value cgenVideoRenditionXML, fallback cgenVideoRenditionXML) cgenVideoRenditionXML {
	value.ID = fallbackText(strings.ToLower(strings.TrimSpace(value.ID)), fallback.ID)
	value.Enabled = normalizeCgenEnabledText(value.Enabled, fallback.Enabled)
	value.Width = cleanPositive(value.Width, fallback.Width)
	value.Height = cleanPositive(value.Height, fallback.Height)
	value.FPS = fallbackText(strings.TrimSpace(value.FPS), fallback.FPS)
	value.Interlaced = boolText(xmlBool(value.Interlaced, xmlBool(fallback.Interlaced, false)))
	value.FieldOrder = normalizeCgenFieldOrder(fallbackText(value.FieldOrder, fallback.FieldOrder))
	value.Standard = normalizeCgenStandard(fallbackText(value.Standard, fallback.Standard))
	value.VCodec = fallbackText(strings.TrimSpace(value.VCodec), fallback.VCodec)
	value.BitrateKbps = cleanPositive(value.BitrateKbps, fallback.BitrateKbps)
	value.Program = cleanPositive(value.Program, fallback.Program)
	value.VideoPID = cleanPositive(value.VideoPID, fallback.VideoPID)
	value.PMTPID = cleanPositive(value.PMTPID, fallback.PMTPID)
	return value
}

func normalizeAudioRendition(value cgenAudioRenditionXML, fallback cgenAudioRenditionXML) cgenAudioRenditionXML {
	value.ID = fallbackText(strings.ToLower(strings.TrimSpace(value.ID)), fallback.ID)
	value.Enabled = boolText(xmlBool(value.Enabled, xmlBool(fallback.Enabled, true)))
	value.Channels = cleanPositive(value.Channels, fallback.Channels)
	value.ACodec = fallbackText(strings.TrimSpace(value.ACodec), fallback.ACodec)
	value.BitrateKbps = cleanPositive(value.BitrateKbps, fallback.BitrateKbps)
	value.Language = fallbackText(strings.TrimSpace(value.Language), fallback.Language)
	return value
}

func normalizeCgenEnabledText(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if strings.EqualFold(value, "auto") {
		return "auto"
	}
	if value == "" && strings.EqualFold(strings.TrimSpace(fallback), "auto") {
		return "auto"
	}
	return boolText(xmlBool(value, xmlBool(fallback, true)))
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
			FeedID:      stringFromAny(source["priority_feed_id"]),
			AudioSource: stringFromAny(source["audio_source"]),
			Format:      stringFromAny(source["priority_input_format"]),
		},
		ProgramOutput: cgenEndpointXML{
			URL:              stringFromAny(source["program_output_url"]),
			Format:           stringFromAny(source["program_output_format"]),
			VCodec:           stringFromAny(source["vcodec"]),
			ACodec:           stringFromAny(source["acodec"]),
			VideoBitrateKbps: firstStringFromAny(source, "hd_bitrate_kbps", "video_bitrate_kbps"),
			AudioBitrateKbps: firstStringFromAny(source, "stereo_bitrate_kbps", "audio_bitrate_kbps"),
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
			Idle:               stringFromAny(source["audio_idle"]),
			AlertMode:          stringFromAny(source["audio_alert_mode"]),
			MuteStandbyRoutine: boolText(boolFromAny(source["mute_standby_routine"], true)),
		},
		Ladder: cgenLadderXML{
			Videos: []cgenVideoRenditionXML{
				{
					ID:          "hd",
					Enabled:     boolOrTextFromAny(source["hd_enabled"], "auto"),
					Width:       stringFromAny(source["width"]),
					Height:      stringFromAny(source["height"]),
					FPS:         stringFromAny(source["fps"]),
					Interlaced:  boolText(boolFromAny(source["interlaced"], false)),
					FieldOrder:  stringFromAny(source["field_order"]),
					Standard:    stringFromAny(source["standard"]),
					VCodec:      stringFromAny(source["vcodec"]),
					BitrateKbps: firstStringFromAny(source, "hd_bitrate_kbps", "video_bitrate_kbps"),
					Program:     "1",
					VideoPID:    "256",
					PMTPID:      "4096",
				},
				{
					ID:          "p720",
					Enabled:     boolText(boolFromAny(source["p720_enabled"], false)),
					Width:       "1280",
					Height:      "720",
					FPS:         stringFromAny(source["fps"]),
					Interlaced:  "false",
					FieldOrder:  "tff",
					Standard:    "atsc",
					VCodec:      stringFromAny(source["vcodec"]),
					BitrateKbps: fallbackText(stringFromAny(source["p720_bitrate_kbps"]), "8000"),
					Program:     "2",
					VideoPID:    "288",
					PMTPID:      "4097",
				},
				{
					ID:          "sd",
					Enabled:     boolText(boolFromAny(source["sd_enabled"], false)),
					Width:       "720",
					Height:      "480",
					FPS:         stringFromAny(source["fps"]),
					Interlaced:  "true",
					FieldOrder:  "tff",
					Standard:    "atsc",
					VCodec:      stringFromAny(source["vcodec"]),
					BitrateKbps: fallbackText(stringFromAny(source["sd_bitrate_kbps"]), "5000"),
					Program:     "3",
					VideoPID:    "320",
					PMTPID:      "4098",
				},
			},
			Audios: []cgenAudioRenditionXML{
				{
					ID:          "stereo",
					Enabled:     boolText(boolFromAny(source["stereo_enabled"], true)),
					Channels:    "2",
					ACodec:      stringFromAny(source["acodec"]),
					BitrateKbps: firstStringFromAny(source, "stereo_bitrate_kbps", "audio_bitrate_kbps"),
					Language:    "eng",
				},
				{
					ID:          "surround_51",
					Enabled:     boolText(boolFromAny(source["surround_enabled"], true)),
					Channels:    "6",
					ACodec:      stringFromAny(source["acodec"]),
					BitrateKbps: fallbackText(stringFromAny(source["surround_bitrate_kbps"]), "384"),
					Language:    "eng",
				},
			},
		},
		Banner: cgenBannerXML{
			Mode:              stringFromAny(source["banner_mode"]),
			TickerHeight:      stringFromAny(source["ticker_height"]),
			Font:              stringFromAny(source["font"]),
			FontSize:          stringFromAny(source["font_size"]),
			ScrollSpeed:       stringFromAny(source["scroll_speed"]),
			BackgroundEnabled: boolText(boolFromAny(source["banner_background_enabled"], true)),
		},
		Graphics: cgenGraphicsXML{
			Font:         stringFromAny(source["font"]),
			FontSize:     stringFromAny(source["font_size"]),
			BannerHeight: stringFromAny(source["ticker_height"]),
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
			FontSize: stringFromAny(source["text_font_size"]),
			Color:    stringFromAny(source["text_color"]),
			Content:  stringFromAny(source["text"]),
		},
		State: cgenStateXML{
			Mode:      stringFromAny(source["mode"]),
			SMPTEBars: boolText(boolFromAny(source["smpte_bars"], false)),
		},
		Standby: cgenStandbyXML{
			Mode:     stringFromAny(source["standby_mode"]),
			Text:     stringFromAny(source["standby_text"]),
			FontSize: stringFromAny(source["standby_font_size"]),
			YPercent: stringFromAny(source["standby_y_percent"]),
		},
		Sync: cgenSyncXML{
			HardResetMS:            stringFromAny(source["sync_hard_reset_ms"]),
			MaxAudioFramesPerVideo: stringFromAny(source["sync_max_audio_frames_per_video"]),
			SourceBufferMS:         stringFromAny(source["sync_source_buffer_ms"]),
			ReconnectInitialMS:     stringFromAny(source["sync_reconnect_initial_ms"]),
			ReconnectMaxMS:         stringFromAny(source["sync_reconnect_max_ms"]),
			StatusIntervalMS:       stringFromAny(source["sync_status_interval_ms"]),
		},
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
	}
	config, err := normalizeCgen(cgenXML{Enabled: "true", Feeds: []cgenFeedXML{feed}})
	if err != nil {
		return cgenFeedXML{}, err
	}
	return config.Feeds[0], nil
}

func cgenPayload(configPath string, path string, config cgenXML) map[string]any {
	rows := make([]map[string]any, 0, len(config.Feeds))
	for _, feed := range config.Feeds {
		runtime := loadCgenRuntimeStatus(configPath, feed.ID)
		rows = append(rows, map[string]any{
			"id":                              feed.ID,
			"name":                            feed.Name,
			"enabled":                         xmlBool(feed.Enabled, false),
			"mode":                            feed.State.Mode,
			"smpte_bars":                      xmlBool(feed.State.SMPTEBars, false),
			"standby_mode":                    feed.Standby.Mode,
			"standby_text":                    feed.Standby.Text,
			"standby_font_size":               feed.Standby.FontSize,
			"standby_y_percent":               feed.Standby.YPercent,
			"program_input_url":               feed.ProgramInput.URL,
			"program_input_format":            feed.ProgramInput.Format,
			"priority_feed_id":                feed.PriorityInput.FeedID,
			"audio_source":                    feed.PriorityInput.AudioSource,
			"priority_input_format":           feed.PriorityInput.Format,
			"program_output_url":              feed.ProgramOutput.URL,
			"program_output_format":           feed.ProgramOutput.Format,
			"vcodec":                          feed.ProgramOutput.VCodec,
			"acodec":                          feed.ProgramOutput.ACodec,
			"video_bitrate_kbps":              feed.ProgramOutput.VideoBitrateKbps,
			"audio_bitrate_kbps":              feed.ProgramOutput.AudioBitrateKbps,
			"hd_enabled":                      videoRendition(feed.Ladder, "hd").Enabled,
			"hd_bitrate_kbps":                 videoRendition(feed.Ladder, "hd").BitrateKbps,
			"p720_enabled":                    xmlBool(videoRendition(feed.Ladder, "p720").Enabled, false),
			"p720_bitrate_kbps":               videoRendition(feed.Ladder, "p720").BitrateKbps,
			"sd_enabled":                      xmlBool(videoRendition(feed.Ladder, "sd").Enabled, false),
			"sd_bitrate_kbps":                 videoRendition(feed.Ladder, "sd").BitrateKbps,
			"surround_enabled":                xmlBool(audioRendition(feed.Ladder, "surround_51").Enabled, true),
			"surround_bitrate_kbps":           audioRendition(feed.Ladder, "surround_51").BitrateKbps,
			"stereo_enabled":                  xmlBool(audioRendition(feed.Ladder, "stereo").Enabled, true),
			"stereo_bitrate_kbps":             audioRendition(feed.Ladder, "stereo").BitrateKbps,
			"width":                           feed.Video.Width,
			"height":                          feed.Video.Height,
			"fps":                             feed.Video.FPS,
			"interlaced":                      xmlBool(feed.Video.Interlaced, false),
			"field_order":                     feed.Video.FieldOrder,
			"standard":                        feed.Video.Standard,
			"audio_idle":                      feed.Audio.Idle,
			"audio_alert_mode":                feed.Audio.AlertMode,
			"mute_standby_routine":            xmlBool(feed.Audio.MuteStandbyRoutine, true),
			"banner_mode":                     feed.Banner.Mode,
			"ticker_height":                   feed.Banner.TickerHeight,
			"scroll_speed":                    feed.Banner.ScrollSpeed,
			"font":                            feed.Graphics.Font,
			"font_size":                       feed.Graphics.FontSize,
			"banner_background_enabled":       xmlBool(feed.Banner.BackgroundEnabled, true),
			"banner_height":                   feed.Graphics.BannerHeight,
			"text_enabled":                    xmlBool(feed.Text.Enabled, false),
			"text":                            feed.Text.Content,
			"text_font_size":                  feed.Text.FontSize,
			"text_color":                      feed.Text.Color,
			"clock_enabled":                   xmlBool(feed.Clock.Enabled, false),
			"clock_format":                    feed.Clock.Format,
			"clock_x":                         feed.Clock.X,
			"clock_y":                         feed.Clock.Y,
			"clock_font_size":                 feed.Clock.FontSize,
			"clock_color":                     feed.Clock.Color,
			"sync_hard_reset_ms":              feed.Sync.HardResetMS,
			"sync_max_audio_frames_per_video": feed.Sync.MaxAudioFramesPerVideo,
			"sync_source_buffer_ms":           feed.Sync.SourceBufferMS,
			"sync_reconnect_initial_ms":       feed.Sync.ReconnectInitialMS,
			"sync_reconnect_max_ms":           feed.Sync.ReconnectMaxMS,
			"sync_status_interval_ms":         feed.Sync.StatusIntervalMS,
			"updated_at":                      fallbackText(feed.State.UpdatedAt, feed.UpdatedAt),
			"runtime":                         runtime,
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

func loadCgenRuntimeStatus(configPath string, feedID string) map[string]any {
	path := resolveConfigPath(configPath, filepath.Join("runtime", "cgen", safeCgenRuntimeID(feedID)+".status.json"))
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func safeCgenRuntimeID(value string) string {
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-' || r == '_' {
			return r
		}
		return '_'
	}, strings.TrimSpace(value))
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

func normalizeCgenAudioSource(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "routine", "feed", "program_feed":
		return "routine"
	case "both", "priority+routine", "priority_routine", "routine_priority":
		return "both"
	default:
		return "priority"
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

func normalizeCgenStandbyMode(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "smpte", "bars", "color_bars", "colour_bars":
		return "smpte"
	case "black", "blank", "off", "disabled":
		return "black"
	default:
		return "banner"
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

func cleanPercent(value string, fallback string) string {
	text := strings.TrimSpace(value)
	if text == "" {
		text = strings.TrimSpace(fallback)
	}
	n, err := strconv.Atoi(text)
	if err != nil {
		n, _ = strconv.Atoi(strings.TrimSpace(fallback))
	}
	if n < 0 {
		n = 0
	}
	if n > 100 {
		n = 100
	}
	return strconv.Itoa(n)
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

func firstStringFromAny(source map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := stringFromAny(source[key]); strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func boolOrTextFromAny(value any, fallback string) string {
	text := strings.TrimSpace(stringFromAny(value))
	if strings.EqualFold(text, "auto") {
		return "auto"
	}
	if text != "" {
		return boolText(boolFromAny(value, xmlBool(fallback, true)))
	}
	return fallback
}

func videoRendition(ladder cgenLadderXML, id string) cgenVideoRenditionXML {
	for _, video := range ladder.Videos {
		if strings.EqualFold(video.ID, id) {
			return video
		}
	}
	return cgenVideoRenditionXML{}
}

func audioRendition(ladder cgenLadderXML, id string) cgenAudioRenditionXML {
	for _, audio := range ladder.Audios {
		if strings.EqualFold(audio.ID, id) {
			return audio
		}
	}
	return cgenAudioRenditionXML{}
}
