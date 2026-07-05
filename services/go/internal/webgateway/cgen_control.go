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
const defaultCgenEncodersFile = "managed/configs/cgen-encoders.xml"

type cgenXML struct {
	XMLName xml.Name      `xml:"cgen"`
	Enabled string        `xml:"enabled,attr,omitempty"`
	Feeds   []cgenFeedXML `xml:"feed"`
}

type cgenFeedXML struct {
	ID            string               `xml:"id,attr"`
	Name          string               `xml:"name,attr,omitempty"`
	Enabled       string               `xml:"enabled,attr,omitempty"`
	ProgramInput  cgenEndpointXML      `xml:"program>input"`
	PriorityInput cgenPriorityInputXML `xml:"priority>input"`
	ProgramOutput cgenEndpointXML      `xml:"program>output"`
	Video         cgenVideoXML         `xml:"media>video"`
	Audio         cgenAudioXML         `xml:"media>audio"`
	Ladder        cgenLadderXML        `xml:"ladder"`
	Banner        cgenBannerXML        `xml:"presentation>banner"`
	Graphics      cgenGraphicsXML      `xml:"presentation>graphics"`
	Clock         cgenClockXML         `xml:"presentation>clock"`
	Text          cgenTextXML          `xml:"presentation>text"`
	State         cgenStateXML         `xml:"presentation>state"`
	Standby       cgenStandbyXML       `xml:"presentation>standby"`
	Sync          cgenSyncXML          `xml:"sync"`
	UpdatedAt     string               `xml:"updated_at,attr,omitempty"`
}

type cgenEndpointXML struct {
	URL               string `xml:"url,attr,omitempty"`
	Type              string `xml:"type,attr,omitempty"`
	Format            string `xml:"format,attr,omitempty"`
	VCodec            string `xml:"vcodec,attr,omitempty"`
	ACodec            string `xml:"acodec,attr,omitempty"`
	VideoBitrateKbps  string `xml:"video_bitrate_kbps,attr,omitempty"`
	AudioBitrateKbps  string `xml:"audio_bitrate_kbps,attr,omitempty"`
	BrowserAutoSize   string `xml:"browser_auto_size,attr,omitempty"`
	BrowserWidth      string `xml:"browser_width,attr,omitempty"`
	BrowserHeight     string `xml:"browser_height,attr,omitempty"`
	BrowserFPS        string `xml:"browser_fps,attr,omitempty"`
	HardwareDecoder   string `xml:"hardware_decoder,attr,omitempty"`
	HardwareDecoderOn string `xml:"hardware_decoder_enabled,attr,omitempty"`
	ServiceName       string `xml:"service_name,attr,omitempty"`
	ProviderName      string `xml:"provider_name,attr,omitempty"`
	ServiceID         string `xml:"service_id,attr,omitempty"`
	TransportStreamID string `xml:"transport_stream_id,attr,omitempty"`
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
	Program     string `xml:"program,attr,omitempty"`
	AudioPID    string `xml:"audio_pid,attr,omitempty"`
	PMTPID      string `xml:"pmt_pid,attr,omitempty"`
}

type cgenEncodersXML struct {
	XMLName   xml.Name             `xml:"cgenEncoders"`
	UpdatedAt string               `xml:"updated_at,attr,omitempty"`
	Feeds     []cgenEncoderFeedXML `xml:"feed"`
}

type cgenEncoderFeedXML struct {
	ID        string              `xml:"id,attr"`
	Video     cgenEncoderCodecXML `xml:"video"`
	Audio     cgenEncoderCodecXML `xml:"audio"`
	UpdatedAt string              `xml:"updated_at,attr,omitempty"`
}

type cgenEncoderCodecXML struct {
	Codec       string                 `xml:"codec,attr,omitempty"`
	BitrateKbps string                 `xml:"bitrate_kbps,attr,omitempty"`
	GOP         string                 `xml:"gop,attr,omitempty"`
	BFrames     string                 `xml:"bframes,attr,omitempty"`
	Preset      string                 `xml:"preset,attr,omitempty"`
	Tune        string                 `xml:"tune,attr,omitempty"`
	Profile     string                 `xml:"profile,attr,omitempty"`
	Level       string                 `xml:"level,attr,omitempty"`
	Options     []cgenEncoderOptionXML `xml:"option"`
}

type cgenEncoderOptionXML struct {
	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr,omitempty"`
}

type cgenPriorityInputXML struct {
	FeedID      string `xml:"feed_id,attr,omitempty"`
	AudioSource string `xml:"audio_source,attr,omitempty"`
	Format      string `xml:"format,attr,omitempty"`
}

func (f cgenFeedXML) MarshalXML(e *xml.Encoder, start xml.StartElement) error {
	start.Name.Local = "feed"
	start.Attr = append(start.Attr, xml.Attr{Name: xml.Name{Local: "id"}, Value: f.ID})
	if f.Name != "" {
		start.Attr = append(start.Attr, xml.Attr{Name: xml.Name{Local: "name"}, Value: f.Name})
	}
	if f.Enabled != "" {
		start.Attr = append(start.Attr, xml.Attr{Name: xml.Name{Local: "enabled"}, Value: f.Enabled})
	}
	if f.UpdatedAt != "" {
		start.Attr = append(start.Attr, xml.Attr{Name: xml.Name{Local: "updated_at"}, Value: f.UpdatedAt})
	}
	if err := e.EncodeToken(start); err != nil {
		return err
	}
	if err := e.EncodeElement(struct {
		Input  cgenEndpointXML `xml:"input"`
		Output cgenEndpointXML `xml:"output"`
	}{
		Input:  f.ProgramInput,
		Output: f.ProgramOutput,
	}, xml.StartElement{Name: xml.Name{Local: "program"}}); err != nil {
		return err
	}
	if err := e.EncodeElement(struct {
		Input cgenPriorityInputXML `xml:"input"`
	}{
		Input: f.PriorityInput,
	}, xml.StartElement{Name: xml.Name{Local: "priority"}}); err != nil {
		return err
	}
	if err := e.EncodeElement(struct {
		Video cgenVideoXML `xml:"video"`
		Audio cgenAudioXML `xml:"audio"`
	}{
		Video: f.Video,
		Audio: f.Audio,
	}, xml.StartElement{Name: xml.Name{Local: "media"}}); err != nil {
		return err
	}
	if err := e.EncodeElement(f.Ladder, xml.StartElement{Name: xml.Name{Local: "ladder"}}); err != nil {
		return err
	}
	if err := e.EncodeElement(struct {
		Banner   cgenBannerXML   `xml:"banner"`
		Graphics cgenGraphicsXML `xml:"graphics"`
		Clock    cgenClockXML    `xml:"clock"`
		Text     cgenTextXML     `xml:"text"`
		State    cgenStateXML    `xml:"state"`
		Standby  cgenStandbyXML  `xml:"standby"`
	}{
		Banner:   f.Banner,
		Graphics: f.Graphics,
		Clock:    f.Clock,
		Text:     f.Text,
		State:    f.State,
		Standby:  f.Standby,
	}, xml.StartElement{Name: xml.Name{Local: "presentation"}}); err != nil {
		return err
	}
	if err := e.EncodeElement(f.Sync, xml.StartElement{Name: xml.Name{Local: "sync"}}); err != nil {
		return err
	}
	return e.EncodeToken(start.End())
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
	FontWeight        string `xml:"font_weight,attr,omitempty"`
	FontSize          string `xml:"font_size,attr,omitempty"`
	ScrollSpeed       string `xml:"scroll_speed,attr,omitempty"`
	ScrollRepeatMode  string `xml:"scroll_repeat_mode,attr,omitempty"`
	AfterEOMRepeats   string `xml:"after_eom_repeats,attr,omitempty"`
	FixedRepeats      string `xml:"fixed_repeats,attr,omitempty"`
	BackgroundEnabled string `xml:"background_enabled,attr,omitempty"`
}

type cgenBannerFontXML struct {
	Family string `xml:"family,attr,omitempty"`
	Weight string `xml:"weight,attr,omitempty"`
	Size   string `xml:"size,attr,omitempty"`
}

type cgenBannerTickerXML struct {
	Height string              `xml:"height,attr,omitempty"`
	Speed  string              `xml:"speed,attr,omitempty"`
	Repeat cgenTickerRepeatXML `xml:"repeat"`
}

type cgenTickerRepeatXML struct {
	Mode     string `xml:"mode,attr,omitempty"`
	AfterEOM string `xml:"after_eom,attr,omitempty"`
	Count    string `xml:"count,attr,omitempty"`
}

func (b cgenBannerXML) MarshalXML(e *xml.Encoder, start xml.StartElement) error {
	start.Attr = append(start.Attr, xml.Attr{Name: xml.Name{Local: "mode"}, Value: b.Mode})
	start.Attr = append(start.Attr, xml.Attr{Name: xml.Name{Local: "background_enabled"}, Value: b.BackgroundEnabled})
	if err := e.EncodeToken(start); err != nil {
		return err
	}
	if err := e.EncodeElement(cgenBannerFontXML{
		Family: b.Font,
		Weight: b.FontWeight,
		Size:   b.FontSize,
	}, xml.StartElement{Name: xml.Name{Local: "font"}}); err != nil {
		return err
	}
	if err := e.EncodeElement(cgenBannerTickerXML{
		Height: b.TickerHeight,
		Speed:  b.ScrollSpeed,
		Repeat: cgenTickerRepeatXML{
			Mode:     b.ScrollRepeatMode,
			AfterEOM: b.AfterEOMRepeats,
			Count:    b.FixedRepeats,
		},
	}, xml.StartElement{Name: xml.Name{Local: "ticker"}}); err != nil {
		return err
	}
	return e.EncodeToken(start.End())
}

func (b *cgenBannerXML) UnmarshalXML(d *xml.Decoder, start xml.StartElement) error {
	for _, attr := range start.Attr {
		switch attr.Name.Local {
		case "mode":
			b.Mode = attr.Value
		case "ticker_height":
			b.TickerHeight = attr.Value
		case "font":
			b.Font = attr.Value
		case "font_weight":
			b.FontWeight = attr.Value
		case "font_size":
			b.FontSize = attr.Value
		case "scroll_speed":
			b.ScrollSpeed = attr.Value
		case "scroll_repeat_mode":
			b.ScrollRepeatMode = attr.Value
		case "after_eom_repeats":
			b.AfterEOMRepeats = attr.Value
		case "fixed_repeats":
			b.FixedRepeats = attr.Value
		case "background_enabled":
			b.BackgroundEnabled = attr.Value
		}
	}
	for {
		tok, err := d.Token()
		if err != nil {
			return err
		}
		switch tok := tok.(type) {
		case xml.StartElement:
			switch tok.Name.Local {
			case "font":
				var font cgenBannerFontXML
				if err := d.DecodeElement(&font, &tok); err != nil {
					return err
				}
				b.Font = fallbackText(font.Family, b.Font)
				b.FontWeight = fallbackText(font.Weight, b.FontWeight)
				b.FontSize = fallbackText(font.Size, b.FontSize)
			case "ticker":
				var ticker cgenBannerTickerXML
				if err := d.DecodeElement(&ticker, &tok); err != nil {
					return err
				}
				b.TickerHeight = fallbackText(ticker.Height, b.TickerHeight)
				b.ScrollSpeed = fallbackText(ticker.Speed, b.ScrollSpeed)
				b.ScrollRepeatMode = fallbackText(ticker.Repeat.Mode, b.ScrollRepeatMode)
				b.AfterEOMRepeats = fallbackText(ticker.Repeat.AfterEOM, b.AfterEOMRepeats)
				b.FixedRepeats = fallbackText(ticker.Repeat.Count, b.FixedRepeats)
			default:
				var discard struct{}
				if err := d.DecodeElement(&discard, &tok); err != nil {
					return err
				}
			}
		case xml.EndElement:
			if tok.Name.Local == start.Name.Local {
				return nil
			}
		}
	}
}

type cgenGraphicsXML struct {
	Font         string `xml:"font,attr,omitempty"`
	FontWeight   string `xml:"font_weight,attr,omitempty"`
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
	SunnyCat  string `xml:"sunny_cat,attr,omitempty"`
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
	encoderPath := cgenEncodersPath(configPath)
	encoders, err := readCgenEncodersXML(encoderPath)
	if err != nil {
		return nil, err
	}
	return cgenPayload(configPath, path, config, encoderPath, encoders), nil
}

func saveCgenPayload(configPath string, payload map[string]any) (map[string]any, error) {
	rawFeeds, ok := payload["feeds"].([]any)
	if !ok {
		return nil, fmt.Errorf("cgen feeds payload is required")
	}
	config := cgenXML{Enabled: boolText(boolFromAny(payload["enabled"], true))}
	encoders := cgenEncodersXML{UpdatedAt: time.Now().UTC().Format(time.RFC3339)}
	for _, raw := range rawFeeds {
		feed, err := cgenFeedFromMap(raw)
		if err != nil {
			return nil, err
		}
		config.Feeds = append(config.Feeds, feed)
		encoderFeed, err := cgenEncoderFeedFromMap(raw, feed)
		if err != nil {
			return nil, err
		}
		encoders.Feeds = append(encoders.Feeds, encoderFeed)
	}
	path := cgenPath(configPath)
	if err := writeCgenXML(path, config); err != nil {
		return nil, err
	}
	if err := writeCgenEncodersXML(cgenEncodersPath(configPath), encoders); err != nil {
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
	applyCgenRuntimeFields(&feed, payload)
	switch action {
	case "release":
		feed.State.Mode = "release"
		feed.State.SMPTEBars = "false"
		feed.State.SunnyCat = "false"
		feed.Clock.Enabled = "false"
		feed.Text.Enabled = "false"
	case "overlay":
		feed.State.Mode = "overlay"
	case "smpte_bars":
		feed.State.Mode = "overlay"
		feed.State.SMPTEBars = boolText(boolFromAny(payload["enabled"], true))
	case "clock":
		feed.Clock.Enabled = boolText(boolFromAny(payload["enabled"], true))
	case "unleash_sunny":
		feed.State.Mode = "overlay"
		feed.State.SunnyCat = "true"
	case "banish_sunny":
		feed.State.SunnyCat = "false"
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

func applyCgenRuntimeFields(feed *cgenFeedXML, source map[string]any) {
	feed.Banner.Mode = fallbackText(stringFromAny(source["banner_mode"]), feed.Banner.Mode)
	feed.Banner.TickerHeight = fallbackText(firstStringFromAny(source, "ticker_height", "banner_height"), feed.Banner.TickerHeight)
	feed.Banner.Font = fallbackText(stringFromAny(source["font"]), feed.Banner.Font)
	feed.Banner.FontWeight = fallbackText(stringFromAny(source["font_weight"]), feed.Banner.FontWeight)
	feed.Banner.FontSize = fallbackText(stringFromAny(source["font_size"]), feed.Banner.FontSize)
	feed.Banner.ScrollSpeed = fallbackText(stringFromAny(source["scroll_speed"]), feed.Banner.ScrollSpeed)
	feed.Banner.ScrollRepeatMode = fallbackText(stringFromAny(source["scroll_repeat_mode"]), feed.Banner.ScrollRepeatMode)
	feed.Banner.AfterEOMRepeats = fallbackText(stringFromAny(source["after_eom_repeats"]), feed.Banner.AfterEOMRepeats)
	feed.Banner.FixedRepeats = fallbackText(stringFromAny(source["fixed_repeats"]), feed.Banner.FixedRepeats)
	if _, ok := source["banner_background_enabled"]; ok {
		feed.Banner.BackgroundEnabled = boolText(boolFromAny(source["banner_background_enabled"], true))
	}
	feed.Graphics.Font = fallbackText(stringFromAny(source["font"]), feed.Graphics.Font)
	feed.Graphics.FontWeight = fallbackText(stringFromAny(source["font_weight"]), feed.Graphics.FontWeight)
	feed.Graphics.FontSize = fallbackText(stringFromAny(source["font_size"]), feed.Graphics.FontSize)
	feed.Graphics.BannerHeight = fallbackText(firstStringFromAny(source, "banner_height", "ticker_height"), feed.Graphics.BannerHeight)
	if _, ok := source["clock_enabled"]; ok {
		feed.Clock.Enabled = boolText(boolFromAny(source["clock_enabled"], false))
	}
	feed.Clock.Format = fallbackText(stringFromAny(source["clock_format"]), feed.Clock.Format)
	feed.Clock.X = fallbackText(stringFromAny(source["clock_x"]), feed.Clock.X)
	feed.Clock.Y = fallbackText(stringFromAny(source["clock_y"]), feed.Clock.Y)
	feed.Clock.FontSize = fallbackText(stringFromAny(source["clock_font_size"]), feed.Clock.FontSize)
	feed.Clock.Color = fallbackText(stringFromAny(source["clock_color"]), feed.Clock.Color)
	if _, ok := source["text_enabled"]; ok {
		feed.Text.Enabled = boolText(boolFromAny(source["text_enabled"], false))
	}
	if _, ok := source["text"]; ok {
		feed.Text.Content = stringFromAny(source["text"])
	}
	feed.Text.FontSize = fallbackText(stringFromAny(source["text_font_size"]), feed.Text.FontSize)
	feed.Text.Color = fallbackText(stringFromAny(source["text_color"]), feed.Text.Color)
	feed.State.Mode = fallbackText(stringFromAny(source["mode"]), feed.State.Mode)
	if _, ok := source["smpte_bars"]; ok {
		feed.State.SMPTEBars = boolText(boolFromAny(source["smpte_bars"], false))
	}
	if _, ok := source["sunny_cat"]; ok {
		feed.State.SunnyCat = boolText(boolFromAny(source["sunny_cat"], false))
	}
	feed.Standby.Mode = fallbackText(stringFromAny(source["standby_mode"]), feed.Standby.Mode)
	feed.Standby.Text = fallbackText(stringFromAny(source["standby_text"]), feed.Standby.Text)
	feed.Standby.FontSize = fallbackText(stringFromAny(source["standby_font_size"]), feed.Standby.FontSize)
	feed.Standby.YPercent = fallbackText(stringFromAny(source["standby_y_percent"]), feed.Standby.YPercent)
}

func cgenPath(configPath string) string {
	return resolveConfigPath(configPath, defaultCgenFile)
}

func cgenEncodersPath(configPath string) string {
	return resolveConfigPath(configPath, defaultCgenEncodersFile)
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

func readCgenEncodersXML(path string) (cgenEncodersXML, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return cgenEncodersXML{}, nil
		}
		return cgenEncodersXML{}, err
	}
	var config cgenEncodersXML
	if err := xml.Unmarshal(raw, &config); err != nil {
		return cgenEncodersXML{}, fmt.Errorf("parse cgen encoders XML: %w", err)
	}
	return normalizeCgenEncoders(config), nil
}

func writeCgenEncodersXML(path string, config cgenEncodersXML) error {
	config = normalizeCgenEncoders(config)
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

func rejectLegacyCgenXML(raw []byte, path string) error {
	decoder := xml.NewDecoder(bytes.NewReader(raw))
	stack := []string{}
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return fmt.Errorf("scan cgen XML %s: %w", path, err)
		}
		switch tok := token.(type) {
		case xml.StartElement:
			name := tok.Name.Local
			parent := ""
			if len(stack) > 0 {
				parent = stack[len(stack)-1]
			}
			if legacyCgenElement(parent, name) {
				return fmt.Errorf("legacy cgen <%s> element is no longer supported in %s; use programInput, priorityInput, and programOutput", name, path)
			}
			for _, attr := range tok.Attr {
				if legacyCgenAttribute(name, attr.Name.Local) {
					return fmt.Errorf("legacy cgen %s @%s attribute is no longer supported in %s", name, attr.Name.Local, path)
				}
			}
			stack = append(stack, name)
		case xml.EndElement:
			if len(stack) > 0 {
				stack = stack[:len(stack)-1]
			}
		default:
			continue
		}
	}
}

func legacyCgenElement(parent string, element string) bool {
	return (element == "output" && parent != "program") ||
		element == "alertOutput" ||
		(element == "input" && parent != "program" && parent != "priority")
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
		feed.Banner.FontWeight = normalizeCgenFontWeight(feed.Banner.FontWeight)
		feed.Banner.FontSize = cleanPositive(feed.Banner.FontSize, "58")
		feed.Banner.ScrollSpeed = cleanPositive(feed.Banner.ScrollSpeed, "8")
		feed.Banner.ScrollRepeatMode = normalizeCgenScrollRepeatMode(feed.Banner.ScrollRepeatMode)
		feed.Banner.AfterEOMRepeats = cleanNonNegative(feed.Banner.AfterEOMRepeats, "0")
		feed.Banner.FixedRepeats = cleanPositive(feed.Banner.FixedRepeats, "1")
		feed.Banner.BackgroundEnabled = boolText(xmlBool(feed.Banner.BackgroundEnabled, true))
		feed.Graphics.Font = fallbackText(strings.TrimSpace(feed.Graphics.Font), feed.Banner.Font)
		feed.Graphics.FontWeight = normalizeCgenFontWeight(fallbackText(feed.Graphics.FontWeight, feed.Banner.FontWeight))
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
		feed.State.SunnyCat = boolText(xmlBool(feed.State.SunnyCat, false))
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

func normalizeCgenEncoders(config cgenEncodersXML) cgenEncodersXML {
	config.UpdatedAt = strings.TrimSpace(config.UpdatedAt)
	seen := map[string]int{}
	out := make([]cgenEncoderFeedXML, 0, len(config.Feeds))
	for _, feed := range config.Feeds {
		feed.ID = cleanCgenID(feed.ID)
		if feed.ID == "" {
			continue
		}
		feed.Video = normalizeCgenEncoderCodec(feed.Video, true)
		feed.Audio = normalizeCgenEncoderCodec(feed.Audio, false)
		feed.UpdatedAt = strings.TrimSpace(feed.UpdatedAt)
		if index, ok := seen[feed.ID]; ok {
			out[index] = feed
			continue
		}
		seen[feed.ID] = len(out)
		out = append(out, feed)
	}
	sort.SliceStable(out, func(i, j int) bool { return strings.ToLower(out[i].ID) < strings.ToLower(out[j].ID) })
	config.Feeds = out
	return config
}

func normalizeCgenEncoderCodec(value cgenEncoderCodecXML, video bool) cgenEncoderCodecXML {
	value.Codec = strings.TrimSpace(value.Codec)
	value.BitrateKbps = cleanOptionalPositive(value.BitrateKbps)
	value.GOP = cleanOptionalPositive(value.GOP)
	value.BFrames = cleanNonNegative(value.BFrames, "")
	value.Preset = normalizeCgenEncoderPreset(value.Preset)
	value.Tune = normalizeCgenEncoderTune(value.Tune)
	value.Profile = strings.TrimSpace(value.Profile)
	value.Level = strings.TrimSpace(value.Level)
	options := make([]cgenEncoderOptionXML, 0, len(value.Options))
	seen := map[string]int{}
	for _, option := range value.Options {
		option.Name = normalizeCgenEncoderOptionName(option.Name)
		if option.Name == "" {
			continue
		}
		option.Value = strings.TrimSpace(option.Value)
		if index, ok := seen[option.Name]; ok {
			options[index] = option
			continue
		}
		seen[option.Name] = len(options)
		options = append(options, option)
	}
	sort.SliceStable(options, func(i, j int) bool { return options[i].Name < options[j].Name })
	value.Options = options
	if !video {
		value.GOP = ""
		value.BFrames = ""
		value.Preset = ""
		value.Tune = ""
	}
	return value
}

func normalizeCgenEncoderPreset(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow":
		return value
	default:
		return ""
	}
}

func normalizeCgenEncoderTune(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "zerolatency", "film", "animation", "grain", "stillimage", "fastdecode", "psnr", "ssim":
		return value
	default:
		return ""
	}
}

func normalizeCgenEncoderOptionName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	for _, ch := range value {
		if ch >= 'a' && ch <= 'z' || ch >= '0' && ch <= '9' || ch == '-' || ch == '_' || ch == '.' {
			continue
		}
		return ""
	}
	return value
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
		Program:     "1",
		AudioPID:    "258",
		PMTPID:      "4096",
	})
	stereo := normalizeAudioRendition(audios["stereo"], cgenAudioRenditionXML{
		ID:          "stereo",
		Enabled:     "true",
		Channels:    "2",
		ACodec:      feed.ProgramOutput.ACodec,
		BitrateKbps: fallbackText(feed.ProgramOutput.AudioBitrateKbps, "192"),
		Language:    "eng",
		Program:     "1",
		AudioPID:    "257",
		PMTPID:      "4096",
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
	value.Program = cleanPositive(value.Program, fallback.Program)
	value.AudioPID = cleanPositive(value.AudioPID, fallback.AudioPID)
	value.PMTPID = cleanPositive(value.PMTPID, fallback.PMTPID)
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
			URL:             stringFromAny(source["program_input_url"]),
			Type:            stringFromAny(source["program_input_type"]),
			Format:          stringFromAny(source["program_input_format"]),
			BrowserAutoSize: boolText(boolFromAny(source["browser_auto_size"], true)),
			BrowserWidth:    stringFromAny(source["browser_width"]),
			BrowserHeight:   stringFromAny(source["browser_height"]),
			BrowserFPS:      stringFromAny(source["browser_fps"]),
			HardwareDecoder: stringFromAny(source["hardware_decoder"]),
			HardwareDecoderOn: boolText(
				boolFromAny(source["hardware_decoder_enabled"], false),
			),
		},
		PriorityInput: cgenPriorityInputXML{
			FeedID:      stringFromAny(source["priority_feed_id"]),
			AudioSource: stringFromAny(source["audio_source"]),
			Format:      stringFromAny(source["priority_input_format"]),
		},
		ProgramOutput: cgenEndpointXML{
			URL:               stringFromAny(source["program_output_url"]),
			Format:            stringFromAny(source["program_output_format"]),
			VCodec:            stringFromAny(source["vcodec"]),
			ACodec:            stringFromAny(source["acodec"]),
			VideoBitrateKbps:  firstStringFromAny(source, "hd_bitrate_kbps", "video_bitrate_kbps"),
			AudioBitrateKbps:  firstStringFromAny(source, "stereo_bitrate_kbps", "audio_bitrate_kbps"),
			ServiceName:       stringFromAny(source["service_name"]),
			ProviderName:      stringFromAny(source["provider_name"]),
			ServiceID:         stringFromAny(source["service_id"]),
			TransportStreamID: stringFromAny(source["transport_stream_id"]),
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
					Program:     fallbackText(stringFromAny(source["hd_program"]), "1"),
					VideoPID:    fallbackText(stringFromAny(source["hd_video_pid"]), "256"),
					PMTPID:      fallbackText(stringFromAny(source["hd_pmt_pid"]), "4096"),
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
					Program:     fallbackText(stringFromAny(source["p720_program"]), "2"),
					VideoPID:    fallbackText(stringFromAny(source["p720_video_pid"]), "288"),
					PMTPID:      fallbackText(stringFromAny(source["p720_pmt_pid"]), "4097"),
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
					Program:     fallbackText(stringFromAny(source["sd_program"]), "3"),
					VideoPID:    fallbackText(stringFromAny(source["sd_video_pid"]), "320"),
					PMTPID:      fallbackText(stringFromAny(source["sd_pmt_pid"]), "4098"),
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
					Program:     fallbackText(stringFromAny(source["stereo_program"]), "1"),
					AudioPID:    fallbackText(stringFromAny(source["stereo_audio_pid"]), "257"),
					PMTPID:      fallbackText(stringFromAny(source["stereo_pmt_pid"]), "4096"),
				},
				{
					ID:          "surround_51",
					Enabled:     boolText(boolFromAny(source["surround_enabled"], true)),
					Channels:    "6",
					ACodec:      stringFromAny(source["acodec"]),
					BitrateKbps: fallbackText(stringFromAny(source["surround_bitrate_kbps"]), "384"),
					Language:    "eng",
					Program:     fallbackText(stringFromAny(source["surround_program"]), "1"),
					AudioPID:    fallbackText(stringFromAny(source["surround_audio_pid"]), "258"),
					PMTPID:      fallbackText(stringFromAny(source["surround_pmt_pid"]), "4096"),
				},
			},
		},
		Banner: cgenBannerXML{
			Mode:              stringFromAny(source["banner_mode"]),
			TickerHeight:      stringFromAny(source["ticker_height"]),
			Font:              stringFromAny(source["font"]),
			FontWeight:        fallbackText(stringFromAny(source["font_weight"]), "regular"),
			FontSize:          stringFromAny(source["font_size"]),
			ScrollSpeed:       stringFromAny(source["scroll_speed"]),
			ScrollRepeatMode:  stringFromAny(source["scroll_repeat_mode"]),
			AfterEOMRepeats:   stringFromAny(source["after_eom_repeats"]),
			FixedRepeats:      stringFromAny(source["fixed_repeats"]),
			BackgroundEnabled: boolText(boolFromAny(source["banner_background_enabled"], true)),
		},
		Graphics: cgenGraphicsXML{
			Font:         stringFromAny(source["font"]),
			FontWeight:   fallbackText(stringFromAny(source["font_weight"]), "regular"),
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
			SunnyCat:  boolText(boolFromAny(source["sunny_cat"], false)),
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

func cgenEncoderFeedFromMap(raw any, feed cgenFeedXML) (cgenEncoderFeedXML, error) {
	source, ok := raw.(map[string]any)
	if !ok {
		return cgenEncoderFeedXML{}, fmt.Errorf("cgen feed entries must be objects")
	}
	encoderFeed := cgenEncoderFeedXML{
		ID:        feed.ID,
		UpdatedAt: time.Now().UTC().Format(time.RFC3339),
		Video: cgenEncoderCodecXML{
			Codec:       firstStringFromAny(source, "video_encoder_codec", "vcodec"),
			BitrateKbps: firstStringFromAny(source, "video_encoder_bitrate_kbps", "hd_bitrate_kbps", "video_bitrate_kbps"),
			GOP:         stringFromAny(source["video_gop"]),
			BFrames:     stringFromAny(source["video_bframes"]),
			Preset:      stringFromAny(source["video_preset"]),
			Tune:        stringFromAny(source["video_tune"]),
			Profile:     stringFromAny(source["video_profile"]),
			Level:       stringFromAny(source["video_level"]),
			Options:     encoderOptionsFromAny(source["video_encoder_options"]),
		},
		Audio: cgenEncoderCodecXML{
			Codec:       firstStringFromAny(source, "audio_encoder_codec", "acodec"),
			BitrateKbps: firstStringFromAny(source, "audio_encoder_bitrate_kbps", "stereo_bitrate_kbps", "audio_bitrate_kbps"),
			Profile:     stringFromAny(source["audio_profile"]),
			Level:       stringFromAny(source["audio_level"]),
			Options:     encoderOptionsFromAny(source["audio_encoder_options"]),
		},
	}
	return normalizeCgenEncoders(cgenEncodersXML{Feeds: []cgenEncoderFeedXML{encoderFeed}}).Feeds[0], nil
}

func encoderOptionsFromAny(raw any) []cgenEncoderOptionXML {
	items, ok := raw.([]any)
	if !ok {
		return nil
	}
	options := make([]cgenEncoderOptionXML, 0, len(items))
	for _, item := range items {
		source, ok := item.(map[string]any)
		if !ok {
			continue
		}
		options = append(options, cgenEncoderOptionXML{
			Name:  stringFromAny(source["name"]),
			Value: stringFromAny(source["value"]),
		})
	}
	return options
}

func cgenPayload(configPath string, path string, config cgenXML, encoderPath string, encoders cgenEncodersXML) map[string]any {
	rows := make([]map[string]any, 0, len(config.Feeds))
	for _, feed := range config.Feeds {
		runtime := loadCgenRuntimeStatus(configPath, feed.ID)
		row := map[string]any{
			"id":                              feed.ID,
			"name":                            feed.Name,
			"enabled":                         xmlBool(feed.Enabled, false),
			"mode":                            feed.State.Mode,
			"smpte_bars":                      xmlBool(feed.State.SMPTEBars, false),
			"sunny_cat":                       xmlBool(feed.State.SunnyCat, false),
			"standby_mode":                    feed.Standby.Mode,
			"standby_text":                    feed.Standby.Text,
			"standby_font_size":               feed.Standby.FontSize,
			"standby_y_percent":               feed.Standby.YPercent,
			"program_input_url":               feed.ProgramInput.URL,
			"program_input_type":              normalizeCgenInputType(feed.ProgramInput.Type),
			"program_input_format":            feed.ProgramInput.Format,
			"browser_auto_size":               xmlBool(feed.ProgramInput.BrowserAutoSize, true),
			"browser_width":                   fallbackText(feed.ProgramInput.BrowserWidth, feed.Video.Width),
			"browser_height":                  fallbackText(feed.ProgramInput.BrowserHeight, feed.Video.Height),
			"browser_fps":                     fallbackText(feed.ProgramInput.BrowserFPS, "60"),
			"hardware_decoder_enabled":        xmlBool(feed.ProgramInput.HardwareDecoderOn, false),
			"hardware_decoder":                feed.ProgramInput.HardwareDecoder,
			"priority_feed_id":                feed.PriorityInput.FeedID,
			"audio_source":                    feed.PriorityInput.AudioSource,
			"priority_input_format":           feed.PriorityInput.Format,
			"program_output_url":              feed.ProgramOutput.URL,
			"program_output_format":           feed.ProgramOutput.Format,
			"vcodec":                          feed.ProgramOutput.VCodec,
			"acodec":                          feed.ProgramOutput.ACodec,
			"video_bitrate_kbps":              feed.ProgramOutput.VideoBitrateKbps,
			"audio_bitrate_kbps":              feed.ProgramOutput.AudioBitrateKbps,
			"service_name":                    feed.ProgramOutput.ServiceName,
			"provider_name":                   feed.ProgramOutput.ProviderName,
			"service_id":                      feed.ProgramOutput.ServiceID,
			"transport_stream_id":             feed.ProgramOutput.TransportStreamID,
			"hd_enabled":                      videoRendition(feed.Ladder, "hd").Enabled,
			"hd_bitrate_kbps":                 videoRendition(feed.Ladder, "hd").BitrateKbps,
			"hd_program":                      videoRendition(feed.Ladder, "hd").Program,
			"hd_video_pid":                    videoRendition(feed.Ladder, "hd").VideoPID,
			"hd_pmt_pid":                      videoRendition(feed.Ladder, "hd").PMTPID,
			"p720_enabled":                    xmlBool(videoRendition(feed.Ladder, "p720").Enabled, false),
			"p720_bitrate_kbps":               videoRendition(feed.Ladder, "p720").BitrateKbps,
			"p720_program":                    videoRendition(feed.Ladder, "p720").Program,
			"p720_video_pid":                  videoRendition(feed.Ladder, "p720").VideoPID,
			"p720_pmt_pid":                    videoRendition(feed.Ladder, "p720").PMTPID,
			"sd_enabled":                      xmlBool(videoRendition(feed.Ladder, "sd").Enabled, false),
			"sd_bitrate_kbps":                 videoRendition(feed.Ladder, "sd").BitrateKbps,
			"sd_program":                      videoRendition(feed.Ladder, "sd").Program,
			"sd_video_pid":                    videoRendition(feed.Ladder, "sd").VideoPID,
			"sd_pmt_pid":                      videoRendition(feed.Ladder, "sd").PMTPID,
			"surround_enabled":                xmlBool(audioRendition(feed.Ladder, "surround_51").Enabled, true),
			"surround_bitrate_kbps":           audioRendition(feed.Ladder, "surround_51").BitrateKbps,
			"surround_program":                audioRendition(feed.Ladder, "surround_51").Program,
			"surround_audio_pid":              audioRendition(feed.Ladder, "surround_51").AudioPID,
			"surround_pmt_pid":                audioRendition(feed.Ladder, "surround_51").PMTPID,
			"stereo_enabled":                  xmlBool(audioRendition(feed.Ladder, "stereo").Enabled, true),
			"stereo_bitrate_kbps":             audioRendition(feed.Ladder, "stereo").BitrateKbps,
			"stereo_program":                  audioRendition(feed.Ladder, "stereo").Program,
			"stereo_audio_pid":                audioRendition(feed.Ladder, "stereo").AudioPID,
			"stereo_pmt_pid":                  audioRendition(feed.Ladder, "stereo").PMTPID,
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
			"scroll_repeat_mode":              feed.Banner.ScrollRepeatMode,
			"after_eom_repeats":               feed.Banner.AfterEOMRepeats,
			"fixed_repeats":                   feed.Banner.FixedRepeats,
			"font":                            feed.Graphics.Font,
			"font_weight":                     fallbackText(fallbackText(feed.Graphics.FontWeight, feed.Banner.FontWeight), "regular"),
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
		}
		for key, value := range cgenEncoderPayload(encoders, feed.ID) {
			row[key] = value
		}
		rows = append(rows, row)
	}
	return map[string]any{
		"path":         filepath.ToSlash(path),
		"encoder_path": filepath.ToSlash(encoderPath),
		"enabled":      xmlBool(config.Enabled, true),
		"feeds":        rows,
		"summary": map[string]any{
			"count": len(rows),
		},
	}
}

func cgenEncoderPayload(encoders cgenEncodersXML, feedID string) map[string]any {
	feed := cgenEncoderFeed(encoders, feedID)
	videoOptions := encoderOptionsPayload(feed.Video.Options)
	audioOptions := encoderOptionsPayload(feed.Audio.Options)
	return map[string]any{
		"video_encoder_codec":        feed.Video.Codec,
		"video_encoder_bitrate_kbps": feed.Video.BitrateKbps,
		"video_gop":                  feed.Video.GOP,
		"video_bframes":              feed.Video.BFrames,
		"video_preset":               feed.Video.Preset,
		"video_tune":                 feed.Video.Tune,
		"video_profile":              feed.Video.Profile,
		"video_level":                feed.Video.Level,
		"video_encoder_options":      videoOptions,
		"audio_encoder_codec":        feed.Audio.Codec,
		"audio_encoder_bitrate_kbps": feed.Audio.BitrateKbps,
		"audio_profile":              feed.Audio.Profile,
		"audio_level":                feed.Audio.Level,
		"audio_encoder_options":      audioOptions,
	}
}

func cgenEncoderFeed(encoders cgenEncodersXML, feedID string) cgenEncoderFeedXML {
	for _, feed := range encoders.Feeds {
		if strings.EqualFold(feed.ID, feedID) {
			return feed
		}
	}
	return cgenEncoderFeedXML{}
}

func encoderOptionsPayload(options []cgenEncoderOptionXML) []map[string]any {
	out := make([]map[string]any, 0, len(options))
	for _, option := range options {
		out = append(out, map[string]any{
			"name":  option.Name,
			"value": option.Value,
		})
	}
	return out
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
	value.Type = normalizeCgenInputType(value.Type)
	value.Format = strings.TrimSpace(value.Format)
	if value.Type == "browser" && value.Format == "" {
		value.Format = "cef"
	}
	value.VCodec = strings.TrimSpace(value.VCodec)
	value.ACodec = strings.TrimSpace(value.ACodec)
	value.VideoBitrateKbps = cleanOptionalPositive(value.VideoBitrateKbps)
	value.AudioBitrateKbps = cleanOptionalPositive(value.AudioBitrateKbps)
	value.BrowserAutoSize = boolText(xmlBool(value.BrowserAutoSize, true))
	value.BrowserWidth = cleanPositive(value.BrowserWidth, "1920")
	value.BrowserHeight = cleanPositive(value.BrowserHeight, "1080")
	value.BrowserFPS = cleanCgenBrowserFPS(value.BrowserFPS, "60")
	value.HardwareDecoder = cleanGstElementName(value.HardwareDecoder)
	value.HardwareDecoderOn = boolText(xmlBool(value.HardwareDecoderOn, false))
	value.ServiceName = strings.TrimSpace(value.ServiceName)
	value.ProviderName = strings.TrimSpace(value.ProviderName)
	value.ServiceID = cleanOptionalPositive(value.ServiceID)
	value.TransportStreamID = cleanOptionalPositive(value.TransportStreamID)
	if value.Type == "stream" {
		value.Type = ""
		value.BrowserAutoSize = ""
		value.BrowserWidth = ""
		value.BrowserHeight = ""
		value.BrowserFPS = ""
		if value.HardwareDecoderOn == "false" {
			value.HardwareDecoderOn = ""
			value.HardwareDecoder = ""
		}
	} else {
		value.HardwareDecoder = ""
		value.HardwareDecoderOn = ""
	}
	return value
}

func cleanGstElementName(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	for _, ch := range value {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '_' || ch == '-' {
			continue
		}
		return ""
	}
	return value
}

func normalizeCgenInputType(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "browser", "cef", "browser_source":
		return "browser"
	default:
		return "stream"
	}
}

func cleanCgenBrowserFPS(value string, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		value = strings.TrimSpace(fallback)
	}
	n, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	if n <= 0 {
		return "0"
	}
	if n < 5 {
		return "5"
	}
	if n > 120 {
		return "120"
	}
	return strconv.Itoa(n)
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

func normalizeCgenFontWeight(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "thin", "100":
		return "thin"
	case "extra-light", "extralight", "ultra-light", "ultralight", "200":
		return "extra-light"
	case "light", "300":
		return "light"
	case "medium", "500":
		return "medium"
	case "semi-bold", "semibold", "demi-bold", "demibold", "600":
		return "semibold"
	case "bold", "700":
		return "bold"
	case "extra-bold", "extrabold", "ultra-bold", "ultrabold", "800":
		return "extra-bold"
	case "black", "heavy", "900":
		return "black"
	default:
		return "regular"
	}
}

func normalizeCgenScrollRepeatMode(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "fixed", "fixed_repeats", "count", "count_only":
		return "fixed"
	default:
		return "until_audio_end"
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

func cleanNonNegative(value string, fallback string) string {
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
