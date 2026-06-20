package webhook

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alerttext"
)

const serviceID = "haze-webhook"

var testEventCodes = map[string]struct{}{"RWT": {}, "RMT": {}, "DMO": {}}
var adminEventCodes = map[string]struct{}{"NMN": {}, "ADR": {}, "TXP": {}, "TXF": {}, "TXO": {}}

var codecExt = map[string]string{
	"libopus":    "ogg",
	"libmp3lame": "mp3",
	"aac":        "aac",
	"libvorbis":  "ogg",
	"pcm_s16le":  "wav",
	"wav":        "wav",
}

const maxDiscordAttachmentBytes = 8 * 1024 * 1024

func Run(ctx context.Context, options Options) error {
	if options.Timeout <= 0 {
		options.Timeout = 15 * time.Second
	}
	if strings.TrimSpace(options.ConfigPath) == "" {
		options.ConfigPath = "config.yaml"
	}
	loadDotEnv(filepath.Join(filepath.Dir(filepath.Clean(options.ConfigPath)), ".env"))
	loadDotEnv(".env")
	for {
		cfg, err := loadConfig(options)
		if err != nil {
			return err
		}
		bridge, err := connectBridge(ctx, options.BridgeAddr)
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			log.Printf("webhook waiting for event bridge: %v", err)
			sleepOrDone(ctx, time.Second)
			continue
		}
		service := &Service{
			cfg:     cfg,
			bridge:  bridge,
			client:  &http.Client{Timeout: options.Timeout},
			timeout: options.Timeout,
		}
		err = service.runConnected(ctx)
		_ = bridge.Close()
		if ctx.Err() != nil {
			return nil
		}
		log.Printf("webhook event bridge disconnected: %v", err)
		sleepOrDone(ctx, time.Second)
	}
}

type Service struct {
	cfg     loadedConfig
	bridge  *bridgeClient
	client  *http.Client
	timeout time.Duration
	cache   configCache
}

func (s *Service) runConnected(ctx context.Context) error {
	log.Printf("Discord webhook service connected to event bridge")
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case event, ok := <-s.bridge.Events():
			if !ok {
				return fmt.Errorf("event bridge closed")
			}
			s.handleEvent(ctx, event)
		}
	}
}

func (s *Service) handleEvent(ctx context.Context, event map[string]any) {
	if firstText(event, nil, "type") != "cap.alert.audio.ready" {
		return
	}
	data := mapAt(event, "data")
	feedID := firstText(event, data, "feed_id")
	if feedID == "" {
		return
	}
	configs, err := s.cache.load(s.cfg.WebhooksPath)
	if err != nil {
		log.Printf("Discord webhook config load failed: %v", err)
		return
	}
	sameEvent := strings.ToUpper(firstNonBlank(firstText(nil, data, "same_event"), firstText(nil, data, "event"), "ADR"))
	_, isTest := testEventCodes[sameEvent]
	_, isAdmin := adminEventCodes[sameEvent]
	for _, cfg := range configs {
		if !cfg.enabled() || cfg.FeedID != feedID {
			continue
		}
		if isTest && !cfg.logTestAlerts() {
			continue
		}
		if isAdmin && !cfg.logAdminAlerts() {
			continue
		}
		if err := s.dispatch(ctx, cfg, feedID, sameEvent, data); err != nil {
			log.Printf("[%s] Discord webhook dispatch failed: %v", feedID, err)
		}
	}
}

func (s *Service) dispatch(ctx context.Context, cfg WebhookConfig, feedID string, sameEvent string, data map[string]any) error {
	payload := buildPayload(cfg, feedID, sameEvent, data, s.cfg.ListenBaseURL)
	audioPath := resolvePath(s.cfg.BaseDir, firstText(nil, data, "audio_path"))
	attached := false
	var audio attachment
	var err error
	if cfg.audioEnabled() && audioPath != "" {
		audio, err = transcodeAlertAudio(ctx, audioPath, cfg.EmbedAudio.Codec, intAt(data, "sample_rate", 48000), intAt(data, "channels", 1))
		if err != nil {
			log.Printf("[%s] Discord webhook audio attachment skipped: %v", feedID, err)
		} else {
			attached = true
		}
	}
	status, detail, err := s.post(ctx, cfg.WebhookURL, payload, audio)
	if err != nil {
		return err
	}
	if attached && (status == http.StatusBadRequest || status == http.StatusRequestEntityTooLarge) {
		log.Printf("[%s] Discord webhook rejected alert attachment, retrying without audio", feedID)
		status, detail, err = s.post(ctx, cfg.WebhookURL, payload, attachment{})
		if err != nil {
			return err
		}
	}
	if status < 200 || status >= 300 {
		if detail != "" {
			return fmt.Errorf("HTTP %d: %s", status, clipText(detail, 240))
		}
		return fmt.Errorf("HTTP %d", status)
	}
	log.Printf("[%s] Discord webhook dispatched: %s (%s)", feedID, sameEvent, firstNonBlank(firstText(nil, data, "alert_id"), firstText(nil, data, "title")))
	return nil
}

type discordPayload struct {
	Username  string         `json:"username,omitempty"`
	AvatarURL string         `json:"avatar_url,omitempty"`
	Embeds    []discordEmbed `json:"embeds"`
}

type discordEmbed struct {
	Title       string              `json:"title,omitempty"`
	Description string              `json:"description,omitempty"`
	Color       int                 `json:"color,omitempty"`
	Fields      []discordEmbedField `json:"fields,omitempty"`
	Footer      *discordFooter      `json:"footer,omitempty"`
	Timestamp   string              `json:"timestamp,omitempty"`
}

type discordEmbedField struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Inline bool   `json:"inline"`
}

type discordFooter struct {
	Text string `json:"text"`
}

func buildPayload(cfg WebhookConfig, feedID string, sameEvent string, data map[string]any, listenBaseURL string) discordPayload {
	title := firstNonBlank(firstText(nil, data, "headline"), firstText(nil, data, "title", "header"), sameEvent)
	description := firstText(nil, data, "description")
	generated := firstText(nil, data, "alert_text", "same_message", "message")
	if description == "" || len(description) > 4096 {
		description = generated
	}
	color := colorInt(firstNonBlank(
		firstText(nil, data, "background_color"),
		alerttext.PickBannerColor([]alerttext.AlertVisualInput{{
			Severity: firstText(nil, data, "severity"),
			Event:    firstNonBlank(firstText(nil, data, "event"), title, sameEvent),
		}}),
	))
	embed := discordEmbed{
		Title:       clipText(title, 256),
		Description: clipText(description, 4096),
		Color:       color,
		Timestamp:   time.Now().UTC().Format(time.RFC3339Nano),
	}
	embed.Fields = append(embed.Fields,
		discordEmbedField{Name: "Feed", Value: clipText(feedID, 1024), Inline: true},
		discordEmbedField{Name: "Event", Value: clipText(firstNonBlank(firstText(nil, data, "event"), sameEvent), 1024), Inline: true},
		discordEmbedField{Name: "Severity", Value: clipText(firstNonBlank(firstText(nil, data, "severity"), "Unknown"), 1024), Inline: true},
	)
	if expires := firstText(nil, data, "expires", "alert_expires_at"); expires != "" {
		embed.Fields = append(embed.Fields, discordEmbedField{Name: "Expires", Value: clipText(expires, 1024), Inline: true})
	}
	if instruction := firstText(nil, data, "instruction"); instruction != "" {
		embed.Fields = append(embed.Fields, discordEmbedField{Name: "Instruction", Value: clipText(instruction, 1024), Inline: false})
	}
	if header := codeBlockText(firstText(nil, data, "same_header"), 1024); header != "" {
		embed.Fields = append(embed.Fields, discordEmbedField{Name: "SAME Header", Value: header, Inline: false})
	}
	if generated != "" {
		embed.Fields = append(embed.Fields, discordEmbedField{Name: "SAME Message", Value: codeBlockText(generated, 1024), Inline: false})
	}
	if listenURL := integratedListenURL(listenBaseURL, feedID); listenURL != "" {
		embed.Fields = append(embed.Fields, discordEmbedField{Name: "Listen", Value: clipText(listenURL, 1024), Inline: false})
	}
	if authoritative := firstText(nil, data, "authoritative_url"); authoritative != "" {
		embed.Fields = append(embed.Fields, discordEmbedField{Name: "CAP Audio", Value: clipText(authoritative, 1024), Inline: false})
	}
	if identifier := firstText(nil, data, "alert_id", "identifier"); identifier != "" {
		embed.Footer = &discordFooter{Text: clipText(identifier, 2048)}
	}
	return discordPayload{
		Username:  cfg.Username,
		AvatarURL: cfg.IconURL,
		Embeds:    []discordEmbed{embed},
	}
}

func integratedListenURL(base string, feedID string) string {
	base = strings.TrimSpace(base)
	feedID = strings.TrimSpace(feedID)
	if base == "" || feedID == "" {
		return ""
	}
	parsed, err := url.Parse(strings.TrimRight(base, "/") + "/listen")
	if err != nil {
		return ""
	}
	query := parsed.Query()
	query.Set("feed", feedID)
	query.Set("codec", "pcm16")
	parsed.RawQuery = query.Encode()
	return parsed.String()
}

func (s *Service) post(ctx context.Context, url string, payload discordPayload, file attachment) (int, string, error) {
	if len(file.Data) == 0 {
		body, err := json.Marshal(payload)
		if err != nil {
			return 0, "", err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
		if err != nil {
			return 0, "", err
		}
		req.Header.Set("Content-Type", "application/json")
		return doRequest(s.client, req)
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return 0, "", err
	}
	if err := writer.WriteField("payload_json", string(payloadBytes)); err != nil {
		return 0, "", err
	}
	part, err := writer.CreateFormFile("files[0]", file.Name)
	if err != nil {
		return 0, "", err
	}
	if _, err := part.Write(file.Data); err != nil {
		return 0, "", err
	}
	if err := writer.Close(); err != nil {
		return 0, "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, &body)
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	return doRequest(s.client, req)
}

func doRequest(client *http.Client, req *http.Request) (int, string, error) {
	resp, err := client.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	detail, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	return resp.StatusCode, strings.TrimSpace(string(detail)), nil
}

type attachment struct {
	Name string
	Data []byte
}

func transcodeAlertAudio(ctx context.Context, src string, codec string, sampleRate int, channels int) (attachment, error) {
	info, err := os.Stat(src)
	if err != nil {
		return attachment{}, err
	}
	if !info.Mode().IsRegular() {
		return attachment{}, fmt.Errorf("audio path is not a regular file")
	}
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if channels <= 0 {
		channels = 1
	}
	codec = firstNonBlank(codec, "libopus")
	ext := codecExt[codec]
	if ext == "" {
		ext = "ogg"
	}
	outCodec := codec
	if codec == "wav" {
		outCodec = "pcm_s16le"
	}
	transcodeCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(
		transcodeCtx,
		"ffmpeg",
		"-loglevel", "error",
		"-f", "s16le",
		"-ar", strconv.Itoa(sampleRate),
		"-ac", strconv.Itoa(channels),
		"-i", src,
		"-c:a", outCodec,
		"-f", ext,
		"pipe:1",
	)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail != "" {
			return attachment{}, fmt.Errorf("ffmpeg: %s", clipText(detail, 200))
		}
		return attachment{}, err
	}
	if stdout.Len() > maxDiscordAttachmentBytes {
		return attachment{}, fmt.Errorf("audio exceeds Discord attachment limit")
	}
	name := "alert." + ext
	if base := strings.TrimSuffix(filepath.Base(src), filepath.Ext(src)); base != "" {
		name = base + "." + ext
	}
	return attachment{Name: name, Data: stdout.Bytes()}, nil
}

func colorInt(value string) int {
	text := strings.TrimPrefix(strings.TrimSpace(value), "#")
	if text == "" {
		return 0x888888
	}
	parsed, err := strconv.ParseInt(text, 16, 32)
	if err != nil {
		return 0x888888
	}
	return int(parsed)
}

func clipText(value string, limit int) string {
	text := strings.TrimSpace(value)
	if limit <= 0 || len(text) <= limit {
		return text
	}
	if limit <= 3 {
		return text[:limit]
	}
	return strings.TrimSpace(text[:limit-3]) + "..."
}

func codeBlockText(value string, limit int) string {
	normalized := strings.ReplaceAll(strings.TrimSpace(value), "```", "'''")
	if normalized == "" {
		return ""
	}
	innerLimit := limit - 8
	if innerLimit < 1 {
		innerLimit = 1
	}
	return "```\n" + clipText(normalized, innerLimit) + "\n```"
}

func sleepOrDone(ctx context.Context, duration time.Duration) {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}
