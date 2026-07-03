package webgateway

import (
	"context"
	"encoding/base64"
	"encoding/xml"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/tts"
)

const (
	defaultReadersFile    = "managed/configs/readers.xml"
	ttsPreviewTimeout     = 90 * time.Second
	ttsPreviewMaxTextLen  = 2000
	defaultTTSPreviewText = "This is a preview of the selected Haze Weather Radio voice."
)

var readerIDCleaner = regexp.MustCompile(`[^a-zA-Z0-9_.-]+`)

type ttsReadersXML struct {
	XMLName xml.Name       `xml:"Readers"`
	Readers []ttsReaderXML `xml:"reader"`
}

type ttsReaderXML struct {
	ID       string `xml:"id,attr"`
	Provider string `xml:"provider,attr"`
	Gender   string `xml:"gender"`
	Language string `xml:"language"`
	VoiceID  string `xml:"voice_id,omitempty"`
}

func loadTTSPayload(configPath string) (map[string]any, error) {
	path, relPath, err := ttsReadersPath(configPath)
	if err != nil {
		return nil, err
	}
	items, err := readTTSReadersXML(path)
	if err != nil {
		return nil, err
	}
	return ttsPayload(path, relPath, items), nil
}

func saveTTSPayload(configPath string, payload map[string]any) (map[string]any, error) {
	rawItems, ok := payload["readers"].([]any)
	if !ok {
		return nil, fmt.Errorf("readers payload is required")
	}
	items := make([]ttsReaderXML, 0, len(rawItems))
	for _, raw := range rawItems {
		item, err := ttsReaderFromMap(raw)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	path, _, err := ttsReadersPath(configPath)
	if err != nil {
		return nil, err
	}
	if err := writeTTSReadersXML(path, items); err != nil {
		return nil, err
	}
	return loadTTSPayload(configPath)
}

func (s *wsSession) previewTTS(payload map[string]any) (map[string]any, error) {
	text := strings.TrimSpace(stringPayload(payload, "text", defaultTTSPreviewText))
	if text == "" {
		return nil, fmt.Errorf("preview text is required")
	}
	if len(text) > ttsPreviewMaxTextLen {
		return nil, fmt.Errorf("preview text is too long")
	}
	readerID := cleanReaderID(stringPayload(payload, "reader_id", ""))
	provider := tts.NormalizeProvider(stringPayload(payload, "provider", ""))
	voiceID := strings.TrimSpace(stringPayload(payload, "voice_id", ""))
	language := tts.NormalizeLanguage(stringPayload(payload, "language", "en-CA"))
	if language == "" {
		language = "en-ca"
	}
	if provider == "" || provider == "auto" {
		provider = ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), ttsPreviewTimeout)
	defer cancel()
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil, fmt.Errorf("event bridge is not available")
	}
	bridge, err := connectWxBridge(ctx, bridgeAddr)
	if err != nil {
		return nil, fmt.Errorf("event bridge connect failed: %w", err)
	}
	defer bridge.Close()
	jobID := safeID(fmt.Sprintf("tts-preview-%d", time.Now().UTC().UnixNano()))
	outputPath := resolveConfigPath(s.configPath, filepath.Join("runtime", "audio", "previews", jobID+".pcm16le"))
	synth, err := bridge.SynthesizeDirect(ctx, jobID, map[string]any{
		"job_id":        jobID,
		"text":          text,
		"reader_id":     readerID,
		"provider":      provider,
		"voice_id":      voiceID,
		"language":      language,
		"output_path":   outputPath,
		"output_format": "pcm_s16le",
		"rate":          intPayload(payload, "rate", 0),
		"volume":        intPayload(payload, "volume", 100),
	})
	if err != nil {
		return nil, err
	}
	path := firstNonBlank(synth.Path, outputPath)
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	pcm, err := normalizePreviewVoicePCM(ctx, raw, synth)
	if err != nil {
		return nil, err
	}
	return map[string]any{
		"audio_base64": base64.StdEncoding.EncodeToString(wavFromPCM16(pcm, 48000, 1)),
		"format":       "wav",
		"content_type": "audio/wav",
		"sample_rate":  48000,
		"channels":     1,
		"reader_id":    readerID,
		"provider":     firstNonBlank(synth.Provider, provider),
		"voice_id":     firstNonBlank(synth.VoiceID, voiceID),
		"language":     language,
	}, nil
}

func (c *wxBridgeClient) SynthesizeDirect(ctx context.Context, jobID string, data map[string]any) (wxSynthResult, error) {
	ch := make(chan wxSynthResult, 1)
	c.mu.Lock()
	c.pendingSynth[jobID] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.pendingSynth, jobID)
		c.mu.Unlock()
	}()
	if outputPath, _ := data["output_path"].(string); outputPath != "" {
		if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
			return wxSynthResult{}, err
		}
	}
	if err := c.Publish(map[string]any{
		"type":    "tts.synthesize",
		"source":  "haze-web",
		"subject": jobID,
		"data":    data,
	}); err != nil {
		return wxSynthResult{}, err
	}
	select {
	case <-ctx.Done():
		return wxSynthResult{}, ctx.Err()
	case result := <-ch:
		return result, result.Err
	}
}

func ttsReadersPath(configPath string) (string, string, error) {
	root, err := loadYAMLMap(configPath)
	if err != nil {
		return "", "", err
	}
	rel := textAt(root, []string{"services", "go", "tts", "readers"}, defaultReadersFile, 240)
	return resolveConfigPath(configPath, rel), rel, nil
}

func readTTSReadersXML(path string) ([]ttsReaderXML, error) {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		if os.IsNotExist(err) {
			return []ttsReaderXML{}, nil
		}
		return nil, err
	}
	var parsed ttsReadersXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("readers XML is invalid: %w", err)
	}
	return normalizeTTSReaders(parsed.Readers)
}

func writeTTSReadersXML(path string, items []ttsReaderXML) error {
	items, err := normalizeTTSReaders(items)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := xml.MarshalIndent(ttsReadersXML{Readers: items}, "", "  ")
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

func normalizeTTSReaders(items []ttsReaderXML) ([]ttsReaderXML, error) {
	out := make([]ttsReaderXML, 0, len(items))
	seen := map[string]struct{}{}
	for _, item := range items {
		item.ID = cleanReaderID(item.ID)
		if item.ID == "" {
			return nil, fmt.Errorf("reader id is required")
		}
		if _, ok := seen[item.ID]; ok {
			return nil, fmt.Errorf("duplicate reader id %q", item.ID)
		}
		seen[item.ID] = struct{}{}
		item.Provider = tts.NormalizeProvider(item.Provider)
		if item.Provider == "" {
			item.Provider = "auto"
		}
		item.Gender = strings.ToLower(strings.TrimSpace(item.Gender))
		if item.Gender != "female" {
			item.Gender = "male"
		}
		item.Language = tts.NormalizeLanguage(item.Language)
		if item.Language == "" {
			item.Language = "en-us"
		}
		item.VoiceID = strings.TrimSpace(item.VoiceID)
		out = append(out, item)
	}
	sort.SliceStable(out, func(i, j int) bool {
		return naturalReaderSortKey(out[i].ID) < naturalReaderSortKey(out[j].ID)
	})
	return out, nil
}

func ttsReaderFromMap(raw any) (ttsReaderXML, error) {
	itemMap, ok := raw.(map[string]any)
	if !ok {
		return ttsReaderXML{}, fmt.Errorf("reader must be an object")
	}
	return normalizeOneTTSReader(ttsReaderXML{
		ID:       stringPayload(itemMap, "id", ""),
		Provider: stringPayload(itemMap, "provider", "auto"),
		Gender:   stringPayload(itemMap, "gender", "male"),
		Language: stringPayload(itemMap, "language", "en-us"),
		VoiceID:  stringPayload(itemMap, "voice_id", ""),
	})
}

func normalizeOneTTSReader(item ttsReaderXML) (ttsReaderXML, error) {
	items, err := normalizeTTSReaders([]ttsReaderXML{item})
	if err != nil {
		return ttsReaderXML{}, err
	}
	return items[0], nil
}

func cleanReaderID(value string) string {
	cleaned := readerIDCleaner.ReplaceAllString(strings.TrimSpace(value), "-")
	cleaned = strings.Trim(cleaned, "-")
	if len(cleaned) > 64 {
		cleaned = cleaned[:64]
	}
	return cleaned
}

func naturalReaderSortKey(id string) string {
	if len(id) == 1 && id[0] >= '0' && id[0] <= '9' {
		return "0" + id
	}
	return id
}

func ttsPayload(path string, relPath string, items []ttsReaderXML) map[string]any {
	readers := make([]map[string]any, 0, len(items))
	for _, item := range items {
		readers = append(readers, map[string]any{
			"id":       item.ID,
			"provider": item.Provider,
			"gender":   item.Gender,
			"language": item.Language,
			"voice_id": item.VoiceID,
			"label":    ttsReaderLabel(item),
		})
	}
	return map[string]any{
		"path":         filepath.ToSlash(path),
		"configured":   relPath,
		"readers":      readers,
		"providers":    []string{"auto", "fast", "piper", "kokoro", "sapi5", "espeak", "speakyapi", "f5tts", "chatterbox"},
		"preview_text": defaultTTSPreviewText,
	}
}

func ttsReaderLabel(item ttsReaderXML) string {
	parts := []string{item.ID}
	for _, part := range []string{item.Provider, item.Gender, item.Language, item.VoiceID} {
		if strings.TrimSpace(part) != "" {
			parts = append(parts, strings.TrimSpace(part))
		}
	}
	return strings.Join(parts, " · ")
}
