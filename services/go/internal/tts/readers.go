package tts

import (
	"encoding/xml"
	"fmt"
	"os"
	"strings"
)

// Reader maps a Haze reader_id to a concrete TTS provider voice.
type Reader struct {
	ID       string `json:"id"`
	Provider string `json:"provider"`
	Gender   string `json:"gender,omitempty"`
	Language string `json:"language,omitempty"`
	VoiceID  string `json:"voice_id"`
}

type readersXML struct {
	Readers []readerXML `xml:"reader"`
}

type readerXML struct {
	ID       string `xml:"id,attr"`
	Provider string `xml:"provider,attr"`
	Gender   string `xml:"gender"`
	Language string `xml:"language"`
	VoiceID  string `xml:"voice_id"`
	Path     string `xml:"path"`
}

// LoadReaders parses managed/configs/readers.xml.
func LoadReaders(path string) ([]Reader, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var parsed readersXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}

	readers := make([]Reader, 0, len(parsed.Readers))
	for _, item := range parsed.Readers {
		readerID := strings.TrimSpace(item.ID)
		if readerID == "" {
			continue
		}
		provider := NormalizeProvider(item.Provider)
		if provider == "" {
			provider = "auto"
		}
		voiceID := strings.TrimSpace(item.VoiceID)
		if voiceID == "" {
			voiceID = strings.TrimSpace(item.Path)
		}
		gender := strings.ToLower(strings.TrimSpace(item.Gender))
		if gender != "female" {
			gender = "male"
		}
		readers = append(readers, Reader{
			ID:       readerID,
			Provider: provider,
			Gender:   gender,
			Language: NormalizeLanguage(item.Language),
			VoiceID:  voiceID,
		})
	}
	return readers, nil
}

// NormalizeProvider maps provider names onto the service provider IDs.
func NormalizeProvider(provider string) string {
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "", "auto", "default":
		return "auto"
	case "fast", "ivr-fast", "ivr_fast", "prompt", "low-latency", "low_latency":
		return "fast"
	case "sapi", "sapi5":
		return "sapi5"
	case "espeak", "espeak-ng", "espeakng":
		return "espeak"
	case "piper", "piper-tts", "pipertts":
		return "piper"
	case "f5", "f5tts", "f5-tts":
		return "f5tts"
	case "chatterbox", "chatterbox-tts", "chatterboxtts":
		return "chatterbox"
	case "kokoro", "kokoro-tts", "kokorotts", "sherpa", "sherpa-onnx":
		return "kokoro"
	case "speaky", "speaky-api", "speakyapi":
		return "speakyapi"
	default:
		return strings.ToLower(strings.TrimSpace(provider))
	}
}

// NormalizeLanguage canonicalizes language tags enough for reader matching.
func NormalizeLanguage(language string) string {
	return strings.ToLower(strings.ReplaceAll(strings.TrimSpace(language), "_", "-"))
}

// SelectReader resolves a requested reader ID, falling back by language/gender.
func SelectReader(readers []Reader, readerID string, language string, gender string) (Reader, bool) {
	requested := strings.TrimSpace(readerID)
	if requested != "" && strings.ToLower(requested) != "male" && strings.ToLower(requested) != "female" {
		for _, reader := range readers {
			if reader.ID == requested {
				return reader, true
			}
		}
	}

	lang := NormalizeLanguage(language)
	prefix := lang
	if idx := strings.Index(prefix, "-"); idx >= 0 {
		prefix = prefix[:idx]
	}
	slot := strings.ToLower(strings.TrimSpace(gender))
	if slot == "" || requested == "male" || requested == "female" {
		slot = strings.ToLower(strings.TrimSpace(requested))
	}
	if slot != "female" {
		slot = "male"
	}

	groups := [][]Reader{
		filterReaders(readers, func(reader Reader) bool { return reader.Language == lang }),
		filterReaders(readers, func(reader Reader) bool {
			return reader.Language != "" && strings.SplitN(reader.Language, "-", 2)[0] == prefix
		}),
		filterReaders(readers, func(reader Reader) bool { return reader.Language == "" }),
		readers,
	}
	for _, group := range groups {
		for _, reader := range group {
			if reader.Gender == slot {
				return reader, true
			}
		}
	}
	for _, group := range groups {
		if len(group) > 0 {
			return group[0], true
		}
	}
	return Reader{}, false
}

func filterReaders(readers []Reader, keep func(Reader) bool) []Reader {
	filtered := make([]Reader, 0, len(readers))
	for _, reader := range readers {
		if keep(reader) {
			filtered = append(filtered, reader)
		}
	}
	return filtered
}

// ProviderForReader resolves the provider for a reader.
func ProviderForReader(providers map[string]Provider, reader Reader) (Provider, error) {
	provider := providers[NormalizeProvider(reader.Provider)]
	if provider == nil {
		return nil, fmt.Errorf("%w: %s", ErrProviderUnavailable, reader.Provider)
	}
	return provider, nil
}
