package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
	"github.com/meowraii/haze-weather-radio/services/go/internal/tts"
)

const serviceID = "haze-tts"

var errSystemShutdown = errors.New("system shutdown requested")

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil {
		log.Fatalf("haze-tts: %v", err)
	}
}

func run() error {
	readersPath := flag.String("readers", filepath.Join("managed", "configs", "readers.xml"), "readers.xml path")
	dictionaryPath := flag.String("dictionary", envOrDefault("HAZE_TTS_DICTIONARY", filepath.Join("managed", "dictionary.json")), "dictionary.json path")
	providerID := flag.String("provider", "auto", "provider to use: auto, piper, sapi5, espeak, f5tts, or chatterbox")
	readerID := flag.String("reader-id", "", "reader id from readers.xml")
	lang := flag.String("lang", "en-CA", "requested language")
	timezone := flag.String("timezone", envOrDefault("HAZE_TTS_TIMEZONE", "Local"), "timezone for spoken timestamps")
	piperExe := flag.String("piper-exe", envOrDefault("HAZE_PIPER_EXE", "piper"), "Piper executable path")
	piperVoicesDir := flag.String("piper-voices-dir", envOrDefault("HAZE_PIPER_VOICES_DIR", filepath.Join("managed", "voices", "piper")), "Piper voice model directory")
	text := flag.String("text", "", "text to synthesize")
	out := flag.String("out", "", "output WAV path")
	listVoices := flag.Bool("list-voices", false, "list provider voices as JSON")
	service := flag.Bool("service", false, "run as a host-bridge TTS service")
	bridge := flag.String("bridge", os.Getenv("HAZE_HOST_BRIDGE_ADDR"), "host bridge address")
	outDir := flag.String("out-dir", filepath.Join("managed", "audio", "tts"), "default service output directory")
	timeout := flag.Duration("timeout", 60*time.Second, "synthesis timeout")
	flag.Parse()
	setTTSRuntimeEnv(*piperExe, *piperVoicesDir)

	if *service {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		return runService(ctx, serviceConfig{
			BridgeAddr: *bridge,
			Readers:    *readersPath,
			Dictionary: *dictionaryPath,
			Provider:   *providerID,
			Language:   *lang,
			Timezone:   *timezone,
			OutDir:     *outDir,
			Timeout:    *timeout,
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	providers := tts.DefaultProviders()
	if *listVoices {
		voices, err := listVoicesForProvider(ctx, providers, *providerID)
		if err != nil {
			return err
		}
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(voices)
	}

	if *text == "" {
		return errors.New("missing --text")
	}
	if *out == "" {
		return errors.New("missing --out")
	}

	reader, err := resolveReader(*readersPath, *readerID, *lang)
	if err != nil {
		return err
	}
	providerName := *providerID
	voiceID := ""
	if reader.ID != "" {
		providerName = reader.Provider
		voiceID = reader.VoiceID
		if reader.Language != "" {
			*lang = reader.Language
		}
	}
	dictionary, err := tts.LoadDictionary(*dictionaryPath, *lang)
	if err != nil {
		return err
	}
	audio, _, err := synthesizeWithProvider(ctx, providers, providerName, tts.Request{
		Text:     tts.NormalizeText(*text, dictionary, *timezone),
		VoiceID:  voiceID,
		Language: *lang,
		Volume:   100,
	})
	if err != nil {
		return err
	}
	if audio.Format != tts.FormatWAV {
		return fmt.Errorf("provider returned unsupported format %q", audio.Format)
	}
	if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
		return err
	}
	return os.WriteFile(*out, audio.Data, 0o644)
}

type serviceConfig struct {
	BridgeAddr string
	Readers    string
	Dictionary string
	Provider   string
	Language   string
	Timezone   string
	OutDir     string
	Timeout    time.Duration
}

func runService(ctx context.Context, cfg serviceConfig) error {
	if strings.TrimSpace(cfg.BridgeAddr) == "" {
		return errors.New("missing host bridge address")
	}
	if strings.TrimSpace(cfg.Language) == "" {
		cfg.Language = "en-CA"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
	if strings.TrimSpace(cfg.Timezone) == "" {
		cfg.Timezone = "Local"
	}

	for ctx.Err() == nil {
		conn, err := net.DialTimeout("tcp", cfg.BridgeAddr, 3*time.Second)
		if err != nil {
			sleepOrDone(ctx, time.Second)
			continue
		}
		log.Printf("connected to host bridge at %s", cfg.BridgeAddr)
		_ = publishServiceEvent(conn, "service.ready", "", map[string]any{
			"service":   serviceID,
			"providers": providerIDs(tts.DefaultProviders()),
			"readers":   cfg.Readers,
		})

		done := make(chan struct{})
		go func() {
			select {
			case <-ctx.Done():
				_ = conn.Close()
			case <-done:
			}
		}()
		err = runServiceConnection(ctx, conn, cfg)
		close(done)
		_ = conn.Close()
		if errors.Is(err, errSystemShutdown) {
			return nil
		}
		if ctx.Err() != nil {
			break
		}
		if err != nil {
			log.Printf("host bridge connection closed: %v", err)
		}
		sleepOrDone(ctx, time.Second)
	}
	return nil
}

func runServiceConnection(ctx context.Context, conn net.Conn, cfg serviceConfig) error {
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		line := strings.TrimPrefix(strings.TrimSpace(scanner.Text()), "\ufeff")
		if line == "" {
			continue
		}
		var message map[string]any
		if err := json.Unmarshal([]byte(line), &message); err != nil {
			continue
		}
		if stringValue(message, "type") == "system.shutdown" {
			return errSystemShutdown
		}
		if stringValue(message, "type") != "tts.synthesize" {
			continue
		}
		handleSynthesisJob(ctx, conn, cfg, message)
	}
	return scanner.Err()
}

func handleSynthesisJob(ctx context.Context, conn net.Conn, cfg serviceConfig, message map[string]any) {
	data := objectValue(message, "data")
	jobID := firstText(message, data, "job_id", "id", "subject")
	if jobID == "" {
		jobID = fmt.Sprintf("tts-%d", time.Now().UnixNano())
	}

	text := firstText(message, data, "text")
	if strings.TrimSpace(text) == "" {
		publishTTSError(conn, jobID, "empty synthesis text")
		return
	}

	jobCtx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()
	providers := tts.DefaultProviders()
	reader, hasReader, err := serviceReader(cfg.Readers, firstText(message, data, "reader_id"), firstText(message, data, "language"))
	if err != nil {
		publishTTSError(conn, jobID, err.Error())
		return
	}

	providerID := firstText(message, data, "provider")
	voiceID := firstText(message, data, "voice_id")
	language := firstText(message, data, "language")
	if language == "" {
		language = cfg.Language
	}
	timezone := firstText(message, data, "timezone")
	if timezone == "" {
		timezone = cfg.Timezone
	}
	if hasReader {
		providerID = reader.Provider
		voiceID = reader.VoiceID
		if reader.Language != "" {
			language = reader.Language
		}
	}
	if providerID == "" {
		providerID = cfg.Provider
	}
	if providerID == "" {
		providerID = "auto"
	}
	dictionary, err := tts.LoadDictionary(cfg.Dictionary, language)
	if err != nil {
		publishTTSError(conn, jobID, err.Error())
		return
	}

	audio, provider, err := synthesizeWithProvider(jobCtx, providers, providerID, tts.Request{
		Text:            tts.NormalizeText(text, dictionary, timezone),
		VoiceID:         voiceID,
		Language:        language,
		Volume:          intValue(message, data, "volume", 100),
		Rate:            intValue(message, data, "rate", 0),
		SentenceSilence: floatValue(message, data, "sentence_silence", 0),
	})
	if err != nil {
		publishTTSError(conn, jobID, err.Error())
		return
	}
	if audio.Format != tts.FormatWAV {
		publishTTSError(conn, jobID, fmt.Sprintf("unsupported audio format %q", audio.Format))
		return
	}

	outputPath := firstText(message, data, "output_path", "out")
	if outputPath == "" {
		outputPath = filepath.Join(cfg.OutDir, sanitizeFileName(jobID)+".wav")
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		publishTTSError(conn, jobID, err.Error())
		return
	}
	if err := os.WriteFile(outputPath, audio.Data, 0o644); err != nil {
		publishTTSError(conn, jobID, err.Error())
		return
	}
	_ = publishServiceEvent(conn, "tts.synthesized", jobID, map[string]any{
		"job_id":      jobID,
		"output_path": outputPath,
		"bytes":       len(audio.Data),
		"provider":    provider.ID(),
		"reader_id":   reader.ID,
		"voice_id":    voiceID,
		"language":    language,
	})
}

func serviceReader(path string, readerID string, language string) (tts.Reader, bool, error) {
	if strings.TrimSpace(readerID) == "" {
		return tts.Reader{}, false, nil
	}
	readers, err := tts.LoadReaders(path)
	if err != nil {
		return tts.Reader{}, false, err
	}
	reader, ok := tts.SelectReader(readers, readerID, language, "")
	if !ok {
		return tts.Reader{}, false, fmt.Errorf("reader %q not found in %s", readerID, path)
	}
	return reader, true, nil
}

func publishTTSError(conn net.Conn, jobID string, detail string) {
	_ = publishServiceEvent(conn, "tts.failed", jobID, map[string]any{
		"job_id": jobID,
		"error":  detail,
	})
}

func publishServiceEvent(conn net.Conn, eventType string, subject string, data map[string]any) error {
	return json.NewEncoder(conn).Encode(events.Event{
		Type:      eventType,
		Source:    serviceID,
		Timestamp: time.Now().UTC(),
		Subject:   subject,
		Data:      data,
	})
}

func providerIDs(providers map[string]tts.Provider) []string {
	ids := make([]string, 0, len(providers))
	for id := range providers {
		ids = append(ids, id)
	}
	return ids
}

func objectValue(message map[string]any, key string) map[string]any {
	if value, ok := message[key].(map[string]any); ok {
		return value
	}
	return nil
}

func firstText(message map[string]any, data map[string]any, keys ...string) string {
	for _, key := range keys {
		if value := stringValue(data, key); value != "" {
			return value
		}
		if value := stringValue(message, key); value != "" {
			return value
		}
	}
	return ""
}

func stringValue(message map[string]any, key string) string {
	if message == nil {
		return ""
	}
	switch value := message[key].(type) {
	case string:
		return strings.TrimSpace(value)
	default:
		return ""
	}
}

func intValue(message map[string]any, data map[string]any, key string, fallback int) int {
	for _, source := range []map[string]any{data, message} {
		switch value := source[key].(type) {
		case float64:
			return int(value)
		case int:
			return value
		case string:
			var parsed int
			if _, err := fmt.Sscanf(value, "%d", &parsed); err == nil {
				return parsed
			}
		}
	}
	return fallback
}

func floatValue(message map[string]any, data map[string]any, key string, fallback float64) float64 {
	for _, source := range []map[string]any{data, message} {
		switch value := source[key].(type) {
		case float64:
			return value
		case int:
			return float64(value)
		case string:
			var parsed float64
			if _, err := fmt.Sscanf(value, "%f", &parsed); err == nil {
				return parsed
			}
		}
	}
	return fallback
}

func sanitizeFileName(value string) string {
	var builder strings.Builder
	for _, char := range value {
		if char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || char == '-' || char == '_' {
			builder.WriteRune(char)
		}
	}
	if builder.Len() == 0 {
		return "tts"
	}
	return builder.String()
}

func sleepOrDone(ctx context.Context, duration time.Duration) {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

func envOrDefault(key string, fallback string) string {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	return value
}

func setTTSRuntimeEnv(piperExe string, piperVoicesDir string) {
	if strings.TrimSpace(piperExe) != "" {
		_ = os.Setenv("HAZE_PIPER_EXE", strings.TrimSpace(piperExe))
	}
	if strings.TrimSpace(piperVoicesDir) != "" {
		_ = os.Setenv("HAZE_PIPER_VOICES_DIR", strings.TrimSpace(piperVoicesDir))
	}
}

func resolveReader(path string, readerID string, lang string) (tts.Reader, error) {
	if readerID == "" {
		return tts.Reader{}, nil
	}
	readers, err := tts.LoadReaders(path)
	if err != nil {
		return tts.Reader{}, err
	}
	reader, ok := tts.SelectReader(readers, readerID, lang, "")
	if !ok {
		return tts.Reader{}, fmt.Errorf("reader %q not found in %s", readerID, path)
	}
	return reader, nil
}

func listVoicesForProvider(ctx context.Context, providers map[string]tts.Provider, providerID string) ([]tts.Voice, error) {
	candidates, err := providerCandidates(providers, providerID)
	if err != nil {
		return nil, err
	}
	var voices []tts.Voice
	var failures []error
	for _, provider := range candidates {
		providerVoices, err := provider.ListVoices(ctx)
		if err != nil {
			failures = append(failures, fmt.Errorf("%s: %w", provider.ID(), err))
			continue
		}
		voices = append(voices, providerVoices...)
	}
	if len(voices) > 0 {
		return voices, nil
	}
	return nil, errors.Join(failures...)
}

func synthesizeWithProvider(ctx context.Context, providers map[string]tts.Provider, providerID string, req tts.Request) (tts.Audio, tts.Provider, error) {
	candidates, err := providerCandidates(providers, providerID)
	if err != nil {
		return tts.Audio{}, nil, err
	}
	var failures []error
	for _, provider := range candidates {
		audio, err := provider.Synthesize(ctx, req)
		if err == nil {
			return audio, provider, nil
		}
		failures = append(failures, fmt.Errorf("%s: %w", provider.ID(), err))
	}
	return tts.Audio{}, nil, errors.Join(failures...)
}

func providerCandidates(providers map[string]tts.Provider, providerID string) ([]tts.Provider, error) {
	normalized := tts.NormalizeProvider(providerID)
	if normalized == "fast" {
		candidates := make([]tts.Provider, 0, 3)
		for _, id := range []string{"sapi5", "espeak", "piper"} {
			if provider := providers[id]; provider != nil {
				candidates = append(candidates, provider)
			}
		}
		if len(candidates) == 0 {
			return nil, fmt.Errorf("%w: fast", tts.ErrProviderUnavailable)
		}
		return candidates, nil
	}
	if normalized == "" || normalized == "auto" {
		candidates := make([]tts.Provider, 0, 2)
		for _, id := range []string{"piper", "sapi5", "espeak"} {
			if provider := providers[id]; provider != nil {
				candidates = append(candidates, provider)
			}
		}
		if len(candidates) == 0 {
			return nil, fmt.Errorf("%w: auto", tts.ErrProviderUnavailable)
		}
		return candidates, nil
	}
	provider := providers[normalized]
	if provider == nil {
		return nil, fmt.Errorf("%w: %s", tts.ErrProviderUnavailable, providerID)
	}
	return []tts.Provider{provider}, nil
}
