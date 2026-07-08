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
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
	"github.com/meowraii/haze-weather-radio/services/go/internal/processguard"
	"github.com/meowraii/haze-weather-radio/services/go/internal/tts"
)

const serviceID = "haze-tts"

const (
	synthesisHighQueueSize   = 64
	synthesisNormalQueueSize = 64
	synthesisLowQueueSize    = 16
)

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
	providerID := flag.String("provider", "auto", "provider to use: auto, piper, kokoro, sapi5, espeak, f5tts, chatterbox, or speakyapi")
	readerID := flag.String("reader-id", "", "reader id from readers.xml")
	lang := flag.String("lang", "en-US", "requested language")
	timezone := flag.String("timezone", envOrDefault("HAZE_TTS_TIMEZONE", "Local"), "timezone for spoken timestamps")
	_ = flag.String("piper-exe", "", "deprecated; Piper uses the native sherpa-onnx runtime")
	piperVoicesDir := flag.String("piper-voices-dir", envOrDefault("HAZE_PIPER_VOICES_DIR", filepath.Join("managed", "voices", "piper")), "Piper voice model directory")
	_ = flag.String("piper-mode", "", "deprecated; Piper is native-only")
	_ = flag.Int("piper-workers", 0, "deprecated; Piper is native-only")
	_ = flag.Bool("piper-cuda", false, "deprecated; use HAZE_PIPER_PROVIDER or HAZE_KOKORO_PROVIDER")
	kokoroModelDir := flag.String("kokoro-model-dir", envOrDefault("HAZE_KOKORO_MODEL_DIR", filepath.Join("managed", "voices", "kokoro-multi-lang-v1_0")), "Kokoro model directory")
	kokoroRuntimeProvider := flag.String("kokoro-runtime-provider", envOrDefault("HAZE_KOKORO_PROVIDER", "cpu"), "Kokoro sherpa-onnx provider: cpu, cuda, or coreml")
	kokoroThreads := flag.Int("kokoro-threads", envIntOrDefault("HAZE_KOKORO_THREADS", 0), "Kokoro neural network worker threads")
	kokoroSpeed := flag.Float64("kokoro-speed", envFloatOrDefault("HAZE_KOKORO_SPEED", 1.0), "Kokoro default generation speed")
	kokoroLengthScale := flag.Float64("kokoro-length-scale", envFloatOrDefault("HAZE_KOKORO_LENGTH_SCALE", 1.0), "Kokoro model length scale")
	speakyAPIURL := flag.String("speakyapi-url", envOrDefault("HAZE_SPEAKYAPI_URL", ""), "SpeakyAPI server base URL, for example http://127.0.0.1:5000")
	runtimeIdleTimeout := flag.Duration("runtime-idle-timeout", envDurationOrDefault("HAZE_TTS_RUNTIME_IDLE", 30*time.Second), "idle time before native TTS model runtimes are unloaded; 0 disables unloading")
	text := flag.String("text", "", "text to synthesize")
	out := flag.String("out", "", "output WAV path")
	listVoices := flag.Bool("list-voices", false, "list provider voices as JSON")
	service := flag.Bool("service", false, "run as a host-bridge TTS service")
	bridge := flag.String("bridge", os.Getenv("HAZE_HOST_BRIDGE_ADDR"), "host bridge address")
	outDir := flag.String("out-dir", filepath.Join("managed", "audio", "tts"), "default service output directory")
	timeout := flag.Duration("timeout", 60*time.Second, "synthesis timeout")
	flag.Parse()
	setTTSRuntimeEnv(ttsRuntimeEnv{
		PiperVoicesDir:        *piperVoicesDir,
		KokoroModelDir:        *kokoroModelDir,
		KokoroLang:            kokoroRuntimeLang(*lang),
		KokoroRuntimeProvider: *kokoroRuntimeProvider,
		KokoroThreads:         *kokoroThreads,
		KokoroSpeed:           *kokoroSpeed,
		KokoroLengthScale:     *kokoroLengthScale,
	})

	if *service {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		defer stop()
		ctx = processguard.WithParent(ctx)
		return runService(ctx, serviceConfig{
			BridgeAddr:   *bridge,
			Readers:      *readersPath,
			Dictionary:   *dictionaryPath,
			Provider:     *providerID,
			Language:     *lang,
			Timezone:     *timezone,
			OutDir:       *outDir,
			Timeout:      *timeout,
			Workers:      1,
			SpeakyAPIURL: *speakyAPIURL,
			RuntimeIdle:  *runtimeIdleTimeout,
		})
	}

	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()

	providers := tts.DefaultProviders()
	configureSpeakyAPIProvider(providers, *speakyAPIURL)
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
	return writeFileAtomic(*out, audio.Data, 0o644)
}

type serviceConfig struct {
	BridgeAddr   string
	Readers      string
	Dictionary   string
	Provider     string
	Language     string
	Timezone     string
	OutDir       string
	Timeout      time.Duration
	Workers      int
	SpeakyAPIURL string
	RuntimeIdle  time.Duration
}

type serviceState struct {
	cfg          serviceConfig
	providers    map[string]tts.Provider
	readers      []tts.Reader
	readersErr   error
	dictionaries map[string]dictionaryResult
	mu           sync.Mutex
	publishMu    sync.Mutex
}

type dictionaryResult struct {
	Dictionary tts.Dictionary
	Err        error
}

func runService(ctx context.Context, cfg serviceConfig) error {
	if strings.TrimSpace(cfg.BridgeAddr) == "" {
		return errors.New("missing host bridge address")
	}
	if strings.TrimSpace(cfg.Language) == "" {
		cfg.Language = "en-US"
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 60 * time.Second
	}
	if strings.TrimSpace(cfg.Timezone) == "" {
		cfg.Timezone = "Local"
	}
	state, err := newServiceState(ctx, cfg)
	if err != nil {
		return err
	}
	stopPruner := startRuntimePruner(ctx, state.providers, cfg.RuntimeIdle)
	defer stopPruner()

	for ctx.Err() == nil {
		conn, err := net.DialTimeout("tcp", cfg.BridgeAddr, 3*time.Second)
		if err != nil {
			sleepOrDone(ctx, time.Second)
			continue
		}
		log.Printf("connected to host bridge at %s", cfg.BridgeAddr)
		_ = publishServiceEvent(conn, "service.ready", "", map[string]any{
			"service":   serviceID,
			"providers": providerIDs(state.providers),
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
		err = runServiceConnection(ctx, conn, state)
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

func startRuntimePruner(ctx context.Context, providers map[string]tts.Provider, maxIdle time.Duration) func() {
	if maxIdle <= 0 {
		return func() {}
	}
	pruners := make(map[string]tts.RuntimePruner)
	for id, provider := range providers {
		if pruner, ok := provider.(tts.RuntimePruner); ok {
			pruners[id] = pruner
		}
	}
	if len(pruners) == 0 {
		return func() {}
	}
	pruneCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		interval := maxIdle / 4
		if interval < 5*time.Second {
			interval = 5 * time.Second
		}
		if interval > time.Minute {
			interval = time.Minute
		}
		timer := time.NewTimer(interval)
		defer timer.Stop()
		for {
			select {
			case <-pruneCtx.Done():
				return
			case <-timer.C:
				total := 0
				for id, pruner := range pruners {
					removed := pruner.PruneIdleRuntime(maxIdle)
					if removed > 0 {
						log.Printf("pruned %d idle %s TTS runtime(s)", removed, id)
						total += removed
					}
				}
				if total > 0 {
					runtime.GC()
					debug.FreeOSMemory()
				}
				timer.Reset(interval)
			}
		}
	}()
	return func() {
		cancel()
		<-done
	}
}

func runServiceConnection(ctx context.Context, conn net.Conn, state *serviceState) error {
	queue := newSynthesisQueue(ctx, conn, state, maxInt(1, state.cfg.Workers))
	defer queue.Close()
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
		if !queue.Enqueue(ctx, message) {
			jobID := firstText(message, objectValue(message, "data"), "job_id", "id", "subject")
			if jobID == "" {
				jobID = "tts"
			}
			state.publishTTSError(conn, jobID, "tts queue is full")
		}
	}
	return scanner.Err()
}

type synthesisQueue struct {
	conn   net.Conn
	state  *serviceState
	high   chan map[string]any
	normal chan map[string]any
	low    chan map[string]any
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func newSynthesisQueue(ctx context.Context, conn net.Conn, state *serviceState, workers int) *synthesisQueue {
	queueCtx, cancel := context.WithCancel(ctx)
	q := &synthesisQueue{
		conn:   conn,
		state:  state,
		high:   make(chan map[string]any, synthesisHighQueueSize),
		normal: make(chan map[string]any, synthesisNormalQueueSize),
		low:    make(chan map[string]any, synthesisLowQueueSize),
		cancel: cancel,
	}
	for range maxInt(1, workers) {
		q.wg.Add(1)
		go q.worker(queueCtx)
	}
	return q
}

func (q *synthesisQueue) Enqueue(ctx context.Context, message map[string]any) bool {
	var target chan map[string]any
	switch synthesisPriority(message) {
	case "high":
		target = q.high
	case "low":
		target = q.low
	default:
		target = q.normal
	}
	select {
	case target <- message:
		return true
	case <-ctx.Done():
		return false
	default:
		return false
	}
}

func (q *synthesisQueue) Close() {
	q.cancel()
	q.wg.Wait()
}

func (q *synthesisQueue) worker(ctx context.Context) {
	defer q.wg.Done()
	for {
		message, ok := q.next(ctx)
		if !ok {
			return
		}
		handleSynthesisJob(ctx, q.conn, q.state, message)
	}
}

func (q *synthesisQueue) next(ctx context.Context) (map[string]any, bool) {
	select {
	case message := <-q.high:
		return message, true
	default:
	}
	select {
	case message := <-q.high:
		return message, true
	case message := <-q.normal:
		return message, true
	default:
	}
	select {
	case message := <-q.high:
		return message, true
	case message := <-q.normal:
		return message, true
	case message := <-q.low:
		return message, true
	case <-ctx.Done():
		return nil, false
	}
}

func synthesisPriority(message map[string]any) string {
	data := objectValue(message, "data")
	switch strings.ToLower(firstText(message, data, "priority", "queue_priority")) {
	case "realtime", "urgent", "high", "radio", "playout":
		return "high"
	case "batch", "background", "low":
		return "low"
	default:
		return "normal"
	}
}

func newServiceState(ctx context.Context, cfg serviceConfig) (*serviceState, error) {
	providers := tts.DefaultProviders()
	configureSpeakyAPIProvider(providers, cfg.SpeakyAPIURL)
	readers, err := tts.LoadReaders(cfg.Readers)
	if err != nil {
		readers = nil
	}
	state := &serviceState{
		cfg:          cfg,
		providers:    providers,
		readers:      readers,
		readersErr:   err,
		dictionaries: map[string]dictionaryResult{},
	}
	return state, nil
}

func configureSpeakyAPIProvider(providers map[string]tts.Provider, baseURL string) {
	if strings.TrimSpace(baseURL) == "" {
		return
	}
	providers["speakyapi"] = tts.NewSpeakyAPIProvider(baseURL)
}

func (s *serviceState) serviceReader(readerID string, language string) (tts.Reader, bool, error) {
	if strings.TrimSpace(readerID) == "" {
		return tts.Reader{}, false, nil
	}
	if s.readersErr != nil {
		return tts.Reader{}, false, s.readersErr
	}
	reader, ok := tts.SelectReader(s.readers, readerID, language, "")
	if !ok {
		return tts.Reader{}, false, fmt.Errorf("reader %q not found in %s", readerID, s.cfg.Readers)
	}
	return reader, true, nil
}

func (s *serviceState) dictionary(language string) (tts.Dictionary, error) {
	key := tts.NormalizeLanguage(language)
	if key == "" {
		key = tts.NormalizeLanguage(s.cfg.Language)
	}
	s.mu.Lock()
	if cached, ok := s.dictionaries[key]; ok {
		s.mu.Unlock()
		return cached.Dictionary, cached.Err
	}
	s.mu.Unlock()

	dictionary, err := tts.LoadDictionary(s.cfg.Dictionary, language)
	s.mu.Lock()
	s.dictionaries[key] = dictionaryResult{Dictionary: dictionary, Err: err}
	s.mu.Unlock()
	return dictionary, err
}

func handleSynthesisJob(ctx context.Context, conn net.Conn, state *serviceState, message map[string]any) {
	cfg := state.cfg
	data := objectValue(message, "data")
	jobID := firstText(message, data, "job_id", "id", "subject")
	if jobID == "" {
		jobID = fmt.Sprintf("tts-%d", time.Now().UnixNano())
	}

	text := firstText(message, data, "text")
	if strings.TrimSpace(text) == "" {
		state.publishTTSError(conn, jobID, "empty synthesis text")
		return
	}

	jobCtx, cancel := context.WithTimeout(ctx, cfg.Timeout)
	defer cancel()
	reader, hasReader, err := state.serviceReader(firstText(message, data, "reader_id"), firstText(message, data, "language"))
	if err != nil {
		state.publishTTSError(conn, jobID, err.Error())
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
	dictionary, err := state.dictionary(language)
	if err != nil {
		state.publishTTSError(conn, jobID, err.Error())
		return
	}
	outputFormat := normalizeOutputFormat(firstText(message, data, "output_format", "format"))

	audio, provider, err := synthesizeWithProvider(jobCtx, state.providers, providerID, tts.Request{
		Text:            tts.NormalizeText(text, dictionary, timezone),
		VoiceID:         voiceID,
		Language:        language,
		OutputFormat:    outputFormat,
		Volume:          intValue(message, data, "volume", 100),
		Rate:            intValue(message, data, "rate", 0),
		SentenceSilence: floatValue(message, data, "sentence_silence", 0),
	})
	if err != nil {
		state.publishTTSError(conn, jobID, err.Error())
		return
	}
	if audio.Format != tts.FormatWAV && audio.Format != tts.FormatPCM16LE {
		state.publishTTSError(conn, jobID, fmt.Sprintf("unsupported audio format %q", audio.Format))
		return
	}

	outputPath := firstText(message, data, "output_path", "out")
	if outputPath == "" {
		outputPath = filepath.Join(cfg.OutDir, sanitizeFileName(jobID)+".wav")
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		state.publishTTSError(conn, jobID, err.Error())
		return
	}
	if err := writeFileAtomic(outputPath, audio.Data, 0o644); err != nil {
		state.publishTTSError(conn, jobID, err.Error())
		return
	}
	_ = state.publishServiceEvent(conn, "tts.synthesized", jobID, map[string]any{
		"job_id":      jobID,
		"output_path": outputPath,
		"bytes":       len(audio.Data),
		"format":      audio.Format,
		"sample_rate": audio.SampleRate,
		"channels":    audio.Channels,
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

func (s *serviceState) publishTTSError(conn net.Conn, jobID string, detail string) {
	_ = s.publishServiceEvent(conn, "tts.failed", jobID, map[string]any{
		"job_id": jobID,
		"error":  detail,
	})
}

func (s *serviceState) publishServiceEvent(conn net.Conn, eventType string, subject string, data map[string]any) error {
	s.publishMu.Lock()
	defer s.publishMu.Unlock()
	return publishServiceEvent(conn, eventType, subject, data)
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

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	tmp := fmt.Sprintf("%s.%d.tmp", path, time.Now().UnixNano())
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
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

func normalizeOutputFormat(format string) tts.AudioFormat {
	switch tts.AudioFormat(strings.ToLower(strings.TrimSpace(format))) {
	case tts.FormatPCM16LE:
		return tts.FormatPCM16LE
	default:
		return tts.FormatWAV
	}
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

func envIntOrDefault(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	var parsed int
	if _, err := fmt.Sscanf(value, "%d", &parsed); err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func envFloatOrDefault(key string, fallback float64) float64 {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	var parsed float64
	if _, err := fmt.Sscanf(value, "%f", &parsed); err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}

func envDurationOrDefault(key string, fallback time.Duration) time.Duration {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed < 0 {
		return fallback
	}
	return parsed
}

func envBoolOrDefault(key string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "":
		return fallback
	case "1", "true", "yes", "on", "enabled":
		return true
	case "0", "false", "no", "off", "disabled":
		return false
	default:
		return fallback
	}
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

type ttsRuntimeEnv struct {
	PiperVoicesDir        string
	KokoroModelDir        string
	KokoroLang            string
	KokoroRuntimeProvider string
	KokoroThreads         int
	KokoroSpeed           float64
	KokoroLengthScale     float64
}

func setTTSRuntimeEnv(options ttsRuntimeEnv) {
	if strings.TrimSpace(options.PiperVoicesDir) != "" {
		_ = os.Setenv("HAZE_PIPER_VOICES_DIR", strings.TrimSpace(options.PiperVoicesDir))
	}
	if strings.TrimSpace(options.KokoroModelDir) != "" {
		_ = os.Setenv("HAZE_KOKORO_MODEL_DIR", strings.TrimSpace(options.KokoroModelDir))
	}
	if strings.TrimSpace(options.KokoroLang) != "" {
		_ = os.Setenv("HAZE_KOKORO_LANG", strings.TrimSpace(options.KokoroLang))
	}
	if strings.TrimSpace(options.KokoroRuntimeProvider) != "" {
		_ = os.Setenv("HAZE_KOKORO_PROVIDER", strings.TrimSpace(options.KokoroRuntimeProvider))
	}
	if options.KokoroThreads > 0 {
		_ = os.Setenv("HAZE_KOKORO_THREADS", fmt.Sprintf("%d", options.KokoroThreads))
	}
	if options.KokoroSpeed > 0 {
		_ = os.Setenv("HAZE_KOKORO_SPEED", fmt.Sprintf("%g", options.KokoroSpeed))
	}
	if options.KokoroLengthScale > 0 {
		_ = os.Setenv("HAZE_KOKORO_LENGTH_SCALE", fmt.Sprintf("%g", options.KokoroLengthScale))
	}
}

func kokoroRuntimeLang(language string) string {
	normalized := tts.NormalizeLanguage(language)
	if strings.HasPrefix(normalized, "en") || normalized == "" {
		return "en-us"
	}
	if strings.HasPrefix(normalized, "zh") || strings.HasPrefix(normalized, "cmn") || strings.HasPrefix(normalized, "yue") {
		return "zh"
	}
	return normalized
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
		for _, id := range []string{"sapi5", "espeak", "piper", "speakyapi"} {
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
		candidates := make([]tts.Provider, 0, 4)
		for _, id := range []string{"piper", "kokoro", "sapi5", "espeak", "speakyapi"} {
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
