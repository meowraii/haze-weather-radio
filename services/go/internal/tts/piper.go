package tts

import (
	"archive/tar"
	"compress/bzip2"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const defaultPiperMetadataURL = "https://raw.githubusercontent.com/rhasspy/piper-samples/master/voices.json"
const defaultPiperVoiceBaseURL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
const defaultPiperEspeakDataURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2"
const hfcMalePiperVoiceID = "en_US-hfc_male-medium"
const hfcMaleLengthScale = 1.012

// PiperProvider uses sherpa-onnx's native VITS runtime with Piper ONNX voice files.
type PiperProvider struct {
	VoicesDir    string
	MetadataURL  string
	VoiceBaseURL string
	HTTPClient   *http.Client

	downloadMu       sync.Mutex
	runtimeMu        sync.Mutex
	prewarm          bool
	voiceIndexCached piperVoiceIndex
	voiceIndexErr    error
	resolvedVoices   map[string]resolvedPiperVoice
	nativeEngines    map[string]piperNativeEngineEntry
}

// PiperRuntimeOptions keeps legacy daemon settings compatible. Piper synthesis is native-only.
type PiperRuntimeOptions struct {
	Mode         string
	Workers      int
	Prewarm      bool
	UseCUDA      bool
	WorkerScript string
}

type resolvedPiperVoice struct {
	ID         string
	ModelPath  string
	ConfigPath string
}

type piperNativeEngine interface {
	Synthesize(context.Context, Request) (Audio, error)
	Close()
}

type piperNativeEngineEntry struct {
	engine   piperNativeEngine
	lastUsed time.Time
}

type piperVoiceConfig struct {
	Audio struct {
		SampleRate int    `json:"sample_rate"`
		Quality    string `json:"quality"`
	} `json:"audio"`
	Espeak struct {
		Voice string `json:"voice"`
	} `json:"espeak"`
	Language struct {
		Code        string `json:"code"`
		NameEnglish string `json:"name_english"`
	} `json:"language"`
	Inference struct {
		NoiseScale  float32 `json:"noise_scale"`
		LengthScale float32 `json:"length_scale"`
		NoiseW      float32 `json:"noise_w"`
	} `json:"inference"`
	NumSpeakers  int              `json:"num_speakers"`
	PhonemeIDMap map[string][]int `json:"phoneme_id_map"`
}

type piperVoiceIndex map[string]piperVoiceInfo

type piperVoiceInfo struct {
	Key      string                    `json:"key"`
	Name     string                    `json:"name"`
	Quality  string                    `json:"quality"`
	Language piperVoiceLanguage        `json:"language"`
	Files    map[string]piperVoiceFile `json:"files"`
	Aliases  []string                  `json:"aliases"`
}

type piperVoiceLanguage struct {
	Code string `json:"code"`
	Name string `json:"name_english"`
}

type piperVoiceFile struct {
	SizeBytes int64  `json:"size_bytes"`
	MD5Digest string `json:"md5_digest"`
}

// NewPiperProvider creates a Piper provider. Empty values use Haze defaults.
func NewPiperProvider(executable string, voicesDir string) *PiperProvider {
	if strings.TrimSpace(voicesDir) == "" {
		voicesDir = strings.TrimSpace(os.Getenv("HAZE_PIPER_VOICES_DIR"))
	}
	if strings.TrimSpace(voicesDir) == "" {
		voicesDir = filepath.Join("managed", "voices", "piper")
	}
	return &PiperProvider{
		VoicesDir:      voicesDir,
		MetadataURL:    defaultPiperMetadataURL,
		VoiceBaseURL:   defaultPiperVoiceBaseURL,
		HTTPClient:     defaultPiperHTTPClient(),
		prewarm:        true,
		resolvedVoices: map[string]resolvedPiperVoice{},
		nativeEngines:  map[string]piperNativeEngineEntry{},
	}
}

func defaultPiperHTTPClient() *http.Client {
	return &http.Client{
		Timeout: 2 * time.Minute,
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			MaxIdleConns:          32,
			MaxIdleConnsPerHost:   8,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ExpectContinueTimeout: time.Second,
		},
	}
}

func (p *PiperProvider) ID() string { return "piper" }

// ConfigureRuntime updates native Piper runtime options that still affect startup.
func (p *PiperProvider) ConfigureRuntime(options PiperRuntimeOptions) {
	p.runtimeMu.Lock()
	defer p.runtimeMu.Unlock()
	p.prewarm = options.Prewarm
}

// Prewarm initializes the native Piper runtime and verifies synthesis for the requested voice.
func (p *PiperProvider) Prewarm(ctx context.Context, req Request) error {
	modelPath, configPath, err := p.ensureVoice(ctx, req.VoiceID)
	if err != nil {
		return err
	}
	voice := resolvedPiperVoice{ID: cleanPiperVoiceID(req.VoiceID), ModelPath: modelPath, ConfigPath: configPath}
	warmReq := req
	if strings.TrimSpace(warmReq.Text) == "" {
		warmReq.Text = "Ready."
	}
	warmReq.OutputFormat = FormatPCM16LE
	_, err = p.synthesizeWithNative(ctx, voice, warmReq)
	return err
}

func (p *PiperProvider) ListVoices(ctx context.Context) ([]Voice, error) {
	index, err := p.voiceIndex(ctx)
	if err != nil {
		return nil, err
	}
	keys := make([]string, 0, len(index))
	for key := range index {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	voices := make([]Voice, 0, len(keys))
	for _, key := range keys {
		item := index[key]
		language := NormalizeLanguage(strings.ReplaceAll(item.Language.Code, "_", "-"))
		voices = append(voices, Voice{
			ID:       key,
			Name:     strings.TrimSpace(item.Name),
			Provider: p.ID(),
			Language: []string{language},
		})
	}
	return voices, nil
}

func (p *PiperProvider) Synthesize(ctx context.Context, req Request) (Audio, error) {
	if strings.TrimSpace(req.Text) == "" {
		return Audio{}, errors.New("empty synthesis text")
	}
	modelPath, configPath, err := p.ensureVoice(ctx, req.VoiceID)
	if err != nil {
		return Audio{}, err
	}
	voice := resolvedPiperVoice{ID: cleanPiperVoiceID(req.VoiceID), ModelPath: modelPath, ConfigPath: configPath}
	return p.synthesizeWithNative(ctx, voice, req)
}

// PruneIdleRuntime closes native Piper voice engines that have not been used recently.
func (p *PiperProvider) PruneIdleRuntime(maxIdle time.Duration) int {
	if maxIdle <= 0 {
		return 0
	}
	cutoff := time.Now().Add(-maxIdle)
	var stale []piperNativeEngine
	p.runtimeMu.Lock()
	for key, entry := range p.nativeEngines {
		if entry.engine == nil || entry.lastUsed.Before(cutoff) {
			delete(p.nativeEngines, key)
			if entry.engine != nil {
				stale = append(stale, entry.engine)
			}
		}
	}
	p.runtimeMu.Unlock()
	for _, engine := range stale {
		engine.Close()
	}
	return len(stale)
}

func loadPiperVoiceConfig(configPath string) (piperVoiceConfig, error) {
	raw, err := os.ReadFile(filepath.Clean(configPath))
	if err != nil {
		return piperVoiceConfig{}, err
	}
	var config piperVoiceConfig
	if err := json.Unmarshal(raw, &config); err != nil {
		return piperVoiceConfig{}, fmt.Errorf("parse piper voice config: %w", err)
	}
	if config.Inference.NoiseScale == 0 {
		config.Inference.NoiseScale = 0.667
	}
	if config.Inference.NoiseW == 0 {
		config.Inference.NoiseW = 0.8
	}
	if config.Inference.LengthScale == 0 {
		config.Inference.LengthScale = 1
	}
	return config, nil
}

func ensurePiperModelMetadata(modelPath string, config piperVoiceConfig) error {
	if config.Audio.SampleRate <= 0 {
		return fmt.Errorf("%w: piper voice config has no audio.sample_rate", ErrProviderUnavailable)
	}
	nSpeakers := config.NumSpeakers
	if nSpeakers <= 0 {
		nSpeakers = 1
	}
	language := strings.TrimSpace(config.Language.NameEnglish)
	if language == "" {
		language = strings.TrimSpace(config.Language.Code)
	}
	voice := strings.TrimSpace(config.Espeak.Voice)
	metadata := map[string]string{
		"model_type":  "vits",
		"comment":     "piper",
		"language":    language,
		"voice":       voice,
		"has_espeak":  "1",
		"n_speakers":  fmt.Sprint(nSpeakers),
		"sample_rate": fmt.Sprint(config.Audio.SampleRate),
	}
	raw, err := os.ReadFile(filepath.Clean(modelPath))
	if err != nil {
		return err
	}
	existing := onnxMetadata(raw)
	missing := map[string]string{}
	for key, value := range metadata {
		if strings.TrimSpace(existing[key]) == "" {
			missing[key] = value
		}
	}
	if len(missing) == 0 {
		return nil
	}
	updated := append([]byte(nil), raw...)
	keys := make([]string, 0, len(missing))
	for key := range missing {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		updated = appendONNXMetadataEntry(updated, key, missing[key])
	}
	return writeFileAtomic(filepath.Clean(modelPath), updated, 0o644)
}

func appendONNXMetadataEntry(raw []byte, key string, value string) []byte {
	entry := []byte{}
	entry = appendProtoString(entry, 1, key)
	entry = appendProtoString(entry, 2, value)
	raw = appendProtoVarint(raw, uint64(14<<3|2))
	raw = appendProtoVarint(raw, uint64(len(entry)))
	raw = append(raw, entry...)
	return raw
}

func appendProtoString(raw []byte, field int, value string) []byte {
	raw = appendProtoVarint(raw, uint64(field<<3|2))
	raw = appendProtoVarint(raw, uint64(len(value)))
	raw = append(raw, value...)
	return raw
}

func appendProtoVarint(raw []byte, value uint64) []byte {
	for value >= 0x80 {
		raw = append(raw, byte(value)|0x80)
		value >>= 7
	}
	return append(raw, byte(value))
}

func onnxMetadata(raw []byte) map[string]string {
	values := map[string]string{}
	for offset := 0; offset < len(raw); {
		tag, next, ok := readProtoVarint(raw, offset)
		if !ok {
			return values
		}
		offset = next
		field := int(tag >> 3)
		wire := int(tag & 0x7)
		if field == 14 && wire == 2 {
			payload, after, ok := readProtoBytes(raw, offset)
			if !ok {
				return values
			}
			key, value := parseONNXMetadataEntry(payload)
			if key != "" {
				values[key] = value
			}
			offset = after
			continue
		}
		nextOffset, ok := skipProtoValue(raw, offset, wire)
		if !ok {
			return values
		}
		offset = nextOffset
	}
	return values
}

func parseONNXMetadataEntry(raw []byte) (string, string) {
	var key string
	var value string
	for offset := 0; offset < len(raw); {
		tag, next, ok := readProtoVarint(raw, offset)
		if !ok {
			return key, value
		}
		offset = next
		field := int(tag >> 3)
		wire := int(tag & 0x7)
		if (field == 1 || field == 2) && wire == 2 {
			payload, after, ok := readProtoBytes(raw, offset)
			if !ok {
				return key, value
			}
			if field == 1 {
				key = string(payload)
			} else {
				value = string(payload)
			}
			offset = after
			continue
		}
		nextOffset, ok := skipProtoValue(raw, offset, wire)
		if !ok {
			return key, value
		}
		offset = nextOffset
	}
	return key, value
}

func readProtoBytes(raw []byte, offset int) ([]byte, int, bool) {
	length, next, ok := readProtoVarint(raw, offset)
	if !ok || length > uint64(len(raw)-next) {
		return nil, offset, false
	}
	end := next + int(length)
	return raw[next:end], end, true
}

func readProtoVarint(raw []byte, offset int) (uint64, int, bool) {
	var value uint64
	for shift := 0; shift < 64 && offset < len(raw); shift += 7 {
		b := raw[offset]
		offset++
		value |= uint64(b&0x7f) << shift
		if b < 0x80 {
			return value, offset, true
		}
	}
	return 0, offset, false
}

func skipProtoValue(raw []byte, offset int, wire int) (int, bool) {
	switch wire {
	case 0:
		_, next, ok := readProtoVarint(raw, offset)
		return next, ok
	case 1:
		if len(raw)-offset < 8 {
			return offset, false
		}
		return offset + 8, true
	case 2:
		_, next, ok := readProtoBytes(raw, offset)
		return next, ok
	case 5:
		if len(raw)-offset < 4 {
			return offset, false
		}
		return offset + 4, true
	default:
		return offset, false
	}
}

func ensurePiperTokensFile(configPath string, config piperVoiceConfig) (string, error) {
	tokensPath := strings.TrimSuffix(configPath, ".json") + ".tokens.txt"
	if _, err := os.Stat(tokensPath); err == nil {
		return tokensPath, nil
	}
	if len(config.PhonemeIDMap) == 0 {
		return "", fmt.Errorf("%w: piper voice config has no phoneme_id_map", ErrProviderUnavailable)
	}
	type tokenID struct {
		Token string
		ID    int
	}
	tokens := []tokenID{}
	for token, values := range config.PhonemeIDMap {
		if len(values) == 0 || values[0] < 0 {
			continue
		}
		tokens = append(tokens, tokenID{Token: token, ID: values[0]})
	}
	if len(tokens) == 0 {
		return "", fmt.Errorf("%w: piper voice config has no usable token IDs", ErrProviderUnavailable)
	}
	sort.Slice(tokens, func(i, j int) bool {
		if tokens[i].ID == tokens[j].ID {
			return tokens[i].Token < tokens[j].Token
		}
		return tokens[i].ID < tokens[j].ID
	})
	var builder strings.Builder
	for _, item := range tokens {
		builder.WriteString(item.Token)
		builder.WriteByte(' ')
		builder.WriteString(fmt.Sprint(item.ID))
		builder.WriteByte('\n')
	}
	tmp := tokensPath + ".tmp"
	if err := os.WriteFile(tmp, []byte(builder.String()), 0o644); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, tokensPath); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	return tokensPath, nil
}

func (p *PiperProvider) ensurePiperDataDir(ctx context.Context, modelPath string) (string, error) {
	if dataDir := piperDataDir(p.VoicesDir, modelPath); dataDir != "" {
		return dataDir, nil
	}
	url := envOrDefault("HAZE_PIPER_ESPEAK_DATA_URL", defaultPiperEspeakDataURL)
	if strings.TrimSpace(url) == "" {
		return "", fmt.Errorf("%w: piper espeak-ng-data directory not found", ErrProviderUnavailable)
	}
	if err := os.MkdirAll(p.VoicesDir, 0o755); err != nil {
		return "", err
	}
	archivePath := filepath.Join(p.VoicesDir, "espeak-ng-data.tar.bz2")
	if _, err := os.Stat(archivePath); err != nil {
		if err := p.downloadPiperDataArchive(ctx, url, archivePath); err != nil {
			return "", err
		}
	}
	if err := extractPiperDataArchive(archivePath, p.VoicesDir); err != nil {
		return "", err
	}
	if dataDir := piperDataDir(p.VoicesDir, modelPath); dataDir != "" {
		return dataDir, nil
	}
	return "", fmt.Errorf("%w: piper espeak-ng-data archive did not contain espeak-ng-data", ErrProviderUnavailable)
}

func (p *PiperProvider) downloadPiperDataArchive(ctx context.Context, url string, archivePath string) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	response, err := p.HTTPClient.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("download piper espeak-ng-data: %s", response.Status)
	}
	tmp := archivePath + ".tmp"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, response.Body)
	closeErr := out.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if err := os.Rename(tmp, archivePath); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func extractPiperDataArchive(archivePath string, targetRoot string) error {
	file, err := os.Open(filepath.Clean(archivePath))
	if err != nil {
		return err
	}
	defer file.Close()
	reader := tar.NewReader(bzip2.NewReader(file))
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		name := filepath.Clean(filepath.FromSlash(header.Name))
		if strings.HasPrefix(name, "..") || filepath.IsAbs(name) {
			return fmt.Errorf("piper espeak-ng-data archive entry escapes target: %s", header.Name)
		}
		target := filepath.Join(targetRoot, name)
		if !kokoroPathWithin(targetRoot, target) {
			return fmt.Errorf("piper espeak-ng-data archive entry escapes target: %s", header.Name)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := kokoroWriteArchiveFile(target, reader, header.FileInfo().Mode()); err != nil {
				return err
			}
		}
	}
}

func piperDataDir(voicesDir string, modelPath string) string {
	for _, candidate := range []string{
		strings.TrimSpace(os.Getenv("HAZE_PIPER_DATA_DIR")),
		strings.TrimSpace(os.Getenv("HAZE_KOKORO_DATA_DIR")),
		filepath.Join(voicesDir, "espeak-ng-data"),
		filepath.Join("managed", "voices", "kokoro-multi-lang-v1_0", "espeak-ng-data"),
		filepath.Join(filepath.Dir(modelPath), "espeak-ng-data"),
	} {
		if candidate == "" {
			continue
		}
		info, err := os.Stat(candidate)
		if err == nil && info.IsDir() {
			return candidate
		}
	}
	return ""
}

func piperRuntimeProvider() string {
	return envOrDefault("HAZE_PIPER_PROVIDER", envOrDefault("HAZE_KOKORO_PROVIDER", "cpu"))
}

func piperThreads() int {
	return kokoroEnvInt("HAZE_PIPER_THREADS", kokoroEnvInt("HAZE_KOKORO_THREADS", defaultKokoroThreads()))
}

func piperDebug() bool {
	return kokoroEnvBool("HAZE_PIPER_DEBUG", kokoroEnvBool("HAZE_KOKORO_DEBUG", false))
}

func piperSpeedForRequest(rate int) float32 {
	if rate <= 0 {
		rate = 100
	}
	return kokoroClampFloat32(float32(rate)/100, 0.5, 2.0)
}

func (p *PiperProvider) ensureVoice(ctx context.Context, voiceID string) (string, string, error) {
	voiceID = cleanPiperVoiceID(voiceID)
	if voiceID == "" {
		return "", "", fmt.Errorf("%w: piper voice_id is required", ErrProviderUnavailable)
	}
	p.runtimeMu.Lock()
	if voice, ok := p.resolvedVoices[voiceID]; ok {
		p.runtimeMu.Unlock()
		return voice.ModelPath, voice.ConfigPath, nil
	}
	p.runtimeMu.Unlock()

	var modelPath string
	var configPath string
	var err error
	if strings.HasSuffix(strings.ToLower(voiceID), ".onnx") {
		modelPath = filepath.Clean(voiceID)
		if _, err := os.Stat(modelPath); err != nil {
			return "", "", err
		}
		configPath = modelPath + ".json"
		if _, err := os.Stat(configPath); err != nil {
			return "", "", err
		}
	} else {
		var index piperVoiceIndex
		index, err = p.voiceIndex(ctx)
		if err != nil {
			return "", "", err
		}
		info, ok := findPiperVoice(index, voiceID)
		if !ok {
			return "", "", fmt.Errorf("%w: piper voice %q", ErrProviderUnavailable, voiceID)
		}
		modelRel, configRel := piperModelFiles(info)
		if modelRel == "" || configRel == "" {
			return "", "", fmt.Errorf("piper voice %q is missing model files", voiceID)
		}
		modelPath, err = p.ensureVoiceFile(ctx, modelRel, info.Files[modelRel])
		if err != nil {
			return "", "", err
		}
		configPath, err = p.ensureVoiceFile(ctx, configRel, info.Files[configRel])
		if err != nil {
			return "", "", err
		}
		if err := applyPiperVoiceOverrides(info.Key, configPath); err != nil {
			return "", "", err
		}
	}
	p.runtimeMu.Lock()
	p.resolvedVoices[voiceID] = resolvedPiperVoice{ID: voiceID, ModelPath: modelPath, ConfigPath: configPath}
	p.runtimeMu.Unlock()
	return modelPath, configPath, nil
}

func cleanPiperVoiceID(voiceID string) string {
	voiceID = strings.TrimSpace(voiceID)
	voiceID = strings.TrimPrefix(voiceID, "piper://")
	return strings.TrimSpace(voiceID)
}

func (p *PiperProvider) voiceIndex(ctx context.Context) (piperVoiceIndex, error) {
	p.runtimeMu.Lock()
	if p.voiceIndexCached != nil || p.voiceIndexErr != nil {
		index := p.voiceIndexCached
		err := p.voiceIndexErr
		p.runtimeMu.Unlock()
		return index, err
	}
	p.runtimeMu.Unlock()

	metadataPath := filepath.Join(p.VoicesDir, "voices.json")
	raw, err := os.ReadFile(metadataPath)
	if err != nil {
		if !os.IsNotExist(err) {
			return nil, err
		}
		if err := os.MkdirAll(p.VoicesDir, 0o755); err != nil {
			return nil, err
		}
		raw, err = p.download(ctx, p.MetadataURL)
		if err != nil {
			return nil, err
		}
		if err := writeFileAtomic(metadataPath, raw, 0o644); err != nil {
			return nil, err
		}
	}
	var index piperVoiceIndex
	if err := json.Unmarshal(raw, &index); err != nil {
		return nil, fmt.Errorf("parse piper voice metadata: %w", err)
	}
	p.runtimeMu.Lock()
	p.voiceIndexCached = index
	p.voiceIndexErr = nil
	p.runtimeMu.Unlock()
	return index, nil
}

func findPiperVoice(index piperVoiceIndex, voiceID string) (piperVoiceInfo, bool) {
	if info, ok := index[voiceID]; ok {
		return info, true
	}
	lower := strings.ToLower(voiceID)
	for key, info := range index {
		if strings.ToLower(key) == lower {
			return info, true
		}
		for _, alias := range info.Aliases {
			if strings.ToLower(strings.TrimSpace(alias)) == lower {
				return info, true
			}
		}
	}
	return piperVoiceInfo{}, false
}

func piperModelFiles(info piperVoiceInfo) (string, string) {
	var modelPath string
	var configPath string
	for path := range info.Files {
		lower := strings.ToLower(path)
		switch {
		case strings.HasSuffix(lower, ".onnx"):
			modelPath = path
		case strings.HasSuffix(lower, ".onnx.json"):
			configPath = path
		}
	}
	return modelPath, configPath
}

func (p *PiperProvider) ensureVoiceFile(ctx context.Context, relativePath string, meta piperVoiceFile) (string, error) {
	localPath := filepath.Join(p.VoicesDir, filepath.FromSlash(relativePath))
	if piperONNXHasSherpaMetadata(localPath) {
		return localPath, nil
	}
	if shouldPreserveLocalPiperConfig(relativePath) {
		if _, err := os.Stat(localPath); err == nil {
			return localPath, nil
		}
	}
	if ok, err := fileMatchesMD5(localPath, meta.MD5Digest); ok && err == nil {
		return localPath, nil
	}
	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return "", err
	}
	p.downloadMu.Lock()
	defer p.downloadMu.Unlock()
	if shouldPreserveLocalPiperConfig(relativePath) {
		if _, err := os.Stat(localPath); err == nil {
			return localPath, nil
		}
	}
	if ok, err := fileMatchesMD5(localPath, meta.MD5Digest); ok && err == nil {
		return localPath, nil
	}
	url := strings.TrimRight(p.VoiceBaseURL, "/") + "/" + relativePath
	if err := p.downloadFile(ctx, url, localPath, relativePath, meta); err != nil {
		return "", err
	}
	return localPath, nil
}

func shouldPreserveLocalPiperConfig(relativePath string) bool {
	normalized := filepath.ToSlash(strings.ToLower(strings.TrimSpace(relativePath)))
	return strings.HasSuffix(normalized, strings.ToLower(hfcMalePiperVoiceID)+".onnx.json")
}

func piperONNXHasSherpaMetadata(path string) bool {
	if !strings.HasSuffix(strings.ToLower(strings.TrimSpace(path)), ".onnx") {
		return false
	}
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return false
	}
	metadata := onnxMetadata(raw)
	return strings.EqualFold(metadata["model_type"], "vits") &&
		strings.EqualFold(metadata["comment"], "piper") &&
		strings.TrimSpace(metadata["sample_rate"]) != ""
}

func applyPiperVoiceOverrides(voiceID string, configPath string) error {
	if !strings.EqualFold(cleanPiperVoiceID(voiceID), hfcMalePiperVoiceID) {
		return nil
	}
	raw, err := os.ReadFile(filepath.Clean(configPath))
	if err != nil {
		return err
	}
	var config map[string]any
	if err := json.Unmarshal(raw, &config); err != nil {
		return err
	}
	inference, _ := config["inference"].(map[string]any)
	if inference == nil {
		inference = map[string]any{}
		config["inference"] = inference
	}
	if existing, ok := inference["length_scale"].(float64); ok && existing == hfcMaleLengthScale {
		return nil
	}
	inference["length_scale"] = hfcMaleLengthScale
	updated, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	updated = append(updated, '\n')
	return writeFileAtomic(filepath.Clean(configPath), updated, 0o644)
}

func (p *PiperProvider) download(ctx context.Context, url string) ([]byte, error) {
	response, err := p.downloadResponse(ctx, url)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	return io.ReadAll(response.Body)
}

func (p *PiperProvider) downloadFile(ctx context.Context, url string, localPath string, relativePath string, meta piperVoiceFile) error {
	response, err := p.downloadResponse(ctx, url)
	if err != nil {
		return err
	}
	defer response.Body.Close()

	tmp := fmt.Sprintf("%s.%d.tmp", localPath, time.Now().UnixNano())
	file, err := os.OpenFile(tmp, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	cleanup := true
	defer func() {
		if cleanup {
			_ = os.Remove(tmp)
		}
	}()

	hash := md5.New()
	reader := io.Reader(response.Body)
	if meta.SizeBytes > 0 {
		reader = io.LimitReader(response.Body, meta.SizeBytes+1)
	}
	written, copyErr := io.Copy(io.MultiWriter(file, hash), reader)
	closeErr := file.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if meta.SizeBytes > 0 && written != meta.SizeBytes {
		return fmt.Errorf("downloaded piper voice file has unexpected size: %s", relativePath)
	}
	if meta.MD5Digest != "" && hex.EncodeToString(hash.Sum(nil)) != strings.ToLower(meta.MD5Digest) {
		return fmt.Errorf("downloaded piper voice file failed checksum: %s", relativePath)
	}
	if err := os.Rename(tmp, localPath); err != nil {
		return err
	}
	cleanup = false
	return nil
}

func (p *PiperProvider) downloadResponse(ctx context.Context, url string) (*http.Response, error) {
	client := p.HTTPClient
	if client == nil {
		client = defaultPiperHTTPClient()
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		response.Body.Close()
		return nil, fmt.Errorf("download %s failed: %s", url, response.Status)
	}
	return response, nil
}

func fileMatchesMD5(path string, expected string) (bool, error) {
	if expected == "" {
		_, err := os.Stat(path)
		return err == nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer file.Close()
	hash := md5.New()
	if _, err := io.Copy(hash, file); err != nil {
		return false, err
	}
	return hex.EncodeToString(hash.Sum(nil)) == strings.ToLower(expected), nil
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
