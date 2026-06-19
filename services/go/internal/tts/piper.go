package tts

import (
	"bufio"
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const defaultPiperMetadataURL = "https://raw.githubusercontent.com/rhasspy/piper-samples/master/voices.json"
const defaultPiperVoiceBaseURL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
const hfcMalePiperVoiceID = "en_US-hfc_male-medium"
const hfcMaleLengthScale = 1.012

// PiperProvider uses a Piper executable with ONNX voice files.
type PiperProvider struct {
	Executable   string
	VoicesDir    string
	MetadataURL  string
	VoiceBaseURL string
	HTTPClient   *http.Client

	runtimeMu        sync.Mutex
	mode             string
	workerCount      int
	prewarm          bool
	useCUDA          bool
	workerScript     string
	commandPath      string
	commandPrefix    []string
	commandErr       error
	voiceIndexCached piperVoiceIndex
	voiceIndexErr    error
	resolvedVoices   map[string]resolvedPiperVoice
	workerPools      map[string]*piperWorkerPool
}

// PiperRuntimeOptions controls low-latency Piper execution.
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
	if strings.TrimSpace(executable) == "" {
		executable = strings.TrimSpace(os.Getenv("HAZE_PIPER_EXE"))
	}
	if strings.TrimSpace(executable) == "" {
		executable = "piper"
	}
	if strings.TrimSpace(voicesDir) == "" {
		voicesDir = strings.TrimSpace(os.Getenv("HAZE_PIPER_VOICES_DIR"))
	}
	if strings.TrimSpace(voicesDir) == "" {
		voicesDir = filepath.Join("managed", "voices", "piper")
	}
	return &PiperProvider{
		Executable:     executable,
		VoicesDir:      voicesDir,
		MetadataURL:    defaultPiperMetadataURL,
		VoiceBaseURL:   defaultPiperVoiceBaseURL,
		HTTPClient:     &http.Client{Timeout: 2 * time.Minute},
		mode:           "auto",
		workerCount:    1,
		prewarm:        true,
		resolvedVoices: map[string]resolvedPiperVoice{},
		workerPools:    map[string]*piperWorkerPool{},
	}
}

func (p *PiperProvider) ID() string { return "piper" }

// ConfigureRuntime updates Piper's low-latency execution mode.
func (p *PiperProvider) ConfigureRuntime(options PiperRuntimeOptions) {
	p.runtimeMu.Lock()
	defer p.runtimeMu.Unlock()
	p.mode = normalizePiperMode(options.Mode)
	if options.Workers > 0 {
		p.workerCount = options.Workers
	}
	p.prewarm = options.Prewarm
	p.useCUDA = options.UseCUDA
	p.workerScript = strings.TrimSpace(options.WorkerScript)
}

// Prewarm starts a persistent worker and verifies the PCM framing path for the requested voice.
func (p *PiperProvider) Prewarm(ctx context.Context, req Request) error {
	if p.workerMode() == "cli" {
		return nil
	}
	modelPath, configPath, err := p.ensureVoice(ctx, req.VoiceID)
	if err != nil {
		return err
	}
	voice := resolvedPiperVoice{ID: cleanPiperVoiceID(req.VoiceID), ModelPath: modelPath, ConfigPath: configPath}
	pool, err := p.workerPool(ctx, voice)
	if err != nil {
		return err
	}
	worker, err := pool.acquire(ctx)
	if err != nil {
		return err
	}
	warmReq := req
	if strings.TrimSpace(warmReq.Text) == "" {
		warmReq.Text = "Ready."
	}
	warmReq.OutputFormat = FormatPCM16LE
	_, err = worker.synthesize(ctx, warmReq)
	pool.release(worker, err == nil)
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
	if req.OutputFormat == FormatPCM16LE && p.workerMode() != "cli" {
		voice := resolvedPiperVoice{ID: cleanPiperVoiceID(req.VoiceID), ModelPath: modelPath, ConfigPath: configPath}
		audio, err := p.synthesizeWithWorker(ctx, voice, req)
		if err == nil {
			return audio, nil
		}
		log.Printf("piper worker failed; falling back to CLI: %v", err)
	}
	return p.synthesizeWithCLI(ctx, modelPath, configPath, req)
}

func (p *PiperProvider) synthesizeWithCLI(ctx context.Context, modelPath string, configPath string, req Request) (Audio, error) {
	tmp, err := os.CreateTemp("", "haze-piper-*.wav")
	if err != nil {
		return Audio{}, err
	}
	outputPath := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(outputPath)

	executable, prefix, err := p.command()
	if err != nil {
		return Audio{}, err
	}
	args := append(prefix, piperSynthesisArgs(modelPath, configPath, outputPath)...)
	cmd := exec.CommandContext(ctx, executable, args...)
	cmd.Stdin = strings.NewReader(req.Text)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail != "" {
			return Audio{}, errors.New(detail)
		}
		return Audio{}, err
	}
	data, err := os.ReadFile(outputPath)
	if err != nil {
		return Audio{}, err
	}
	return Audio{Format: FormatWAV, Data: data}, nil
}

func (p *PiperProvider) synthesizeWithWorker(ctx context.Context, voice resolvedPiperVoice, req Request) (Audio, error) {
	pool, err := p.workerPool(ctx, voice)
	if err != nil {
		return Audio{}, err
	}
	worker, err := pool.acquire(ctx)
	if err != nil {
		return Audio{}, err
	}
	audio, err := worker.synthesize(ctx, req)
	pool.release(worker, err == nil)
	if err != nil {
		return Audio{}, err
	}
	return audio, nil
}

func piperSynthesisArgs(modelPath string, configPath string, outputPath string) []string {
	return []string{
		"--model", modelPath,
		"--config", configPath,
		"--output_file", outputPath,
	}
}

func normalizePiperMode(mode string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "worker":
		return "worker"
	case "cli":
		return "cli"
	default:
		return "auto"
	}
}

func (p *PiperProvider) workerMode() string {
	p.runtimeMu.Lock()
	defer p.runtimeMu.Unlock()
	return normalizePiperMode(p.mode)
}

func (p *PiperProvider) command() (string, []string, error) {
	p.runtimeMu.Lock()
	if p.commandPath != "" || p.commandErr != nil {
		path := p.commandPath
		prefix := append([]string(nil), p.commandPrefix...)
		err := p.commandErr
		p.runtimeMu.Unlock()
		return path, prefix, err
	}
	p.runtimeMu.Unlock()

	executable := strings.TrimSpace(p.Executable)
	if executable == "" {
		executable = "piper"
	}
	var path string
	var prefix []string
	var err error
	if path, err = exec.LookPath(executable); err == nil {
		prefix = nil
	} else if filepath.IsAbs(executable) {
		err = fmt.Errorf("%w: piper executable %q", ErrProviderUnavailable, executable)
	} else {
		for _, candidate := range []string{"py", "python", "python3"} {
			if path, err = exec.LookPath(candidate); err == nil {
				prefix = []string{"-m", "piper"}
				break
			}
		}
		if path == "" && err == nil {
			err = fmt.Errorf("%w: piper executable %q or python -m piper", ErrProviderUnavailable, executable)
		}
	}
	p.runtimeMu.Lock()
	p.commandPath = path
	p.commandPrefix = append([]string(nil), prefix...)
	p.commandErr = err
	p.runtimeMu.Unlock()
	return path, prefix, err
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
		if err := os.WriteFile(metadataPath, raw, 0o644); err != nil {
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

type piperWorkerPool struct {
	provider *PiperProvider
	voice    resolvedPiperVoice
	python   string
	script   string
	useCUDA  bool
	size     int
	idle     chan *piperWorker
	mu       sync.Mutex
	total    int
}

type piperWorker struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	stderr *bytes.Buffer
}

type piperWorkerRequest struct {
	ID              string   `json:"id"`
	Text            string   `json:"text"`
	Volume          float64  `json:"volume"`
	SpeakerID       *int     `json:"speaker_id,omitempty"`
	LengthScale     *float64 `json:"length_scale,omitempty"`
	NoiseScale      *float64 `json:"noise_scale,omitempty"`
	NoiseWScale     *float64 `json:"noise_w_scale,omitempty"`
	NormalizeAudio  bool     `json:"normalize_audio"`
	SentenceSilence float64  `json:"sentence_silence"`
}

type piperWorkerHeader struct {
	ID          string `json:"id,omitempty"`
	Ready       bool   `json:"ready,omitempty"`
	OK          bool   `json:"ok"`
	Error       string `json:"error,omitempty"`
	Format      string `json:"format,omitempty"`
	SampleRate  int    `json:"sample_rate,omitempty"`
	Channels    int    `json:"channels,omitempty"`
	SampleWidth int    `json:"sample_width,omitempty"`
	Bytes       int    `json:"bytes,omitempty"`
}

func (p *PiperProvider) workerPool(ctx context.Context, voice resolvedPiperVoice) (*piperWorkerPool, error) {
	p.runtimeMu.Lock()
	key := strings.Join([]string{voice.ModelPath, voice.ConfigPath, fmt.Sprint(p.useCUDA)}, "\x00")
	if pool := p.workerPools[key]; pool != nil {
		p.runtimeMu.Unlock()
		return pool, nil
	}
	size := maxInt(1, p.workerCount)
	useCUDA := p.useCUDA
	p.runtimeMu.Unlock()

	python, err := p.workerPython()
	if err != nil {
		return nil, err
	}
	script, err := p.workerScriptPath()
	if err != nil {
		return nil, err
	}
	pool := &piperWorkerPool{
		provider: p,
		voice:    voice,
		python:   python,
		script:   script,
		useCUDA:  useCUDA,
		size:     size,
		idle:     make(chan *piperWorker, size),
	}
	p.runtimeMu.Lock()
	if existing := p.workerPools[key]; existing != nil {
		p.runtimeMu.Unlock()
		return existing, nil
	}
	p.workerPools[key] = pool
	p.runtimeMu.Unlock()
	worker, err := pool.start(ctx)
	if err != nil {
		p.runtimeMu.Lock()
		delete(p.workerPools, key)
		p.runtimeMu.Unlock()
		return nil, err
	}
	pool.release(worker, true)
	return pool, nil
}

func (p *PiperProvider) workerPython() (string, error) {
	if configured := strings.TrimSpace(os.Getenv("HAZE_PIPER_PYTHON")); configured != "" {
		if path, err := exec.LookPath(configured); err == nil {
			return path, nil
		}
		if filepath.IsAbs(configured) {
			if _, err := os.Stat(configured); err == nil {
				return configured, nil
			}
		}
	}
	for _, candidate := range []string{"python", "python3", "py"} {
		if path, err := exec.LookPath(candidate); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("%w: python for piper worker", ErrProviderUnavailable)
}

func (p *PiperProvider) workerScriptPath() (string, error) {
	candidates := []string{}
	if p.workerScript != "" {
		candidates = append(candidates, p.workerScript)
	}
	if env := strings.TrimSpace(os.Getenv("HAZE_PIPER_WORKER_SCRIPT")); env != "" {
		candidates = append(candidates, env)
	}
	candidates = append(candidates,
		filepath.Join("managed", "scripts", "piper_worker.py"),
		filepath.Join("scripts", "tts", "piper_worker.py"),
	)
	for _, candidate := range candidates {
		path := filepath.Clean(candidate)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}
	return "", fmt.Errorf("%w: piper worker script", ErrProviderUnavailable)
}

func (p *PiperProvider) synthID() string {
	return fmt.Sprintf("piper-%d", time.Now().UnixNano())
}

func (pool *piperWorkerPool) acquire(ctx context.Context) (*piperWorker, error) {
	select {
	case worker := <-pool.idle:
		return worker, nil
	default:
	}
	pool.mu.Lock()
	if pool.total < pool.size {
		pool.total++
		pool.mu.Unlock()
		worker, err := pool.newWorker(ctx)
		if err != nil {
			pool.mu.Lock()
			pool.total--
			pool.mu.Unlock()
			return nil, err
		}
		return worker, nil
	}
	pool.mu.Unlock()
	select {
	case worker := <-pool.idle:
		return worker, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (pool *piperWorkerPool) release(worker *piperWorker, healthy bool) {
	if worker == nil {
		return
	}
	if !healthy {
		worker.close()
		pool.mu.Lock()
		if pool.total > 0 {
			pool.total--
		}
		pool.mu.Unlock()
		return
	}
	select {
	case pool.idle <- worker:
	default:
		worker.close()
		pool.mu.Lock()
		if pool.total > 0 {
			pool.total--
		}
		pool.mu.Unlock()
	}
}

func (pool *piperWorkerPool) start(ctx context.Context) (*piperWorker, error) {
	pool.mu.Lock()
	pool.total++
	pool.mu.Unlock()
	worker, err := pool.newWorker(ctx)
	if err != nil {
		pool.mu.Lock()
		pool.total--
		pool.mu.Unlock()
		return nil, err
	}
	return worker, nil
}

func (pool *piperWorkerPool) newWorker(ctx context.Context) (*piperWorker, error) {
	args := []string{
		pool.script,
		"--model", pool.voice.ModelPath,
		"--config", pool.voice.ConfigPath,
	}
	if pool.useCUDA {
		args = append(args, "--cuda")
	}
	cmd := exec.Command(pool.python, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	worker := &piperWorker{
		cmd:    cmd,
		stdin:  stdin,
		stdout: bufio.NewReader(stdoutPipe),
		stderr: stderr,
	}
	type readyResult struct {
		Header piperWorkerHeader
		Err    error
	}
	readyCh := make(chan readyResult, 1)
	go func() {
		line, err := worker.stdout.ReadBytes('\n')
		if err != nil {
			readyCh <- readyResult{Err: fmt.Errorf("piper worker did not become ready: %w: %s", err, strings.TrimSpace(stderr.String()))}
			return
		}
		var header piperWorkerHeader
		if err := json.Unmarshal(bytes.TrimSpace(line), &header); err != nil {
			readyCh <- readyResult{Err: fmt.Errorf("piper worker ready frame: %w", err)}
			return
		}
		if !header.Ready || !header.OK {
			readyCh <- readyResult{Err: fmt.Errorf("piper worker failed: %s", strings.TrimSpace(firstNonBlank(header.Error, stderr.String())))}
			return
		}
		readyCh <- readyResult{Header: header}
	}()
	select {
	case ready := <-readyCh:
		if ready.Err != nil {
			worker.close()
			return nil, ready.Err
		}
	case <-ctx.Done():
		worker.close()
		return nil, ctx.Err()
	}
	return worker, nil
}

func (worker *piperWorker) synthesize(ctx context.Context, req Request) (Audio, error) {
	type result struct {
		Audio Audio
		Err   error
	}
	done := make(chan result, 1)
	go func() {
		audio, err := worker.synthesizeSync(req)
		done <- result{Audio: audio, Err: err}
	}()
	select {
	case result := <-done:
		return result.Audio, result.Err
	case <-ctx.Done():
		worker.close()
		return Audio{}, ctx.Err()
	}
}

func (worker *piperWorker) synthesizeSync(req Request) (Audio, error) {
	requestID := fmt.Sprintf("piper-%d", time.Now().UnixNano())
	payload := piperWorkerRequest{
		ID:              requestID,
		Text:            req.Text,
		Volume:          piperVolume(req.Volume),
		NormalizeAudio:  true,
		SentenceSilence: req.SentenceSilence,
	}
	if err := json.NewEncoder(worker.stdin).Encode(payload); err != nil {
		return Audio{}, err
	}
	line, err := worker.stdout.ReadBytes('\n')
	if err != nil {
		return Audio{}, fmt.Errorf("piper worker response: %w: %s", err, strings.TrimSpace(worker.stderr.String()))
	}
	var header piperWorkerHeader
	if err := json.Unmarshal(bytes.TrimSpace(line), &header); err != nil {
		return Audio{}, fmt.Errorf("piper worker response frame: %w", err)
	}
	if !header.OK {
		return Audio{}, errors.New(firstNonBlank(header.Error, "piper worker synthesis failed"))
	}
	if header.ID != requestID {
		return Audio{}, fmt.Errorf("piper worker response id %q did not match %q", header.ID, requestID)
	}
	if header.Format != string(FormatPCM16LE) || header.Bytes < 0 || header.SampleRate <= 0 || header.Channels <= 0 || header.SampleWidth != 2 {
		return Audio{}, fmt.Errorf("piper worker returned invalid audio metadata: %+v", header)
	}
	if frameBytes := header.SampleWidth * header.Channels; frameBytes <= 0 || header.Bytes%frameBytes != 0 {
		return Audio{}, fmt.Errorf("piper worker returned unaligned PCM bytes: bytes=%d sample_width=%d channels=%d", header.Bytes, header.SampleWidth, header.Channels)
	}
	data := make([]byte, header.Bytes)
	if _, err := io.ReadFull(worker.stdout, data); err != nil {
		return Audio{}, err
	}
	return Audio{
		Format:     FormatPCM16LE,
		SampleRate: header.SampleRate,
		Channels:   header.Channels,
		Data:       data,
	}, nil
}

func (worker *piperWorker) close() {
	if worker == nil || worker.cmd == nil || worker.cmd.Process == nil {
		return
	}
	_ = worker.stdin.Close()
	_ = worker.cmd.Process.Kill()
	_, _ = worker.cmd.Process.Wait()
}

func piperVolume(volume int) float64 {
	if volume <= 0 {
		return 1
	}
	return float64(volume) / 100
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func (p *PiperProvider) ensureVoiceFile(ctx context.Context, relativePath string, meta piperVoiceFile) (string, error) {
	localPath := filepath.Join(p.VoicesDir, filepath.FromSlash(relativePath))
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
	url := strings.TrimRight(p.VoiceBaseURL, "/") + "/" + relativePath
	raw, err := p.download(ctx, url)
	if err != nil {
		return "", err
	}
	if meta.MD5Digest != "" {
		sum := md5.Sum(raw)
		if hex.EncodeToString(sum[:]) != strings.ToLower(meta.MD5Digest) {
			return "", fmt.Errorf("downloaded piper voice file failed checksum: %s", relativePath)
		}
	}
	if err := os.WriteFile(localPath, raw, 0o644); err != nil {
		return "", err
	}
	return localPath, nil
}

func shouldPreserveLocalPiperConfig(relativePath string) bool {
	normalized := filepath.ToSlash(strings.ToLower(strings.TrimSpace(relativePath)))
	return strings.HasSuffix(normalized, strings.ToLower(hfcMalePiperVoiceID)+".onnx.json")
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
	return os.WriteFile(filepath.Clean(configPath), updated, 0o644)
}

func (p *PiperProvider) download(ctx context.Context, url string) ([]byte, error) {
	client := p.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("download %s failed: %s", url, response.Status)
	}
	return io.ReadAll(response.Body)
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
