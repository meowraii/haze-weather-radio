package tts

import (
	"bytes"
	"context"
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
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
		Executable:   executable,
		VoicesDir:    voicesDir,
		MetadataURL:  defaultPiperMetadataURL,
		VoiceBaseURL: defaultPiperVoiceBaseURL,
		HTTPClient:   &http.Client{Timeout: 2 * time.Minute},
	}
}

func (p *PiperProvider) ID() string { return "piper" }

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
	args := append(prefix, []string{
		"--model", modelPath,
		"--config", configPath,
		"--output_file", outputPath,
	}...)
	if req.SentenceSilence > 0 {
		args = append(args, "--sentence-silence", strconv.FormatFloat(req.SentenceSilence, 'f', -1, 64))
	}
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

func (p *PiperProvider) command() (string, []string, error) {
	executable := strings.TrimSpace(p.Executable)
	if executable == "" {
		executable = "piper"
	}
	if path, err := exec.LookPath(executable); err == nil {
		return path, nil, nil
	}
	if filepath.IsAbs(executable) {
		return "", nil, fmt.Errorf("%w: piper executable %q", ErrProviderUnavailable, executable)
	}
	for _, candidate := range []string{"py", "python", "python3"} {
		if path, err := exec.LookPath(candidate); err == nil {
			return path, []string{"-m", "piper"}, nil
		}
	}
	return "", nil, fmt.Errorf("%w: piper executable %q or python -m piper", ErrProviderUnavailable, executable)
}

func (p *PiperProvider) ensureVoice(ctx context.Context, voiceID string) (string, string, error) {
	voiceID = cleanPiperVoiceID(voiceID)
	if voiceID == "" {
		return "", "", fmt.Errorf("%w: piper voice_id is required", ErrProviderUnavailable)
	}
	if strings.HasSuffix(strings.ToLower(voiceID), ".onnx") {
		modelPath := filepath.Clean(voiceID)
		if _, err := os.Stat(modelPath); err != nil {
			return "", "", err
		}
		configPath := modelPath + ".json"
		if _, err := os.Stat(configPath); err != nil {
			return "", "", err
		}
		return modelPath, configPath, nil
	}

	index, err := p.voiceIndex(ctx)
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
	modelPath, err := p.ensureVoiceFile(ctx, modelRel, info.Files[modelRel])
	if err != nil {
		return "", "", err
	}
	configPath, err := p.ensureVoiceFile(ctx, configRel, info.Files[configRel])
	if err != nil {
		return "", "", err
	}
	if err := applyPiperVoiceOverrides(info.Key, configPath); err != nil {
		return "", "", err
	}
	return modelPath, configPath, nil
}

func cleanPiperVoiceID(voiceID string) string {
	voiceID = strings.TrimSpace(voiceID)
	voiceID = strings.TrimPrefix(voiceID, "piper://")
	return strings.TrimSpace(voiceID)
}

func (p *PiperProvider) voiceIndex(ctx context.Context) (piperVoiceIndex, error) {
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
