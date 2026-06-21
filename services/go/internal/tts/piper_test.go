package tts

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPiperProviderDownloadsMissingVoiceFiles(t *testing.T) {
	model := []byte("model")
	config := []byte(`{"audio":{"sample_rate":22050}}`)
	modelPath := "en/en_US/amy/medium/en_US-amy-medium.onnx"
	configPath := "en/en_US/amy/medium/en_US-amy-medium.onnx.json"
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/voices.json":
			_, _ = fmt.Fprintf(writer, `{
				"en_US-amy-medium": {
					"key": "en_US-amy-medium",
					"name": "amy",
					"quality": "medium",
					"language": {"code": "en_US", "name_english": "English"},
					"files": {
						"%s": {"size_bytes": %d, "md5_digest": "%s"},
						"%s": {"size_bytes": %d, "md5_digest": "%s"}
					},
					"aliases": ["amy"]
				}
			}`, modelPath, len(model), md5Hex(model), configPath, len(config), md5Hex(config))
		case "/" + modelPath:
			_, _ = writer.Write(model)
		case "/" + configPath:
			_, _ = writer.Write(config)
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()

	provider := NewPiperProvider("piper", filepath.Join(t.TempDir(), "voices"))
	provider.MetadataURL = server.URL + "/voices.json"
	provider.VoiceBaseURL = server.URL

	resolvedModel, resolvedConfig, err := provider.ensureVoice(context.Background(), "amy")
	if err != nil {
		t.Fatal(err)
	}
	if resolvedModel != filepath.Join(provider.VoicesDir, filepath.FromSlash(modelPath)) {
		t.Fatalf("model path = %s", resolvedModel)
	}
	if resolvedConfig != filepath.Join(provider.VoicesDir, filepath.FromSlash(configPath)) {
		t.Fatalf("config path = %s", resolvedConfig)
	}
	if got, err := os.ReadFile(resolvedModel); err != nil || string(got) != string(model) {
		t.Fatalf("model file = %q err=%v", got, err)
	}
	if got, err := os.ReadFile(resolvedConfig); err != nil || string(got) != string(config) {
		t.Fatalf("config file = %q err=%v", got, err)
	}
}

func TestPiperProviderSlowsHFCMaleConfig(t *testing.T) {
	model := []byte("model")
	config := []byte(`{"audio":{"sample_rate":22050},"inference":{"length_scale":1,"noise_scale":0.667}}`)
	modelPath := "en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx"
	configPath := "en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx.json"
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/voices.json":
			_, _ = fmt.Fprintf(writer, `{
				"en_US-hfc_male-medium": {
					"key": "en_US-hfc_male-medium",
					"name": "hfc_male",
					"quality": "medium",
					"language": {"code": "en_US", "name_english": "English"},
					"files": {
						"%s": {"size_bytes": %d, "md5_digest": "%s"},
						"%s": {"size_bytes": %d, "md5_digest": "%s"}
					}
				}
			}`, modelPath, len(model), md5Hex(model), configPath, len(config), md5Hex(config))
		case "/" + modelPath:
			_, _ = writer.Write(model)
		case "/" + configPath:
			_, _ = writer.Write(config)
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()

	provider := NewPiperProvider("piper", filepath.Join(t.TempDir(), "voices"))
	provider.MetadataURL = server.URL + "/voices.json"
	provider.VoiceBaseURL = server.URL

	_, resolvedConfig, err := provider.ensureVoice(context.Background(), "en_US-hfc_male-medium")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(resolvedConfig)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"length_scale": 1.012`) {
		t.Fatalf("hfc_male config was not slowed:\n%s", string(raw))
	}

	if _, _, err := provider.ensureVoice(context.Background(), "en_US-hfc_male-medium"); err != nil {
		t.Fatalf("local overridden config should be preserved: %v", err)
	}
}

func TestPiperProviderUsesDirectOnnxPath(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "voice.onnx")
	configPath := modelPath + ".json"
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	provider := NewPiperProvider("piper", filepath.Join(dir, "voices"))

	model, config, err := provider.ensureVoice(context.Background(), modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if model != modelPath || config != configPath {
		t.Fatalf("model=%s config=%s", model, config)
	}
}

func TestPiperProviderCachesResolvedVoice(t *testing.T) {
	model := []byte("model")
	config := []byte(`{"audio":{"sample_rate":22050}}`)
	modelPath := "en/en_US/amy/medium/en_US-amy-medium.onnx"
	configPath := "en/en_US/amy/medium/en_US-amy-medium.onnx.json"
	metadataHits := 0
	modelHits := 0
	configHits := 0
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/voices.json":
			metadataHits++
			_, _ = fmt.Fprintf(writer, `{
				"en_US-amy-medium": {
					"key": "en_US-amy-medium",
					"name": "amy",
					"quality": "medium",
					"language": {"code": "en_US", "name_english": "English"},
					"files": {
						"%s": {"size_bytes": %d, "md5_digest": "%s"},
						"%s": {"size_bytes": %d, "md5_digest": "%s"}
					}
				}
			}`, modelPath, len(model), md5Hex(model), configPath, len(config), md5Hex(config))
		case "/" + modelPath:
			modelHits++
			_, _ = writer.Write(model)
		case "/" + configPath:
			configHits++
			_, _ = writer.Write(config)
		default:
			http.NotFound(writer, request)
		}
	}))
	defer server.Close()

	provider := NewPiperProvider("piper", filepath.Join(t.TempDir(), "voices"))
	provider.MetadataURL = server.URL + "/voices.json"
	provider.VoiceBaseURL = server.URL
	for i := 0; i < 2; i++ {
		if _, _, err := provider.ensureVoice(context.Background(), "en_US-amy-medium"); err != nil {
			t.Fatal(err)
		}
	}
	if metadataHits != 1 || modelHits != 1 || configHits != 1 {
		t.Fatalf("hits metadata/model/config = %d/%d/%d", metadataHits, modelHits, configHits)
	}
}

func TestEnsurePiperTokensFileFromConfig(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "voice.onnx.json")
	config := piperVoiceConfig{
		PhonemeIDMap: map[string][]int{
			" ": []int{3},
			"a": []int{14},
			"b": []int{15, 99},
		},
	}
	tokensPath, err := ensurePiperTokensFile(configPath, config)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(tokensPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != "  3\na 14\nb 15\n" {
		t.Fatalf("tokens = %q", string(raw))
	}
}

func TestEnsurePiperModelMetadataAppendsRequiredONNXEntries(t *testing.T) {
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "voice.onnx")
	if err := os.WriteFile(modelPath, []byte{0x08, 0x01}, 0o600); err != nil {
		t.Fatal(err)
	}
	config := piperVoiceConfig{
		NumSpeakers: 2,
	}
	config.Audio.SampleRate = 22050
	config.Espeak.Voice = "en-us"
	config.Language.NameEnglish = "English"

	if err := ensurePiperModelMetadata(modelPath, config); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	metadata := onnxMetadata(raw)
	for key, want := range map[string]string{
		"model_type":  "vits",
		"comment":     "piper",
		"language":    "English",
		"voice":       "en-us",
		"has_espeak":  "1",
		"n_speakers":  "2",
		"sample_rate": "22050",
	} {
		if got := metadata[key]; got != want {
			t.Fatalf("metadata[%s] = %q, want %q", key, got, want)
		}
	}
}

func md5Hex(value []byte) string {
	sum := md5.Sum(value)
	return hex.EncodeToString(sum[:])
}
