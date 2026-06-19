package tts

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
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

func TestPiperCommandCachesResolvedExecutable(t *testing.T) {
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	provider := NewPiperProvider(executable, filepath.Join(t.TempDir(), "voices"))

	path, prefix, err := provider.command()
	if err != nil {
		t.Fatal(err)
	}
	if path == "" {
		t.Fatal("command path is empty")
	}
	if len(prefix) != 0 {
		t.Fatalf("prefix = %v, want none for direct executable", prefix)
	}

	cachedPath, _, err := provider.command()
	if err != nil {
		t.Fatal(err)
	}
	if cachedPath != path {
		t.Fatalf("cached path = %q, want %q", cachedPath, path)
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

func TestPiperWorkerFramingSuccess(t *testing.T) {
	python := requirePython(t)
	script := writeWorkerScript(t, `
import argparse, json, sys
argparse.ArgumentParser().parse_known_args()
sys.stdout.write('{"ready":true,"ok":true}\n')
sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    data = b"\x00\x00\x01\x00"
    sys.stdout.write(json.dumps({"id": req["id"], "ok": True, "format": "pcm_s16le", "sample_rate": 22050, "channels": 1, "sample_width": 2, "bytes": len(data)}) + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
`)
	provider := workerTestProvider(t, python, script)
	audio, err := provider.Synthesize(context.Background(), Request{
		Text:         "hello",
		VoiceID:      workerTestModel(t),
		OutputFormat: FormatPCM16LE,
		Volume:       100,
	})
	if err != nil {
		t.Fatal(err)
	}
	if audio.Format != FormatPCM16LE || audio.SampleRate != 22050 || audio.Channels != 1 || len(audio.Data) != 4 {
		t.Fatalf("audio = %+v len=%d", audio, len(audio.Data))
	}
}

func TestPiperWorkerFramingError(t *testing.T) {
	python := requirePython(t)
	script := writeWorkerScript(t, `
import argparse, json, sys
argparse.ArgumentParser().parse_known_args()
sys.stdout.write('{"ready":true,"ok":true}\n')
sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    sys.stdout.write(json.dumps({"id": req["id"], "ok": False, "error": "boom"}) + "\n")
    sys.stdout.flush()
`)
	provider := workerTestProvider(t, python, script)
	model := workerTestModel(t)
	modelPath, configPath, err := provider.ensureVoice(context.Background(), model)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := provider.workerPool(context.Background(), resolvedPiperVoice{ID: model, ModelPath: modelPath, ConfigPath: configPath})
	if err != nil {
		t.Fatal(err)
	}
	worker, err := pool.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_, err = worker.synthesize(context.Background(), Request{Text: "hello"})
	pool.release(worker, false)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected worker error, got %v", err)
	}
}

func TestPiperWorkerRejectsUnalignedPCM(t *testing.T) {
	python := requirePython(t)
	script := writeWorkerScript(t, `
import argparse, json, sys
argparse.ArgumentParser().parse_known_args()
sys.stdout.write('{"ready":true,"ok":true}\n')
sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    sys.stdout.write(json.dumps({"id": req["id"], "ok": True, "format": "pcm_s16le", "sample_rate": 22050, "channels": 1, "sample_width": 2, "bytes": 3}) + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(b"\x00\x00\x00")
    sys.stdout.buffer.flush()
`)
	provider := workerTestProvider(t, python, script)
	model := workerTestModel(t)
	modelPath, configPath, err := provider.ensureVoice(context.Background(), model)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := provider.workerPool(context.Background(), resolvedPiperVoice{ID: model, ModelPath: modelPath, ConfigPath: configPath})
	if err != nil {
		t.Fatal(err)
	}
	worker, err := pool.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	_, err = worker.synthesize(context.Background(), Request{Text: "hello"})
	pool.release(worker, false)
	if err == nil || !strings.Contains(err.Error(), "unaligned PCM bytes") {
		t.Fatalf("expected unaligned PCM error, got %v", err)
	}
}

func TestPiperWorkerTimeoutKillsWorker(t *testing.T) {
	python := requirePython(t)
	script := writeWorkerScript(t, `
import argparse, json, sys, time
argparse.ArgumentParser().parse_known_args()
sys.stdout.write('{"ready":true,"ok":true}\n')
sys.stdout.flush()
for line in sys.stdin:
    time.sleep(5)
`)
	provider := workerTestProvider(t, python, script)
	model := workerTestModel(t)
	modelPath, configPath, err := provider.ensureVoice(context.Background(), model)
	if err != nil {
		t.Fatal(err)
	}
	pool, err := provider.workerPool(context.Background(), resolvedPiperVoice{ID: model, ModelPath: modelPath, ConfigPath: configPath})
	if err != nil {
		t.Fatal(err)
	}
	worker, err := pool.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	_, err = worker.synthesize(ctx, Request{Text: "hello"})
	pool.release(worker, false)
	if err == nil || !strings.Contains(err.Error(), "deadline") {
		t.Fatalf("expected deadline error, got %v", err)
	}
}

func TestPiperWorkerSurvivesStartupContextCancel(t *testing.T) {
	python := requirePython(t)
	script := writeWorkerScript(t, `
import argparse, json, sys
argparse.ArgumentParser().parse_known_args()
sys.stdout.write('{"ready":true,"ok":true}\n')
sys.stdout.flush()
for line in sys.stdin:
    req = json.loads(line)
    data = b"\x00\x00"
    sys.stdout.write(json.dumps({"id": req["id"], "ok": True, "format": "pcm_s16le", "sample_rate": 22050, "channels": 1, "sample_width": 2, "bytes": len(data)}) + "\n")
    sys.stdout.flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
`)
	provider := workerTestProvider(t, python, script)
	model := workerTestModel(t)
	modelPath, configPath, err := provider.ensureVoice(context.Background(), model)
	if err != nil {
		t.Fatal(err)
	}
	startCtx, cancel := context.WithCancel(context.Background())
	pool, err := provider.workerPool(startCtx, resolvedPiperVoice{ID: model, ModelPath: modelPath, ConfigPath: configPath})
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	worker, err := pool.acquire(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	audio, err := worker.synthesize(context.Background(), Request{Text: "hello"})
	pool.release(worker, err == nil)
	if err != nil {
		t.Fatalf("worker should survive startup context cancellation: %v", err)
	}
	if audio.Format != FormatPCM16LE || len(audio.Data) != 2 {
		t.Fatalf("audio = %+v len=%d", audio, len(audio.Data))
	}
}

func TestPiperSynthesisArgsDoNotUseSentenceSilence(t *testing.T) {
	args := piperSynthesisArgs("voice.onnx", "voice.onnx.json", "out.wav")
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "sentence-silence") {
		t.Fatalf("piper args must not include --sentence-silence: %v", args)
	}
	for _, wanted := range []string{"--model", "voice.onnx", "--config", "voice.onnx.json", "--output_file", "out.wav"} {
		if !strings.Contains(joined, wanted) {
			t.Fatalf("piper args missing %q: %v", wanted, args)
		}
	}
}

func requirePython(t *testing.T) string {
	t.Helper()
	for _, candidate := range []string{"python", "python3", "py"} {
		if path, err := exec.LookPath(candidate); err == nil {
			return path
		}
	}
	t.Skip("python is not available")
	return ""
}

func writeWorkerScript(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "worker.py")
	if err := os.WriteFile(path, []byte(strings.TrimSpace(body)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func workerTestModel(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	modelPath := filepath.Join(dir, "voice.onnx")
	if err := os.WriteFile(modelPath, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(modelPath+".json", []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	return modelPath
}

func workerTestProvider(t *testing.T, python string, script string) *PiperProvider {
	t.Helper()
	t.Setenv("HAZE_PIPER_PYTHON", python)
	provider := NewPiperProvider("piper", filepath.Join(t.TempDir(), "voices"))
	provider.ConfigureRuntime(PiperRuntimeOptions{
		Mode:         "worker",
		Workers:      1,
		Prewarm:      false,
		WorkerScript: script,
	})
	return provider
}

func md5Hex(value []byte) string {
	sum := md5.Sum(value)
	return hex.EncodeToString(sum[:])
}
