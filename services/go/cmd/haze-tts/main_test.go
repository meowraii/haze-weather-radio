package main

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/tts"
)

type fakeProvider struct {
	id       string
	audio    tts.Audio
	mu       sync.Mutex
	requests []tts.Request
}

func (p *fakeProvider) ID() string { return p.id }

func (p *fakeProvider) ListVoices(context.Context) ([]tts.Voice, error) {
	return nil, nil
}

func (p *fakeProvider) Synthesize(_ context.Context, req tts.Request) (tts.Audio, error) {
	p.mu.Lock()
	p.requests = append(p.requests, req)
	p.mu.Unlock()
	return p.audio, nil
}

func (p *fakeProvider) requestAt(index int) tts.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.requests[index]
}

func TestHandleSynthesisJobDefaultsToWAV(t *testing.T) {
	provider := &fakeProvider{id: "piper", audio: tts.Audio{Format: tts.FormatWAV, Data: []byte("wav")}}
	state := &serviceState{
		cfg:          serviceConfig{Timeout: time.Second},
		providers:    map[string]tts.Provider{"piper": provider},
		dictionaries: map[string]dictionaryResult{},
	}
	conn, peer := net.Pipe()
	defer conn.Close()
	defer peer.Close()
	outputPath := filepath.Join(t.TempDir(), "out.wav")

	go handleSynthesisJob(context.Background(), conn, state, map[string]any{
		"type": "tts.synthesize",
		"data": map[string]any{
			"job_id":      "job-1",
			"provider":    "piper",
			"text":        "hello",
			"output_path": outputPath,
		},
	})

	var event map[string]any
	if err := json.NewDecoder(peer).Decode(&event); err != nil {
		t.Fatal(err)
	}
	if event["type"] != "tts.synthesized" {
		t.Fatalf("event type = %v", event["type"])
	}
	data := event["data"].(map[string]any)
	if data["format"] != string(tts.FormatWAV) {
		t.Fatalf("format = %v", data["format"])
	}
	if got := provider.requestAt(0).OutputFormat; got != tts.FormatWAV {
		t.Fatalf("request output format = %q", got)
	}
	if raw, err := os.ReadFile(outputPath); err != nil || string(raw) != "wav" {
		t.Fatalf("output = %q err=%v", raw, err)
	}
}

func TestHandleSynthesisJobPublishesPCMMetadata(t *testing.T) {
	provider := &fakeProvider{id: "piper", audio: tts.Audio{
		Format:     tts.FormatPCM16LE,
		SampleRate: 22050,
		Channels:   1,
		Data:       []byte{0, 0, 1, 0},
	}}
	state := &serviceState{
		cfg:          serviceConfig{Timeout: time.Second},
		providers:    map[string]tts.Provider{"piper": provider},
		dictionaries: map[string]dictionaryResult{},
	}
	conn, peer := net.Pipe()
	defer conn.Close()
	defer peer.Close()
	outputPath := filepath.Join(t.TempDir(), "out.pcm16")

	go handleSynthesisJob(context.Background(), conn, state, map[string]any{
		"type": "tts.synthesize",
		"data": map[string]any{
			"job_id":        "job-2",
			"provider":      "piper",
			"text":          "hello",
			"output_path":   outputPath,
			"output_format": "pcm_s16le",
		},
	})

	var event map[string]any
	if err := json.NewDecoder(peer).Decode(&event); err != nil {
		t.Fatal(err)
	}
	data := event["data"].(map[string]any)
	if data["format"] != string(tts.FormatPCM16LE) || int(data["sample_rate"].(float64)) != 22050 || int(data["channels"].(float64)) != 1 {
		t.Fatalf("pcm metadata = %#v", data)
	}
	if got := provider.requestAt(0).OutputFormat; got != tts.FormatPCM16LE {
		t.Fatalf("request output format = %q", got)
	}
	if raw, err := os.ReadFile(outputPath); err != nil || len(raw) != 4 {
		t.Fatalf("output bytes = %d err=%v", len(raw), err)
	}
}

func TestSynthesisQueuePrioritizesRealtimeJobs(t *testing.T) {
	queue := &synthesisQueue{
		high:   make(chan map[string]any, 3),
		normal: make(chan map[string]any, 3),
		low:    make(chan map[string]any, 3),
	}
	if !queue.Enqueue(context.Background(), map[string]any{"data": map[string]any{"job_id": "low", "priority": "low"}}) {
		t.Fatal("low enqueue failed")
	}
	if !queue.Enqueue(context.Background(), map[string]any{"data": map[string]any{"job_id": "normal"}}) {
		t.Fatal("normal enqueue failed")
	}
	if !queue.Enqueue(context.Background(), map[string]any{"data": map[string]any{"job_id": "high", "priority": "high"}}) {
		t.Fatal("high enqueue failed")
	}
	for _, want := range []string{"high", "normal", "low"} {
		message, ok := queue.next(context.Background())
		if !ok {
			t.Fatalf("next returned false for %s", want)
		}
		if got := firstText(message, objectValue(message, "data"), "job_id"); got != want {
			t.Fatalf("next job = %q, want %q", got, want)
		}
	}
}
