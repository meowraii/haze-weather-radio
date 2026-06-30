package webgateway

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCgenSaveAndActionsRoundTripXML(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: test\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}

	saved, err := saveCgenPayload(configPath, map[string]any{
		"enabled": true,
		"feeds": []any{
			map[string]any{
				"id":                              "CAP-IT-ALL",
				"name":                            "CAP CGEN",
				"enabled":                         true,
				"mode":                            "release",
				"program_input_url":               "udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1",
				"program_input_format":            "mpegts",
				"priority_feed_id":                "CAP-IT-ALL",
				"audio_source":                    "both",
				"mute_standby_routine":            false,
				"program_output_url":              "udp://239.0.0.2:9001?pkt_size=1316",
				"program_output_format":           "mpegts",
				"vcodec":                          "libx264",
				"acodec":                          "aac",
				"width":                           "1280",
				"height":                          "720",
				"fps":                             "source",
				"interlaced":                      true,
				"field_order":                     "bff",
				"standard":                        "atsc",
				"banner_background_enabled":       true,
				"scroll_speed":                    "6",
				"standby_mode":                    "smpte",
				"standby_text":                    "EAS Details Channel",
				"standby_font_size":               "64",
				"standby_y_percent":               "10",
				"sync_hard_reset_ms":              "500",
				"sync_max_audio_frames_per_video": "10",
				"sync_source_buffer_ms":           "400",
				"sync_reconnect_initial_ms":       "250",
				"sync_reconnect_max_ms":           "5000",
				"sync_status_interval_ms":         "750",
			},
		},
	})
	if err != nil {
		t.Fatalf("save cgen: %v", err)
	}
	feeds := saved["feeds"].([]map[string]any)
	if len(feeds) != 1 {
		t.Fatalf("feeds = %#v", saved["feeds"])
	}
	if feeds[0]["program_input_url"] != "udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1" {
		t.Fatalf("program input = %#v", feeds[0]["program_input_url"])
	}
	if feeds[0]["audio_source"] != "both" {
		t.Fatalf("audio source = %#v", feeds[0]["audio_source"])
	}
	if feeds[0]["mute_standby_routine"] != false {
		t.Fatalf("standby routine mute = %#v", feeds[0]["mute_standby_routine"])
	}
	if feeds[0]["interlaced"] != true || feeds[0]["field_order"] != "bff" || feeds[0]["standard"] != "atsc" {
		t.Fatalf("video flags = %#v", feeds[0])
	}
	if feeds[0]["scroll_speed"] != "6" {
		t.Fatalf("banner controls = %#v", feeds[0])
	}
	if feeds[0]["standby_mode"] != "smpte" ||
		feeds[0]["standby_text"] != "EAS Details Channel" ||
		feeds[0]["standby_font_size"] != "64" ||
		feeds[0]["standby_y_percent"] != "10" {
		t.Fatalf("standby controls = %#v", feeds[0])
	}
	if feeds[0]["sync_hard_reset_ms"] != "500" ||
		feeds[0]["sync_max_audio_frames_per_video"] != "10" ||
		feeds[0]["sync_source_buffer_ms"] != "400" ||
		feeds[0]["sync_reconnect_initial_ms"] != "250" ||
		feeds[0]["sync_reconnect_max_ms"] != "5000" ||
		feeds[0]["sync_status_interval_ms"] != "750" {
		t.Fatalf("sync controls = %#v", feeds[0])
	}

	acted, err := cgenActionPayload(configPath, map[string]any{
		"feed_id": "CAP-IT-ALL",
		"action":  "insert_text",
		"text":    "Weather alert",
	})
	if err != nil {
		t.Fatalf("action: %v", err)
	}
	feeds = acted["feeds"].([]map[string]any)
	if feeds[0]["text"] != "Weather alert" || feeds[0]["text_enabled"] != true || feeds[0]["mode"] != "overlay" {
		t.Fatalf("action feed = %#v", feeds[0])
	}
}

func TestCgenPayloadIncludesRuntimeCompositorStatus(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: test\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	statusDir := filepath.Join(dir, "runtime", "cgen")
	if err := os.MkdirAll(statusDir, 0o700); err != nil {
		t.Fatalf("mkdir status: %v", err)
	}
	rawStatus, _ := json.Marshal(map[string]any{
		"visual_mode":          "ticker_alert",
		"overlay_text":         "The National Weather Service has issued a test.",
		"audio_video_drift_ms": 12.5,
		"output_active":        true,
	})
	if err := os.WriteFile(filepath.Join(statusDir, "CAP-IT-ALL.status.json"), rawStatus, 0o600); err != nil {
		t.Fatalf("write status: %v", err)
	}

	loaded, err := saveCgenPayload(configPath, map[string]any{
		"enabled": true,
		"feeds": []any{
			map[string]any{
				"id":                 "CAP-IT-ALL",
				"name":               "CAP CGEN",
				"enabled":            true,
				"program_input_url":  "udp://239.0.0.1:9000",
				"priority_feed_id":   "*",
				"program_output_url": "udp://239.0.0.2:9001?pkt_size=1316",
			},
		},
	})
	if err != nil {
		t.Fatalf("save cgen: %v", err)
	}
	feeds := loaded["feeds"].([]map[string]any)
	runtime := feeds[0]["runtime"].(map[string]any)
	if runtime["visual_mode"] != "ticker_alert" || runtime["output_active"] != true {
		t.Fatalf("runtime status = %#v", runtime)
	}
}

func TestCgenReadRejectsLegacyXML(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: test\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cgenPath := filepath.Join(dir, defaultCgenFile)
	if err := os.MkdirAll(filepath.Dir(cgenPath), 0o700); err != nil {
		t.Fatalf("mkdir cgen: %v", err)
	}
	if err := os.WriteFile(cgenPath, []byte(`<?xml version="1.0" encoding="UTF-8"?>
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <input></input>
    <programInput url="udp://127.0.0.1:5000" format="mpegts"></programInput>
    <priorityInput feed_id="*" url="udp://127.0.0.1:5002"></priorityInput>
    <programOutput url="udp://127.0.0.1:5001" format="mpegts"></programOutput>
    <video width="1280" height="720"></video>
  </feed>
</cgen>`), 0o600); err != nil {
		t.Fatalf("write cgen: %v", err)
	}

	_, err := loadCgenPayload(configPath)
	if err == nil || !strings.Contains(err.Error(), "legacy cgen <input> element") {
		t.Fatalf("expected legacy XML error, got %v", err)
	}
}
