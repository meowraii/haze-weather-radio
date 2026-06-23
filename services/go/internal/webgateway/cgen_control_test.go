package webgateway

import (
	"os"
	"path/filepath"
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
				"id":                               "CAP-IT-ALL",
				"name":                             "CAP CGEN",
				"enabled":                          true,
				"mode":                             "release",
				"program_input_url":                "udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1",
				"program_input_format":             "mpegts",
				"priority_feed_id":                 "CAP-IT-ALL",
				"program_output_url":               "udp://239.0.0.2:9001?pkt_size=1316",
				"program_output_format":            "mpegts",
				"alert_output_url":                 "udp://239.0.0.2:9001?pkt_size=1316",
				"alert_output_format":              "mpegts",
				"vcodec":                           "libx264",
				"acodec":                           "aac",
				"width":                            "1280",
				"height":                           "720",
				"fps":                              "source",
				"interlaced":                       true,
				"field_order":                      "bff",
				"standard":                         "atsc",
				"banner_background_enabled":        true,
				"banner_background_color":          "#b45309",
				"banner_background_gradient_color": "#7f1d1d",
				"scroll_speed":                     "6",
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
	if feeds[0]["interlaced"] != true || feeds[0]["field_order"] != "bff" || feeds[0]["standard"] != "atsc" {
		t.Fatalf("video flags = %#v", feeds[0])
	}
	if feeds[0]["scroll_speed"] != "6" || feeds[0]["banner_background_gradient_color"] != "#7f1d1d" {
		t.Fatalf("banner controls = %#v", feeds[0])
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
