package webgateway

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestCgenCatalogScansBundledGStreamerPlugins(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "")
	pluginDir := filepath.Join(dir, "bin", "gstreamer-1.0")
	for _, name := range []string{
		"libgstmpegtsmux.dll",
		"libgstx264.dll",
		"libgstopus.dll",
		"libgstmatroska.dll",
	} {
		mustWrite(t, filepath.Join(pluginDir, name), "")
	}

	payload, err := cgenCatalogPayload(configPath)
	if err != nil {
		t.Fatalf("cgenCatalogPayload failed: %v", err)
	}
	if !catalogPayloadContains(payload, "formats", "mpegts") {
		t.Fatalf("expected mpegts format from bundled plugin scan: %#v", payload["formats"])
	}
	capabilities, ok := payload["capabilities"].(map[string]any)
	if !ok {
		t.Fatalf("expected runtime capability map: %#v", payload["capabilities"])
	}
	audioTopologies, ok := capabilities["audio_topologies"].(map[string]bool)
	if !ok || !audioTopologies["force_layout"] || audioTopologies["preserve_native_tracks"] {
		t.Fatalf("unexpected audio topology capabilities: %#v", capabilities["audio_topologies"])
	}
	if !catalogPayloadContains(payload, "formats", "matroska") {
		t.Fatalf("expected matroska format from bundled plugin scan: %#v", payload["formats"])
	}
	if !catalogPayloadContains(payload, "video_codecs", "x264enc") {
		t.Fatalf("expected x264enc video encoder from bundled plugin scan: %#v", payload["video_codecs"])
	}
	if !catalogPayloadContains(payload, "audio_codecs", "opusenc") {
		t.Fatalf("expected opusenc audio encoder from bundled plugin scan: %#v", payload["audio_codecs"])
	}
}

func TestCgenInputDevicePayloadKeepsSupportedPersistentDevices(t *testing.T) {
	devices := cgenInputDevicePayload([]cgenInputDevice{
		{
			ID:      "/dev/v4l/by-id/camera-b",
			Label:   "Camera B",
			Backend: "Video4Linux2",
			Element: "v4l2src",
			Caps:    "video/x-raw,width=1920,height=1080",
		},
		{
			ID:      "capture-a",
			Label:   "Camera A",
			Backend: "DirectShow",
			Element: "dshowvideosrc",
		},
		{
			ID:      "capture-a",
			Label:   "Duplicate Camera A",
			Backend: "dshow",
		},
		{
			ID:      "pipewire-camera",
			Label:   "Unsupported Camera",
			Backend: "pipewire",
		},
	})

	if len(devices) != 2 {
		t.Fatalf("device count = %d, want 2: %#v", len(devices), devices)
	}
	if got := stringValue(devices[0], "label"); got != "Camera A" {
		t.Fatalf("first device label = %q, want Camera A", got)
	}
	if got := stringValue(devices[0], "backend"); got != "directshow" {
		t.Fatalf("first device backend = %q, want directshow", got)
	}
	if got := stringValue(devices[1], "backend"); got != "v4l2" {
		t.Fatalf("second device backend = %q, want v4l2", got)
	}
}

func TestFontFamilyFromFilename(t *testing.T) {
	tests := map[string]string{
		"Arial-Bold.ttf":                  "Arial",
		"NotoSansDisplay-Regular.otf":     "NotoSansDisplay",
		"Bahnschrift-SemiBoldCond.ttf":    "Bahnschrift",
		"Inter-55.ttf":                    "Inter",
		"Roboto_Condensed-BoldItalic.ttf": "Roboto Condensed",
		"not-a-font.txt":                  "",
	}
	for name, want := range tests {
		t.Run(name, func(t *testing.T) {
			if got := fontFamilyFromFilename(name); got != want {
				t.Fatalf("fontFamilyFromFilename(%q) = %q, want %q", name, got, want)
			}
		})
	}
}

func TestManagedFontFamilyFromFilename(t *testing.T) {
	tests := map[string]string{
		"AlertSans-Bold.woff2":         "AlertSans",
		"Station_Display-Regular.woff": "Station Display",
		"HelveticaNeueLTPro-Md.otf":    "HelveticaNeueLTPro Md",
		"not-a-font.txt":               "",
	}
	for name, want := range tests {
		t.Run(name, func(t *testing.T) {
			if got := fontFamilyFromManagedFilename(name); got != want {
				t.Fatalf("fontFamilyFromManagedFilename(%q) = %q, want %q", name, got, want)
			}
		})
	}
}

func TestCgenCatalogIncludesManagedFonts(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "")
	mustWrite(t, filepath.Join(dir, "managed", "fonts", "AlertSans-Bold.woff2"), "")
	mustWrite(t, filepath.Join(dir, "managed", "fonts", "StationDisplay-Regular.ttf"), "")
	mustWrite(t, filepath.Join(dir, "managed", "fonts", "not-a-font.txt"), "")

	payload, err := cgenCatalogPayload(configPath)
	if err != nil {
		t.Fatalf("cgenCatalogPayload failed: %v", err)
	}
	font := catalogPayloadEntry(payload, "fonts", "AlertSans")
	if font == nil {
		t.Fatalf("expected managed font entry in catalog: %#v", payload["fonts"])
	}
	if got := stringValue(font, "label"); got != "(*) AlertSans (woff2)" {
		t.Fatalf("managed font label = %q", got)
	}
	if got := stringValue(font, "source"); got != "managed" {
		t.Fatalf("managed font source = %q", got)
	}
	if got := stringValue(font, "extension"); got != "woff2" {
		t.Fatalf("managed font extension = %q", got)
	}
	if path := stringValue(font, "path"); !strings.HasPrefix(path, "managed/fonts/") {
		t.Fatalf("managed font path = %q", path)
	}
	if got := stringValue(font, "url"); got != "/api/v1/cgen/fonts/AlertSans-Bold.woff2" {
		t.Fatalf("managed font url = %q", got)
	}
}

func TestCgenCatalogGstInspectClassification(t *testing.T) {
	builder := cgenCatalogBuilder{
		formats:       map[string]cgenCatalogEntry{},
		video:         map[string]cgenCatalogEntry{},
		audio:         map[string]cgenCatalogEntry{},
		videoDecoders: map[string]cgenCatalogEntry{},
	}

	builder.addGstFactory("avdec_h265", "libav HEVC / H.265 decoder")
	builder.addGstFactory("avenc_eac3", "libav E-AC-3 encoder")

	if _, ok := builder.video["avdec_h265"]; ok {
		t.Fatalf("decoder factory should not be cataloged as an HEVC output encoder")
	}
	if _, ok := builder.videoDecoders["avdec_h265"]; !ok {
		t.Fatalf("decoder factory should be cataloged as an HEVC input decoder: %#v", builder.videoDecoders)
	}
	if _, ok := builder.audio["avenc_eac3"]; !ok {
		t.Fatalf("E-AC-3 encoder should be cataloged separately from AC-3: %#v", builder.audio)
	}
	if _, ok := builder.audio["avenc_ac3"]; ok {
		t.Fatalf("E-AC-3 should not be collapsed into AC-3: %#v", builder.audio)
	}
}

func TestFontFamilyFromRegistryLine(t *testing.T) {
	tests := map[string]string{
		"    Arial Bold (TrueType)    REG_SZ    arialbd.ttf":                    "Arial",
		"    Bahnschrift SemiBold Cond (TrueType)    REG_SZ    bahnschrift.ttf": "Bahnschrift",
		"    Segoe UI Variable (TrueType)    REG_SZ    SegUIVar.ttf":            "Segoe UI",
		"    Noto Sans Display (OpenType)    REG_SZ    NotoSans.otf":            "Noto Sans Display",
		"HKEY_LOCAL_MACHINE\\Software\\Fonts":                                   "",
	}
	for line, want := range tests {
		t.Run(line, func(t *testing.T) {
			if got := fontFamilyFromRegistryLine(line); got != want {
				t.Fatalf("fontFamilyFromRegistryLine(%q) = %q, want %q", line, got, want)
			}
		})
	}
}

func catalogPayloadContains(payload map[string]any, key string, id string) bool {
	return catalogPayloadEntry(payload, key, id) != nil
}

func catalogPayloadEntry(payload map[string]any, key string, id string) map[string]any {
	values, ok := payload[key].([]map[string]any)
	if !ok {
		return nil
	}
	for _, value := range values {
		if stringValue(value, "id") == id {
			return value
		}
	}
	return nil
}
