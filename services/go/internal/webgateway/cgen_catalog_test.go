package webgateway

import (
	"path/filepath"
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

func TestCgenCatalogGstInspectClassification(t *testing.T) {
	builder := cgenCatalogBuilder{
		formats:       map[string]cgenCatalogEntry{},
		video:         map[string]cgenCatalogEntry{},
		audio:         map[string]cgenCatalogEntry{},
		videoDecoders: map[string]cgenCatalogEntry{},
		browser:       map[string]cgenCatalogEntry{},
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
	values, ok := payload[key].([]map[string]any)
	if !ok {
		return false
	}
	for _, value := range values {
		if stringValue(value, "id") == id {
			return true
		}
	}
	return false
}
