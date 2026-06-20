package webhook

import "testing"

func TestIntegratedListenURLUsesPublicListenPage(t *testing.T) {
	got := integratedListenURL("https://radio.example.test/base", "sk-0001")
	if got != "https://radio.example.test/base/listen?codec=pcm16&feed=sk-0001" {
		t.Fatalf("listen URL = %q", got)
	}
}

func TestBuildPayloadAddsListenAndBannerColor(t *testing.T) {
	payload := buildPayload(WebhookConfig{Username: "Haze"}, "sk-0001", "SVR", map[string]any{
		"title":    "Severe Thunderstorm Warning",
		"severity": "Severe",
	}, "https://radio.example.test")
	if len(payload.Embeds) != 1 {
		t.Fatalf("embeds = %d", len(payload.Embeds))
	}
	embed := payload.Embeds[0]
	if embed.Color != 0x931102 {
		t.Fatalf("embed color = %#x", embed.Color)
	}
	for _, field := range embed.Fields {
		if field.Name == "Listen" && field.Value == "https://radio.example.test/listen?codec=pcm16&feed=sk-0001" {
			return
		}
	}
	t.Fatalf("missing integrated Listen field: %#v", embed.Fields)
}
