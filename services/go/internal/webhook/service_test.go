package webhook

import (
	"net/http"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alertmodel"
)

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

func TestBuildPayloadPrefersAlertPacketFields(t *testing.T) {
	payload := buildPayload(WebhookConfig{Username: "Haze"}, "CAP-IT-ALL", "SVR", map[string]any{
		"alert_packet": alertmodel.Packet{
			ID: "urn:test:packet",
			Content: alertmodel.Content{
				Headline:    "Severe Thunderstorm Warning",
				Event:       "Severe Thunderstorm Warning",
				Severity:    "Severe",
				Description: "Packet description.",
				Instruction: "Packet instruction.",
			},
			Timing: alertmodel.Timing{ExpiresAt: "2026-06-22T22:30:00Z"},
			SAME:   &alertmodel.SAME{Event: "SVR"},
			Audio:  &alertmodel.Audio{URL: "https://alerts.example.test/audio.mp3", Authoritative: true},
		},
		"description": "Stale flat description.",
	}, "https://radio.example.test")

	embed := payload.Embeds[0]
	if embed.Title != "Severe Thunderstorm Warning" || embed.Description != "Packet description." {
		t.Fatalf("embed = %#v", embed)
	}
	if embed.Footer == nil || embed.Footer.Text != "urn:test:packet" {
		t.Fatalf("footer = %#v", embed.Footer)
	}
	assertField(t, embed.Fields, "CAP Audio", "https://alerts.example.test/audio.mp3")
	assertField(t, embed.Fields, "Instruction", "Packet instruction.")
	assertField(t, embed.Fields, "Expires", "2026-06-22T22:30:00Z")
}

func TestWebhookHTTPClientUsesReusableTransport(t *testing.T) {
	client := webhookHTTPClient(3 * time.Second)
	if client.Timeout != 3*time.Second {
		t.Fatalf("timeout = %s", client.Timeout)
	}
	transport, ok := client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("transport = %T", client.Transport)
	}
	if transport.MaxIdleConnsPerHost < 8 || !transport.ForceAttemptHTTP2 {
		t.Fatalf("transport not tuned: %#v", transport)
	}
}

func assertField(t *testing.T, fields []discordEmbedField, name string, value string) {
	t.Helper()
	for _, field := range fields {
		if field.Name == name && field.Value == value {
			return
		}
	}
	t.Fatalf("missing field %s=%q in %#v", name, value, fields)
}
