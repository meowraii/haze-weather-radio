package productrender

import (
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
)

func TestRenderableCAPEntryWithoutExpiryAgesOut(t *testing.T) {
	now := time.Date(2026, 7, 4, 18, 0, 0, 0, time.UTC)
	entry := capRegistryEntry{
		ID:        "urn:test",
		UpdatedAt: now.Add(-alertRegistryNoExpiryMaxAge - time.Minute),
		Alert: capmodel.Alert{
			Identifier:  "urn:test",
			MessageType: "Alert",
			Infos: []capmodel.AlertInfo{{
				Event: "Special Weather Statement",
			}},
		},
	}

	if isRenderableCAPEntry(entry, now) {
		t.Fatal("CAP entry without expiry stayed renderable past fallback age")
	}
}

func TestRenderableCAPEntryWithoutExpiryStaysRenderableWithinFallbackAge(t *testing.T) {
	now := time.Date(2026, 7, 4, 18, 0, 0, 0, time.UTC)
	entry := capRegistryEntry{
		ID:        "urn:test",
		UpdatedAt: now.Add(-time.Hour),
		Alert: capmodel.Alert{
			Identifier:  "urn:test",
			MessageType: "Alert",
			Infos: []capmodel.AlertInfo{{
				Event: "Special Weather Statement",
			}},
		},
	}

	if !isRenderableCAPEntry(entry, now) {
		t.Fatal("CAP entry without expiry aged out too early")
	}
}
