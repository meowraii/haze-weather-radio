//go:build windows

package tts

import (
	"context"
	"errors"
	"testing"
)

func TestSAPI5SynthesizeHonorsCanceledContextBeforeCOM(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := NewSAPI5Provider().Synthesize(ctx, Request{Text: "test"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
}

func TestPrepareSAPI5TextFixesWeatherPronunciations(t *testing.T) {
	got := prepareSAPI5Text("Winds north at 40 kilometers per hour and 10 kilometres near Windsor.")
	want := "windz north at 40 killometers per hour and 10 killometers near Windsor."
	if got != want {
		t.Fatalf("SAPI5 text = %q, want %q", got, want)
	}
}

func TestSAPIRateAndVolumeClampToCOMRange(t *testing.T) {
	if got := sapiRate(-99); got != -10 {
		t.Fatalf("low rate = %d", got)
	}
	if got := sapiRate(99); got != 10 {
		t.Fatalf("high rate = %d", got)
	}
	if got := sapiVolume(-1); got != 0 {
		t.Fatalf("low volume = %d", got)
	}
	if got := sapiVolume(150); got != 100 {
		t.Fatalf("high volume = %d", got)
	}
}

func TestSAPILanguagesNormalizesWindowsLCIDs(t *testing.T) {
	got := sapiLanguages("0409;0x1009;c0c")
	want := []string{"en-US", "en-CA", "fr-CA"}
	if len(got) != len(want) {
		t.Fatalf("languages = %#v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("languages = %#v, want %#v", got, want)
		}
	}
}

func TestMergeSAPIVoicesDeduplicatesNativeAndShim(t *testing.T) {
	native := []Voice{
		{ID: "native-id", Name: "Microsoft Linda", Provider: "sapi5"},
	}
	shim := []Voice{
		{ID: "shim-duplicate", Name: "Microsoft Linda", Provider: "sapi5"},
		{ID: "shim-only", Name: "Old 32 Bit Voice", Provider: "sapi5"},
	}

	got := mergeSAPIVoices(native, shim)

	if len(got) != 2 {
		t.Fatalf("voices = %#v", got)
	}
	if got[1].ID != "shim-only" {
		t.Fatalf("shim voice was not preserved: %#v", got)
	}
}

func TestSAPIVoiceNotFoundMatcher(t *testing.T) {
	if !sapiVoiceNotFound(errors.New(`SAPI5 voice "Bob" not found`)) {
		t.Fatal("expected SAPI voice not found error to match")
	}
	if sapiVoiceNotFound(errors.New("COM exploded")) {
		t.Fatal("generic COM error should not match voice fallback")
	}
}
