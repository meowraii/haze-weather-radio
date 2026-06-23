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
