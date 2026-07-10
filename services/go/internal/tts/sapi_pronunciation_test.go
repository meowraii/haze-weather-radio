package tts

import "testing"

func TestPrepareSAPITextRewritesOnlyStandaloneWinds(t *testing.T) {
	got := prepareSAPIText("Winds increase near windows and windspeed sensors.")
	want := "windz increase near windows and windspeed sensors."
	if got != want {
		t.Fatalf("prepared text = %q, want %q", got, want)
	}
}
