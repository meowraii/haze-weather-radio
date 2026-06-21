package webgateway

import (
	"bytes"
	"testing"
	"time"
)

func TestAssembleAlertPreviewPCMIncludesLeadGapVoiceAndTail(t *testing.T) {
	lead := []byte{1, 2}
	voice := []byte{3, 4}
	tail := []byte{5, 6}

	raw := assembleAlertPreviewPCM(lead, voice, tail, 48000, 1)

	expectedLen := len(lead) + len(silencePCMBytes(48000, 1, time.Second)) + len(voice) + len(tail)
	if len(raw) != expectedLen {
		t.Fatalf("len = %d, want %d", len(raw), expectedLen)
	}
	if !bytes.Equal(raw[:len(lead)], lead) {
		t.Fatalf("lead not preserved: %#v", raw[:len(lead)])
	}
	if !bytes.Equal(raw[len(raw)-len(tail):], tail) {
		t.Fatalf("tail not preserved: %#v", raw[len(raw)-len(tail):])
	}
	voiceStart := len(lead) + len(silencePCMBytes(48000, 1, time.Second))
	if !bytes.Equal(raw[voiceStart:voiceStart+len(voice)], voice) {
		t.Fatalf("voice not preserved")
	}
}

func TestAlertPreviewAttentionToneEnabled(t *testing.T) {
	if !alertPreviewAttentionToneEnabled(map[string]any{"tone_type": "NPAS"}) {
		t.Fatal("selected tone should be enabled")
	}
	if alertPreviewAttentionToneEnabled(map[string]any{"tone_type": "NONE"}) {
		t.Fatal("NONE should disable tone")
	}
}
