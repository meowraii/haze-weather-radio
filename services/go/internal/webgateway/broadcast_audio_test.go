package webgateway

import "testing"

func TestBroadcastAlertMediaExtensionAllowsUnknownMediaTypes(t *testing.T) {
	for _, tc := range []struct {
		name string
		want string
	}{
		{name: "alert.mp4", want: ".mp4"},
		{name: "dash.mpd", want: ".mpd"},
		{name: "unknown.custommedia", want: ".custommedia"},
		{name: "no-extension", want: ".bin"},
		{name: "unsafe.bad-name", want: ".bin"},
		{name: "too.longextensionname", want: ".bin"},
	} {
		if got := broadcastAlertMediaExtension(tc.name); got != tc.want {
			t.Fatalf("broadcastAlertMediaExtension(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestBroadcastAlertMediaUploadLimitUsesConfig(t *testing.T) {
	var config Config
	if got := broadcastAlertMediaUploadBytes(config); got != broadcastAlertMediaMaxBytes {
		t.Fatalf("default upload limit = %d, want %d", got, broadcastAlertMediaMaxBytes)
	}

	config.Webpanel.Authentication.MaxAudioUploadBytes = 50 << 20
	if got := broadcastAlertMediaUploadBytes(config); got != 50<<20 {
		t.Fatalf("configured upload limit = %d, want %d", got, 50<<20)
	}

	config.Webpanel.Authentication.MaxAudioUploadBytes = 32
	if got := broadcastAlertMediaUploadBytes(config); got != 1<<20 {
		t.Fatalf("small upload limit = %d, want %d", got, 1<<20)
	}
}
