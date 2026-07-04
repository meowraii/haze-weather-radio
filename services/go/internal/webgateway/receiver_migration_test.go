package webgateway

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func TestMediaServiceWebRTCAnswerFromBaseDoesNotBlockOnHungService(t *testing.T) {
	release := make(chan struct{})
	mediaService := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/api/v1/webrtc/offer" {
			http.NotFound(writer, request)
			return
		}
		<-release
		writeJSON(writer, map[string]any{
			"sdp":      "v=0\r\n",
			"sdp_type": "answer",
		})
	}))
	defer mediaService.Close()

	done := make(chan bool, 1)
	go func() {
		_, ok := mediaServiceWebRTCAnswerFromBase(context.Background(), mediaService.URL, map[string]any{
			"feed_id": "sk-0001",
			"sdp":     "v=0\r\n",
		})
		done <- ok
	}()

	select {
	case ok := <-done:
		if ok {
			t.Fatal("hung media service returned a usable WebRTC answer")
		}
	case <-time.After(3 * time.Second):
		close(release)
		t.Fatal("media service WebRTC offer proxy did not enforce an internal timeout")
	}
	close(release)
}

func TestReceiverSessionRequiresCredentialWhenPairingConfigured(t *testing.T) {
	dir := t.TempDir()
	writeReceiverFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	addReceiverPairingToken(t, configPath)
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	status, payload := postReceiverJSON(t, httpServer.URL+"/api/receiver/v1/session", map[string]any{
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
	})
	if status != http.StatusUnauthorized && status != http.StatusForbidden {
		t.Fatalf("receiver session status = %d payload=%#v, want 401 or 403 without credential proof", status, payload)
	}
}

func TestReceiverCredentialSessionSurvivesDaemonRestart(t *testing.T) {
	dir := t.TempDir()
	writeReceiverFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	addReceiverPairingToken(t, configPath)
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())

	receiverNonce := "receiver-nonce"
	challenge := receiverChallengeForTest(t, httpServer.URL, "sk-0001", "rx-1", "pi-one", receiverNonce)
	challengeID := stringMapValue(challenge, "challenge_id")
	serverNonce := stringMapValue(challenge, "server_nonce")
	proof := receiverHMACHex("test-pairing-token", receiverProofMessage("pair-v1", map[string]string{
		"challenge_id":      challengeID,
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
		"receiver_nonce":    receiverNonce,
		"server_nonce":      serverNonce,
	}))
	status, completed := postReceiverJSON(t, httpServer.URL+"/api/receiver/v1/pair/complete", map[string]any{
		"challenge_id":      challengeID,
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
		"nonce":             receiverNonce,
		"proof":             proof,
	})
	if status != http.StatusOK {
		t.Fatalf("pair complete status = %d payload=%#v", status, completed)
	}
	credentialID := stringMapValue(completed, "credential_id")
	credentialSecret := stringMapValue(completed, "credential_secret")
	if credentialID == "" || credentialSecret == "" {
		t.Fatalf("pair complete missing credential fields: %#v", completed)
	}
	httpServer.Close()

	restarted := NewServerWithConfigPath(config, configPath, ".")
	restartedHTTP := httptest.NewServer(restarted.Handler())
	defer restartedHTTP.Close()
	sessionNonce := "session-nonce"
	sessionProof := receiverHMACHex(credentialSecret, receiverProofMessage("session-v1", map[string]string{
		"credential_id":     credentialID,
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
		"nonce":             sessionNonce,
	}))
	sessionStatus, session := postReceiverJSON(t, restartedHTTP.URL+"/api/receiver/v1/session", map[string]any{
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
		"credential_id":     credentialID,
		"nonce":             sessionNonce,
		"proof":             sessionProof,
	})
	if sessionStatus != http.StatusOK {
		t.Fatalf("post-restart session status = %d payload=%#v", sessionStatus, session)
	}
	cookie := stringMapValue(session, "cookie")
	if cookie == "" {
		t.Fatalf("post-restart session missing cookie: %#v", session)
	}

	headers := http.Header{}
	headers.Set("Authorization", "HazeReceiverCookie "+cookie)
	conn, _, err := websocket.Dial(context.Background(), stringMapValue(session, "ws_url"), &websocket.DialOptions{HTTPHeader: headers})
	if err != nil {
		t.Fatalf("post-restart receiver ws dial: %v", err)
	}
	defer conn.CloseNow()
	ready := readType(t, context.Background(), conn, "receiver_ready")
	if ready["feed_id"] != "sk-0001" {
		t.Fatalf("ready = %#v", ready)
	}
}

func TestReceiverRequireTLSRejectsPlainHTTPSession(t *testing.T) {
	dir := t.TempDir()
	writeReceiverFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	raw, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	updated := strings.Replace(string(raw), "    require_tls: false\n", "    require_tls: true\n", 1)
	if updated == string(raw) {
		t.Fatal("receiver fixture was not updated with require_tls")
	}
	mustWrite(t, configPath, updated)
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	status, payload := postReceiverJSON(t, httpServer.URL+"/api/receiver/v1/session", map[string]any{
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
	})
	if status == http.StatusOK {
		t.Fatalf("plain HTTP receiver session was accepted with require_tls enabled: %#v", payload)
	}
}

func TestReceiverWebSocketRejectsInvalidWebRTCOffersWithoutClosing(t *testing.T) {
	dir := t.TempDir()
	writeReceiverFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	sessionStatus, session := postReceiverJSON(t, httpServer.URL+"/api/receiver/v1/session", map[string]any{
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
	})
	if sessionStatus != http.StatusOK {
		t.Fatalf("session status = %d payload=%#v", sessionStatus, session)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, _, err := websocket.Dial(ctx, stringMapValue(session, "ws_url"), nil)
	if err != nil {
		t.Fatalf("receiver ws dial: %v", err)
	}
	defer conn.CloseNow()
	_ = readType(t, ctx, conn, "receiver_ready")

	tests := []struct {
		name    string
		message map[string]any
		want    string
	}{
		{
			name: "oversized sdp",
			message: map[string]any{
				"type": "webrtc_offer",
				"sdp":  strings.Repeat("v", webRTCOfferMaxSDPLength+1),
			},
			want: "sdp is too long",
		},
		{
			name: "oversized codec",
			message: map[string]any{
				"type":            "webrtc_offer",
				"sdp":             "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n",
				"preferred_codec": strings.Repeat("x", webRTCOfferMaxCodecLength+1),
			},
			want: "codec is too long",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			writeWS(t, ctx, conn, tc.message)
			reply := readType(t, ctx, conn, "webrtc_error")
			if detail := fmt.Sprint(reply["detail"]); detail != tc.want {
				t.Fatalf("detail = %q, want %q", detail, tc.want)
			}
		})
	}
}

func addReceiverPairingToken(t *testing.T, configPath string) {
	t.Helper()
	raw, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	old := "    transmitter_defaults:\n      bandwidth_khz: 12.5\n      deviation_hz: 5000\n      preemphasis: none\n"
	newText := old + "    pairing_tokens:\n      - id: rx-one\n        token: test-pairing-token\n        feed_ids: [sk-0001]\n"
	updated := strings.Replace(string(raw), old, newText, 1)
	if updated == string(raw) {
		t.Fatal("receiver fixture was not updated with pairing token config")
	}
	mustWrite(t, configPath, updated)
}
