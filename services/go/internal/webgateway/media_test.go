package webgateway

import (
	"encoding/base64"
	"strings"
	"testing"
	"time"

	"github.com/pion/webrtc/v4"
)

func TestDecodePCMBridgeEvent(t *testing.T) {
	rawPCM := []byte{0, 0, 1, 0, 2, 0, 3, 0}
	event := `{"type":"playout.pcm","feed_id":"sk-0001","data":{"feed_id":"sk-0001","sample_rate":48000,"channels":1,"duration_ms":20,"pcm":"` +
		base64.StdEncoding.EncodeToString(rawPCM) + `"}}`
	chunk, ok := decodePCMBridgeEvent([]byte(event))
	if !ok {
		t.Fatal("event was not decoded")
	}
	if chunk.FeedID != "sk-0001" || chunk.SampleRate != 48000 || chunk.Channels != 1 || chunk.Duration != 20*time.Millisecond {
		t.Fatalf("chunk metadata = %#v", chunk)
	}
	if string(chunk.Data) != string(rawPCM) {
		t.Fatalf("pcm = %#v", chunk.Data)
	}
}

func TestPCM16ToG722SilenceFrame(t *testing.T) {
	pcm := make([]byte, 960)
	frame := pcm16ToG722(PCMChunk{FeedID: "sk-0001", SampleRate: 48000, Channels: 1, Data: pcm})
	if len(frame) == 0 {
		t.Fatalf("frame length = %d", len(frame))
	}
	if len(frame) > g722FrameSamples {
		t.Fatalf("frame length = %d, want <= %d", len(frame), g722FrameSamples)
	}
}

func TestPreferredWebRTCAudioCodecFallsBackForReceiverOffers(t *testing.T) {
	if got, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 0\r\na=rtpmap:0 PCMU/8000\r\n", WebRTCAnswerOptions{}); err != nil || got != webRTCAudioPCMU {
		t.Fatal("PCMU-only offers should use PCMU")
	}
	if got, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 9\r\na=rtpmap:9 G722/8000\r\n", WebRTCAnswerOptions{}); err != nil || got != webRTCAudioG722 {
		t.Fatal("G.722-capable offers should use G.722")
	}
	got, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 111 9\r\na=rtpmap:111 opus/48000/2\r\na=rtpmap:9 G722/8000\r\n", WebRTCAnswerOptions{})
	if !opusBackendAvailable() && (err != nil || got != webRTCAudioG722) {
		t.Fatal("default builds should not advertise Opus without the native encoder")
	}
	if got, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 9 0\r\na=rtpmap:9 G722/8000\r\na=rtpmap:0 PCMU/8000\r\n", WebRTCAnswerOptions{DisableG722: true}); err != nil || got != webRTCAudioPCMU {
		t.Fatal("G.722 can still be disabled for emergency compatibility fallback")
	}
}

func TestPreferredWebRTCAudioCodecRequiresOpus(t *testing.T) {
	_, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 9\r\na=rtpmap:9 G722/8000\r\n", WebRTCAnswerOptions{RequireOpus: true})
	if err == nil {
		t.Fatal("Opus-required offers without Opus should fail")
	}
	if !opusBackendAvailable() {
		_, err = preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n", WebRTCAnswerOptions{RequireOpus: true})
		if err == nil {
			t.Fatal("Opus-required offers should fail without an encoder backend")
		}
	}
}

func TestOfferedAudioPayloadTypeUsesDynamicOpusPayload(t *testing.T) {
	offer := "m=audio 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 opus/48000/2\r\n"
	if got := offeredAudioPayloadType(offer, webRTCAudioOpus); got != 96 {
		t.Fatalf("Opus payload type = %d, want 96", got)
	}
}

func TestMediaSafeID(t *testing.T) {
	if got := mediaSafeID("sk-0001 / XLF322"); got != "sk-0001XLF322" || strings.ContainsAny(got, " /") {
		t.Fatalf("mediaSafeID = %q", got)
	}
}

func TestMediaHubAnswersAudioOffer(t *testing.T) {
	hub := newMemoryMediaHub()
	offerPeer, err := newWebRTCPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer offerPeer.Close()
	if _, err := offerPeer.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionRecvonly,
	}); err != nil {
		t.Fatal(err)
	}
	offer, err := offerPeer.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gatheringComplete := webrtc.GatheringCompletePromise(offerPeer)
	if err := offerPeer.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gatheringComplete

	localOffer := offerPeer.LocalDescription()
	if localOffer == nil || strings.TrimSpace(localOffer.SDP) == "" {
		t.Fatalf("empty local offer: created=%#v local=%#v", offer, localOffer)
	}
	answer, err := hub.Answer(t.Context(), "sk-0001", localOffer.SDP)
	if err != nil {
		t.Fatal(err)
	}
	wantCodec := "G722"
	if opusBackendAvailable() {
		wantCodec = "opus"
	}
	if !strings.Contains(answer, wantCodec) {
		t.Fatalf("answer did not include %s: %s", wantCodec, answer)
	}
	if err := offerPeer.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}); err != nil {
		t.Fatal(err)
	}
}

func TestMediaHubReceiverAnswerUsesG722WhenAvailable(t *testing.T) {
	hub := newMemoryMediaHub()
	offerPeer, err := newWebRTCPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer offerPeer.Close()
	if _, err := offerPeer.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionRecvonly,
	}); err != nil {
		t.Fatal(err)
	}
	offer, err := offerPeer.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gatheringComplete := webrtc.GatheringCompletePromise(offerPeer)
	if err := offerPeer.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gatheringComplete

	answer, err := hub.Answer(t.Context(), "sk-0001", offerPeer.LocalDescription().SDP)
	if err != nil {
		t.Fatal(err)
	}
	wantCodec := "G722"
	if opusBackendAvailable() {
		wantCodec = "opus"
	}
	if !strings.Contains(answer, wantCodec) {
		t.Fatalf("receiver answer should include %s: %s", wantCodec, answer)
	}
}

func TestMediaHubStreamsRTPToPeer(t *testing.T) {
	hub := newMemoryMediaHub()
	offerPeer, err := newWebRTCPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer offerPeer.Close()
	tracks := make(chan *webrtc.TrackRemote, 1)
	offerPeer.OnTrack(func(track *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		tracks <- track
	})
	if _, err := offerPeer.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionRecvonly,
	}); err != nil {
		t.Fatal(err)
	}
	offer, err := offerPeer.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gatheringComplete := webrtc.GatheringCompletePromise(offerPeer)
	if err := offerPeer.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gatheringComplete
	answer, err := hub.Answer(t.Context(), "sk-0001", offerPeer.LocalDescription().SDP)
	if err != nil {
		t.Fatal(err)
	}
	if err := offerPeer.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}); err != nil {
		t.Fatal(err)
	}

	done := make(chan error, 1)
	go func() {
		track := <-tracks
		packet, _, err := track.ReadRTP()
		if err != nil {
			done <- err
			return
		}
		if len(packet.Payload) == 0 {
			done <- errEmptyRTPPayload
			return
		}
		done <- nil
	}()
	pcm := make([]byte, 960)
	for index := 0; index+1 < len(pcm); index += 2 {
		pcm[index] = byte(index)
	}
	for i := 0; i < 20; i++ {
		hub.publish(PCMChunk{FeedID: "sk-0001", SampleRate: 48000, Channels: 1, Duration: 20 * time.Millisecond, Data: pcm})
		time.Sleep(20 * time.Millisecond)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for WebRTC RTP")
	}
}

var errEmptyRTPPayload = &testError{"empty RTP payload"}

type testError struct {
	message string
}

func (e *testError) Error() string {
	return e.message
}

func newMemoryMediaHub() *MediaHub {
	return &MediaHub{
		addr:        "memory",
		subscribers: map[string]map[chan PCMChunk]struct{}{},
		peers:       map[string]*webrtc.PeerConnection{},
		last:        map[string]PCMChunk{},
		lastAt:      map[string]time.Time{},
		seenLogged:  map[string]bool{},
	}
}
