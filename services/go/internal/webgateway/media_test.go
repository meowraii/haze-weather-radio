package webgateway

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"math"
	"net/http/httptest"
	"strings"
	"sync/atomic"
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

func TestValidatePCMChunkRejectsImpossibleShape(t *testing.T) {
	_, ok, _ := validatePCMChunk(PCMChunk{
		FeedID:     "sk-0001",
		SampleRate: 384000,
		Channels:   1,
		Duration:   20 * time.Millisecond,
		Data:       make([]byte, 960),
	})
	if ok {
		t.Fatal("invalid sample rate should be rejected")
	}
	chunk, ok, _ := validatePCMChunk(PCMChunk{
		FeedID:     "sk-0001",
		SampleRate: 48000,
		Channels:   1,
		Duration:   20 * time.Millisecond,
		Data:       []byte{0, 0, 1},
	})
	if !ok {
		t.Fatal("chunk with a trailing partial sample should be repaired")
	}
	if len(chunk.Data) != 2 {
		t.Fatalf("repaired PCM length = %d, want 2", len(chunk.Data))
	}
}

func TestMediaHubPreservesPublishedPCM(t *testing.T) {
	hub := newMemoryMediaHub()
	pcm := alternatingFullScalePCM(960)
	hub.publish(PCMChunk{
		FeedID:     "sk-0001",
		SampleRate: 48000,
		Channels:   1,
		Duration:   20 * time.Millisecond,
		Data:       pcm,
	})
	updates, unsubscribe := hub.Subscribe("sk-0001")
	defer unsubscribe()
	select {
	case chunk := <-updates:
		if string(chunk.Data) != string(pcm) {
			t.Fatal("media hub must not replace published PCM with silence")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for PCM chunk")
	}
}

func TestFrameConcealerBridgesShortUnderruns(t *testing.T) {
	queue := [][]byte{{1, 2}, {3, 4}}
	head := 0
	concealer := frameConcealer{}
	silence := []byte{0}

	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{1, 2}) {
		t.Fatalf("first frame = %v", got)
	}
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{3, 4}) {
		t.Fatalf("second frame = %v", got)
	}
	for i := 0; i < webrtcConcealmentFrames; i++ {
		if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{3, 4}) {
			t.Fatalf("concealed frame %d = %v", i, got)
		}
	}
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string(silence) {
		t.Fatalf("long underrun frame = %v, want silence", got)
	}

	queue = append(queue, []byte{5, 6})
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string(silence) {
		t.Fatalf("single recovery frame should be held for priming, got %v", got)
	}
	queue = append(queue, []byte{7, 8})
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{5, 6}) {
		t.Fatalf("new frame after underrun = %v", got)
	}
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{7, 8}) {
		t.Fatalf("second recovery frame = %v", got)
	}
}

func TestFrameConcealerPrimesAfterUnderrun(t *testing.T) {
	queue := [][]byte{}
	head := 0
	concealer := frameConcealer{}
	silence := []byte{0}

	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string(silence) {
		t.Fatalf("empty startup frame = %v, want silence", got)
	}
	queue = append(queue, []byte{1})
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string(silence) {
		t.Fatalf("single recovery frame should be held for priming, got %v", got)
	}
	if queuedFrameCount(queue, head) != 1 {
		t.Fatalf("single recovery frame was consumed before priming")
	}
	queue = append(queue, []byte{2})
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{1}) {
		t.Fatalf("primed recovery frame = %v, want first queued frame", got)
	}
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{2}) {
		t.Fatalf("second primed frame = %v, want second queued frame", got)
	}
}

func TestFrameConcealerReportsFrameKind(t *testing.T) {
	queue := [][]byte{{1}}
	head := 0
	concealer := frameConcealer{}
	silence := []byte{0}

	if _, kind := concealer.nextWithKind(&queue, &head, func() []byte { return silence }); kind != webRTCFrameReal {
		t.Fatalf("first frame kind = %v, want real", kind)
	}
	if _, kind := concealer.nextWithKind(&queue, &head, func() []byte { return silence }); kind != webRTCFrameConcealed {
		t.Fatalf("short underrun frame kind = %v, want concealed", kind)
	}
	for i := 1; i < webrtcConcealmentFrames; i++ {
		concealer.next(&queue, &head, func() []byte { return silence })
	}
	if _, kind := concealer.nextWithKind(&queue, &head, func() []byte { return silence }); kind != webRTCFrameIdle {
		t.Fatalf("long underrun frame kind = %v, want idle", kind)
	}
}

func TestWebRTCFrameSourceBroadcastCountsSlowSubscribers(t *testing.T) {
	source := &webRTCFrameSource{subs: map[chan []byte]struct{}{make(chan []byte): {}}}
	dropped, subscribers := source.broadcast([]byte{1})
	if subscribers != 1 {
		t.Fatalf("subscribers = %d, want 1", subscribers)
	}
	if dropped != 1 {
		t.Fatalf("dropped = %d, want 1", dropped)
	}
}

func TestLatestWebRTCFrameSkipsStaleFrames(t *testing.T) {
	frames := make(chan []byte, 4)
	frames <- []byte{2}
	frames <- nil
	frames <- []byte{3}
	frames <- []byte{4}

	latest, skipped := latestWebRTCFrame([]byte{1}, frames)
	if string(latest) != string([]byte{4}) {
		t.Fatalf("latest frame = %v, want [4]", latest)
	}
	if skipped != 3 {
		t.Fatalf("skipped = %d, want 3", skipped)
	}
}

func TestWatchWebRTCSampleWritesClosesStalledPeer(t *testing.T) {
	var inFlight atomic.Bool
	var startedAt atomic.Int64
	inFlight.Store(true)
	startedAt.Store(time.Now().Add(-time.Second).UnixNano())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go watchWebRTCSampleWrites(ctx, "sk-0001", webRTCAudioPCMU, &inFlight, &startedAt, 10*time.Millisecond, func() {
		close(done)
	})

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for stalled WebRTC write watchdog")
	}
}

func TestWatchWebRTCSampleWritesIgnoresIdleWriter(t *testing.T) {
	var inFlight atomic.Bool
	var startedAt atomic.Int64
	startedAt.Store(time.Now().Add(-time.Second).UnixNano())

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan struct{})
	go watchWebRTCSampleWrites(ctx, "sk-0001", webRTCAudioPCMU, &inFlight, &startedAt, 10*time.Millisecond, func() {
		close(done)
	})

	select {
	case <-done:
		t.Fatal("idle WebRTC write watchdog should not fire")
	case <-time.After(40 * time.Millisecond):
	}
}

func TestMediaHubUsesIndependentFeedIngressQueues(t *testing.T) {
	hub := newMemoryMediaHub()
	left := hub.feedIngress("sk-0001")
	right := hub.feedIngress("CAP-IT-ALL")
	if left == right {
		t.Fatal("feeds should not share media ingress queues")
	}
	for i := 0; i < feedIngressCapacity*4; i++ {
		hub.publish(PCMChunk{
			FeedID:     "CAP-IT-ALL",
			SampleRate: 48000,
			Channels:   1,
			Duration:   20 * time.Millisecond,
			Data:       make([]byte, 960),
		})
	}
	hub.publish(PCMChunk{
		FeedID:     "sk-0001",
		SampleRate: 48000,
		Channels:   1,
		Duration:   20 * time.Millisecond,
		Data:       sinePCM(480, 1000, 48000, 8000),
	})
	if !waitForRecentPCM(hub, "sk-0001", time.Second) {
		t.Fatal("sk-0001 should still receive PCM while another feed is busy")
	}
}

func TestMediaHubSharesWebRTCFrameSourcePerFeedCodec(t *testing.T) {
	hub := newMemoryMediaHub()
	left, unsubscribeLeft, err := hub.SubscribeWebRTCFrames("sk-0001", webRTCAudioG722)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribeLeft()
	right, unsubscribeRight, err := hub.SubscribeWebRTCFrames("sk-0001", webRTCAudioG722)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribeRight()

	hub.mu.Lock()
	sourceCount := len(hub.frameSources)
	hub.mu.Unlock()
	if sourceCount != 1 {
		t.Fatalf("frame source count = %d, want 1", sourceCount)
	}

	hub.publish(PCMChunk{
		FeedID:     "sk-0001",
		SampleRate: 48000,
		Channels:   1,
		Duration:   20 * time.Millisecond,
		Data:       sinePCM(960, 1000, 48000, 8000),
	})
	if frame := waitForWebRTCFrame(t, left); len(frame) == 0 {
		t.Fatal("left subscriber received an empty frame")
	}
	if frame := waitForWebRTCFrame(t, right); len(frame) == 0 {
		t.Fatal("right subscriber received an empty frame")
	}

	unsubscribeLeft()
	hub.mu.Lock()
	sourceCount = len(hub.frameSources)
	hub.mu.Unlock()
	if sourceCount != 1 {
		t.Fatalf("frame source should stay alive for the remaining subscriber, got %d", sourceCount)
	}
	unsubscribeRight()
	if !waitForWebRTCFrameSources(hub, 0, time.Second) {
		t.Fatal("frame source was not removed after the final subscriber left")
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
	if err != nil || got != webRTCAudioG722 {
		t.Fatal("auto codec should prefer G.722 for low-latency radio streams")
	}
	got, err = preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n", WebRTCAnswerOptions{})
	if opusBackendAvailable() && (err != nil || got != webRTCAudioOpus) {
		t.Fatal("Opus-only offers should use Opus when the native encoder is available")
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

func TestPreferredWebRTCAudioCodecHonorsExplicitSelection(t *testing.T) {
	offer := "m=audio 9 UDP/TLS/RTP/SAVPF 111 9 0\r\na=rtpmap:111 opus/48000/2\r\na=rtpmap:9 G722/8000\r\na=rtpmap:0 PCMU/8000\r\n"
	if got, err := preferredWebRTCAudioCodec(offer, WebRTCAnswerOptions{PreferredCodec: "g722"}); err != nil || got != webRTCAudioG722 {
		t.Fatalf("explicit G.722 codec = %v, %v", got, err)
	}
	if got, err := preferredWebRTCAudioCodec(offer, WebRTCAnswerOptions{PreferredCodec: "pcmu"}); err != nil || got != webRTCAudioPCMU {
		t.Fatalf("explicit PCMU codec = %v, %v", got, err)
	}
	if _, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 9\r\na=rtpmap:9 G722/8000\r\n", WebRTCAnswerOptions{PreferredCodec: "pcmu"}); err == nil {
		t.Fatal("explicit PCMU should fail when the receiver offer does not include PCMU")
	}
}

func TestOfferedAudioPayloadTypeUsesDynamicOpusPayload(t *testing.T) {
	offer := "m=audio 9 UDP/TLS/RTP/SAVPF 96\r\na=rtpmap:96 opus/48000/2\r\n"
	if got := offeredAudioPayloadType(offer, webRTCAudioOpus); got != 96 {
		t.Fatalf("Opus payload type = %d, want 96", got)
	}
}

func TestWebRTCPeerStateCleanupPolicyKeepsTransientDisconnects(t *testing.T) {
	if shouldCleanupWebRTCPeer(webrtc.PeerConnectionStateDisconnected) {
		t.Fatal("transient WebRTC disconnects should be given a recovery window")
	}
	if !shouldCleanupWebRTCPeer(webrtc.PeerConnectionStateFailed) {
		t.Fatal("failed WebRTC peers should be cleaned up")
	}
	if !shouldCleanupWebRTCPeer(webrtc.PeerConnectionStateClosed) {
		t.Fatal("closed WebRTC peers should be cleaned up")
	}
}

func TestWriteWAVStreamHeader(t *testing.T) {
	recorder := httptest.NewRecorder()
	if err := writeWAVStreamHeader(recorder, 48000, 1, 16); err != nil {
		t.Fatal(err)
	}
	header := recorder.Body.Bytes()
	if len(header) != 44 {
		t.Fatalf("header length = %d, want 44", len(header))
	}
	if string(header[:4]) != "RIFF" || string(header[8:12]) != "WAVE" || string(header[36:40]) != "data" {
		t.Fatalf("invalid WAV header markers: %q", header)
	}
	if got := binary.LittleEndian.Uint32(header[24:28]); got != 48000 {
		t.Fatalf("sample rate = %d", got)
	}
	if got := binary.LittleEndian.Uint32(header[40:44]); got != 0xffffffff {
		t.Fatalf("stream data size = %#x", got)
	}
}

func TestHTTPAudioFormatByID(t *testing.T) {
	cases := []struct {
		raw         string
		id          string
		contentType string
		usesFFmpeg  bool
	}{
		{"", "pcm16", "audio/wav", false},
		{"wav", "pcm16", "audio/wav", false},
		{"opus", "opus", "audio/ogg; codecs=opus", true},
		{"webm-opus", "webm_opus", "audio/webm; codecs=opus", true},
		{"aac", "aac", "audio/aac", true},
		{"m4a", "m4a", "audio/mp4", true},
		{"mp3", "mp3", "audio/mpeg", true},
		{"vorbis", "vorbis", "audio/ogg; codecs=vorbis", true},
		{"flac", "flac", "audio/flac", true},
		{"ulaw", "ulaw", "audio/basic", true},
	}
	for _, tc := range cases {
		got, ok := httpAudioFormatByID(tc.raw)
		if !ok {
			t.Fatalf("%q was not accepted", tc.raw)
		}
		if got.ID != tc.id || got.ContentType != tc.contentType || (got.FFmpegFormat != "") != tc.usesFFmpeg {
			t.Fatalf("%q => %#v", tc.raw, got)
		}
	}
	if _, ok := httpAudioFormatByID("definitely-not-real"); ok {
		t.Fatal("unknown HTTP audio format should be rejected")
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

func TestMediaHubPeerOutlivesOfferContext(t *testing.T) {
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

	offerCtx, cancelOffer := context.WithCancel(context.Background())
	answer, err := hub.Answer(offerCtx, "sk-0001", offerPeer.LocalDescription().SDP)
	cancelOffer()
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
		t.Fatal("timed out waiting for WebRTC RTP after offer context cancellation")
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
		addr:         "memory",
		subscribers:  map[string]map[chan PCMChunk]struct{}{},
		peers:        map[string]*webrtc.PeerConnection{},
		ingress:      map[string]chan PCMChunk{},
		frameSources: map[webRTCFrameSourceKey]*webRTCFrameSource{},
		last:         map[string]PCMChunk{},
		lastAt:       map[string]time.Time{},
		seenLogged:   map[string]bool{},
	}
}

func waitForWebRTCFrame(t *testing.T, frames <-chan []byte) []byte {
	t.Helper()
	select {
	case frame := <-frames:
		return frame
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for WebRTC frame")
		return nil
	}
}

func waitForWebRTCFrameSources(hub *MediaHub, want int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		hub.mu.Lock()
		got := len(hub.frameSources)
		hub.mu.Unlock()
		if got == want {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func waitForRecentPCM(hub *MediaHub, feedID string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if hub.HasRecentPCM(feedID, timeout) {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func alternatingFullScalePCM(samples int) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		sample := int16(32767)
		if i%2 == 1 {
			sample = -32768
		}
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], uint16(sample))
	}
	return out
}

func sinePCM(samples int, frequency float64, sampleRate float64, amplitude float64) []byte {
	out := make([]byte, samples*2)
	for i := 0; i < samples; i++ {
		value := math.Sin(2*math.Pi*frequency*float64(i)/sampleRate) * amplitude
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], uint16(int16(value)))
	}
	return out
}
