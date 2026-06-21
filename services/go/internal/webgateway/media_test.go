package webgateway

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"math"
	"net/http/httptest"
	"strings"
	"sync"
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

func TestWebRTCIdleFramesCarryDither(t *testing.T) {
	g722Idle := g722IdleFrameSamples()
	if len(g722Idle) != g722FrameSamples {
		t.Fatalf("G.722 idle samples = %d, want %d", len(g722Idle), g722FrameSamples)
	}
	if g722Idle[0] == 0 || g722Idle[1] == 0 || g722Idle[0] == g722Idle[1] {
		t.Fatalf("G.722 idle dither should alternate tiny non-zero samples: %v %v", g722Idle[0], g722Idle[1])
	}
	pcmuIdle := pcmuIdleFrame()
	if len(pcmuIdle) != pcmuFrameSamples {
		t.Fatalf("PCMU idle frame = %d, want %d", len(pcmuIdle), pcmuFrameSamples)
	}
	if pcmuIdle[0] == pcmuIdle[1] {
		t.Fatalf("PCMU idle dither should not collapse to a constant byte: %x %x", pcmuIdle[0], pcmuIdle[1])
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
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{5, 6}) {
		t.Fatalf("new frame after underrun = %v", got)
	}
	queue = append(queue, []byte{7, 8})
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
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{1}) {
		t.Fatalf("single recovery frame = %v, want queued frame", got)
	}
	queue = append(queue, []byte{2})
	if got := concealer.next(&queue, &head, func() []byte { return silence }); string(got) != string([]byte{2}) {
		t.Fatalf("second recovery frame = %v, want second queued frame", got)
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
	source := &webRTCFrameSource{subs: map[chan webRTCFrame]struct{}{make(chan webRTCFrame): {}}}
	dropped, subscribers := source.broadcast([]byte{1})
	if subscribers != 1 {
		t.Fatalf("subscribers = %d, want 1", subscribers)
	}
	if dropped != 1 {
		t.Fatalf("dropped = %d, want 1", dropped)
	}
}

func TestWebRTCFrameSourceMailboxKeepsLatestFrame(t *testing.T) {
	source := &webRTCFrameSource{subs: map[chan webRTCFrame]struct{}{}}
	frames, unsubscribe, ok := source.subscribe()
	if !ok {
		t.Fatal("subscriber should attach")
	}
	_ = unsubscribe

	if dropped, _ := source.broadcast([]byte{1}); dropped != 0 {
		t.Fatalf("first broadcast dropped = %d, want 0 for empty mailbox", dropped)
	}
	if dropped, _ := source.broadcast([]byte{2}); dropped != 1 {
		t.Fatalf("second broadcast dropped = %d, want 1 for stale mailbox replacement", dropped)
	}
	select {
	case frame := <-frames:
		if string(frame.payload) != string([]byte{2}) {
			t.Fatalf("mailbox frame = %v, want latest [2]", frame)
		}
		if frame.sequence != 2 {
			t.Fatalf("mailbox frame sequence = %d, want 2", frame.sequence)
		}
	default:
		t.Fatal("mailbox did not retain latest frame")
	}
}

func TestWebRTCFrameSourceBroadcastCopiesFramePayload(t *testing.T) {
	source := &webRTCFrameSource{subs: map[chan webRTCFrame]struct{}{}}
	frames, unsubscribe, ok := source.subscribe()
	if !ok {
		t.Fatal("subscriber should attach")
	}
	defer unsubscribe()

	frame := []byte{1, 2, 3}
	if dropped, _ := source.broadcast(frame); dropped != 0 {
		t.Fatalf("broadcast dropped = %d, want 0", dropped)
	}
	frame[0] = 9
	source.mu.Lock()
	source.lastFrame.payload[1] = 8
	source.mu.Unlock()

	select {
	case got := <-frames:
		if string(got.payload) != string([]byte{1, 2, 3}) {
			t.Fatalf("subscriber frame = %v, want copied [1 2 3]", got)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for copied frame")
	}
}

func TestPCMSubscriberMailboxKeepsLatestChunk(t *testing.T) {
	subscriber := make(chan PCMChunk, 1)
	deliverPCMToSubscriber(subscriber, PCMChunk{FeedID: "sk-0001", Data: []byte{1}})
	deliverPCMToSubscriber(subscriber, PCMChunk{FeedID: "sk-0001", Data: []byte{2}})

	select {
	case chunk := <-subscriber:
		if string(chunk.Data) != string([]byte{2}) {
			t.Fatalf("mailbox chunk = %v, want latest [2]", chunk.Data)
		}
	default:
		t.Fatal("mailbox did not retain latest PCM chunk")
	}
}

func TestLatestWebRTCFrameSkipsStaleFrames(t *testing.T) {
	frames := make(chan webRTCFrame, 4)
	frames <- webRTCFrame{sequence: 2, payload: []byte{2}}
	frames <- webRTCFrame{sequence: 3}
	frames <- webRTCFrame{sequence: 4, payload: []byte{3}}
	frames <- webRTCFrame{sequence: 5, payload: []byte{4}}

	latest, skipped, ok := latestWebRTCFrame(webRTCFrame{sequence: 1, payload: []byte{1}}, frames)
	if !ok {
		t.Fatal("open frame channel was reported closed")
	}
	if string(latest.payload) != string([]byte{4}) {
		t.Fatalf("latest frame = %v, want [4]", latest)
	}
	if skipped != 3 {
		t.Fatalf("skipped = %d, want 3", skipped)
	}
}

func TestLatestWebRTCFrameDoesNotSkipStartupFrame(t *testing.T) {
	frames := make(chan webRTCFrame, 2)
	frames <- webRTCFrame{sequence: 1, payload: []byte{1}}

	latest, skipped, ok := latestWebRTCFrame(webRTCFrame{}, frames)
	if !ok {
		t.Fatal("open frame channel was reported closed")
	}
	if string(latest.payload) != string([]byte{1}) {
		t.Fatalf("latest frame = %v, want [1]", latest)
	}
	if skipped != 0 {
		t.Fatalf("skipped = %d, want 0", skipped)
	}
}

func TestRTPTimestampForFrameIncludesSkippedFramesBeforeWrite(t *testing.T) {
	base := uint32(4000)
	if got := rtpTimestampForFrame(webRTCAudioPCMU, base, 0); got != base {
		t.Fatalf("PCMU timestamp without skips = %d, want %d", got, base)
	}
	if got := rtpTimestampForFrame(webRTCAudioPCMU, base, 2); got != base+2*rtpTimestampStep(webRTCAudioPCMU) {
		t.Fatalf("PCMU timestamp with skips = %d, want %d", got, base+2*rtpTimestampStep(webRTCAudioPCMU))
	}
	if got := rtpTimestampForFrame(webRTCAudioOpus, base, 1); got != base+rtpTimestampStep(webRTCAudioOpus) {
		t.Fatalf("Opus timestamp with skips = %d, want %d", got, base+rtpTimestampStep(webRTCAudioOpus))
	}
	if got := rtpTimestampForFrame(webRTCAudioPCMU, base, -1); got != base {
		t.Fatalf("negative skipped timestamp = %d, want %d", got, base)
	}
}

func TestRTPTimestampAfterFrameAdvancesOneFrame(t *testing.T) {
	packetTimestamp := uint32(4000)
	if got := rtpTimestampAfterFrame(webRTCAudioPCMU, packetTimestamp); got != packetTimestamp+rtpTimestampStep(webRTCAudioPCMU) {
		t.Fatalf("PCMU next timestamp = %d, want %d", got, packetTimestamp+rtpTimestampStep(webRTCAudioPCMU))
	}
}

func TestShouldSendWebRTCFillerAtFrameCadence(t *testing.T) {
	now := time.Now()
	if !shouldSendWebRTCFiller(time.Time{}, now) {
		t.Fatal("filler should be allowed before the first write")
	}
	if shouldSendWebRTCFiller(now.Add(-webrtcFrameDuration+time.Millisecond), now) {
		t.Fatal("filler should wait until one frame duration has elapsed")
	}
	if !shouldSendWebRTCFiller(now.Add(-webrtcFrameDuration), now) {
		t.Fatal("filler should be sent at the normal WebRTC frame cadence")
	}
	if !shouldSendWebRTCFiller(now.Add(-2*webrtcFrameDuration+time.Millisecond), now) {
		t.Fatal("filler should not wait for the old two-frame threshold")
	}
}

func TestWebRTCTimestampSkippedFramesUsesSourceSequenceGap(t *testing.T) {
	if got := webRTCTimestampSkippedFrames(0, 4); got != 0 {
		t.Fatalf("first sent frame skipped count = %d, want 0", got)
	}
	if got := webRTCTimestampSkippedFrames(1, 2); got != 0 {
		t.Fatalf("contiguous frame skipped count = %d, want 0", got)
	}
	if got := webRTCTimestampSkippedFrames(1, 4); got != 2 {
		t.Fatalf("source sequence gap skipped count = %d, want 2", got)
	}
	if got := webRTCTimestampSkippedFrames(4, 4); got != 0 {
		t.Fatalf("duplicate frame skipped count = %d, want 0", got)
	}
}

func TestWebRTCTimestampSkippedFramesSubtractsFillerPackets(t *testing.T) {
	if got := webRTCTimestampSkippedFramesAfterFiller(3, 0); got != 3 {
		t.Fatalf("skipped count without filler = %d, want 3", got)
	}
	if got := webRTCTimestampSkippedFramesAfterFiller(3, 2); got != 1 {
		t.Fatalf("skipped count after partial filler = %d, want 1", got)
	}
	if got := webRTCTimestampSkippedFramesAfterFiller(2, 4); got != 0 {
		t.Fatalf("skipped count after covering filler = %d, want 0", got)
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
	hub.mu.Lock()
	source := hub.frameSources[webRTCFrameSourceKey{feedID: "sk-0001", codec: webRTCAudioG722}]
	hub.mu.Unlock()
	if source == nil {
		t.Fatal("frame source disappeared before idle cleanup")
	}
	source.mu.Lock()
	idleEpoch := source.idleEpoch
	source.mu.Unlock()
	hub.removeWebRTCFrameSourceIfIdle(source, idleEpoch)
	if !waitForWebRTCFrameSources(hub, 0, 2*time.Second) {
		t.Fatal("frame source was not removed by matching idle cleanup")
	}
}

func TestWebRTCFrameSourcePrimesIdleFrame(t *testing.T) {
	hub := newMemoryMediaHub()
	frames, unsubscribe, err := hub.SubscribeWebRTCFrames("sk-0001", webRTCAudioPCMU)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()
	select {
	case frame := <-frames:
		if len(frame.payload) != pcmuFrameSamples {
			t.Fatalf("primed idle frame length = %d, want %d", len(frame.payload), pcmuFrameSamples)
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("timed out waiting for primed idle frame")
	}
}

func TestInitialWebRTCFrameSeedsPacketWriter(t *testing.T) {
	if frame := initialWebRTCFrame(webRTCAudioPCMU); len(frame) != pcmuFrameSamples {
		t.Fatalf("PCMU initial frame length = %d, want %d", len(frame), pcmuFrameSamples)
	}
	if frame := initialWebRTCFrame(webRTCAudioG722); len(frame) == 0 {
		t.Fatal("G.722 initial frame should not be empty")
	}
	if frame := initialWebRTCFrame(webRTCAudioOpus); opusBackendAvailable() && len(frame) == 0 {
		t.Fatal("Opus initial frame should not be empty when the encoder is available")
	} else if !opusBackendAvailable() && len(frame) != 0 {
		t.Fatalf("Opus initial frame length = %d, want 0 without an encoder", len(frame))
	}
}

func TestWebRTCFrameSourceSeedsLateSubscriber(t *testing.T) {
	hub := newMemoryMediaHub()
	source, err := hub.webRTCFrameSource("sk-0001", webRTCAudioPCMU)
	if err != nil {
		t.Fatal(err)
	}
	cached := waitForCachedWebRTCFrame(t, source)
	frames, unsubscribe, ok := source.subscribe()
	if !ok {
		t.Fatal("late subscriber could not attach to frame source")
	}
	defer unsubscribe()
	select {
	case frame := <-frames:
		if string(frame.payload) != string(cached) {
			t.Fatalf("seeded frame = %v, want cached %v", frame, cached)
		}
		if frame.sequence == 0 {
			t.Fatal("seeded frame sequence should be set")
		}
	case <-time.After(50 * time.Millisecond):
		t.Fatal("timed out waiting for seeded frame")
	}
}

func TestWebRTCFrameSourceReusesIdleSourceDuringGrace(t *testing.T) {
	hub := newMemoryMediaHub()
	_, unsubscribe, err := hub.SubscribeWebRTCFrames("sk-0001", webRTCAudioPCMU)
	if err != nil {
		t.Fatal(err)
	}
	key := webRTCFrameSourceKey{feedID: "sk-0001", codec: webRTCAudioPCMU}
	hub.mu.Lock()
	first := hub.frameSources[key]
	hub.mu.Unlock()
	if first == nil {
		t.Fatal("frame source was not registered")
	}
	unsubscribe()

	_, unsubscribeAgain, err := hub.SubscribeWebRTCFrames("sk-0001", webRTCAudioPCMU)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribeAgain()
	hub.mu.Lock()
	second := hub.frameSources[key]
	hub.mu.Unlock()
	if second != first {
		t.Fatal("frame source should be reused during idle grace")
	}
}

func TestWebRTCFrameSourceStopClosesSubscribers(t *testing.T) {
	hub := newMemoryMediaHub()
	source, err := hub.webRTCFrameSource("sk-0001", webRTCAudioPCMU)
	if err != nil {
		t.Fatal(err)
	}
	frames, unsubscribe, ok := source.subscribe()
	if !ok {
		t.Fatal("subscriber could not attach to frame source")
	}

	source.stop()
	select {
	case _, ok := <-frames:
		if ok {
			t.Fatal("frame source subscriber channel stayed open after stop")
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for frame source subscriber close")
	}
	unsubscribe()
}

func TestMediaHubClosesPeerWhenWebRTCFrameSourceEnds(t *testing.T) {
	hub := newMemoryMediaHub()
	frames := make(chan webRTCFrame)
	close(frames)
	ready := make(chan struct{})
	close(ready)
	cleanup := make(chan struct{})
	var cleanupOnce sync.Once

	go hub.streamWebRTCFrames(t.Context(), "sk-0001", webRTCAudioPCMU, 0, nil, frames, func() {}, ready, func() {
		cleanupOnce.Do(func() {
			close(cleanup)
		})
	})

	select {
	case <-cleanup:
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for peer cleanup after frame source close")
	}
}

func TestWebRTCFrameSourceIgnoresStaleIdleCleanup(t *testing.T) {
	hub := newMemoryMediaHub()
	_, unsubscribe, err := hub.SubscribeWebRTCFrames("sk-0001", webRTCAudioPCMU)
	if err != nil {
		t.Fatal(err)
	}
	key := webRTCFrameSourceKey{feedID: "sk-0001", codec: webRTCAudioPCMU}
	hub.mu.Lock()
	source := hub.frameSources[key]
	hub.mu.Unlock()
	if source == nil {
		t.Fatal("frame source was not registered")
	}
	unsubscribe()
	source.mu.Lock()
	staleEpoch := source.idleEpoch
	source.mu.Unlock()

	_, unsubscribeAgain, err := hub.SubscribeWebRTCFrames("sk-0001", webRTCAudioPCMU)
	if err != nil {
		t.Fatal(err)
	}
	unsubscribeAgain()
	hub.removeWebRTCFrameSourceIfIdle(source, staleEpoch)
	hub.mu.Lock()
	stillRegistered := hub.frameSources[key] == source
	hub.mu.Unlock()
	if !stillRegistered {
		t.Fatal("stale idle cleanup removed a reused frame source")
	}

	source.mu.Lock()
	currentEpoch := source.idleEpoch
	source.mu.Unlock()
	hub.removeWebRTCFrameSourceIfIdle(source, currentEpoch)
	if !waitForWebRTCFrameSources(hub, 0, 2*time.Second) {
		t.Fatal("current idle cleanup did not remove the frame source")
	}
}

func TestPreferredWebRTCAudioCodecFallsBackForReceiverOffers(t *testing.T) {
	if got, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 0\r\na=rtpmap:0 PCMU/8000\r\n", WebRTCAnswerOptions{}); err != nil || got != webRTCAudioPCMU {
		t.Fatal("PCMU-only offers should use PCMU")
	}
	if got, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 9\r\na=rtpmap:9 G722/8000\r\n", WebRTCAnswerOptions{}); err != nil || got != webRTCAudioG722 {
		t.Fatal("G.722-capable offers should use G.722")
	}
	got, err := preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 111 9 0\r\na=rtpmap:111 opus/48000/2\r\na=rtpmap:9 G722/8000\r\na=rtpmap:0 PCMU/8000\r\n", WebRTCAnswerOptions{})
	if err != nil || got != webRTCAudioPCMU {
		t.Fatal("auto codec should prefer PCMU for receiver stability")
	}
	got, err = preferredWebRTCAudioCodec("m=audio 9 UDP/TLS/RTP/SAVPF 111 9\r\na=rtpmap:111 opus/48000/2\r\na=rtpmap:9 G722/8000\r\n", WebRTCAnswerOptions{})
	if err != nil || got != webRTCAudioG722 {
		t.Fatal("auto codec should use G.722 when PCMU is unavailable")
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

func TestWebRTCICECleanupPolicyKeepsTransientDisconnects(t *testing.T) {
	if shouldCleanupWebRTCICE(webrtc.ICEConnectionStateDisconnected) {
		t.Fatal("transient ICE disconnects should be given a recovery window")
	}
	if !shouldCleanupWebRTCICE(webrtc.ICEConnectionStateFailed) {
		t.Fatal("failed ICE peers should be cleaned up")
	}
	if !shouldCleanupWebRTCICE(webrtc.ICEConnectionStateClosed) {
		t.Fatal("closed ICE peers should be cleaned up")
	}
}

func TestWebRTCDisconnectGraceCleanupIncludesICEState(t *testing.T) {
	if !shouldCleanupDisconnectedWebRTC(webrtc.PeerConnectionStateConnected, webrtc.ICEConnectionStateDisconnected) {
		t.Fatal("ICE disconnect should be enough to clean up after the grace window")
	}
	if !shouldCleanupDisconnectedWebRTC(webrtc.PeerConnectionStateDisconnected, webrtc.ICEConnectionStateConnected) {
		t.Fatal("peer disconnect should be enough to clean up after the grace window")
	}
	if shouldCleanupDisconnectedWebRTC(webrtc.PeerConnectionStateConnected, webrtc.ICEConnectionStateConnected) {
		t.Fatal("connected peer and ICE states should not be cleaned up by the grace timer")
	}
}

func TestWebRTCMediaStartPolicyUsesPeerOrICEReadiness(t *testing.T) {
	if !shouldStartWebRTCMedia(webrtc.PeerConnectionStateConnected, webrtc.ICEConnectionStateNew) {
		t.Fatal("connected peer state should start media")
	}
	if !shouldStartWebRTCMedia(webrtc.PeerConnectionStateConnecting, webrtc.ICEConnectionStateConnected) {
		t.Fatal("connected ICE state should start media")
	}
	if !shouldStartWebRTCMedia(webrtc.PeerConnectionStateConnecting, webrtc.ICEConnectionStateCompleted) {
		t.Fatal("completed ICE state should start media")
	}
	if shouldStartWebRTCMedia(webrtc.PeerConnectionStateConnecting, webrtc.ICEConnectionStateChecking) {
		t.Fatal("checking ICE should not start media")
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
	wantCodec := "PCMU"
	if !strings.Contains(answer, wantCodec) {
		t.Fatalf("answer did not include %s: %s", wantCodec, answer)
	}
	if err := offerPeer.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer}); err != nil {
		t.Fatal(err)
	}
}

func TestMediaHubReceiverAnswerUsesPCMUWhenAvailable(t *testing.T) {
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
	wantCodec := "PCMU"
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

func TestMediaHubStreamsIdleRTPWithoutPublishedPCM(t *testing.T) {
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

	track := waitForRemoteTrack(t, tracks)
	for i := 0; i < 3; i++ {
		payload := waitForRTPPacket(t, track)
		if len(payload) == 0 {
			t.Fatal("idle RTP payload must not be empty")
		}
	}
}

func TestMediaHubStreamsIdleRTPContinuously(t *testing.T) {
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

	track := waitForRemoteTrack(t, tracks)
	var previousSequence uint16
	var previousTimestamp uint32
	for i := 0; i < 8; i++ {
		sequence, timestamp, payloadLength := waitForRTPPacketHeaderInfo(t, track)
		if payloadLength == 0 {
			t.Fatal("idle RTP payload must not be empty")
		}
		if i > 0 {
			if delta := sequence - previousSequence; delta != 1 {
				t.Fatalf("RTP sequence delta = %d, want 1", delta)
			}
			if delta := timestamp - previousTimestamp; delta != uint32(webrtcRTPClockRate/50) {
				t.Fatalf("RTP timestamp delta = %d, want %d", delta, webrtcRTPClockRate/50)
			}
		}
		previousSequence = sequence
		previousTimestamp = timestamp
	}
}

func TestMediaHubWritesNegotiatedRTPPayloadType(t *testing.T) {
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
	answer, err := hub.AnswerWithOptions(t.Context(), "sk-0001", offerPeer.LocalDescription().SDP, WebRTCAnswerOptions{PreferredCodec: "g722"})
	if err != nil {
		t.Fatal(err)
	}
	if answer.Codec != webRTCAudioG722 || answer.PayloadType != 9 {
		t.Fatalf("negotiated codec=%s payload=%d, want g722/9", answer.Codec, answer.PayloadType)
	}
	if err := offerPeer.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: answer.SDP}); err != nil {
		t.Fatal(err)
	}

	track := waitForRemoteTrack(t, tracks)
	payloadType, payloadLength := waitForRTPPacketPayloadType(t, track)
	if payloadLength == 0 {
		t.Fatal("RTP payload must not be empty")
	}
	if payloadType != answer.PayloadType {
		t.Fatalf("RTP payload type = %d, want negotiated payload %d", payloadType, answer.PayloadType)
	}
}

func TestNewWebRTCPeerStreamStatsSeedsRTPStart(t *testing.T) {
	stats := newWebRTCPeerStreamStats(time.Unix(1700000000, 123456789))
	if stats.lastReport.IsZero() {
		t.Fatal("last report time should be initialized")
	}
	if stats.sequenceNumber == 0 && stats.timestamp == 0 {
		t.Fatal("RTP sequence and timestamp should not both start at zero")
	}
}

func TestMediaHubKeepsRTPContinuousAfterPublishedPCMStops(t *testing.T) {
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

	track := waitForRemoteTrack(t, tracks)
	pcm := sinePCM(960, 1000, 48000, 8000)
	for i := 0; i < 5; i++ {
		hub.publish(PCMChunk{FeedID: "sk-0001", SampleRate: 48000, Channels: 1, Duration: 20 * time.Millisecond, Data: pcm})
		time.Sleep(20 * time.Millisecond)
	}

	var previousTimestamp uint32
	for i := 0; i < 14; i++ {
		timestamp, payloadLength := waitForRTPPacketInfo(t, track)
		if payloadLength == 0 {
			t.Fatal("RTP payload must not be empty after PCM stops")
		}
		if i > 0 {
			if delta := timestamp - previousTimestamp; delta != uint32(webrtcRTPClockRate/50) {
				t.Fatalf("RTP timestamp delta after PCM stopped = %d, want %d", delta, webrtcRTPClockRate/50)
			}
		}
		previousTimestamp = timestamp
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

func waitForWebRTCFrame(t *testing.T, frames <-chan webRTCFrame) []byte {
	t.Helper()
	select {
	case frame := <-frames:
		return frame.payload
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for WebRTC frame")
		return nil
	}
}

func waitForCachedWebRTCFrame(t *testing.T, source *webRTCFrameSource) []byte {
	t.Helper()
	deadline := time.Now().Add(250 * time.Millisecond)
	for time.Now().Before(deadline) {
		source.mu.Lock()
		frame := append([]byte(nil), source.lastFrame.payload...)
		source.mu.Unlock()
		if len(frame) > 0 {
			return frame
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("timed out waiting for cached WebRTC frame")
	return nil
}

func waitForRemoteTrack(t *testing.T, tracks <-chan *webrtc.TrackRemote) *webrtc.TrackRemote {
	t.Helper()
	select {
	case track := <-tracks:
		return track
	case <-time.After(5 * time.Second):
		t.Fatal("timed out waiting for WebRTC track")
		return nil
	}
}

func waitForRTPPacket(t *testing.T, track *webrtc.TrackRemote) []byte {
	t.Helper()
	packet, _, err := track.ReadRTP()
	if err != nil {
		t.Fatal(err)
	}
	return packet.Payload
}

func waitForRTPPacketInfo(t *testing.T, track *webrtc.TrackRemote) (uint32, int) {
	t.Helper()
	packet, _, err := track.ReadRTP()
	if err != nil {
		t.Fatal(err)
	}
	return packet.Timestamp, len(packet.Payload)
}

func waitForRTPPacketHeaderInfo(t *testing.T, track *webrtc.TrackRemote) (uint16, uint32, int) {
	t.Helper()
	packet, _, err := track.ReadRTP()
	if err != nil {
		t.Fatal(err)
	}
	return packet.SequenceNumber, packet.Timestamp, len(packet.Payload)
}

func waitForRTPPacketPayloadType(t *testing.T, track *webrtc.TrackRemote) (uint8, int) {
	t.Helper()
	packet, _, err := track.ReadRTP()
	if err != nil {
		t.Fatal(err)
	}
	return packet.PayloadType, len(packet.Payload)
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
