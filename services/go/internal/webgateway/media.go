package webgateway

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/gotranspile/g722"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

const (
	opusSampleRate            = 48000
	opusFrameSamples          = opusSampleRate / 50
	g722SampleRate            = 16000
	webrtcRTPClockRate        = 8000
	pcmuSampleRate            = 8000
	pcmuFrameSamples          = pcmuSampleRate / 50
	webrtcChannels            = 1
	opusRTPChannels           = 2
	opusEncoderChannels       = 1
	webrtcFrameDuration       = 20 * time.Millisecond
	webrtcMaxQueuedFrames     = 10
	webrtcResumeQueuedFrames  = 2
	webrtcConcealmentFrames   = 6
	feedIngressCapacity       = 4
	g722FrameSamples          = g722SampleRate / 50
	bridgeReconnectDelay      = 750 * time.Millisecond
	webrtcDisconnectGrace     = 15 * time.Second
	webrtcDiagnosticsInterval = 30 * time.Second
)

type webRTCAudioCodec int

const (
	webRTCAudioOpus webRTCAudioCodec = iota
	webRTCAudioG722
	webRTCAudioPCMU
)

func (c webRTCAudioCodec) String() string {
	switch c {
	case webRTCAudioOpus:
		return "opus"
	case webRTCAudioG722:
		return "g722"
	case webRTCAudioPCMU:
		return "pcmu"
	default:
		return "unknown"
	}
}

type WebRTCAnswerOptions struct {
	DisableG722    bool
	RequireOpus    bool
	PreferredCodec string
}

type opusFrameEncoder interface {
	Encode([]int16) ([]byte, error)
}

type PCMChunk struct {
	FeedID     string
	SampleRate int
	Channels   int
	Duration   time.Duration
	Data       []byte
}

type MediaHub struct {
	addr         string
	mu           sync.Mutex
	subscribers  map[string]map[chan PCMChunk]struct{}
	peers        map[string]*webrtc.PeerConnection
	ingress      map[string]chan PCMChunk
	frameSources map[webRTCFrameSourceKey]*webRTCFrameSource
	last         map[string]PCMChunk
	lastAt       map[string]time.Time
	seenLogged   map[string]bool
}

func NewMediaHub(addr string) *MediaHub {
	hub := &MediaHub{
		addr:         strings.TrimSpace(addr),
		subscribers:  map[string]map[chan PCMChunk]struct{}{},
		peers:        map[string]*webrtc.PeerConnection{},
		ingress:      map[string]chan PCMChunk{},
		frameSources: map[webRTCFrameSourceKey]*webRTCFrameSource{},
		last:         map[string]PCMChunk{},
		lastAt:       map[string]time.Time{},
		seenLogged:   map[string]bool{},
	}
	if hub.addr != "" {
		go hub.run(context.Background())
	}
	return hub
}

func (h *MediaHub) Available() bool {
	return h != nil && h.addr != ""
}

func (h *MediaHub) Answer(ctx context.Context, feedID string, offerSDP string) (string, error) {
	return h.AnswerWithOptions(ctx, feedID, offerSDP, WebRTCAnswerOptions{})
}

func (h *MediaHub) AnswerWithOptions(ctx context.Context, feedID string, offerSDP string, options WebRTCAnswerOptions) (string, error) {
	feedID = strings.TrimSpace(feedID)
	if !h.Available() {
		return "", errors.New("media bridge is not available")
	}
	if feedID == "" {
		return "", errors.New("feed_id is required")
	}
	if strings.TrimSpace(offerSDP) == "" {
		return "", errors.New("sdp is required")
	}

	peerConnection, err := newWebRTCPeerConnection(webrtc.Configuration{})
	if err != nil {
		return "", err
	}
	if !h.HasRecentPCM(feedID, 5*time.Second) {
		log.Printf("media bridge has no recent PCM for feed %s; WebRTC peer will receive silence until playout publishes audio", feedID)
	}
	peerCtx, cancelPeer := context.WithCancel(ctx)
	peerID := fmt.Sprintf("%s-%d", mediaSafeID(feedID), time.Now().UnixNano())
	h.mu.Lock()
	h.peers[peerID] = peerConnection
	h.mu.Unlock()
	var cleanupOnce sync.Once
	var disconnectMu sync.Mutex
	var disconnectTimer *time.Timer
	var cleanup func()
	stopDisconnectTimer := func() {
		disconnectMu.Lock()
		defer disconnectMu.Unlock()
		if disconnectTimer != nil {
			disconnectTimer.Stop()
			disconnectTimer = nil
		}
	}
	scheduleDisconnectCleanup := func() {
		disconnectMu.Lock()
		defer disconnectMu.Unlock()
		if disconnectTimer != nil {
			return
		}
		disconnectTimer = time.AfterFunc(webrtcDisconnectGrace, func() {
			if peerConnection.ConnectionState() == webrtc.PeerConnectionStateDisconnected {
				cleanup()
			}
			disconnectMu.Lock()
			disconnectTimer = nil
			disconnectMu.Unlock()
		})
	}
	cleanup = func() {
		cleanupOnce.Do(func() {
			stopDisconnectTimer()
			cancelPeer()
			h.mu.Lock()
			delete(h.peers, peerID)
			h.mu.Unlock()
			_ = peerConnection.Close()
		})
	}

	codec, err := preferredWebRTCAudioCodec(offerSDP, options)
	if err != nil {
		cleanup()
		return "", err
	}
	payloadType := offeredAudioPayloadType(offerSDP, codec)
	capability := webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: opusSampleRate, Channels: opusRTPChannels}
	if codec == webRTCAudioG722 {
		capability = webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeG722, ClockRate: webrtcRTPClockRate, Channels: webrtcChannels}
	} else if codec == webRTCAudioPCMU {
		capability = webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypePCMU, ClockRate: pcmuSampleRate, Channels: webrtcChannels}
	}
	track, err := webrtc.NewTrackLocalStaticSample(capability, "haze-"+mediaSafeID(feedID), "haze-"+mediaSafeID(feedID))
	if err != nil {
		cleanup()
		return "", err
	}
	transceiver, err := peerConnection.AddTransceiverFromTrack(track, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionSendonly,
	})
	if err != nil {
		cleanup()
		return "", err
	}
	if err := transceiver.SetCodecPreferences(codecPreferences(codec, payloadType)); err != nil {
		cleanup()
		return "", err
	}
	sender := transceiver.Sender()
	go drainRTCP(sender)

	peerConnection.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		if shouldCleanupWebRTCPeer(state) {
			cleanup()
			return
		}
		switch state {
		case webrtc.PeerConnectionStateConnected:
			stopDisconnectTimer()
		case webrtc.PeerConnectionStateDisconnected:
			scheduleDisconnectCleanup()
		}
	})

	if err := peerConnection.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  offerSDP,
	}); err != nil {
		cleanup()
		return "", err
	}
	answer, err := peerConnection.CreateAnswer(nil)
	if err != nil {
		cleanup()
		return "", err
	}
	gatheringComplete := webrtc.GatheringCompletePromise(peerConnection)
	if err := peerConnection.SetLocalDescription(answer); err != nil {
		cleanup()
		return "", err
	}
	select {
	case <-gatheringComplete:
	case <-ctx.Done():
		cleanup()
		return "", ctx.Err()
	case <-time.After(5 * time.Second):
	}
	localDescription := peerConnection.LocalDescription()
	if localDescription == nil {
		cleanup()
		return "", errors.New("could not create local WebRTC description")
	}

	frames, unsubscribeFrames, err := h.SubscribeWebRTCFrames(feedID, codec)
	if err != nil {
		cleanup()
		return "", err
	}
	go h.streamWebRTCFrames(peerCtx, feedID, codec, track, frames, unsubscribeFrames)
	return localDescription.SDP, nil
}

func shouldCleanupWebRTCPeer(state webrtc.PeerConnectionState) bool {
	return state == webrtc.PeerConnectionStateClosed || state == webrtc.PeerConnectionStateFailed
}

func newWebRTCPeerConnection(configuration webrtc.Configuration) (*webrtc.PeerConnection, error) {
	var mediaEngine webrtc.MediaEngine
	if err := mediaEngine.RegisterDefaultCodecs(); err != nil {
		return nil, err
	}
	api := webrtc.NewAPI(webrtc.WithMediaEngine(&mediaEngine))
	return api.NewPeerConnection(configuration)
}

func preferredWebRTCAudioCodec(offerSDP string, options WebRTCAnswerOptions) (webRTCAudioCodec, error) {
	upper := strings.ToUpper(offerSDP)
	if preferred, ok := parseWebRTCAudioCodec(options.PreferredCodec); ok {
		return requiredWebRTCAudioCodec(upper, preferred)
	}
	if options.RequireOpus {
		return requiredWebRTCAudioCodec(upper, webRTCAudioOpus)
	}
	if !options.DisableG722 && (strings.Contains(upper, "G722/8000") || strings.Contains(upper, " G722")) {
		return webRTCAudioG722, nil
	}
	if opusBackendAvailable() && strings.Contains(upper, "OPUS/48000") {
		return webRTCAudioOpus, nil
	}
	return webRTCAudioPCMU, nil
}

func parseWebRTCAudioCodec(value string) (webRTCAudioCodec, bool) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "auto":
		return webRTCAudioOpus, false
	case "opus":
		return webRTCAudioOpus, true
	case "g722", "g.722":
		return webRTCAudioG722, true
	case "pcmu", "mulaw", "ulaw", "u-law":
		return webRTCAudioPCMU, true
	default:
		return webRTCAudioOpus, false
	}
}

func requiredWebRTCAudioCodec(upperOfferSDP string, codec webRTCAudioCodec) (webRTCAudioCodec, error) {
	switch codec {
	case webRTCAudioOpus:
		if !strings.Contains(upperOfferSDP, "OPUS/48000") {
			return webRTCAudioOpus, errors.New("receiver requires Opus but the offer did not include Opus")
		}
		if !opusBackendAvailable() {
			return webRTCAudioOpus, errors.New("receiver requires Opus but this gateway was built without an Opus encoder")
		}
		return webRTCAudioOpus, nil
	case webRTCAudioG722:
		if !strings.Contains(upperOfferSDP, "G722/8000") && !strings.Contains(upperOfferSDP, " G722") {
			return webRTCAudioG722, errors.New("receiver requires G.722 but the offer did not include G.722")
		}
		return webRTCAudioG722, nil
	case webRTCAudioPCMU:
		if !strings.Contains(upperOfferSDP, "PCMU/8000") {
			return webRTCAudioPCMU, errors.New("receiver requires PCMU but the offer did not include PCMU")
		}
		return webRTCAudioPCMU, nil
	default:
		return webRTCAudioOpus, errors.New("unsupported requested WebRTC audio codec")
	}
}

func offeredAudioPayloadType(offerSDP string, codec webRTCAudioCodec) uint8 {
	target := ""
	fallback := uint8(0)
	switch codec {
	case webRTCAudioOpus:
		target = "OPUS/48000"
		fallback = 111
	case webRTCAudioG722:
		target = "G722/8000"
		fallback = 9
	default:
		target = "PCMU/8000"
		fallback = 0
	}
	for _, line := range strings.Split(offerSDP, "\n") {
		line = strings.TrimSpace(line)
		upper := strings.ToUpper(line)
		if !strings.HasPrefix(upper, "A=RTPMAP:") || !strings.Contains(upper, target) {
			continue
		}
		start := len("a=rtpmap:")
		end := strings.Index(line[start:], " ")
		if end < 0 {
			continue
		}
		var parsed uint64
		if _, err := fmt.Sscanf(line[start:start+end], "%d", &parsed); err == nil && parsed <= 127 {
			return uint8(parsed)
		}
	}
	return fallback
}

func codecPreferences(codec webRTCAudioCodec, payloadType uint8) []webrtc.RTPCodecParameters {
	switch codec {
	case webRTCAudioOpus:
		if payloadType == 0 {
			payloadType = 111
		}
		return []webrtc.RTPCodecParameters{{
			RTPCodecCapability: webrtc.RTPCodecCapability{
				MimeType:    webrtc.MimeTypeOpus,
				ClockRate:   opusSampleRate,
				Channels:    opusRTPChannels,
				SDPFmtpLine: "minptime=10;useinbandfec=1",
			},
			PayloadType: webrtc.PayloadType(payloadType),
		}}
	case webRTCAudioG722:
		if payloadType == 0 {
			payloadType = 9
		}
		return []webrtc.RTPCodecParameters{{
			RTPCodecCapability: webrtc.RTPCodecCapability{
				MimeType:  webrtc.MimeTypeG722,
				ClockRate: webrtcRTPClockRate,
				Channels:  webrtcChannels,
			},
			PayloadType: webrtc.PayloadType(payloadType),
		}}
	default:
		return []webrtc.RTPCodecParameters{{
			RTPCodecCapability: webrtc.RTPCodecCapability{
				MimeType:  webrtc.MimeTypePCMU,
				ClockRate: pcmuSampleRate,
				Channels:  webrtcChannels,
			},
			PayloadType: 0,
		}}
	}
}

func (h *MediaHub) run(ctx context.Context) {
	for {
		if ctx.Err() != nil {
			return
		}
		conn, err := net.DialTimeout("tcp", h.addr, 3*time.Second)
		if err != nil {
			sleepMedia(ctx, bridgeReconnectDelay)
			continue
		}
		h.readBridge(ctx, conn)
		_ = conn.Close()
		sleepMedia(ctx, bridgeReconnectDelay)
	}
}

func (h *MediaHub) readBridge(ctx context.Context, conn net.Conn) {
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		if ctx.Err() != nil {
			return
		}
		chunk, ok := decodePCMBridgeEvent(scanner.Bytes())
		if !ok {
			continue
		}
		h.publish(chunk)
	}
}

func decodePCMBridgeEvent(raw []byte) (PCMChunk, bool) {
	var event struct {
		Type   string `json:"type"`
		FeedID string `json:"feed_id"`
		Data   struct {
			FeedID     string `json:"feed_id"`
			SampleRate int    `json:"sample_rate"`
			Channels   int    `json:"channels"`
			DurationMS int    `json:"duration_ms"`
			PCM        string `json:"pcm"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &event); err != nil || event.Type != "playout.pcm" {
		return PCMChunk{}, false
	}
	feedID := strings.TrimSpace(event.Data.FeedID)
	if feedID == "" {
		feedID = strings.TrimSpace(event.FeedID)
	}
	if feedID == "" || event.Data.PCM == "" {
		return PCMChunk{}, false
	}
	pcm, err := base64.StdEncoding.DecodeString(event.Data.PCM)
	if err != nil {
		return PCMChunk{}, false
	}
	sampleRate := event.Data.SampleRate
	if sampleRate <= 0 {
		sampleRate = opusSampleRate
	}
	channels := event.Data.Channels
	if channels <= 0 {
		channels = 1
	}
	duration := time.Duration(event.Data.DurationMS) * time.Millisecond
	if duration <= 0 {
		duration = webrtcFrameDuration
	}
	chunk := PCMChunk{
		FeedID:     feedID,
		SampleRate: sampleRate,
		Channels:   channels,
		Duration:   duration,
		Data:       pcm,
	}
	chunk, ok, _ := validatePCMChunk(chunk)
	return chunk, ok
}

func (h *MediaHub) publish(chunk PCMChunk) {
	chunk, ok, reason := validatePCMChunk(chunk)
	if !ok {
		if reason != "" {
			log.Printf("media bridge rejected PCM for feed %s: %s", chunk.FeedID, reason)
		}
		return
	}
	ingress := h.feedIngress(chunk.FeedID)
	select {
	case ingress <- chunk:
	default:
		select {
		case <-ingress:
		default:
		}
		select {
		case ingress <- chunk:
		default:
		}
	}
}

func (h *MediaHub) feedIngress(feedID string) chan PCMChunk {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.ingress == nil {
		h.ingress = map[string]chan PCMChunk{}
	}
	ingress := h.ingress[feedID]
	if ingress == nil {
		ingress = make(chan PCMChunk, feedIngressCapacity)
		h.ingress[feedID] = ingress
		go h.runFeedIngress(feedID, ingress)
	}
	return ingress
}

func (h *MediaHub) runFeedIngress(_ string, ingress <-chan PCMChunk) {
	for chunk := range ingress {
		h.publishReady(chunk)
	}
}

func (h *MediaHub) publishReady(chunk PCMChunk) {
	now := time.Now()
	h.mu.Lock()
	if h.last == nil {
		h.last = map[string]PCMChunk{}
	}
	if h.lastAt == nil {
		h.lastAt = map[string]time.Time{}
	}
	if h.seenLogged == nil {
		h.seenLogged = map[string]bool{}
	}
	h.last[chunk.FeedID] = chunk
	h.lastAt[chunk.FeedID] = now
	if !h.seenLogged[chunk.FeedID] {
		h.seenLogged[chunk.FeedID] = true
		log.Printf("media bridge receiving PCM for feed %s (%d Hz, %d channel)", chunk.FeedID, chunk.SampleRate, chunk.Channels)
	}
	for subscriber := range h.subscribers[chunk.FeedID] {
		select {
		case subscriber <- chunk:
		default:
			select {
			case <-subscriber:
			default:
			}
			select {
			case subscriber <- chunk:
			default:
			}
		}
	}
	h.mu.Unlock()
}

func validatePCMChunk(chunk PCMChunk) (PCMChunk, bool, string) {
	chunk.FeedID = strings.TrimSpace(chunk.FeedID)
	if chunk.FeedID == "" {
		return chunk, false, "missing feed_id"
	}
	if len(chunk.Data) == 0 {
		return chunk, false, "empty PCM payload"
	}
	if chunk.SampleRate <= 0 {
		chunk.SampleRate = opusSampleRate
	}
	if chunk.SampleRate < 8000 || chunk.SampleRate > 96000 {
		return chunk, false, fmt.Sprintf("unsupported sample rate %d", chunk.SampleRate)
	}
	if chunk.Channels <= 0 {
		chunk.Channels = 1
	}
	if chunk.Channels > 8 {
		return chunk, false, fmt.Sprintf("unsupported channel count %d", chunk.Channels)
	}
	bytesPerFrame := chunk.Channels * 2
	if len(chunk.Data) < bytesPerFrame {
		return chunk, false, "PCM payload is shorter than one frame"
	}
	if remainder := len(chunk.Data) % bytesPerFrame; remainder != 0 {
		chunk.Data = chunk.Data[:len(chunk.Data)-remainder]
	}
	if len(chunk.Data) == 0 {
		return chunk, false, "PCM payload has no aligned frames"
	}
	maxBytes := chunk.SampleRate * chunk.Channels * 2 * 2
	if len(chunk.Data) > maxBytes {
		return chunk, false, fmt.Sprintf("PCM payload is too large (%d bytes)", len(chunk.Data))
	}
	if chunk.Duration <= 0 {
		chunk.Duration = webrtcFrameDuration
	}
	if chunk.Duration > 2*time.Second {
		return chunk, false, fmt.Sprintf("PCM duration is too large (%s)", chunk.Duration)
	}
	return chunk, true, ""
}

func (h *MediaHub) HasRecentPCM(feedID string, maxAge time.Duration) bool {
	if h == nil {
		return false
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	lastAt, ok := h.lastAt[strings.TrimSpace(feedID)]
	return ok && time.Since(lastAt) <= maxAge
}

func (h *MediaHub) Subscribe(feedID string) (<-chan PCMChunk, func()) {
	feedID = strings.TrimSpace(feedID)
	ch := make(chan PCMChunk, 8)
	h.mu.Lock()
	if h.subscribers[feedID] == nil {
		h.subscribers[feedID] = map[chan PCMChunk]struct{}{}
	}
	h.subscribers[feedID][ch] = struct{}{}
	if last, ok := h.last[feedID]; ok {
		ch <- last
	}
	h.mu.Unlock()
	var once sync.Once
	return ch, func() {
		once.Do(func() {
			h.mu.Lock()
			if h.subscribers[feedID] != nil {
				delete(h.subscribers[feedID], ch)
				if len(h.subscribers[feedID]) == 0 {
					delete(h.subscribers, feedID)
				}
			}
			h.mu.Unlock()
			close(ch)
		})
	}
}

type webRTCFrameSourceKey struct {
	feedID string
	codec  webRTCAudioCodec
}

type webRTCFrameSource struct {
	hub      *MediaHub
	key      webRTCFrameSourceKey
	encoder  opusFrameEncoder
	mu       sync.Mutex
	subs     map[chan []byte]struct{}
	stopCh   chan struct{}
	stopOnce sync.Once
	closed   bool
}

type webRTCFrameKind int

const (
	webRTCFrameReal webRTCFrameKind = iota
	webRTCFrameConcealed
	webRTCFrameIdle
)

type webRTCFrameSourceStats struct {
	produced   uint64
	real       uint64
	concealed  uint64
	idle       uint64
	dropped    uint64
	lastReport time.Time
}

func (h *MediaHub) SubscribeWebRTCFrames(feedID string, codec webRTCAudioCodec) (<-chan []byte, func(), error) {
	for {
		source, err := h.webRTCFrameSource(feedID, codec)
		if err != nil {
			return nil, nil, err
		}
		frames, unsubscribe, ok := source.subscribe()
		if ok {
			return frames, unsubscribe, nil
		}
		h.removeWebRTCFrameSource(source)
	}
}

func (h *MediaHub) webRTCFrameSource(feedID string, codec webRTCAudioCodec) (*webRTCFrameSource, error) {
	key := webRTCFrameSourceKey{feedID: strings.TrimSpace(feedID), codec: codec}
	h.mu.Lock()
	if h.frameSources == nil {
		h.frameSources = map[webRTCFrameSourceKey]*webRTCFrameSource{}
	}
	source := h.frameSources[key]
	h.mu.Unlock()
	if source != nil {
		return source, nil
	}

	var encoder opusFrameEncoder
	var err error
	if codec == webRTCAudioOpus {
		encoder, err = newOpusFrameEncoder(opusSampleRate, opusEncoderChannels)
		if err != nil {
			return nil, err
		}
	}
	source = &webRTCFrameSource{
		hub:     h,
		key:     key,
		encoder: encoder,
		subs:    map[chan []byte]struct{}{},
		stopCh:  make(chan struct{}),
	}

	h.mu.Lock()
	if existing := h.frameSources[key]; existing != nil {
		h.mu.Unlock()
		source.stop()
		return existing, nil
	}
	h.frameSources[key] = source
	h.mu.Unlock()

	go source.run()
	return source, nil
}

func (s *webRTCFrameSource) subscribe() (<-chan []byte, func(), bool) {
	ch := make(chan []byte, webrtcMaxQueuedFrames)
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil, nil, false
	}
	s.subs[ch] = struct{}{}
	s.mu.Unlock()
	var once sync.Once
	return ch, func() {
		once.Do(func() {
			s.mu.Lock()
			delete(s.subs, ch)
			empty := len(s.subs) == 0
			s.mu.Unlock()
			if empty {
				s.hub.removeWebRTCFrameSource(s)
			}
		})
	}, true
}

func (h *MediaHub) removeWebRTCFrameSource(source *webRTCFrameSource) {
	shouldStop := false
	h.mu.Lock()
	source.mu.Lock()
	if h.frameSources[source.key] == source && len(source.subs) == 0 {
		delete(h.frameSources, source.key)
		source.closed = true
		shouldStop = true
	}
	source.mu.Unlock()
	h.mu.Unlock()
	if shouldStop {
		source.stop()
	}
}

func (s *webRTCFrameSource) stop() {
	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()
	s.stopOnce.Do(func() {
		close(s.stopCh)
	})
}

func (s *webRTCFrameSource) run() {
	updates, unsubscribe := s.hub.Subscribe(s.key.feedID)
	defer unsubscribe()
	ticker := time.NewTicker(webrtcFrameDuration)
	defer ticker.Stop()

	frameQueue := make([][]byte, 0, 32)
	frameHead := 0
	concealer := frameConcealer{}
	g722Encoder := g722.NewEncoder(g722.Rate64000, 0)
	g722Silence := make([]int16, g722FrameSamples)
	opusIdle := opusIdleFrameSamples()
	pcmuIdle := pcmuSilence()
	loggedFirstFrame := false
	stats := webRTCFrameSourceStats{lastReport: time.Now()}

	for {
		select {
		case <-s.stopCh:
			return
		case <-ticker.C:
			for drained := 0; drained < cap(updates); drained++ {
				select {
				case chunk, ok := <-updates:
					if !ok {
						return
					}
					compactQueuedFrames(&frameQueue, &frameHead)
					frameQueue = s.appendFrames(frameQueue, g722Encoder, chunk)
				default:
					drained = cap(updates)
				}
			}
			frame, kind := concealer.nextWithKind(&frameQueue, &frameHead, func() []byte {
				return s.idleFrame(g722Encoder, g722Silence, opusIdle, pcmuIdle)
			})
			if len(frame) == 0 {
				continue
			}
			dropped, subscribers := s.broadcast(frame)
			stats.record(kind, dropped)
			s.maybeLogDiagnostics(&stats, subscribers, queuedFrameCount(frameQueue, frameHead))
			if !loggedFirstFrame {
				log.Printf("media bridge WebRTC frame source started for feed %s codec=%s (%d bytes)", s.key.feedID, s.key.codec, len(frame))
				loggedFirstFrame = true
			}
		}
	}
}

func (s *webRTCFrameSource) appendFrames(queue [][]byte, g722Encoder *g722.Encoder, chunk PCMChunk) [][]byte {
	switch s.key.codec {
	case webRTCAudioOpus:
		return appendOpusFrames(queue, s.encoder, chunk)
	case webRTCAudioPCMU:
		return appendPCMUFrames(queue, chunk)
	default:
		return appendG722Frames(queue, g722Encoder, chunk)
	}
}

func (s *webRTCFrameSource) idleFrame(g722Encoder *g722.Encoder, g722Silence []int16, opusIdle []int16, pcmuIdle []byte) []byte {
	switch s.key.codec {
	case webRTCAudioOpus:
		encoded, err := s.encoder.Encode(opusIdle)
		if err != nil {
			return nil
		}
		return encoded
	case webRTCAudioPCMU:
		return pcmuIdle
	default:
		return encodeG722Frame(g722Encoder, g722Silence)
	}
}

func (s *webRTCFrameSource) maybeLogDiagnostics(stats *webRTCFrameSourceStats, subscribers int, queuedFrames int) {
	now := time.Now()
	if now.Sub(stats.lastReport) < webrtcDiagnosticsInterval {
		return
	}
	if stats.dropped == 0 && stats.concealed == 0 && stats.idle == 0 {
		stats.lastReport = now
		stats.resetInterval()
		return
	}
	log.Printf("media bridge WebRTC diagnostics feed=%s codec=%s frames=%d real=%d concealed=%d idle=%d subscriber_drops=%d subscribers=%d queue_frames=%d",
		s.key.feedID,
		s.key.codec,
		stats.produced,
		stats.real,
		stats.concealed,
		stats.idle,
		stats.dropped,
		subscribers,
		queuedFrames,
	)
	stats.lastReport = now
	stats.resetInterval()
}

func (s *webRTCFrameSourceStats) record(kind webRTCFrameKind, dropped int) {
	s.produced++
	switch kind {
	case webRTCFrameConcealed:
		s.concealed++
	case webRTCFrameIdle:
		s.idle++
	default:
		s.real++
	}
	if dropped > 0 {
		s.dropped += uint64(dropped)
	}
}

func (s *webRTCFrameSourceStats) resetInterval() {
	s.produced = 0
	s.real = 0
	s.concealed = 0
	s.idle = 0
	s.dropped = 0
}

func (s *webRTCFrameSource) broadcast(frame []byte) (int, int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	dropped := 0
	for subscriber := range s.subs {
		select {
		case subscriber <- frame:
		default:
			select {
			case <-subscriber:
				dropped++
			default:
			}
			select {
			case subscriber <- frame:
			default:
				dropped++
			}
		}
	}
	return dropped, len(s.subs)
}

func (h *MediaHub) streamWebRTCFrames(ctx context.Context, feedID string, codec webRTCAudioCodec, track *webrtc.TrackLocalStaticSample, frames <-chan []byte, unsubscribe func()) {
	defer unsubscribe()
	loggedWrite := false
	for {
		select {
		case <-ctx.Done():
			return
		case frame, ok := <-frames:
			if !ok {
				return
			}
			if len(frame) == 0 {
				continue
			}
			if err := track.WriteSample(media.Sample{Data: append([]byte(nil), frame...), Duration: webrtcFrameDuration}); err != nil {
				return
			}
			if !loggedWrite {
				log.Printf("media bridge WebRTC stream wrote first frame for feed %s codec=%s (%d bytes)", feedID, codec, len(frame))
				loggedWrite = true
			}
		}
	}
}

type frameConcealer struct {
	last       []byte
	repeated   int
	needsPrime bool
}

func (c *frameConcealer) next(queue *[][]byte, head *int, silence func() []byte) []byte {
	frame, _ := c.nextWithKind(queue, head, silence)
	return frame
}

func (c *frameConcealer) nextWithKind(queue *[][]byte, head *int, silence func() []byte) ([]byte, webRTCFrameKind) {
	if c.needsPrime && queuedFrameCount(*queue, *head) < webrtcResumeQueuedFrames {
		return c.fallback(silence)
	}
	if frame, ok := popQueuedFrame(queue, head); ok {
		c.last = append([]byte(nil), frame...)
		c.repeated = 0
		c.needsPrime = false
		return frame, webRTCFrameReal
	}
	c.needsPrime = true
	return c.fallback(silence)
}

func (c *frameConcealer) fallback(silence func() []byte) ([]byte, webRTCFrameKind) {
	if len(c.last) > 0 && c.repeated < webrtcConcealmentFrames {
		c.repeated++
		return append([]byte(nil), c.last...), webRTCFrameConcealed
	}
	return silence(), webRTCFrameIdle
}

func queuedFrameCount(queue [][]byte, head int) int {
	if head < 0 || head >= len(queue) {
		return 0
	}
	return len(queue) - head
}

func popQueuedFrame(queue *[][]byte, head *int) ([]byte, bool) {
	if *head >= len(*queue) {
		*head = 0
		*queue = (*queue)[:0]
		return nil, false
	}
	frame := (*queue)[*head]
	(*queue)[*head] = nil
	*head = *head + 1
	if *head > 32 && *head*2 >= len(*queue) {
		copy(*queue, (*queue)[*head:])
		*queue = (*queue)[:len(*queue)-*head]
		*head = 0
	}
	return frame, true
}

func compactQueuedFrames(queue *[][]byte, head *int) {
	if *head <= 0 {
		return
	}
	if *head >= len(*queue) {
		*head = 0
		*queue = (*queue)[:0]
		return
	}
	copy(*queue, (*queue)[*head:])
	*queue = (*queue)[:len(*queue)-*head]
	*head = 0
}

func opusIdleFrameSamples() []int16 {
	samples := make([]int16, opusFrameSamples*opusEncoderChannels)
	for i := range samples {
		if i%2 == 0 {
			samples[i] = 1
		} else {
			samples[i] = -1
		}
	}
	return samples
}

func appendG722Frames(queue [][]byte, encoder *g722.Encoder, chunk PCMChunk) [][]byte {
	frames := pcm16ToG722Frames(encoder, chunk)
	if len(frames) == 0 {
		return queue
	}
	queue = append(queue, frames...)
	if len(queue) > webrtcMaxQueuedFrames {
		queue = queue[len(queue)-webrtcMaxQueuedFrames:]
	}
	return queue
}

func appendOpusFrames(queue [][]byte, encoder opusFrameEncoder, chunk PCMChunk) [][]byte {
	frames := pcm16ToOpusFrames(encoder, chunk)
	if len(frames) == 0 {
		return queue
	}
	queue = append(queue, frames...)
	if len(queue) > webrtcMaxQueuedFrames {
		queue = queue[len(queue)-webrtcMaxQueuedFrames:]
	}
	return queue
}

func pcm16ToOpusFrames(encoder opusFrameEncoder, chunk PCMChunk) [][]byte {
	samples := resamplePCM16ToMono(chunk, opusSampleRate)
	if len(samples) == 0 {
		return nil
	}
	frameCount := (len(samples) + opusFrameSamples - 1) / opusFrameSamples
	frames := make([][]byte, 0, frameCount)
	for frameIndex := 0; frameIndex < frameCount; frameIndex++ {
		start := frameIndex * opusFrameSamples
		end := start + opusFrameSamples
		frameSamples := make([]int16, opusFrameSamples*opusEncoderChannels)
		if start < len(samples) {
			monoEnd := min(end, len(samples))
			for sourceIndex, sample := range samples[start:monoEnd] {
				out := sourceIndex * opusEncoderChannels
				for channel := 0; channel < opusEncoderChannels; channel++ {
					frameSamples[out+channel] = sample
				}
			}
		}
		encoded, err := encoder.Encode(frameSamples)
		if err != nil || len(encoded) == 0 {
			continue
		}
		frames = append(frames, encoded)
	}
	return frames
}

func pcm16ToG722(chunk PCMChunk) []byte {
	encoder := g722.NewEncoder(g722.Rate64000, 0)
	frames := pcm16ToG722Frames(encoder, chunk)
	if len(frames) == 0 {
		return encodeG722Frame(encoder, make([]int16, g722FrameSamples))
	}
	return frames[0]
}

func pcm16ToG722Frames(encoder *g722.Encoder, chunk PCMChunk) [][]byte {
	samples := resamplePCM16ToMono(chunk, g722SampleRate)
	if len(samples) == 0 {
		return nil
	}
	frameCount := (len(samples) + g722FrameSamples - 1) / g722FrameSamples
	frames := make([][]byte, 0, frameCount)
	for frameIndex := 0; frameIndex < frameCount; frameIndex++ {
		start := frameIndex * g722FrameSamples
		end := start + g722FrameSamples
		frameSamples := make([]int16, g722FrameSamples)
		if start < len(samples) {
			copy(frameSamples, samples[start:min(end, len(samples))])
		}
		frames = append(frames, encodeG722Frame(encoder, frameSamples))
	}
	return frames
}

func encodeG722Frame(encoder *g722.Encoder, samples []int16) []byte {
	out := make([]byte, len(samples))
	n := encoder.Encode(out, samples)
	return out[:n]
}

func appendPCMUFrames(queue [][]byte, chunk PCMChunk) [][]byte {
	frames := pcm16ToPCMUFrames(chunk)
	if len(frames) == 0 {
		return queue
	}
	queue = append(queue, frames...)
	if len(queue) > webrtcMaxQueuedFrames {
		queue = queue[len(queue)-webrtcMaxQueuedFrames:]
	}
	return queue
}

func pcm16ToPCMU(chunk PCMChunk) []byte {
	frames := pcm16ToPCMUFrames(chunk)
	if len(frames) == 0 {
		return pcmuSilence()
	}
	return frames[0]
}

func pcm16ToPCMUFrames(chunk PCMChunk) [][]byte {
	samples := resamplePCM16ToMono(chunk, pcmuSampleRate)
	if len(samples) == 0 {
		return nil
	}
	frameCount := (len(samples) + pcmuFrameSamples - 1) / pcmuFrameSamples
	frames := make([][]byte, 0, frameCount)
	for frameIndex := 0; frameIndex < frameCount; frameIndex++ {
		start := frameIndex * pcmuFrameSamples
		end := min(start+pcmuFrameSamples, len(samples))
		out := make([]byte, pcmuFrameSamples)
		for i := range out {
			sampleIndex := start + i
			if sampleIndex >= end {
				out[i] = 0xff
			} else {
				out[i] = linearToMuLaw(samples[sampleIndex])
			}
		}
		frames = append(frames, out)
	}
	return frames
}

func resamplePCM16ToMono(chunk PCMChunk, outputSampleRate int) []int16 {
	if len(chunk.Data) == 0 {
		return nil
	}
	if outputSampleRate <= 0 {
		outputSampleRate = opusSampleRate
	}
	sampleRate := chunk.SampleRate
	if sampleRate <= 0 {
		sampleRate = opusSampleRate
	}
	channels := chunk.Channels
	if channels <= 0 {
		channels = 1
	}
	bytesPerFrame := channels * 2
	sourceFrames := len(chunk.Data) / bytesPerFrame
	if sourceFrames <= 0 {
		return nil
	}
	outputSamples := int((int64(sourceFrames)*int64(outputSampleRate) + int64(sampleRate/2)) / int64(sampleRate))
	if outputSamples <= 0 {
		return nil
	}
	out := make([]int16, outputSamples)
	for outputIndex := range out {
		sourceNumerator := int64(outputIndex) * int64(sampleRate)
		left := int(sourceNumerator / int64(outputSampleRate))
		if left >= sourceFrames {
			left = sourceFrames - 1
		}
		right := min(left+1, sourceFrames-1)
		fraction := float64(sourceNumerator%int64(outputSampleRate)) / float64(outputSampleRate)
		leftSample := monoSampleAt(chunk.Data, left, bytesPerFrame, channels)
		rightSample := monoSampleAt(chunk.Data, right, bytesPerFrame, channels)
		value := float64(leftSample) + (float64(rightSample)-float64(leftSample))*fraction
		out[outputIndex] = int16(clampPCMInt(int(value), int(i16Min), int(i16Max)))
	}
	return out
}

func linearToMuLaw(sample int16) byte {
	const bias = 0x84
	const clip = 32635
	pcm := int(sample)
	sign := 0
	if pcm < 0 {
		sign = 0x80
		pcm = -pcm
	}
	if pcm > clip {
		pcm = clip
	}
	pcm += bias
	exponent := 7
	for mask := 0x4000; exponent > 0 && pcm&mask == 0; exponent-- {
		mask >>= 1
	}
	mantissa := (pcm >> (exponent + 3)) & 0x0f
	return ^byte(sign | (exponent << 4) | mantissa)
}

func pcmuSilence() []byte {
	frame := make([]byte, pcmuFrameSamples)
	for i := range frame {
		frame[i] = 0xff
	}
	return frame
}

func monoSampleAt(data []byte, frame int, bytesPerFrame int, channels int) int {
	offset := frame * bytesPerFrame
	sum := 0
	for channel := 0; channel < channels; channel++ {
		sampleOffset := offset + channel*2
		sum += int(int16(binary.LittleEndian.Uint16(data[sampleOffset : sampleOffset+2])))
	}
	return sum / channels
}

func clampPCMInt(value int, minValue int, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

const (
	i16Min = -32768
	i16Max = 32767
)

func drainRTCP(sender *webrtc.RTPSender) {
	buffer := make([]byte, 1500)
	for {
		if _, _, err := sender.Read(buffer); err != nil {
			return
		}
	}
}

func mediaSafeID(value string) string {
	var builder strings.Builder
	for _, ch := range value {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '-' || ch == '_' {
			builder.WriteRune(ch)
		}
	}
	if builder.Len() == 0 {
		return "feed"
	}
	return builder.String()
}

func sleepMedia(ctx context.Context, duration time.Duration) {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}
