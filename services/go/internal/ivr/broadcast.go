package ivr

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gotranspile/g722"
)

const broadcastFrameDuration = 20 * time.Millisecond
const broadcastMaxDuration = 10 * time.Minute

type broadcastPCMChunk struct {
	FeedID     string
	SampleRate int
	Channels   int
	Data       []byte
}

type broadcastHub struct {
	mu          sync.Mutex
	subscribers map[string]map[chan broadcastPCMChunk]struct{}
	last        map[string]broadcastPCMChunk
	lastAt      map[string]time.Time
	seenLogged  map[string]bool
}

func newBroadcastHub() *broadcastHub {
	return &broadcastHub{
		subscribers: map[string]map[chan broadcastPCMChunk]struct{}{},
		last:        map[string]broadcastPCMChunk{},
		lastAt:      map[string]time.Time{},
		seenLogged:  map[string]bool{},
	}
}

func (h *broadcastHub) run(ctx context.Context, events <-chan map[string]any) {
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-events:
			if !ok {
				return
			}
			chunk, ok := decodeBroadcastPCMEvent(event)
			if ok {
				h.publish(chunk)
			}
		}
	}
}

func (h *broadcastHub) Subscribe(feedID string) (<-chan broadcastPCMChunk, func()) {
	feedID = strings.TrimSpace(feedID)
	ch := make(chan broadcastPCMChunk, 8)
	h.mu.Lock()
	if h.subscribers[feedID] == nil {
		h.subscribers[feedID] = map[chan broadcastPCMChunk]struct{}{}
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

func (h *broadcastHub) HasRecent(feedID string, maxAge time.Duration) bool {
	if h == nil {
		return false
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	lastAt, ok := h.lastAt[strings.TrimSpace(feedID)]
	return ok && time.Since(lastAt) <= maxAge
}

func (h *broadcastHub) publish(chunk broadcastPCMChunk) {
	now := time.Now()
	h.mu.Lock()
	h.last[chunk.FeedID] = chunk
	h.lastAt[chunk.FeedID] = now
	if !h.seenLogged[chunk.FeedID] {
		h.seenLogged[chunk.FeedID] = true
		log.Printf("IVR live broadcast receiving PCM for feed %s (%d Hz, %d channel)", chunk.FeedID, chunk.SampleRate, chunk.Channels)
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

func decodeBroadcastPCMEvent(event map[string]any) (broadcastPCMChunk, bool) {
	if stringAt(event, "type") != "playout.pcm" {
		return broadcastPCMChunk{}, false
	}
	data := mapAt(event, "data")
	feedID := firstNonBlank(stringAt(data, "feed_id"), stringAt(event, "feed_id"))
	pcmText := stringAt(data, "pcm")
	if feedID == "" || pcmText == "" {
		return broadcastPCMChunk{}, false
	}
	pcm, err := base64.StdEncoding.DecodeString(pcmText)
	if err != nil {
		return broadcastPCMChunk{}, false
	}
	sampleRate := intFromAny(data["sample_rate"], 48000)
	channels := intFromAny(data["channels"], 1)
	if sampleRate <= 0 {
		sampleRate = 48000
	}
	if channels <= 0 {
		channels = 1
	}
	return broadcastPCMChunk{
		FeedID:     feedID,
		SampleRate: sampleRate,
		Channels:   channels,
		Data:       pcm,
	}, true
}

func intFromAny(value any, fallback int) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case string:
		if parsed, err := strconv.Atoi(strings.TrimSpace(typed)); err == nil {
			return parsed
		}
	}
	return fallback
}

func pcmChunkToCodecFrames(codec sipAudioCodec, encoder *g722.Encoder, chunk broadcastPCMChunk) [][]byte {
	if codec == sipAudioCodecG722 {
		return pcmChunkToG722Frames(encoder, chunk)
	}
	return pcmChunkToPCMUFrames(chunk)
}

func pcmChunkToPCMUFrames(chunk broadcastPCMChunk) [][]byte {
	samples := pcmChunkMonoSamples(chunk, sipPCMUSampleRate)
	if len(samples) == 0 {
		return nil
	}
	frameCount := (len(samples) + sipPacketSamples - 1) / sipPacketSamples
	frames := make([][]byte, 0, frameCount)
	for frameIndex := 0; frameIndex < frameCount; frameIndex++ {
		start := frameIndex * sipPacketSamples
		end := minInt(start+sipPacketSamples, len(samples))
		frame := make([]byte, sipPacketSamples)
		for i := range frame {
			sampleIndex := start + i
			if sampleIndex >= end {
				frame[i] = 0xff
				continue
			}
			frame[i] = linearToULaw(samples[sampleIndex])
		}
		frames = append(frames, frame)
	}
	return frames
}

func pcmChunkToG722Frames(encoder *g722.Encoder, chunk broadcastPCMChunk) [][]byte {
	samples := pcmChunkMonoSamples(chunk, sipG722SampleRate)
	if len(samples) == 0 {
		return nil
	}
	frameCount := (len(samples) + sipG722FrameSamples - 1) / sipG722FrameSamples
	frames := make([][]byte, 0, frameCount)
	for frameIndex := 0; frameIndex < frameCount; frameIndex++ {
		start := frameIndex * sipG722FrameSamples
		end := minInt(start+sipG722FrameSamples, len(samples))
		frameSamples := make([]int16, sipG722FrameSamples)
		if start < len(samples) {
			copy(frameSamples, samples[start:end])
		}
		frames = append(frames, encodeG722Samples(encoder, frameSamples))
	}
	return frames
}

func pcmChunkMonoSamples(chunk broadcastPCMChunk, outputRate int) []int16 {
	if len(chunk.Data) == 0 {
		return nil
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
	mono := make([]int16, sourceFrames)
	for frame := 0; frame < sourceFrames; frame++ {
		offset := frame * bytesPerFrame
		sum := 0
		for channel := 0; channel < channels; channel++ {
			sampleOffset := offset + channel*2
			sum += int(int16(binary.LittleEndian.Uint16(chunk.Data[sampleOffset : sampleOffset+2])))
		}
		mono[frame] = int16(sum / channels)
	}
	return resampleLinear(mono, chunk.SampleRate, outputRate)
}

func appendBroadcastFrames(queue [][]byte, frames [][]byte) [][]byte {
	if len(frames) == 0 {
		return queue
	}
	queue = append(queue, frames...)
	const maxQueuedFrames = 100
	if len(queue) > maxQueuedFrames {
		queue = queue[len(queue)-maxQueuedFrames:]
	}
	return queue
}

func popBroadcastFrame(queue *[][]byte, head *int) ([]byte, bool) {
	if *head >= len(*queue) {
		*head = 0
		*queue = (*queue)[:0]
		return nil, false
	}
	frame := (*queue)[*head]
	(*queue)[*head] = nil
	*head++
	if *head > 32 && *head*2 >= len(*queue) {
		copy(*queue, (*queue)[*head:])
		*queue = (*queue)[:len(*queue)-*head]
		*head = 0
	}
	return frame, true
}

func minInt(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

func (c *sipCall) playLiveBroadcast(feedID string) bool {
	feedID = strings.TrimSpace(feedID)
	if feedID == "" || c.service == nil || c.service.broadcast == nil {
		return false
	}
	if !c.service.broadcast.HasRecent(feedID, 5*time.Second) {
		log.Printf("IVR live broadcast has no recent PCM for feed %s; sending silence until playout publishes audio", feedID)
	}
	updates, unsubscribe := c.service.broadcast.Subscribe(feedID)
	defer unsubscribe()

	ticker := time.NewTicker(broadcastFrameDuration)
	defer ticker.Stop()
	timer := time.NewTimer(broadcastMaxDuration)
	defer timer.Stop()
	silence := c.audioCodec.silenceFrame()
	frameQueue := make([][]byte, 0, 32)
	frameHead := 0
	encoder := g722.NewEncoder(g722.Rate64000, 0)
	for {
		select {
		case <-c.ctx.Done():
			return false
		case <-timer.C:
			return true
		case digit := <-c.digits:
			if digit == "#" {
				return true
			}
		case <-ticker.C:
			if digit, ok := c.pendingInterruptDigit(digitInterruptPound); ok && digit == "#" {
				return true
			}
			for drained := 0; drained < cap(updates); drained++ {
				select {
				case chunk, ok := <-updates:
					if !ok {
						return false
					}
					frameQueue = appendBroadcastFrames(frameQueue, pcmChunkToCodecFrames(c.audioCodec, encoder, chunk))
				default:
					drained = cap(updates)
				}
			}
			frame := silence
			if next, ok := popBroadcastFrame(&frameQueue, &frameHead); ok {
				frame = next
			}
			c.sendRTP(frame)
		}
	}
}
