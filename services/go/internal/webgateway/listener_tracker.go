package webgateway

import (
	"context"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

type FeedListenerStats struct {
	Current int
	Peak    int
	PeakAt  time.Time
}

type ListenerTracker struct {
	mu    sync.Mutex
	feeds map[string]*listenerFeedState
}

type listenerFeedState struct {
	active map[string]time.Time
	peak   int
	peakAt time.Time
}

func NewListenerTracker() *ListenerTracker {
	return &ListenerTracker{feeds: map[string]*listenerFeedState{}}
}

func (t *ListenerTracker) TryAcquire(feedID string, clientID string) (func(), FeedListenerStats, bool) {
	if t == nil {
		return func() {}, FeedListenerStats{}, true
	}
	feedID = strings.TrimSpace(feedID)
	clientID = strings.TrimSpace(clientID)
	if feedID == "" {
		return func() {}, FeedListenerStats{}, true
	}
	if clientID == "" {
		clientID = "unknown"
	}
	now := time.Now().UTC()
	t.mu.Lock()
	defer t.mu.Unlock()
	state := t.feeds[feedID]
	if state == nil {
		state = &listenerFeedState{active: map[string]time.Time{}}
		t.feeds[feedID] = state
	}
	if _, exists := state.active[clientID]; exists {
		return nil, listenerStatsLocked(state), false
	}
	state.active[clientID] = now
	if current := len(state.active); current > state.peak {
		state.peak = current
		state.peakAt = now
	}
	var once sync.Once
	release := func() {
		once.Do(func() {
			t.mu.Lock()
			defer t.mu.Unlock()
			current := t.feeds[feedID]
			if current == nil {
				return
			}
			delete(current.active, clientID)
		})
	}
	return release, listenerStatsLocked(state), true
}

func (t *ListenerTracker) Snapshot() map[string]FeedListenerStats {
	return t.SnapshotWithExternal(nil)
}

func (t *ListenerTracker) SnapshotWithExternal(external map[string][]string) map[string]FeedListenerStats {
	out := map[string]FeedListenerStats{}
	if t == nil {
		return out
	}
	now := time.Now().UTC()
	t.mu.Lock()
	defer t.mu.Unlock()
	for feedID := range external {
		feedID = strings.TrimSpace(feedID)
		if feedID == "" {
			continue
		}
		if t.feeds[feedID] == nil {
			t.feeds[feedID] = &listenerFeedState{active: map[string]time.Time{}}
		}
	}
	for feedID, state := range t.feeds {
		current := len(state.active)
		seen := map[string]struct{}{}
		for _, clientID := range external[feedID] {
			clientID = strings.TrimSpace(clientID)
			if clientID == "" {
				clientID = "unknown"
			}
			if _, exists := state.active[clientID]; exists {
				continue
			}
			if _, exists := seen[clientID]; exists {
				continue
			}
			seen[clientID] = struct{}{}
			current++
		}
		if current > state.peak {
			state.peak = current
			state.peakAt = now
		}
		out[feedID] = listenerStatsLocked(state)
		stats := out[feedID]
		stats.Current = current
		out[feedID] = stats
	}
	return out
}

func listenerStatsLocked(state *listenerFeedState) FeedListenerStats {
	if state == nil {
		return FeedListenerStats{}
	}
	return FeedListenerStats{
		Current: len(state.active),
		Peak:    state.peak,
		PeakAt:  state.peakAt,
	}
}

func listenerStatsPayload(stats FeedListenerStats) map[string]any {
	payload := map[string]any{
		"current": stats.Current,
		"peak":    stats.Peak,
	}
	if !stats.PeakAt.IsZero() {
		payload["peak_at"] = stats.PeakAt.UTC().Format(time.RFC3339)
	}
	return payload
}

func listenerStatsForFeed(stats map[string]FeedListenerStats, feedID string) FeedListenerStats {
	if stats == nil {
		return FeedListenerStats{}
	}
	return stats[strings.TrimSpace(feedID)]
}

func listenerClientID(request *http.Request) string {
	if ip := clientIPForMediaRequest(request); ip != "" {
		return ip
	}
	if request != nil {
		if ip := addrIP(request.RemoteAddr); ip != "" {
			return ip
		}
	}
	return "unknown"
}

func (s *Server) listenerSnapshot(ctx context.Context) map[string]FeedListenerStats {
	if s == nil || s.listeners == nil {
		return map[string]FeedListenerStats{}
	}
	external := map[string][]string{}
	if health, ok := mediaServiceHealth(ctx, s.config); ok {
		external = mediaServiceWebRTCListenerClients(health)
	}
	return s.listeners.SnapshotWithExternal(external)
}

func (s *Server) acquireFeedListener(writer http.ResponseWriter, request *http.Request, feedID string) (func(), bool) {
	if s == nil || s.listeners == nil {
		return func() {}, true
	}
	release, _, ok := s.listeners.TryAcquire(feedID, listenerClientID(request))
	if !ok {
		http.Error(writer, "listener already active for this IP and feed", http.StatusTooManyRequests)
		return nil, false
	}
	return release, true
}

func mediaServiceWebRTCListenerClients(health map[string]any) map[string][]string {
	out := map[string][]string{}
	peers, _ := health["webrtc_peers"].([]any)
	for _, item := range peers {
		peer, _ := item.(map[string]any)
		if peer == nil {
			continue
		}
		feedID := strings.TrimSpace(stringValue(peer, "feed_id"))
		if feedID == "" {
			continue
		}
		clientID := strings.TrimSpace(stringValue(peer, "client_id"))
		if clientID == "" {
			switch value := peer["id"].(type) {
			case string:
				clientID = value
			case float64:
				clientID = strconv.FormatInt(int64(value), 10)
			}
		}
		if clientID == "" {
			clientID = "unknown"
		}
		out[feedID] = append(out[feedID], clientID)
	}
	return out
}
