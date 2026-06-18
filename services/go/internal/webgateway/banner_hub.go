package webgateway

import (
	"bufio"
	"context"
	"encoding/json"
	"log"
	"net"
	"strings"
	"sync"
	"time"
)

const bannerBridgeReconnectDelay = 750 * time.Millisecond

type BannerHub struct {
	configPath string
	addr       string
	mu         sync.Mutex
	onAir      map[string]bannerOnAirAlert
}

type bannerOnAirAlert struct {
	FeedID    string
	AlertID   string
	QueueID   string
	Event     string
	Header    string
	ExpiresAt time.Time
	UpdatedAt time.Time
}

func NewBannerHub(configPath string, addr string) *BannerHub {
	hub := &BannerHub{
		configPath: configPath,
		addr:       strings.TrimSpace(addr),
		onAir:      map[string]bannerOnAirAlert{},
	}
	if hub.addr != "" {
		go hub.run(context.Background())
	}
	return hub
}

func (h *BannerHub) run(ctx context.Context) {
	for {
		if ctx.Err() != nil {
			return
		}
		conn, err := net.DialTimeout("tcp", h.addr, 3*time.Second)
		if err != nil {
			sleepBanner(ctx, bannerBridgeReconnectDelay)
			continue
		}
		h.readBridge(ctx, conn)
		_ = conn.Close()
		sleepBanner(ctx, bannerBridgeReconnectDelay)
	}
}

func (h *BannerHub) readBridge(ctx context.Context, conn net.Conn) {
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		if ctx.Err() != nil {
			return
		}
		h.handleEvent(scanner.Bytes(), time.Now().UTC())
	}
}

func (h *BannerHub) handleEvent(raw []byte, now time.Time) {
	var event struct {
		Type    string   `json:"type"`
		FeedID  string   `json:"feed_id"`
		FeedIDs []string `json:"feed_ids"`
		QueueID string   `json:"queue_id"`
		Header  string   `json:"header"`
		Event   string   `json:"event"`
		Data    struct {
			FeedID  string   `json:"feed_id"`
			FeedIDs []string `json:"feed_ids"`
			QueueID string   `json:"queue_id"`
			Header  string   `json:"header"`
			Event   string   `json:"event"`
			AlertID string   `json:"alert_id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(raw, &event); err != nil {
		return
	}
	eventType := strings.TrimSpace(event.Type)
	if !strings.HasPrefix(eventType, "alert.playout.") && eventType != "playout.interrupted" {
		return
	}
	feedIDs := eventFeedIDs(event.FeedID, event.FeedIDs, event.Data.FeedID, event.Data.FeedIDs)
	queueID := fallbackString(event.Data.QueueID, event.QueueID)
	if len(feedIDs) == 0 && strings.TrimSpace(event.Data.FeedID) != "" {
		feedIDs = []string{strings.TrimSpace(event.Data.FeedID)}
	}
	if len(feedIDs) == 0 {
		return
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	for _, feedID := range feedIDs {
		switch eventType {
		case "alert.playout.started":
			item := h.queueItem(queueID, feedID)
			alertID := fallbackString(event.Data.AlertID, item.AlertID, alertIDFromQueueID(queueID, feedID))
			if alertID == "" {
				alertID = fallbackString(queueID, event.Data.Event, event.Event)
			}
			h.onAir[feedID] = bannerOnAirAlert{
				FeedID:    feedID,
				AlertID:   alertID,
				QueueID:   queueID,
				Event:     fallbackString(event.Data.Event, event.Event, item.Event),
				Header:    fallbackString(event.Data.Header, event.Header, item.Header),
				ExpiresAt: now.Add(30 * time.Minute),
				UpdatedAt: now,
			}
		case "alert.playout.completed":
			item := h.queueItem(queueID, feedID)
			active := h.onAir[feedID]
			if active.QueueID != "" && active.QueueID != queueID && active.AlertID != "" && item.AlertID != active.AlertID {
				continue
			}
			if bannerQueueItemEndsAlert(item, queueID) {
				delete(h.onAir, feedID)
			} else if active.AlertID != "" {
				active.ExpiresAt = now.Add(15 * time.Second)
				active.UpdatedAt = now
				h.onAir[feedID] = active
			}
		case "playout.interrupted":
			delete(h.onAir, feedID)
		}
	}
}

func (h *BannerHub) queueItem(queueID string, feedID string) sameQueueItem {
	items, err := loadAlertQueueItems(h.configPath)
	if err != nil {
		return sameQueueItem{}
	}
	for _, item := range items {
		if strings.TrimSpace(item.ID) == strings.TrimSpace(queueID) && itemTargetsFeed(item, feedID) {
			return item
		}
	}
	return sameQueueItem{}
}

func (h *BannerHub) Active(feedID string, now time.Time) []bannerOnAirAlert {
	if h == nil {
		return nil
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	out := []bannerOnAirAlert{}
	for key, active := range h.onAir {
		if !active.ExpiresAt.IsZero() && now.After(active.ExpiresAt) {
			delete(h.onAir, key)
			continue
		}
		if strings.TrimSpace(feedID) != "" && key != feedID {
			continue
		}
		out = append(out, active)
	}
	return out
}

func eventFeedIDs(values ...any) []string {
	out := []string{}
	seen := map[string]struct{}{}
	add := func(value string) {
		value = strings.TrimSpace(value)
		if value == "" {
			return
		}
		if _, ok := seen[value]; ok {
			return
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	for _, value := range values {
		switch typed := value.(type) {
		case string:
			add(typed)
		case []string:
			for _, item := range typed {
				add(item)
			}
		}
	}
	return out
}

func alertIDFromQueueID(queueID string, feedID string) string {
	queueID = strings.TrimSpace(queueID)
	feedID = strings.TrimSpace(feedID)
	if queueID == "" || feedID == "" {
		return ""
	}
	for _, prefix := range []string{"000_" + feedID + "_", "001_" + feedID + "_", "002_" + feedID + "_"} {
		if strings.HasPrefix(queueID, prefix) {
			value := strings.TrimPrefix(queueID, prefix)
			for _, suffix := range []string{"_same_header", "_same_eom", "_same_full", "_same", "_cap"} {
				value = strings.TrimSuffix(value, suffix)
			}
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func bannerQueueItemEndsAlert(item sameQueueItem, queueID string) bool {
	itemType := strings.ToLower(strings.TrimSpace(item.Type))
	queueID = strings.ToLower(strings.TrimSpace(queueID))
	return strings.Contains(itemType, "eom") ||
		strings.Contains(queueID, "same_eom") ||
		strings.Contains(itemType, "full") ||
		strings.Contains(queueID, "same_full")
}

func sleepBanner(ctx context.Context, duration time.Duration) {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

func logBannerEventDebug(message string) {
	log.Print(message)
}
