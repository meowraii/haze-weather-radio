package webgateway

import (
	"encoding/xml"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

const bannerStatePublishInterval = 750 * time.Millisecond
const bannerStateFeedRefreshInterval = 15 * time.Second

func (s *Server) startBannerStatePublisher(bridgeAddr string) {
	bridgeAddr = strings.TrimSpace(bridgeAddr)
	if bridgeAddr == "" {
		return
	}
	publisher := events.NewHostBridgePublisher(bridgeAddr)
	go func() {
		defer publisher.Close()
		ticker := time.NewTicker(bannerStatePublishInterval)
		defer ticker.Stop()
		last := map[string]string{}
		feedIDs := []string{"*"}
		nextFeedRefresh := time.Time{}
		for {
			now := time.Now()
			if now.After(nextFeedRefresh) {
				feedIDs = s.bannerStateFeedIDs()
				nextFeedRefresh = now.Add(bannerStateFeedRefreshInterval)
			}
			for _, feedID := range feedIDs {
				payload := buildBannerPayload(s.configPath, feedID, s.bannerHub)
				if last[feedID] == payload.Signature {
					continue
				}
				last[feedID] = payload.Signature
				_ = publisher.Publish(events.Event{
					Type:    "banner.state.updated",
					Source:  "haze-web",
					Subject: feedID,
					Data:    bannerPayloadMap(payload),
				})
			}
			<-ticker.C
		}
	}()
}

func (s *Server) bannerStateFeedIDs() []string {
	out := []string{"*"}
	seen := map[string]struct{}{"*": {}}
	add := func(feedID string) {
		feedID = strings.TrimSpace(feedID)
		if feedID == "" {
			return
		}
		if _, ok := seen[feedID]; ok {
			return
		}
		seen[feedID] = struct{}{}
		out = append(out, feedID)
	}
	for _, feedID := range bannerFeedIDsFromConfig(s.configPath, time.Now().UTC()) {
		add(feedID)
	}
	for _, feedID := range loadCgenFeedIDs(s.configPath) {
		add(feedID)
	}
	return out
}

func bannerPayloadMap(payload bannerPayload) map[string]any {
	return map[string]any{
		"active":           payload.Active,
		"signature":        payload.Signature,
		"feed_id":          payload.FeedID,
		"feed_name":        payload.FeedName,
		"generated_at":     payload.GeneratedAt,
		"primary_color":    payload.PrimaryColor,
		"primary_gradient": payload.PrimaryGradient,
		"alerts":           payload.Alerts,
	}
}

func loadCgenFeedIDs(configPath string) []string {
	type cgenFeed struct {
		ID string `xml:"id,attr"`
	}
	type cgenConfig struct {
		Feeds []cgenFeed `xml:"feed"`
	}
	base := filepath.Dir(filepath.Clean(configPath))
	path := filepath.Join(base, "managed", "configs", "cgen.xml")
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var parsed cgenConfig
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil
	}
	out := make([]string, 0, len(parsed.Feeds))
	for _, feed := range parsed.Feeds {
		if strings.TrimSpace(feed.ID) != "" {
			out = append(out, strings.TrimSpace(feed.ID))
		}
	}
	return out
}
