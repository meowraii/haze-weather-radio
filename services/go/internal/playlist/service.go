package playlist

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/alerttext"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

const serviceID = "haze-playlist"
const routineRetryDelay = 30 * time.Second
const startupPrimerDelay = 2 * time.Second
const pendingReplayInterval = 2 * time.Second
const cachedRoutineFallbackMaxAge = 15 * time.Minute
const cachedStartupFallbackMaxAge = 30 * time.Minute

var errSystemShutdown = errors.New("system shutdown requested")

func Run(ctx context.Context, options Options) error {
	if strings.TrimSpace(options.ConfigPath) == "" {
		options.ConfigPath = "config.yaml"
	}
	if options.Tick <= 0 {
		options.Tick = 500 * time.Millisecond
	}
	if options.Lookahead <= 0 {
		options.Lookahead = 2 * time.Minute
	}
	for {
		cfg, err := loadConfig(options.ConfigPath, options.OutDir)
		if err != nil {
			return err
		}
		store, err := datastore.Open(ctx, cfg.Root.Storage, cfg.BaseDir)
		if err != nil {
			return err
		}
		cfg.Store = store
		bridge, err := connectBridge(ctx, options.BridgeAddr)
		if err != nil {
			store.Close()
			if ctx.Err() != nil {
				return ctx.Err()
			}
			log.Printf("playlist waiting for event bridge: %v", err)
			sleepOrDone(ctx, time.Second)
			continue
		}
		service := newService(cfg, bridge, options)
		err = service.runConnected(ctx)
		_ = bridge.Close()
		store.Close()
		if errors.Is(err, errSystemShutdown) {
			return nil
		}
		if ctx.Err() != nil {
			return nil
		}
		log.Printf("playlist event bridge disconnected: %v", err)
		sleepOrDone(ctx, time.Second)
	}
}

type Service struct {
	cfg     loadedConfig
	bridge  *bridgeClient
	options Options
	feeds   map[string]*feedPlanner
}

func newService(cfg loadedConfig, bridge *bridgeClient, options Options) *Service {
	feeds := map[string]*feedPlanner{}
	for _, feed := range cfg.enabledFeeds() {
		feeds[feed.ID] = newFeedPlanner(cfg, bridge, feed)
	}
	return &Service{cfg: cfg, bridge: bridge, options: options, feeds: feeds}
}

func (s *Service) runConnected(ctx context.Context) error {
	_ = s.bridge.Publish(map[string]any{
		"type":   "service.ready",
		"source": serviceID,
		"data": map[string]any{
			"service": serviceID,
			"feeds":   len(s.feeds),
		},
	})
	ticker := time.NewTicker(s.options.Tick)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case event, ok := <-s.bridge.Events():
			if !ok {
				return fmt.Errorf("bridge event stream closed")
			}
			if stringAt(event, "type") == "system.shutdown" {
				return errSystemShutdown
			}
			s.handleEvent(ctx, event)
		case now := <-ticker.C:
			for _, feed := range s.feeds {
				feed.tick(ctx, now, s.options.Lookahead)
			}
		}
	}
}

func (s *Service) handleEvent(ctx context.Context, event map[string]any) {
	switch stringAt(event, "type") {
	case "playlist.control":
		data := mapAt(event, "data")
		feedID := firstText(event, data, "feed_id")
		action := firstText(event, data, "action")
		for _, planner := range s.matchFeeds(feedID) {
			planner.applyControl(action)
		}
	case "playlist.insert":
		data := mapAt(event, "data")
		feedID := firstText(event, data, "feed_id")
		for _, planner := range s.matchFeeds(feedID) {
			planner.insert(ctx, data)
		}
	case "cap.alert.broadcast.requested":
		data := mapAt(event, "data")
		for _, planner := range s.matchFeedsFromEvent(event) {
			planner.queuePriorityAlert(ctx, data)
		}
	case "cap.alert.cancelled":
		data := mapAt(event, "data")
		for _, planner := range s.matchFeedsFromEvent(event) {
			planner.cancelPriorityAlerts(data)
		}
	case "playout.started":
		data := mapAt(event, "data")
		feedID := firstText(event, data, "feed_id")
		if feedID == "" {
			feedID = stringAt(event, "feed_id")
		}
		if planner := s.feeds[feedID]; planner != nil {
			planner.markStarted(firstText(event, data, "queue_id"))
		}
	case "playout.interrupted":
		data := mapAt(event, "data")
		feedID := firstText(event, data, "feed_id")
		if feedID == "" {
			feedID = stringAt(event, "feed_id")
		}
		if planner := s.feeds[feedID]; planner != nil {
			planner.markInterrupted(firstText(event, data, "queue_id"))
		}
	case "playout.completed":
		data := mapAt(event, "data")
		feedID := firstText(event, data, "feed_id")
		if feedID == "" {
			feedID = stringAt(event, "feed_id")
		}
		if planner := s.feeds[feedID]; planner != nil {
			planner.markCompleted(firstText(event, data, "queue_id"))
		}
	case "alert.playout.started":
		for _, planner := range s.matchFeedsFromEvent(event) {
			planner.markPriorityStarted()
		}
	case "alert.playout.completed":
		for _, planner := range s.matchFeedsFromEvent(event) {
			planner.markPriorityCompleted()
		}
	case "service.ready":
		if !isPlayoutReadyEvent(event) {
			return
		}
		for _, planner := range s.feeds {
			planner.replayPendingItems()
		}
	}
}

func (s *Service) matchFeeds(feedID string) []*feedPlanner {
	feedID = strings.TrimSpace(feedID)
	if feedID == "" || feedID == "*" {
		out := make([]*feedPlanner, 0, len(s.feeds))
		for _, planner := range s.feeds {
			out = append(out, planner)
		}
		return out
	}
	if planner := s.feeds[feedID]; planner != nil {
		return []*feedPlanner{planner}
	}
	return nil
}

func (s *Service) matchFeedsFromEvent(event map[string]any) []*feedPlanner {
	data := mapAt(event, "data")
	out := []*feedPlanner{}
	seen := map[string]struct{}{}
	add := func(feedID string) {
		for _, planner := range s.matchFeeds(feedID) {
			if _, ok := seen[planner.feed.ID]; ok {
				continue
			}
			seen[planner.feed.ID] = struct{}{}
			out = append(out, planner)
		}
	}
	if feedID := firstText(event, data, "feed_id"); feedID != "" {
		add(feedID)
	}
	for _, feedID := range stringListAny(firstValue(event, data, "feed_ids")) {
		add(feedID)
	}
	return out
}

type feedPlanner struct {
	cfg                  loadedConfig
	bridge               *bridgeClient
	feed                 feedXML
	mode                 string
	modeBeforePriority   string
	priorityActive       int
	startupPrimerAt      time.Time
	startupPrimerPending bool
	pendingAfterCurrent  string
	cursor               int
	nextRoutineRetryAt   time.Time
	lastPendingReplayAt  time.Time
	queue                []playlistItem
	current              *playlistItem
	lastFixed            map[string]time.Time
	lastError            string
}

type playlistItem struct {
	QueueID           string `json:"queue_id"`
	FeedID            string `json:"feed_id"`
	Kind              string `json:"kind"`
	PackageID         string `json:"package_id,omitempty"`
	Title             string `json:"title"`
	AudioPath         string `json:"audio_path"`
	DurationMS        int64  `json:"duration_ms"`
	QueuedAt          string `json:"queued_at"`
	TargetStartAt     string `json:"target_start_at"`
	PredictedStartAt  string `json:"predicted_start_at"`
	PredictedFinishAt string `json:"predicted_finish_at"`
	Status            string `json:"status"`
	Source            string `json:"source"`
}

type priorityAlertManifest struct {
	ID                 string   `json:"id"`
	AlertID            string   `json:"alert_id,omitempty"`
	Type               string   `json:"type"`
	Status             string   `json:"status"`
	CreatedAt          string   `json:"created_at"`
	FeedIDs            []string `json:"feed_ids"`
	Header             string   `json:"header"`
	Event              string   `json:"event"`
	AlertText          string   `json:"alert_text,omitempty"`
	BannerText         string   `json:"banner_text,omitempty"`
	AlertSentAt        string   `json:"alert_sent_at,omitempty"`
	AlertExpiresAt     string   `json:"alert_expires_at,omitempty"`
	MessageType        string   `json:"message_type,omitempty"`
	BroadcastImmediate bool     `json:"broadcast_immediate,omitempty"`
	AudioPath          string   `json:"audio_path"`
	Format             string   `json:"format"`
	SampleRate         int      `json:"sample_rate"`
	Channels           int      `json:"channels"`
	AudioBytes         int      `json:"audio_bytes"`
	Source             string   `json:"source"`
	Priority           string   `json:"priority"`
	AuthoritativeURL   string   `json:"authoritative_url,omitempty"`
	LastError          string   `json:"last_error,omitempty"`
}

type fixedEvent struct {
	Kind      string
	PackageID string
	Title     string
	Target    time.Time
}

func newFeedPlanner(cfg loadedConfig, bridge *bridgeClient, feed feedXML) *feedPlanner {
	return &feedPlanner{
		cfg:                  cfg,
		bridge:               bridge,
		feed:                 feed,
		mode:                 "running",
		startupPrimerAt:      time.Now().Add(startupPrimerDelay),
		startupPrimerPending: true,
		lastFixed:            map[string]time.Time{},
	}
}

func (p *feedPlanner) tick(ctx context.Context, now time.Time, lookahead time.Duration) {
	if !feedRoutineEnabled(p.feed) {
		return
	}
	if p.mode != "running" {
		p.writeState()
		return
	}
	if p.priorityActive > 0 {
		p.writeState()
		return
	}
	if p.startupPrimerPending {
		if now.Before(p.startupPrimerAt) {
			p.writeState()
			return
		}
		p.queueStartupPrimer(ctx, now)
		p.startupPrimerPending = false
		p.writeState()
		return
	}
	p.dropCompleted()
	p.replayPendingItemsIfDue(now)
	maxQueued := p.cfg.Root.Services.Go.Playlist.MaxQueued
	for p.queuedCount() < maxQueued && p.timelineEnd(now).Before(now.Add(lookahead)) {
		next, ok := p.nextPlannedItem(ctx, now)
		if !ok {
			break
		}
		p.queue = append(p.queue, next)
		if err := p.publishReady(next); err != nil {
			p.lastError = err.Error()
			break
		}
	}
	p.writeState()
}

func (p *feedPlanner) nextPlannedItem(ctx context.Context, now time.Time) (playlistItem, bool) {
	if !p.nextRoutineRetryAt.IsZero() && now.Before(p.nextRoutineRetryAt) {
		return playlistItem{}, false
	}
	timelineEnd := p.timelineEnd(now)
	nextFixed, hasFixed := p.nextFixedEvent(now, timelineEnd)
	if hasFixed && shouldFrontLoadFixed(now, nextFixed, p.cfg.Root.Services.Go.Playlist.FixedToleranceS) {
		item, err := p.buildFixed(ctx, nextFixed, timelineEnd, now)
		if err != nil {
			p.lastError = err.Error()
			return playlistItem{}, false
		}
		p.lastFixed[fixedEventKey(nextFixed)] = nextFixed.Target
		return item, true
	}
	pkgID := p.nextRoutinePackage()
	for attempts := 0; pkgID != "" && attempts < len(p.cfg.Root.Playout.PlaylistOrder); attempts++ {
		item, err := p.buildProduct(ctx, pkgID, "routine", "", timelineEnd, now)
		if err == nil {
			p.nextRoutineRetryAt = time.Time{}
			p.lastError = ""
			if hasFixed &&
				shouldFrontLoadFixed(now, nextFixed, p.cfg.Root.Services.Go.Playlist.FixedToleranceS) &&
				routineItemWouldCrowdFixed(item, nextFixed, p.cfg.Root.Services.Go.Playlist.FixedToleranceS) {
				fixed, fixedErr := p.buildFixed(ctx, nextFixed, timelineEnd, now)
				if fixedErr != nil {
					p.lastError = fixedErr.Error()
					return playlistItem{}, false
				}
				p.lastFixed[fixedEventKey(nextFixed)] = nextFixed.Target
				return fixed, true
			}
			return item, true
		}
		p.lastError = err.Error()
		pkgID = p.nextRoutinePackage()
	}
	p.nextRoutineRetryAt = now.Add(routineRetryDelay)
	p.lastError = fmt.Sprintf("routine products unavailable; retrying at %s", p.nextRoutineRetryAt.UTC().Format(time.RFC3339))
	return playlistItem{}, false
}

func routineItemWouldCrowdFixed(item playlistItem, event fixedEvent, toleranceSeconds int) bool {
	if event.Target.IsZero() {
		return false
	}
	finish := parseTime(item.PredictedFinishAt)
	if finish.IsZero() {
		return false
	}
	tolerance := time.Duration(toleranceSeconds) * time.Second
	return finish.After(event.Target.Add(tolerance))
}

func shouldFrontLoadFixed(timelineEnd time.Time, event fixedEvent, toleranceSeconds int) bool {
	if event.Target.IsZero() {
		return false
	}
	tolerance := time.Duration(toleranceSeconds) * time.Second
	return !timelineEnd.Before(event.Target.Add(-tolerance))
}

func (p *feedPlanner) nextRoutinePackage() string {
	order := p.cfg.Root.Playout.PlaylistOrder
	for range order {
		pkgID := strings.TrimSpace(order[p.cursor%len(order)])
		p.cursor++
		if pkgID == "" {
			continue
		}
		if pkgID == "alerts" && !p.hasRoutineAlerts(time.Now()) {
			continue
		}
		return pkgID
	}
	return ""
}

func (p *feedPlanner) nextFixedEvent(now time.Time, timelineEnd time.Time) (fixedEvent, bool) {
	loc := feedLocation(p.feed)
	from := now.In(loc).Add(-time.Second)
	until := now.In(loc).Add(3 * time.Minute)
	candidates := []fixedEvent{}
	if p.cfg.Root.Playout.StationIDSchedule.enabled(true) {
		for _, minute := range p.cfg.Root.Playout.StationIDSchedule.minutesOr([]int{0, 15, 30, 45}) {
			if target, ok := minuteTarget(from, until, minute); ok {
				candidates = append(candidates, fixedEvent{Kind: "station_id", PackageID: "station_id", Title: "Station Identification", Target: target})
			}
		}
	}
	if p.cfg.Root.Playout.DateTimeSchedule.enabled(true) {
		for _, minute := range p.cfg.Root.Playout.DateTimeSchedule.minutesOr([]int{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}) {
			if target, ok := minuteTarget(from, until, minute); ok {
				candidates = append(candidates, fixedEvent{Kind: "date_time", PackageID: "date_time", Title: "Date and Time", Target: target})
			}
		}
	}
	if p.cfg.Root.Playout.Chimes.Enabled {
		for _, minute := range []int{0, 30} {
			if minute == 0 && !p.cfg.Root.Playout.Chimes.TopOfHour.Enabled {
				continue
			}
			if minute == 30 && !p.cfg.Root.Playout.Chimes.HalfHour.Enabled {
				continue
			}
			if target, ok := minuteTarget(from, until, minute); ok {
				candidates = append(candidates, fixedEvent{Kind: "chime", PackageID: "chime", Title: "Chime", Target: target})
			}
		}
	}
	var best fixedEvent
	for _, candidate := range candidates {
		if last, ok := p.lastFixed[fixedEventKey(candidate)]; ok && last.Equal(candidate.Target) {
			continue
		}
		if !candidate.Target.After(now.Add(-2 * time.Second)) {
			continue
		}
		if best.Target.IsZero() || candidate.Target.Before(best.Target) {
			best = candidate
		}
	}
	if best.Target.IsZero() {
		return fixedEvent{}, false
	}
	_ = timelineEnd
	return best, true
}

func fixedEventKey(event fixedEvent) string {
	return event.Kind + ":" + event.Target.UTC().Format(time.RFC3339)
}

func minuteTarget(from time.Time, until time.Time, minute int) (time.Time, bool) {
	base := time.Date(from.Year(), from.Month(), from.Day(), from.Hour(), minute, 0, 0, from.Location())
	for base.Before(from) {
		base = base.Add(time.Hour)
	}
	if base.After(until) {
		return time.Time{}, false
	}
	return base, true
}

func (p *feedPlanner) buildFixed(ctx context.Context, event fixedEvent, timelineEnd time.Time, now time.Time) (playlistItem, error) {
	if event.Kind == "chime" {
		return p.buildChime(event, timelineEnd, now)
	}
	return p.buildProduct(ctx, event.PackageID, "fixed", event.Target.Format(time.RFC3339Nano), timelineEnd, now)
}

func (p *feedPlanner) buildProduct(ctx context.Context, pkgID string, source string, targetRaw string, timelineEnd time.Time, now time.Time) (playlistItem, error) {
	queueID := queueID(pkgID)
	product, err := p.renderProductForBuild(ctx, queueID, pkgID, source, now)
	if err != nil {
		return playlistItem{}, err
	}
	if item, ok, err := p.buildProductAudioItem(ctx, product, pkgID, source, targetRaw, timelineEnd, now, queueID); ok || err != nil {
		return item, err
	}
	if strings.TrimSpace(product.Text) == "" {
		return playlistItem{}, fmt.Errorf("product %s rendered empty text", pkgID)
	}
	if cached, ok := p.startupCachedProductItem(pkgID, product, source, targetRaw, timelineEnd, now); ok {
		return cached, nil
	}
	outputPath := filepath.Join(p.cfg.OutputDir, safeID(p.feed.ID), queueID+".wav")
	synthCtx, cancel := context.WithTimeout(ctx, p.synthesisTimeoutForBuild(pkgID, source))
	defer cancel()
	wavPath, err := p.bridge.Synthesize(synthCtx, synthJob{
		ID:         queueID,
		Text:       product.Text,
		ReaderID:   product.ReaderID,
		Language:   fallbackText(product.Language, feedLanguage(p.feed)),
		Timezone:   feedTimezone(p.feed),
		OutputPath: outputPath,
	})
	if err != nil {
		if cached, ok := p.cachedProductItem(pkgID, product, source, targetRaw, timelineEnd, now, p.cacheFallbackMaxAge(source)); ok {
			p.lastError = fmt.Sprintf("using cached %s audio after synthesis failed: %v", pkgID, err)
			return cached, nil
		}
		return playlistItem{}, err
	}
	info, err := wavInfo(wavPath)
	if err != nil {
		if cached, ok := p.cachedProductItem(pkgID, product, source, targetRaw, timelineEnd, now, p.cacheFallbackMaxAge(source)); ok {
			p.lastError = fmt.Sprintf("using cached %s audio after synthesized WAV could not be read: %v", pkgID, err)
			return cached, nil
		}
		return playlistItem{}, err
	}
	target := parseTime(targetRaw)
	start := predictedStart(now, timelineEnd, target)
	finish := start.Add(p.itemScheduleDuration(info.DurationMS))
	return playlistItem{
		QueueID:           queueID,
		FeedID:            p.feed.ID,
		Kind:              "product",
		PackageID:         pkgID,
		Title:             fallbackText(product.Title, pkgID),
		AudioPath:         wavPath,
		DurationMS:        info.DurationMS,
		QueuedAt:          now.UTC().Format(time.RFC3339Nano),
		TargetStartAt:     formatOptionalTime(target),
		PredictedStartAt:  start.UTC().Format(time.RFC3339Nano),
		PredictedFinishAt: finish.UTC().Format(time.RFC3339Nano),
		Status:            "queued",
		Source:            source,
	}, nil
}

func (p *feedPlanner) synthesisTimeoutForBuild(pkgID string, source string) time.Duration {
	if strings.EqualFold(source, "startup") {
		return 8 * time.Second
	}
	if p.cachedProductAudioEligible(pkgID) {
		if _, _, ok := p.latestCachedProductAudio(pkgID, cachedRoutineFallbackMaxAge); ok {
			return 12 * time.Second
		}
	}
	return 90 * time.Second
}

func (p *feedPlanner) startupCachedProductItem(pkgID string, product renderedProduct, source string, targetRaw string, timelineEnd time.Time, now time.Time) (playlistItem, bool) {
	if !strings.EqualFold(source, "startup") || strings.EqualFold(pkgID, "date_time") {
		return playlistItem{}, false
	}
	return p.cachedProductItem(pkgID, product, source, targetRaw, timelineEnd, now, cachedStartupFallbackMaxAge)
}

func (p *feedPlanner) cacheFallbackMaxAge(source string) time.Duration {
	if strings.EqualFold(source, "startup") {
		return cachedStartupFallbackMaxAge
	}
	return cachedRoutineFallbackMaxAge
}

func (p *feedPlanner) cachedProductItem(pkgID string, product renderedProduct, source string, targetRaw string, timelineEnd time.Time, now time.Time, maxAge time.Duration) (playlistItem, bool) {
	if !p.cachedProductAudioEligible(pkgID) {
		return playlistItem{}, false
	}
	path, info, ok := p.latestCachedProductAudio(pkgID, maxAge)
	if !ok {
		return playlistItem{}, false
	}
	target := parseTime(targetRaw)
	start := predictedStart(now, timelineEnd, target)
	finish := start.Add(p.itemScheduleDuration(info.DurationMS))
	return playlistItem{
		QueueID:           queueID(pkgID),
		FeedID:            p.feed.ID,
		Kind:              "product",
		PackageID:         pkgID,
		Title:             fallbackText(product.Title, pkgID),
		AudioPath:         path,
		DurationMS:        info.DurationMS,
		QueuedAt:          now.UTC().Format(time.RFC3339Nano),
		TargetStartAt:     formatOptionalTime(target),
		PredictedStartAt:  start.UTC().Format(time.RFC3339Nano),
		PredictedFinishAt: finish.UTC().Format(time.RFC3339Nano),
		Status:            "queued",
		Source:            fallbackText(source, "cached"),
	}, true
}

func (p *feedPlanner) cachedProductAudioEligible(pkgID string) bool {
	switch strings.ToLower(strings.TrimSpace(pkgID)) {
	case "alerts", "date_time":
		return false
	default:
		return true
	}
}

func (p *feedPlanner) latestCachedProductAudio(pkgID string, maxAge time.Duration) (string, audioInfo, bool) {
	dir := filepath.Join(p.cfg.OutputDir, safeID(p.feed.ID))
	prefix := safeID(pkgID) + "-"
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", audioInfo{}, false
	}
	var bestPath string
	var bestMod time.Time
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(strings.ToLower(name), ".wav") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		mod := info.ModTime()
		if maxAge > 0 && time.Since(mod) > maxAge {
			continue
		}
		if bestPath == "" || mod.After(bestMod) {
			bestPath = filepath.Join(dir, name)
			bestMod = mod
		}
	}
	if bestPath == "" {
		return "", audioInfo{}, false
	}
	info, err := wavInfo(bestPath)
	if err != nil {
		return "", audioInfo{}, false
	}
	return bestPath, info, true
}

func (p *feedPlanner) renderProductForBuild(ctx context.Context, queueID string, pkgID string, source string, now time.Time) (renderedProduct, error) {
	if source == "startup" && (pkgID == "station_id" || pkgID == "date_time") {
		return p.staticProduct(pkgID, now)
	}
	renderCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	product, err := p.bridge.RenderProduct(renderCtx, queueID, p.feed.ID, pkgID)
	if err != nil {
		fallback, fallbackErr := p.staticProduct(pkgID, now)
		if fallbackErr != nil {
			return renderedProduct{}, err
		}
		product = fallback
	}
	return product, nil
}

func (p *feedPlanner) queueStartupPrimer(ctx context.Context, now time.Time) {
	timelineEnd := p.timelineEnd(now)
	for index, pkgID := range []string{"station_id", "date_time", "current_conditions"} {
		if ctx.Err() != nil || p.mode != "running" || p.priorityActive > 0 {
			return
		}
		targetRaw := ""
		if index == 0 && !p.startupPrimerAt.IsZero() {
			targetRaw = p.startupPrimerAt.UTC().Format(time.RFC3339Nano)
		}
		item, err := p.buildProduct(ctx, pkgID, "startup", targetRaw, timelineEnd, now)
		if err != nil {
			if pkgID != "current_conditions" {
				p.lastError = err.Error()
			}
			continue
		}
		p.queue = append(p.queue, item)
		if err := p.publishReady(item); err != nil {
			p.lastError = err.Error()
			return
		}
		if finish := parseTime(item.PredictedFinishAt); !finish.IsZero() {
			timelineEnd = finish
		}
		now = time.Now()
	}
}

func (p *feedPlanner) buildProductAudioItem(ctx context.Context, product renderedProduct, pkgID string, source string, targetRaw string, timelineEnd time.Time, now time.Time, queueID string) (playlistItem, bool, error) {
	if !strings.EqualFold(metadataText(product.Metadata, "content_type"), "audio") {
		return playlistItem{}, false, nil
	}
	audioPath := metadataText(product.Metadata, "audio_path")
	audioURL := metadataText(product.Metadata, "audio_url")
	if audioPath == "" && audioURL == "" {
		return playlistItem{}, true, fmt.Errorf("product %s declared audio content without audio_path or audio_url", pkgID)
	}
	outputPath := filepath.Join(p.cfg.OutputDir, safeID(p.feed.ID), queueID+".wav")
	finalPath := ""
	if audioURL != "" {
		path, err := p.downloadRoutineAudio(ctx, audioURL, queueID, outputPath)
		if err != nil {
			return playlistItem{}, true, err
		}
		finalPath = path
	} else {
		path, err := p.prepareRoutineAudio(ctx, audioPath, outputPath)
		if err != nil {
			return playlistItem{}, true, err
		}
		finalPath = path
	}
	info, err := wavInfo(finalPath)
	if err != nil {
		return playlistItem{}, true, err
	}
	target := parseTime(targetRaw)
	start := predictedStart(now, timelineEnd, target)
	finish := start.Add(p.itemScheduleDuration(info.DurationMS))
	return playlistItem{
		QueueID:           queueID,
		FeedID:            p.feed.ID,
		Kind:              "audio",
		PackageID:         pkgID,
		Title:             fallbackText(product.Title, pkgID),
		AudioPath:         finalPath,
		DurationMS:        info.DurationMS,
		QueuedAt:          now.UTC().Format(time.RFC3339Nano),
		TargetStartAt:     formatOptionalTime(target),
		PredictedStartAt:  start.UTC().Format(time.RFC3339Nano),
		PredictedFinishAt: finish.UTC().Format(time.RFC3339Nano),
		Status:            "queued",
		Source:            source,
	}, true, nil
}

func (p *feedPlanner) downloadRoutineAudio(ctx context.Context, sourceURL string, queueID string, outputPath string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(sourceURL))
	if err != nil || parsed == nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return "", fmt.Errorf("audio URL must be http or https")
	}
	ctx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	inputPath := filepath.Join(filepath.Dir(outputPath), queueID+".download")
	if err := downloadFile(ctx, parsed.String(), inputPath, 20<<20); err != nil {
		return "", err
	}
	if _, err := wavInfo(inputPath); err == nil {
		return inputPath, nil
	}
	if err := convertAudioToWAV(ctx, inputPath, outputPath, p.cfg.Root.Playout.SampleRate, p.cfg.Root.Playout.Channels); err != nil {
		return "", err
	}
	return outputPath, nil
}

func (p *feedPlanner) prepareRoutineAudio(ctx context.Context, rawPath string, outputPath string) (string, error) {
	sourcePath, err := p.resolveRoutineAudioPath(rawPath)
	if err != nil {
		return "", err
	}
	if _, err := wavInfo(sourcePath); err == nil {
		return sourcePath, nil
	}
	convertCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	if err := convertAudioToWAV(convertCtx, sourcePath, outputPath, p.cfg.Root.Playout.SampleRate, p.cfg.Root.Playout.Channels); err != nil {
		return "", err
	}
	return outputPath, nil
}

func (p *feedPlanner) resolveRoutineAudioPath(rawPath string) (string, error) {
	rawPath = strings.TrimSpace(rawPath)
	if rawPath == "" {
		return "", fmt.Errorf("audio path is required")
	}
	clean := filepath.Clean(rawPath)
	if filepath.IsAbs(clean) {
		return clean, nil
	}
	slash := filepath.ToSlash(clean)
	if slash == ".." || strings.HasPrefix(slash, "../") || strings.HasPrefix(slash, "/") {
		return "", fmt.Errorf("audio path must stay inside the bundle directory")
	}
	return filepath.Join(p.cfg.BaseDir, clean), nil
}

func metadataText(metadata map[string]string, key string) string {
	if len(metadata) == 0 {
		return ""
	}
	for rawKey, value := range metadata {
		if strings.EqualFold(strings.TrimSpace(rawKey), key) {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func (p *feedPlanner) staticProduct(pkgID string, now time.Time) (renderedProduct, error) {
	switch strings.ToLower(strings.TrimSpace(pkgID)) {
	case "station_id":
		onAirName := displayText(p.cfg.Root.Operator.OnAirName)
		if onAirName == "" {
			onAirName = "Haze Weather Radio"
		}
		parts := []string{fmt.Sprintf("You are listening to %s.", onAirName)}
		site := feedName(p.feed)
		callsign := feedCallsign(p.feed)
		frequency := feedFrequencyMHz(p.feed)
		if callsignSpoken := spokenCallsign(callsign); callsignSpoken != "" {
			parts = append(parts, fmt.Sprintf("Callsign %s.", callsignSpoken))
		}
		if frequency != "" {
			parts = append(parts, fmt.Sprintf("Broadcasting from %s on a frequency of %s megahertz.", fallbackText(site, p.feed.ID), frequency))
		} else if site != "" {
			parts = append(parts, fmt.Sprintf("Serving %s.", site))
		}
		if replacement := replacementStationStatement(p.feed); replacement != "" {
			parts = append(parts, replacement)
		}
		return renderedProduct{
			ID:        fmt.Sprintf("%s-%s-%d", p.feed.ID, pkgID, now.UnixNano()),
			FeedID:    p.feed.ID,
			PackageID: pkgID,
			Title:     "Station Identification",
			Text:      strings.Join(parts, " "),
			ReaderID:  "00",
			Language:  feedLanguage(p.feed),
		}, nil
	case "date_time":
		local := now
		if loc := feedLocation(p.feed); loc != nil {
			local = now.In(loc)
		}
		return renderedProduct{
			ID:        fmt.Sprintf("%s-%s-%d", p.feed.ID, pkgID, now.UnixNano()),
			FeedID:    p.feed.ID,
			PackageID: pkgID,
			Title:     "Date and Time",
			Text:      dateTimeAnnouncement(local, feedLanguage(p.feed)),
			ReaderID:  "00",
			Language:  feedLanguage(p.feed),
		}, nil
	default:
		return renderedProduct{}, fmt.Errorf("no static fallback for %s", pkgID)
	}
}

func replacementStationStatement(feed feedXML) string {
	transmitter, ok := replacementTransmitter(feed)
	if !ok {
		return ""
	}
	callsign := strings.TrimSpace(transmitter.Callsign)
	site := strings.TrimSpace(transmitter.SiteName)
	if callsign == "" && site == "" {
		return ""
	}
	network := strings.TrimSpace(transmitter.Network.Name)
	if network == "" {
		network = "Weatheradio Canada"
	}
	parts := []string{"This station replaces former", network, "station"}
	if callsign != "" {
		parts = append(parts, callsign)
	}
	if site != "" {
		parts = append(parts, "in", site)
	}
	return strings.Join(parts, " ") + "."
}

func (p *feedPlanner) buildChime(event fixedEvent, timelineEnd time.Time, now time.Time) (playlistItem, error) {
	source := filepath.Join(p.cfg.BaseDir, "audio", "8-step_chime.wav")
	info, err := wavInfo(source)
	if err != nil {
		return playlistItem{}, err
	}
	queueID := queueID("chime")
	start := predictedStart(now, timelineEnd, event.Target)
	finish := start.Add(p.itemScheduleDuration(info.DurationMS))
	return playlistItem{
		QueueID:           queueID,
		FeedID:            p.feed.ID,
		Kind:              "audio",
		PackageID:         "chime",
		Title:             "Chime",
		AudioPath:         source,
		DurationMS:        info.DurationMS,
		QueuedAt:          now.UTC().Format(time.RFC3339Nano),
		TargetStartAt:     event.Target.UTC().Format(time.RFC3339Nano),
		PredictedStartAt:  start.UTC().Format(time.RFC3339Nano),
		PredictedFinishAt: finish.UTC().Format(time.RFC3339Nano),
		Status:            "queued",
		Source:            "fixed",
	}, nil
}

func (p *feedPlanner) insert(ctx context.Context, data map[string]any) {
	kind := strings.ToLower(firstText(nil, data, "kind"))
	position := strings.ToLower(firstText(nil, data, "position"))
	if position == "next" {
		p.queue = nil
		_ = p.bridge.Publish(map[string]any{
			"type":    "playlist.control",
			"source":  serviceID,
			"feed_id": p.feed.ID,
			"data":    map[string]any{"feed_id": p.feed.ID, "action": "flush_pending"},
		})
	}
	var item playlistItem
	var err error
	now := time.Now()
	switch kind {
	case "product":
		item, err = p.buildProduct(ctx, firstText(nil, data, "package_id", "pkg_id"), "operator", "", p.timelineEnd(now), now)
	case "tts":
		item, err = p.buildTextItem(ctx, firstText(nil, data, "title"), firstText(nil, data, "text"), now)
	case "audio":
		item, err = p.buildAudioItem(firstText(nil, data, "title"), firstText(nil, data, "audio_path"), now)
	case "same":
		p.lastError = "SAME playlist insertion is staged for the SAME queue integration path"
		return
	default:
		p.lastError = "unknown insert kind"
		return
	}
	if err != nil {
		p.lastError = err.Error()
		return
	}
	p.queue = append(p.queue, item)
	_ = p.publishReady(item)
	p.writeState()
}

func (p *feedPlanner) queuePriorityAlert(ctx context.Context, data map[string]any) {
	alertID := firstText(nil, data, "alert_id", "id", "subject")
	if alertID == "" {
		alertID = fmt.Sprintf("cap-%d", time.Now().UnixNano())
	}
	if priorityAlertRequestStale(data, time.Now().UTC()) {
		cleanupSupersededAlertQueueParts(p.cfg.BaseDir, p.feed.ID, alertID, "")
		p.lastError = ""
		p.writeState()
		return
	}
	includeSame := includeSameAlert(data)
	includeAttentionTone := !includeSame && alertAttentionToneEnabled(data)
	var sameRequest sameGenerateRequest
	var sameHeader sameAudioPayload
	var sameEOM sameAudioPayload
	var attentionTone sameAudioPayload
	alertSampleRate := p.cfg.Root.Playout.SampleRate
	alertChannels := p.cfg.Root.Playout.Channels
	if includeSame {
		var err error
		var sameHeaderResult map[string]any
		sameRequest, sameHeaderResult, err = p.generatePrioritySAME(ctx, data, "header")
		if err != nil {
			p.lastError = "SAME header generation failed: " + err.Error()
			p.writeState()
			return
		}
		sameHeader, err = sameAudioFromResult(sameHeaderResult, p.cfg.Root.Playout.SampleRate, p.cfg.Root.Playout.Channels)
		if err != nil {
			p.lastError = "SAME header generation failed: " + err.Error()
			p.writeState()
			return
		}
		var sameEOMResult map[string]any
		_, sameEOMResult, err = p.generatePrioritySAME(ctx, data, "eom")
		if err != nil {
			p.lastError = "SAME EOM generation failed: " + err.Error()
			p.writeState()
			return
		}
		sameEOM, err = sameAudioFromResult(sameEOMResult, sameHeader.SampleRate, sameHeader.Channels)
		if err != nil {
			p.lastError = "SAME EOM generation failed: " + err.Error()
			p.writeState()
			return
		}
		if sameHeader.SampleRate != sameEOM.SampleRate || sameHeader.Channels != sameEOM.Channels {
			p.lastError = "SAME header and EOM formats do not match"
			p.writeState()
			return
		}
		alertSampleRate = sameHeader.SampleRate
		alertChannels = sameHeader.Channels
	} else if includeAttentionTone {
		var err error
		var toneResult map[string]any
		_, toneResult, err = p.generatePrioritySAME(ctx, data, "tone")
		if err != nil {
			p.lastError = "attention tone generation failed: " + err.Error()
			p.writeState()
			return
		}
		attentionTone, err = sameAudioFromResult(toneResult, p.cfg.Root.Playout.SampleRate, p.cfg.Root.Playout.Channels)
		if err != nil {
			p.lastError = "attention tone generation failed: " + err.Error()
			p.writeState()
			return
		}
		alertSampleRate = attentionTone.SampleRate
		alertChannels = attentionTone.Channels
	}
	queueID := safeID("001_" + p.feed.ID + "_" + alertID + "_cap")
	if includeSame {
		queueID = safeID("000_" + p.feed.ID + "_" + alertID + "_same")
	}
	audioRel := filepath.ToSlash(filepath.Join("runtime", "audio", "alerts", queueID+".pcm16le"))
	audioPath := filepath.Join(p.cfg.BaseDir, filepath.FromSlash(audioRel))
	voicePath := audioPath
	if includeSame || includeAttentionTone {
		voicePath = audioPath + ".voice"
		defer os.Remove(voicePath)
	}
	title := fallbackText(firstText(nil, data, "title", "header"), "Weather Alert")
	eventName := fallbackText(firstText(nil, data, "event"), "CAP")
	alertText := p.alertTextFromData(data)
	bannerText := p.bannerTextFromData(data, alertText)
	source := "cap-tts"
	authoritativeURL := strings.TrimSpace(firstText(nil, data, "audio_url", "authoritative_url"))
	var lastErr error
	if authoritativeURL != "" {
		if err := p.downloadAndConvertAlertAudio(ctx, authoritativeURL, voicePath, alertSampleRate, alertChannels); err == nil {
			source = "cap-broadcast-audio"
		} else {
			lastErr = err
		}
	}
	if source != "cap-broadcast-audio" {
		if err := p.renderAlertTTSAsPCM(ctx, queueID, voicePath, alertText, alertSampleRate, alertChannels); err != nil {
			if lastErr != nil {
				p.lastError = fmt.Sprintf("broadcast audio failed: %v; TTS fallback failed: %v", lastErr, err)
			} else {
				p.lastError = fmt.Sprintf("alert TTS fallback failed: %v", err)
			}
			p.writeState()
			return
		}
	}
	if includeSame {
		if err := combineSAMEAlertAudio(audioPath, sameHeader.Audio, voicePath, sameEOM.Audio, alertSampleRate, alertChannels); err != nil {
			p.lastError = "SAME alert assembly failed: " + err.Error()
			p.writeState()
			return
		}
		if source == "cap-broadcast-audio" {
			source = "cap-same-broadcast-audio"
		} else {
			source = "cap-same-tts"
		}
	} else if includeAttentionTone {
		if err := combineAttentionAlertAudio(audioPath, attentionTone.Audio, voicePath, alertSampleRate, alertChannels); err != nil {
			p.lastError = "attention tone alert assembly failed: " + err.Error()
			p.writeState()
			return
		}
		if source == "cap-broadcast-audio" {
			source = "cap-tone-broadcast-audio"
		} else {
			source = "cap-tone-tts"
		}
	}
	info, err := pcmInfo(audioPath, alertSampleRate, alertChannels)
	if err != nil {
		p.lastError = err.Error()
		p.writeState()
		return
	}
	manifest := priorityAlertManifest{
		ID:                 queueID,
		AlertID:            alertID,
		Type:               "cap_alert",
		Status:             "pending",
		CreatedAt:          time.Now().UTC().Format(time.RFC3339Nano),
		FeedIDs:            []string{p.feed.ID},
		Header:             title,
		Event:              eventName,
		AlertText:          strings.TrimSpace(alertText),
		BannerText:         strings.TrimSpace(bannerText),
		AlertSentAt:        firstText(nil, data, "alert_sent_at", "sent"),
		AlertExpiresAt:     firstText(nil, data, "alert_expires_at", "expires"),
		MessageType:        firstText(nil, data, "message_type", "msg_type"),
		BroadcastImmediate: boolAny(firstValue(nil, data, "broadcast_immediate")),
		AudioPath:          audioRel,
		Format:             "pcm_s16le",
		SampleRate:         info.SampleRate,
		Channels:           info.Channels,
		AudioBytes:         int(info.Bytes),
		Source:             source,
		Priority:           "cap",
		AuthoritativeURL:   authoritativeURL,
	}
	if includeSame {
		manifest.Type = "same_alert"
		manifest.Event = sameRequest.Event
		manifest.Priority = "same"
	}
	if lastErr != nil && source != "cap-broadcast-audio" {
		manifest.LastError = "broadcast audio fallback: " + lastErr.Error()
	}
	cleanupSupersededAlertQueueParts(p.cfg.BaseDir, p.feed.ID, alertID, queueID)
	if err := writePriorityAlertManifest(filepath.Join(p.cfg.BaseDir, "runtime", "queues", "alerts", queueID+".json"), manifest); err != nil {
		p.lastError = err.Error()
		p.writeState()
		return
	}
	p.publishAlertAudioReady(manifest, data, alertText, sameHeader.Header)
	p.lastError = ""
	p.writeState()
}

func (p *feedPlanner) publishAlertAudioReady(manifest priorityAlertManifest, data map[string]any, alertText string, sameHeader string) {
	payload := map[string]any{
		"feed_id":              p.feed.ID,
		"alert_id":             manifest.AlertID,
		"queue_id":             manifest.ID,
		"manifest_id":          manifest.ID,
		"audio_path":           manifest.AudioPath,
		"audio_format":         manifest.Format,
		"sample_rate":          manifest.SampleRate,
		"channels":             manifest.Channels,
		"audio_bytes":          manifest.AudioBytes,
		"source":               manifest.Source,
		"priority":             manifest.Priority,
		"message_type":         manifest.MessageType,
		"alert_sent_at":        manifest.AlertSentAt,
		"alert_expires_at":     manifest.AlertExpiresAt,
		"broadcast_immediate":  manifest.BroadcastImmediate,
		"authoritative_url":    manifest.AuthoritativeURL,
		"title":                fallbackText(firstText(nil, data, "headline", "title", "header"), manifest.Header),
		"header":               manifest.Header,
		"event":                fallbackText(firstText(nil, data, "same_event", "event"), manifest.Event),
		"alert_text":           fallbackText(alertText, firstText(nil, data, "alert_text", "tts_text", "text", "message")),
		"banner_text":          fallbackText(manifest.BannerText, firstText(nil, data, "banner_text")),
		"description":          firstText(nil, data, "description"),
		"instruction":          firstText(nil, data, "instruction"),
		"severity":             firstText(nil, data, "severity"),
		"urgency":              firstText(nil, data, "urgency"),
		"certainty":            firstText(nil, data, "certainty"),
		"same_event":           firstText(nil, data, "same_event"),
		"same_event_name":      firstText(nil, data, "same_event_name"),
		"same_originator":      firstText(nil, data, "same_originator"),
		"same_originator_name": firstText(nil, data, "same_originator_name"),
		"same_weather_service": firstText(nil, data, "same_weather_service"),
		"same_duration":        firstText(nil, data, "same_duration"),
		"same_sent_at":         firstText(nil, data, "same_sent_at"),
		"same_begins_at":       firstText(nil, data, "same_begins_at"),
		"same_expires_at":      firstText(nil, data, "same_expires_at"),
		"same_tone":            firstText(nil, data, "same_tone"),
		"same_header":          sameHeader,
		"background_color":     firstText(nil, data, "background_color"),
	}
	if locations := stringListAny(firstValue(nil, data, "same_locations", "locations")); len(locations) > 0 {
		payload["same_locations"] = locations
	}
	if err := p.bridge.Publish(map[string]any{
		"type":    "cap.alert.audio.ready",
		"source":  serviceID,
		"feed_id": p.feed.ID,
		"subject": manifest.AlertID,
		"data":    payload,
	}); err != nil {
		log.Printf("[%s] alert webhook event publish failed: %v", p.feed.ID, err)
	}
}

func (p *feedPlanner) cancelPriorityAlerts(data map[string]any) {
	ids := stringListAny(firstValue(nil, data, "alert_ids"))
	if id := firstText(nil, data, "alert_id", "id", "subject"); id != "" {
		ids = append(ids, id)
	}
	ids = uniqueStrings(ids)
	for _, alertID := range ids {
		cleanupSupersededAlertQueueParts(p.cfg.BaseDir, p.feed.ID, alertID, "")
	}
	p.writeState()
}

func priorityAlertRequestStale(data map[string]any, now time.Time) bool {
	if strings.EqualFold(firstText(nil, data, "message_type", "msg_type"), "Cancel") {
		return true
	}
	header := strings.ToLower(firstText(nil, data, "title", "header"))
	if strings.Contains(header, "ended") || strings.Contains(header, "cancelled") || strings.Contains(header, "canceled") {
		return true
	}
	if expires := parseTime(firstText(nil, data, "alert_expires_at", "expires")); !expires.IsZero() && now.After(expires) {
		return true
	}
	sent := parseTime(firstText(nil, data, "alert_sent_at", "sent"))
	if sent.IsZero() || now.Before(sent) {
		return false
	}
	event := strings.ToUpper(firstText(nil, data, "same_event", "event"))
	limit := time.Hour
	if event == "SVR" || event == "TOR" || strings.Contains(header, "severe thunderstorm warning") || strings.Contains(header, "tornado warning") {
		limit = 30 * time.Minute
	}
	return now.Sub(sent) > limit
}

func cleanupSupersededAlertQueueParts(baseDir string, feedID string, alertID string, keepID string) {
	ids := []string{
		safeID("000_" + feedID + "_" + alertID + "_same_header"),
		safeID("000_" + feedID + "_" + alertID + "_same"),
		safeID("001_" + feedID + "_" + alertID + "_cap"),
		safeID("002_" + feedID + "_" + alertID + "_same_eom"),
		safeID("cap_alert_" + feedID + "_" + alertID),
	}
	for _, id := range ids {
		if id == "" || id == keepID {
			continue
		}
		_ = os.Remove(filepath.Join(baseDir, "runtime", "queues", "alerts", id+".json"))
		_ = os.Remove(filepath.Join(baseDir, "runtime", "audio", "alerts", id+".pcm16le"))
	}
}

func (p *feedPlanner) downloadAndConvertAlertAudio(ctx context.Context, sourceURL string, outputPath string, sampleRate int, channels int) error {
	ctx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	tempDir := filepath.Join(p.cfg.BaseDir, "runtime", "audio", "alerts")
	if err := os.MkdirAll(tempDir, 0o755); err != nil {
		return err
	}
	inputPath := filepath.Join(tempDir, safeID(fmt.Sprintf("download-%d", time.Now().UnixNano()))+".bin")
	defer os.Remove(inputPath)
	if err := downloadFile(ctx, sourceURL, inputPath, 10<<20); err != nil {
		return err
	}
	return convertAudioToPCM(ctx, inputPath, outputPath, sampleRate, channels)
}

func alertTextFromData(data map[string]any) string {
	return alerttext.SpeechFromData(data)
}

func (p *feedPlanner) alertTextFromData(data map[string]any) string {
	if strings.EqualFold(firstText(nil, data, "cap_source"), "nws") {
		if intro := p.sameSpeechIntroFromData(data); intro != "" {
			parts := []string{intro}
			if description := strings.TrimSpace(firstText(nil, data, "description")); description != "" {
				parts = append(parts, alerttext.CleanAlertText(description))
			}
			if instruction := strings.TrimSpace(firstText(nil, data, "instruction")); instruction != "" {
				parts = append(parts, alerttext.CleanAlertText(instruction))
			}
			return strings.TrimSpace(strings.Join(nonEmptyStrings(parts), " "))
		}
	}
	return alertTextFromData(data)
}

func (p *feedPlanner) bannerTextFromData(data map[string]any, alertText string) string {
	if text := strings.TrimSpace(firstText(nil, data, "banner_text")); text != "" {
		return text
	}
	parts := []string{}
	if intro := p.sameIntroFromData(data); intro != "" {
		parts = append(parts, intro)
	}
	if description := strings.TrimSpace(firstText(nil, data, "description")); description != "" {
		parts = append(parts, alerttext.CleanAlertText(description))
	}
	if instruction := strings.TrimSpace(firstText(nil, data, "instruction")); instruction != "" {
		parts = append(parts, alerttext.CleanAlertText(instruction))
	}
	if len(parts) <= 1 {
		if custom := customAlertTextFromData(data); custom != "" && !sameText(parts, custom) {
			parts = append(parts, custom)
		}
	}
	if len(parts) == 0 {
		return strings.TrimSpace(alertText)
	}
	return strings.TrimSpace(strings.Join(nonEmptyStrings(parts), " "))
}

func (p *feedPlanner) sameIntroFromData(data map[string]any) string {
	return p.sameIntroFromDataWithOptions(data, true)
}

func (p *feedPlanner) sameSpeechIntroFromData(data map[string]any) string {
	return p.sameIntroFromDataWithOptions(data, false)
}

func (p *feedPlanner) sameIntroFromDataWithOptions(data map[string]any, includeSourceLabel bool) string {
	if intro := strings.TrimSpace(firstText(nil, data, "same_translation", "same_intro")); intro != "" {
		if !includeSourceLabel {
			return stripTrailingSourceLabel(intro)
		}
		return intro
	}
	event := strings.TrimSpace(firstText(nil, data, "same_event", "event"))
	locations := stringListAny(firstValue(nil, data, "same_locations", "locations"))
	if event == "" || len(locations) == 0 {
		return ""
	}
	configPath := filepath.Join(p.cfg.BaseDir, "config.yaml")
	duration := normalizeSAMEDuration(fallbackText(firstText(nil, data, "same_duration", "duration"), "0015"))
	beginsAt := dataTimeValue(data, "same_begins_at", "alert_begins_at", "onset", "effective", "same_sent_at", "alert_sent_at", "sent")
	sentAt := dataTimeValue(data, "same_sent_at", "alert_sent_at", "sent")
	expiresAt := dataTimeValue(data, "same_expires_at", "alert_expires_at", "expires")
	if expiresAt.IsZero() && !beginsAt.IsZero() {
		expiresAt = beginsAt.Add(sameDurationToDuration(duration))
	}
	if beginsAt.IsZero() && !sentAt.IsZero() {
		beginsAt = sentAt
	}
	return alerttext.BuildSAMETranslation(alerttext.SAMERequest{
		Originator:     fallbackText(firstText(nil, data, "same_originator", "originator"), "WXR"),
		OriginatorName: strings.TrimSpace(firstText(nil, data, "same_originator_name", "originator_name", "sender_name")),
		Event:          event,
		EventName:      fallbackText(firstText(nil, data, "same_event_name", "event_name"), alerttext.EventName(configPath, event)),
		Locations:      locations,
		AreaNames:      alerttext.ResolveAreaNames(configPath, stringListAny(firstValue(nil, data, "area_names")), locations),
		Callsign:       sameIntroCallsign(data, p.feed, includeSourceLabel),
		WeatherService: strings.TrimSpace(firstText(nil, data, "same_weather_service", "weather_service")),
		SentAt:         sentAt,
		BeginsAt:       beginsAt,
		ExpiresAt:      expiresAt,
		MimicENDEC:     fallbackText(firstText(nil, data, "mimic_endec"), "SAGE"),
	})
}

func sameIntroCallsign(data map[string]any, feed feedXML, includeSourceLabel bool) string {
	if !includeSourceLabel {
		return ""
	}
	return fallbackText(firstText(nil, data, "same_callsign", "callsign"), feedCallsign(feed))
}

func stripTrailingSourceLabel(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" || !strings.HasSuffix(trimmed, ")") && !strings.HasSuffix(trimmed, ").") {
		return trimmed
	}
	withoutFinalPeriod := strings.TrimSpace(strings.TrimSuffix(trimmed, "."))
	if !strings.HasSuffix(withoutFinalPeriod, ")") {
		return trimmed
	}
	open := strings.LastIndex(withoutFinalPeriod, "(")
	if open < 0 || open == 0 || strings.TrimSpace(withoutFinalPeriod[open+1:len(withoutFinalPeriod)-1]) == "" {
		return trimmed
	}
	prefix := strings.TrimSpace(withoutFinalPeriod[:open])
	if prefix == "" {
		return trimmed
	}
	return alerttext.CleanFragment(prefix)
}

func dataTimeValue(data map[string]any, keys ...string) time.Time {
	for _, key := range keys {
		raw := strings.TrimSpace(firstText(nil, data, key))
		if raw == "" {
			continue
		}
		for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02 15:04:05", "2006-01-02T15:04:05"} {
			if parsed, err := time.Parse(layout, raw); err == nil {
				return parsed
			}
		}
		if parsed := alerttext.ParseCAPTime(raw); !parsed.IsZero() {
			return parsed
		}
	}
	return time.Time{}
}

func customAlertTextFromData(data map[string]any) string {
	text := strings.TrimSpace(firstText(nil, data, "alert_text", "tts_text", "text", "message"))
	if text == "" {
		return ""
	}
	for _, intro := range []string{
		firstText(nil, data, "same_translation"),
		firstText(nil, data, "same_intro"),
	} {
		intro = strings.TrimSpace(intro)
		if intro != "" && strings.HasPrefix(strings.ToLower(text), strings.ToLower(intro)) {
			return strings.TrimSpace(text[len(intro):])
		}
	}
	return text
}

func sameDurationToDuration(raw string) time.Duration {
	raw = normalizeSAMEDuration(raw)
	hours := intFromString(raw[:2], 0)
	minutes := intFromString(raw[2:], 15)
	if hours == 0 && minutes == 0 {
		minutes = 15
	}
	return time.Duration(hours)*time.Hour + time.Duration(minutes)*time.Minute
}

func intFromString(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return fallback
	}
	return value
}

func nonEmptyStrings(values []string) []string {
	out := values[:0]
	for _, value := range values {
		if clean := strings.TrimSpace(value); clean != "" {
			out = append(out, clean)
		}
	}
	return out
}

func sameText(parts []string, value string) bool {
	value = strings.TrimSpace(value)
	for _, part := range parts {
		if strings.EqualFold(strings.TrimSpace(part), value) {
			return true
		}
	}
	return false
}

func alertAttentionToneEnabled(data map[string]any) bool {
	tone := strings.ToUpper(strings.TrimSpace(firstText(nil, data, "same_tone", "tone_type", "attention_tone")))
	switch tone {
	case "", "NONE", "NO", "OFF", "DISABLED":
		return false
	default:
		return true
	}
}

func (p *feedPlanner) renderAlertTTSAsPCM(ctx context.Context, queueID string, outputPath string, alertText string, sampleRate int, channels int) error {
	readerID := "00"
	language := feedLanguage(p.feed)
	if strings.TrimSpace(alertText) == "" {
		renderCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		product, err := p.bridge.RenderProduct(renderCtx, queueID, p.feed.ID, "alerts")
		cancel()
		if err != nil {
			return err
		}
		alertText = product.Text
		readerID = fallbackText(product.ReaderID, readerID)
		language = fallbackText(product.Language, language)
	}
	if strings.TrimSpace(alertText) == "" {
		return fmt.Errorf("rendered alert text is empty")
	}
	wavPath := filepath.Join(p.cfg.OutputDir, safeID(p.feed.ID), queueID+".wav")
	synthCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()
	wavPath, err := p.bridge.Synthesize(synthCtx, synthJob{
		ID:         queueID,
		Text:       alertText,
		ReaderID:   readerID,
		Language:   language,
		Timezone:   feedTimezone(p.feed),
		OutputPath: wavPath,
	})
	if err != nil {
		return err
	}
	if err := wavToPCM16File(wavPath, outputPath, sampleRate, channels); err == nil {
		return nil
	}
	return convertAudioToPCM(ctx, wavPath, outputPath, sampleRate, channels)
}

func combineSAMEAlertAudio(outputPath string, header []byte, voicePath string, eom []byte, sampleRate int, channels int) error {
	return combineAlertAudio(outputPath, header, voicePath, eom, sampleRate, channels)
}

func combineAttentionAlertAudio(outputPath string, tone []byte, voicePath string, sampleRate int, channels int) error {
	return combineAlertAudio(outputPath, tone, voicePath, nil, sampleRate, channels)
}

func combineAlertAudio(outputPath string, lead []byte, voicePath string, tail []byte, sampleRate int, channels int) error {
	voice, err := os.ReadFile(voicePath)
	if err != nil {
		return err
	}
	if len(voice) == 0 {
		return fmt.Errorf("alert voice audio is empty")
	}
	if err := os.MkdirAll(filepath.Dir(outputPath), 0o755); err != nil {
		return err
	}
	tmp := outputPath + ".tmp"
	file, err := os.Create(tmp)
	if err != nil {
		return err
	}
	writeErr := func() error {
		if len(lead) > 0 {
			if _, err := file.Write(lead); err != nil {
				return err
			}
			if _, err := file.Write(silencePCM(sampleRate, channels, time.Second)); err != nil {
				return err
			}
		}
		if _, err := file.Write(voice); err != nil {
			return err
		}
		if len(tail) > 0 {
			if _, err := file.Write(tail); err != nil {
				return err
			}
		}
		return nil
	}()
	closeErr := file.Close()
	if writeErr != nil {
		_ = os.Remove(tmp)
		return writeErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	return os.Rename(tmp, outputPath)
}

func (p *feedPlanner) buildTextItem(ctx context.Context, title string, text string, now time.Time) (playlistItem, error) {
	if strings.TrimSpace(text) == "" {
		return playlistItem{}, fmt.Errorf("text is required")
	}
	queueID := queueID("operator_text")
	outputPath := filepath.Join(p.cfg.OutputDir, safeID(p.feed.ID), queueID+".wav")
	wavPath, err := p.bridge.Synthesize(ctx, synthJob{
		ID:         queueID,
		Text:       text,
		Language:   feedLanguage(p.feed),
		Timezone:   feedTimezone(p.feed),
		OutputPath: outputPath,
	})
	if err != nil {
		return playlistItem{}, err
	}
	info, err := wavInfo(wavPath)
	if err != nil {
		return playlistItem{}, err
	}
	start := predictedStart(now, p.timelineEnd(now), time.Time{})
	return playlistItem{
		QueueID:           queueID,
		FeedID:            p.feed.ID,
		Kind:              "tts",
		Title:             fallbackText(title, "Operator Text"),
		AudioPath:         wavPath,
		DurationMS:        info.DurationMS,
		QueuedAt:          now.UTC().Format(time.RFC3339Nano),
		PredictedStartAt:  start.UTC().Format(time.RFC3339Nano),
		PredictedFinishAt: start.Add(p.itemScheduleDuration(info.DurationMS)).UTC().Format(time.RFC3339Nano),
		Status:            "queued",
		Source:            "operator",
	}, nil
}

func (p *feedPlanner) buildAudioItem(title string, path string, now time.Time) (playlistItem, error) {
	if strings.TrimSpace(path) == "" {
		return playlistItem{}, fmt.Errorf("audio path is required")
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(p.cfg.BaseDir, path)
	}
	info, err := wavInfo(path)
	if err != nil {
		return playlistItem{}, err
	}
	start := predictedStart(now, p.timelineEnd(now), time.Time{})
	return playlistItem{
		QueueID:           queueID("operator_audio"),
		FeedID:            p.feed.ID,
		Kind:              "audio",
		Title:             fallbackText(title, filepath.Base(path)),
		AudioPath:         path,
		DurationMS:        info.DurationMS,
		QueuedAt:          now.UTC().Format(time.RFC3339Nano),
		PredictedStartAt:  start.UTC().Format(time.RFC3339Nano),
		PredictedFinishAt: start.Add(p.itemScheduleDuration(info.DurationMS)).UTC().Format(time.RFC3339Nano),
		Status:            "queued",
		Source:            "operator",
	}, nil
}

func (p *feedPlanner) applyControl(action string) {
	action = strings.ToLower(strings.TrimSpace(action))
	switch action {
	case "pause":
		p.mode = "paused"
	case "resume", "restart":
		p.mode = "running"
		if action == "restart" {
			p.queue = nil
			p.current = nil
			p.cursor = 0
		}
	case "flush_restart":
		p.queue = nil
		p.current = nil
		p.cursor = 0
		p.mode = "running"
	case "flush_stop":
		p.queue = nil
		p.current = nil
		p.mode = "stopped"
	case "pause_after_current", "flush_restart_after_current", "flush_stop_after_current":
		p.pendingAfterCurrent = action
	}
	p.writeState()
}

func (p *feedPlanner) markPriorityStarted() {
	if p.priorityActive == 0 {
		p.modeBeforePriority = p.mode
		p.publishPlayoutControl("pause")
	}
	p.priorityActive++
	p.mode = "priority"
	p.writeState()
}

func (p *feedPlanner) markPriorityCompleted() {
	if p.priorityActive > 0 {
		p.priorityActive--
	}
	if p.priorityActive == 0 {
		p.mode = fallbackText(p.modeBeforePriority, "running")
		p.modeBeforePriority = ""
		if p.mode == "running" {
			p.publishPlayoutControl("resume")
		}
	}
	p.writeState()
}

func (p *feedPlanner) publishPlayoutControl(action string) {
	if p.bridge == nil {
		return
	}
	if err := p.bridge.Publish(map[string]any{
		"type":    "playlist.control",
		"source":  serviceID,
		"feed_id": p.feed.ID,
		"data":    map[string]any{"feed_id": p.feed.ID, "action": action},
	}); err != nil {
		p.lastError = err.Error()
	}
}

func (p *feedPlanner) markStarted(queueID string) {
	if queueID == "" {
		return
	}
	if p.current != nil && p.current.QueueID == queueID {
		p.current.Status = "playing"
		p.writeState()
		return
	}
	for i := range p.queue {
		if p.queue[i].QueueID == queueID {
			item := p.queue[i]
			item.Status = "playing"
			p.current = &item
			p.queue = append(p.queue[:i], p.queue[i+1:]...)
			p.writeState()
			return
		}
	}
}

func (p *feedPlanner) markInterrupted(queueID string) {
	if p.current != nil && (queueID == "" || p.current.QueueID == queueID) {
		p.current.Status = "interrupted"
		p.writeState()
		return
	}
}

func (p *feedPlanner) markCompleted(queueID string) {
	if p.current != nil && (queueID == "" || p.current.QueueID == queueID) {
		p.current = nil
	}
	switch p.pendingAfterCurrent {
	case "pause_after_current":
		p.mode = "paused"
		p.queue = nil
	case "flush_restart_after_current":
		p.mode = "running"
		p.queue = nil
		p.cursor = 0
	case "flush_stop_after_current":
		p.mode = "stopped"
		p.queue = nil
	}
	p.pendingAfterCurrent = ""
	p.writeState()
}

func (p *feedPlanner) dropCompleted() {
	now := time.Now()
	filtered := p.queue[:0]
	for _, item := range p.queue {
		finish := parseTime(item.PredictedFinishAt)
		if !finish.IsZero() && finish.Before(now.Add(-5*time.Minute)) {
			continue
		}
		filtered = append(filtered, item)
	}
	p.queue = filtered
}

func (p *feedPlanner) queuedCount() int {
	if p.current != nil {
		return len(p.queue) + 1
	}
	return len(p.queue)
}

func (p *feedPlanner) itemScheduleDuration(durationMS int64) time.Duration {
	return time.Duration(durationMS)*time.Millisecond + p.packageGap()
}

func (p *feedPlanner) packageGap() time.Duration {
	gap := p.cfg.Root.Playout.Pacing.PackageGapS
	if gap <= 0 {
		return 0
	}
	return time.Duration(gap * float64(time.Second))
}

func (p *feedPlanner) timelineEnd(now time.Time) time.Time {
	end := now
	if p.current != nil {
		if finish := parseTime(p.current.PredictedFinishAt); finish.After(end) {
			end = finish
		}
	}
	for _, item := range p.queue {
		if finish := parseTime(item.PredictedFinishAt); finish.After(end) {
			end = finish
		}
	}
	return end
}

func (p *feedPlanner) publishReady(item playlistItem) error {
	if err := p.bridge.Publish(map[string]any{
		"type":    "playlist.item.ready",
		"source":  serviceID,
		"feed_id": item.FeedID,
		"data": map[string]any{
			"queue_id":            item.QueueID,
			"feed_id":             item.FeedID,
			"kind":                item.Kind,
			"package_id":          item.PackageID,
			"title":               item.Title,
			"audio_path":          item.AudioPath,
			"duration_ms":         item.DurationMS,
			"queued_at":           item.QueuedAt,
			"target_start_at":     item.TargetStartAt,
			"predicted_start_at":  item.PredictedStartAt,
			"predicted_finish_at": item.PredictedFinishAt,
			"not_before":          item.PredictedStartAt,
			"source":              item.Source,
		},
	}); err != nil {
		return err
	}
	p.lastPendingReplayAt = time.Now()
	return nil
}

func (p *feedPlanner) replayPendingItems() {
	p.replayPendingItemsAt(time.Now())
}

func (p *feedPlanner) replayPendingItemsIfDue(now time.Time) {
	if len(p.queue) == 0 {
		return
	}
	if !p.lastPendingReplayAt.IsZero() && now.Sub(p.lastPendingReplayAt) < pendingReplayInterval {
		return
	}
	p.replayQueuedItemsAt(now)
}

func (p *feedPlanner) replayPendingItemsAt(now time.Time) {
	p.replayItemsAt(now, true)
}

func (p *feedPlanner) replayQueuedItemsAt(now time.Time) {
	p.replayItemsAt(now, false)
}

func (p *feedPlanner) replayItemsAt(now time.Time, includeCurrent bool) {
	p.lastPendingReplayAt = now
	items := make([]playlistItem, 0, len(p.queue)+1)
	if includeCurrent && p.current != nil {
		current := *p.current
		current.Status = "queued"
		items = append(items, current)
	}
	items = append(items, p.queue...)
	for _, item := range items {
		if err := p.publishReady(item); err != nil {
			p.lastError = err.Error()
			return
		}
	}
}

func (p *feedPlanner) writeState() {
	payload := map[string]any{
		"feed_id":               p.feed.ID,
		"feed_name":             feedName(p.feed),
		"mode":                  p.mode,
		"priority_active":       p.priorityActive > 0,
		"pending_after_current": p.pendingAfterCurrent,
		"current":               p.current,
		"queue":                 p.queue,
		"next":                  firstQueued(p.queue),
		"last_error":            p.lastError,
		"updated_at":            time.Now().UTC().Format(time.RFC3339Nano),
	}
	path := filepath.Join(p.cfg.BaseDir, "runtime", "playlists", safeID(p.feed.ID)+".json")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	raw, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return
	}
	tmp := fmt.Sprintf("%s.%d.tmp", path, time.Now().UnixNano())
	if os.WriteFile(tmp, append(raw, '\n'), 0o644) == nil {
		if err := os.Rename(tmp, path); err != nil {
			_ = os.Remove(path)
			_ = os.Rename(tmp, path)
		}
	}
}

func firstQueued(queue []playlistItem) *playlistItem {
	if len(queue) == 0 {
		return nil
	}
	item := queue[0]
	return &item
}

func isPlayoutReadyEvent(event map[string]any) bool {
	if strings.EqualFold(stringAt(event, "source"), "haze-playout") {
		return true
	}
	data := mapAt(event, "data")
	return strings.EqualFold(firstText(event, data, "service"), "haze-playout")
}

func predictedStart(now time.Time, timelineEnd time.Time, target time.Time) time.Time {
	start := timelineEnd
	if start.Before(now) {
		start = now
	}
	if !target.IsZero() && start.Before(target) {
		start = target
	}
	return start
}

func parseTime(raw string) time.Time {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return time.Time{}
	}
	value, err := time.Parse(time.RFC3339Nano, raw)
	if err == nil {
		return value
	}
	value, err = time.Parse(time.RFC3339, raw)
	if err == nil {
		return value
	}
	return time.Time{}
}

func formatOptionalTime(value time.Time) string {
	if value.IsZero() {
		return ""
	}
	return value.UTC().Format(time.RFC3339Nano)
}

func queueID(prefix string) string {
	var raw [6]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return safeID(fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano()))
	}
	return safeID(fmt.Sprintf("%s-%d-%s", prefix, time.Now().UnixNano(), hex.EncodeToString(raw[:])))
}

func safeID(value string) string {
	var builder strings.Builder
	for _, ch := range value {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '-' || ch == '_' || ch == '.' {
			builder.WriteRune(ch)
		}
	}
	if builder.Len() == 0 {
		return "item"
	}
	return builder.String()
}

func uniqueStrings(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]struct{}{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func sleepOrDone(ctx context.Context, duration time.Duration) {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}
