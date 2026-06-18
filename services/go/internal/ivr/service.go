package ivr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

const serviceID = "haze-ivr"

type Service struct {
	cfg      loadedConfig
	resolver *Resolver
	cache    *ProductCache
	bridge   *bridgeClient
	metrics  metrics
}

type metrics struct {
	Lookups       atomic.Uint64 `json:"-"`
	CacheRequests atomic.Uint64 `json:"-"`
	CacheErrors   atomic.Uint64 `json:"-"`
	SIPMessages   atomic.Uint64 `json:"-"`
	HTTPRequests  atomic.Uint64 `json:"-"`
}

func Run(ctx context.Context, options Options) error {
	cfg, err := loadConfig(options.ConfigPath, options)
	if err != nil {
		return err
	}
	if !cfg.enabled() {
		log.Printf("IVR service disabled")
		<-ctx.Done()
		return nil
	}
	if err := os.MkdirAll(cfg.cacheDir(), 0o755); err != nil {
		return err
	}
	for ctx.Err() == nil {
		bridge, err := connectBridge(ctx, options.BridgeAddr)
		if err != nil {
			log.Printf("IVR waiting for event bridge: %v", err)
			sleepOrDone(ctx, time.Second)
			continue
		}
		service := &Service{
			cfg:      cfg,
			resolver: NewResolver(cfg),
			bridge:   bridge,
		}
		service.cache = NewProductCache(cfg, bridge)
		err = service.runConnected(ctx)
		_ = bridge.Close()
		if ctx.Err() != nil {
			return nil
		}
		log.Printf("IVR bridge disconnected: %v", err)
		sleepOrDone(ctx, time.Second)
	}
	return nil
}

func (s *Service) runConnected(ctx context.Context) error {
	if err := s.bridge.Publish(map[string]any{
		"type":   "service.ready",
		"source": serviceID,
		"data": map[string]any{
			"service":             serviceID,
			"mode":                s.cfg.IVR.Mode,
			"http_addr":           s.cfg.IVR.HTTP.Addr,
			"sip_addr":            s.cfg.IVR.SIP.Listen,
			"cache_dir":           s.cfg.cacheDir(),
			"max_render_inflight": s.cfg.IVR.MaxRenderInflight,
		},
	}); err != nil {
		return err
	}

	errCh := make(chan error, 2)
	if s.cfg.IVR.HTTP.Enabled {
		server := &http.Server{
			Addr:              s.cfg.IVR.HTTP.Addr,
			Handler:           s.routes(),
			ReadHeaderTimeout: 5 * time.Second,
		}
		go func() {
			<-ctx.Done()
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = server.Shutdown(shutdownCtx)
		}()
		go func() {
			log.Printf("IVR provider webhook listening on %s", s.cfg.IVR.HTTP.Addr)
			err := server.ListenAndServe()
			if !errors.Is(err, http.ErrServerClosed) {
				errCh <- err
			}
		}()
	}
	if s.cfg.IVR.SIP.Enabled {
		go func() {
			if err := s.runSIP(ctx); err != nil && ctx.Err() == nil {
				errCh <- err
			}
		}()
	}
	if s.cfg.IVR.Cache.RefreshOnStartup {
		go s.prewarm(ctx, s.cfg.IVR.Cache.PrewarmCodes, false)
	}

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-errCh:
		return err
	}
}

func (s *Service) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/ivr/v1/health", s.handleHealth)
	mux.HandleFunc("/ivr/v1/lookup", s.handleLookup)
	mux.HandleFunc("/ivr/v1/prompt", s.handlePrompt)
	mux.HandleFunc("/ivr/v1/audio", s.handleAudio)
	mux.HandleFunc("/ivr/v1/twiml", s.handleTwiML)
	mux.HandleFunc("/ivr/v1/cache/prewarm", s.handlePrewarm)
	mux.HandleFunc("/ivr/v1/metrics", s.handleMetrics)
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		s.metrics.HTTPRequests.Add(1)
		mux.ServeHTTP(writer, request)
	})
}

func (s *Service) handleHealth(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, map[string]any{
		"ok":      true,
		"service": serviceID,
		"mode":    s.cfg.IVR.Mode,
	})
}

func (s *Service) handleLookup(writer http.ResponseWriter, request *http.Request) {
	code := request.URL.Query().Get("code")
	location, err := s.resolver.Resolve(code)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusNotFound)
		return
	}
	s.metrics.Lookups.Add(1)
	writeJSON(writer, location)
}

func (s *Service) handlePrompt(writer http.ResponseWriter, request *http.Request) {
	menuID := request.URL.Query().Get("menu")
	lineKey := fallbackText(request.URL.Query().Get("line"), "main")
	format := strings.ToLower(strings.TrimSpace(request.URL.Query().Get("format")))
	force := request.URL.Query().Get("force") == "1"
	audio, err := s.cache.GetPrompt(request.Context(), menuID, lineKey, promptValuesFromRequest(request), force)
	if err != nil {
		http.Error(writer, err.Error(), http.StatusBadGateway)
		return
	}
	if format == "pcmu" {
		writer.Header().Set("Content-Type", "audio/basic")
		http.ServeFile(writer, request, audio.PCMUPath)
		return
	}
	writer.Header().Set("Content-Type", "audio/wav")
	http.ServeFile(writer, request, audio.WAVPath)
}

func (s *Service) handleAudio(writer http.ResponseWriter, request *http.Request) {
	format := strings.ToLower(strings.TrimSpace(request.URL.Query().Get("format")))
	force := request.URL.Query().Get("force") == "1"
	packages := packagesFromRequest(request)
	location, err := s.locationFromRequest(request)
	if err != nil {
		s.servePromptAudio(writer, request, "error", "invalid_code", nil, http.StatusNotFound)
		return
	}
	product, err := s.productForLocation(request.Context(), location, packages, force)
	if err != nil {
		log.Printf("IVR product audio unavailable for %s packages=%s: %v", firstNonBlank(location.Code, location.FeedID), strings.Join(normalizePackages(packages, s.cfg.IVR.DefaultPackages), ","), err)
		s.servePromptAudio(writer, request, "weather_product", "unavailable", nil, http.StatusBadGateway)
		return
	}
	if format == "pcmu" {
		writer.Header().Set("Content-Type", "audio/basic")
		http.ServeFile(writer, request, product.PCMUPath)
		return
	}
	writer.Header().Set("Content-Type", "audio/wav")
	http.ServeFile(writer, request, product.WAVPath)
}

func (s *Service) handleTwiML(writer http.ResponseWriter, request *http.Request) {
	state := strings.ToLower(strings.TrimSpace(request.URL.Query().Get("state")))
	switch state {
	case "", "entry":
		if state == "" || strings.TrimSpace(request.FormValue("Digits")) == "" {
			s.writeEntryTwiML(writer, request)
			return
		}
		s.handleEntryDigit(writer, request)
	case "location_code":
		s.handleLocationCodeTwiML(writer, request)
	case "location_menu", "location":
		s.writeLocationMenuTwiML(writer, request)
	case "location_option":
		s.handleLocationOptionTwiML(writer, request)
	default:
		s.writeEntryTwiML(writer, request)
	}
}

func (s *Service) writeEntryTwiML(writer http.ResponseWriter, request *http.Request) {
	entry, _ := s.cfg.Prompts.Menu("entry")
	body := twimlGather(twimlURL(request, "/ivr/v1/twiml", map[string]string{"state": "entry"}), "1", "", entry.Timeout, []string{
		promptURL(request, "entry", "main", nil),
	}, []string{
		twimlPlay(promptURL(request, "error", "timeout", nil)),
	})
	writeTwiML(writer, body)
}

func (s *Service) handleEntryDigit(writer http.ResponseWriter, request *http.Request) {
	digit := strings.TrimSpace(request.FormValue("Digits"))
	option, ok := s.cfg.Prompts.Option("entry", digit)
	if !ok {
		s.writeEntryErrorTwiML(writer, request)
		return
	}
	switch option.Action {
	case "language":
		locationMenu, _ := s.cfg.Prompts.Menu("location_code")
		lang := fallbackText(option.Language, s.cfg.IVR.DefaultLanguage)
		body := twimlGather(twimlURL(request, "/ivr/v1/twiml", map[string]string{
			"state": "location_code",
			"lang":  lang,
		}), "", "#", locationMenu.Timeout, []string{
			promptURL(request, "location_code", "main", nil),
		}, []string{
			twimlPlay(promptURL(request, "error", "timeout", nil)),
		})
		writeTwiML(writer, body)
	case "product":
		location, err := s.defaultFeedLocation()
		if err != nil {
			s.writeUnavailableTwiML(writer, request)
			return
		}
		if option.Language != "" {
			location.Language = option.Language
		}
		s.writeProductTwiML(writer, request, location, splitCSV(option.Packages), "")
	case "broadcast":
		location, err := s.defaultFeedLocation()
		if err != nil {
			writeTwiML(writer, twimlPlay(promptURL(request, "broadcast_menu", "main", nil))+twimlRedirect(twimlURL(request, "/ivr/v1/twiml", nil)))
			return
		}
		s.writeProductTwiML(writer, request, location, s.broadcastPackages(option), "")
	case "operator":
		s.writeOperatorTwiML(writer, request)
	default:
		s.writeEntryErrorTwiML(writer, request)
	}
}

func (s *Service) handleLocationCodeTwiML(writer http.ResponseWriter, request *http.Request) {
	code := firstNonBlank(request.FormValue("Digits"), request.URL.Query().Get("code"))
	if code == "" {
		writeTwiML(writer, twimlPlay(promptURL(request, "error", "timeout", nil)))
		return
	}
	location, err := s.resolver.Resolve(code)
	if err != nil {
		body := twimlPlay(promptURL(request, "error", "invalid_code", nil)) +
			twimlRedirect(twimlURL(request, "/ivr/v1/twiml", nil))
		writeTwiML(writer, body)
		return
	}
	if lang := strings.TrimSpace(request.URL.Query().Get("lang")); lang != "" {
		location.Language = lang
	}
	s.writeLocationMenu(writer, request, location)
}

func (s *Service) writeLocationMenuTwiML(writer http.ResponseWriter, request *http.Request) {
	location, err := s.locationFromRequest(request)
	if err != nil {
		s.writeEntryErrorTwiML(writer, request)
		return
	}
	s.writeLocationMenu(writer, request, location)
}

func (s *Service) writeLocationMenu(writer http.ResponseWriter, request *http.Request, location ResolvedLocation) {
	menu, _ := s.cfg.Prompts.Menu("location_menu")
	body := twimlGather(twimlURL(request, "/ivr/v1/twiml", map[string]string{
		"state": "location_option",
		"code":  location.Code,
		"lang":  location.Language,
	}), "1", "", menu.Timeout, []string{
		promptURL(request, "location_menu", "main", map[string]string{"location": location.Name}),
	}, []string{
		twimlPlay(promptURL(request, "error", "timeout", nil)),
	})
	writeTwiML(writer, body)
}

func (s *Service) handleLocationOptionTwiML(writer http.ResponseWriter, request *http.Request) {
	location, err := s.locationFromRequest(request)
	if err != nil {
		s.writeEntryErrorTwiML(writer, request)
		return
	}
	digit := strings.TrimSpace(request.FormValue("Digits"))
	option, ok := s.cfg.Prompts.Option("location_menu", digit)
	if !ok {
		s.writeLocationMenu(writer, request, location)
		return
	}
	switch option.Action {
	case "product":
		s.writeProductTwiML(writer, request, location, splitCSV(option.Packages), twimlURL(request, "/ivr/v1/twiml", map[string]string{
			"state": "location_menu",
			"code":  location.Code,
			"lang":  location.Language,
		}))
	case "broadcast":
		s.writeProductTwiML(writer, request, location, s.broadcastPackages(option), twimlURL(request, "/ivr/v1/twiml", map[string]string{
			"state": "location_menu",
			"code":  location.Code,
			"lang":  location.Language,
		}))
	default:
		s.writeLocationMenu(writer, request, location)
	}
}

func (s *Service) writeProductTwiML(writer http.ResponseWriter, request *http.Request, location ResolvedLocation, packages []string, afterURL string) {
	queryLocation := location.Code
	packages = normalizePackages(packages, s.cfg.IVR.DefaultPackages)
	params := map[string]string{
		"code":     queryLocation,
		"feed_id":  location.FeedID,
		"lang":     location.Language,
		"packages": strings.Join(packages, ","),
	}
	if queryLocation == "" {
		delete(params, "code")
	}
	body := ""
	if _, ok := s.cache.Fresh(location, packages); !ok {
		body += twimlPlay(promptURL(request, "", "one_moment", nil))
	}
	body += twimlPlay(twimlURL(request, "/ivr/v1/audio", params))
	if afterURL != "" {
		body += twimlRedirect(afterURL)
	}
	writeTwiML(writer, body)
}

func (s *Service) broadcastPackages(option menuOption) []string {
	if packages := splitCSV(option.Packages); len(packages) > 0 {
		return packages
	}
	return append([]string(nil), s.cfg.IVR.BroadcastPackages...)
}

func (s *Service) servePromptAudio(writer http.ResponseWriter, request *http.Request, menuID string, lineKey string, values map[string]string, status int) {
	audio, err := s.cache.GetPrompt(request.Context(), menuID, lineKey, values, false)
	if err != nil {
		http.Error(writer, err.Error(), status)
		return
	}
	format := strings.ToLower(strings.TrimSpace(request.URL.Query().Get("format")))
	if format == "pcmu" {
		writer.Header().Set("Content-Type", "audio/basic")
		http.ServeFile(writer, request, audio.PCMUPath)
		return
	}
	writer.Header().Set("Content-Type", "audio/wav")
	http.ServeFile(writer, request, audio.WAVPath)
}

func (s *Service) writeOperatorTwiML(writer http.ResponseWriter, request *http.Request) {
	menu, _ := s.cfg.Prompts.Menu("operator")
	if strings.TrimSpace(menu.TransferURI) != "" {
		writeTwiML(writer, "<Dial>"+html.EscapeString(strings.TrimSpace(menu.TransferURI))+"</Dial>")
		return
	}
	writeTwiML(writer, twimlPlay(promptURL(request, "operator", "main", nil))+twimlRedirect(twimlURL(request, "/ivr/v1/twiml", nil)))
}

func (s *Service) writeEntryErrorTwiML(writer http.ResponseWriter, request *http.Request) {
	writeTwiML(writer, twimlPlay(promptURL(request, "error", "invalid_code", nil))+twimlRedirect(twimlURL(request, "/ivr/v1/twiml", nil)))
}

func (s *Service) writeUnavailableTwiML(writer http.ResponseWriter, request *http.Request) {
	writeTwiML(writer, twimlPlay(promptURL(request, "error", "unavailable", nil))+twimlRedirect(twimlURL(request, "/ivr/v1/twiml", nil)))
}

func (s *Service) handlePrewarm(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		http.Error(writer, "POST required", http.StatusMethodNotAllowed)
		return
	}
	codes := s.cfg.IVR.Cache.PrewarmCodes
	if raw := request.URL.Query().Get("codes"); raw != "" {
		codes = splitCSV(raw)
	}
	s.prewarm(request.Context(), codes, request.URL.Query().Get("force") == "1")
	writeJSON(writer, map[string]any{"accepted": true, "codes": codes})
}

func (s *Service) handleMetrics(writer http.ResponseWriter, _ *http.Request) {
	writeJSON(writer, map[string]any{
		"lookups":        s.metrics.Lookups.Load(),
		"cache_requests": s.metrics.CacheRequests.Load(),
		"cache_errors":   s.metrics.CacheErrors.Load(),
		"sip_messages":   s.metrics.SIPMessages.Load(),
		"http_requests":  s.metrics.HTTPRequests.Load(),
	})
}

func (s *Service) productForCode(ctx context.Context, code string, packages []string, force bool) (CachedProduct, error) {
	location, err := s.resolver.Resolve(code)
	if err != nil {
		return CachedProduct{}, err
	}
	return s.productForLocation(ctx, location, packages, force)
}

func (s *Service) productForLocation(ctx context.Context, location ResolvedLocation, packages []string, force bool) (CachedProduct, error) {
	s.metrics.CacheRequests.Add(1)
	product, err := s.cache.Get(ctx, location, packages, force)
	if err != nil {
		s.metrics.CacheErrors.Add(1)
		return CachedProduct{}, err
	}
	return product, nil
}

func (s *Service) locationFromRequest(request *http.Request) (ResolvedLocation, error) {
	code := request.URL.Query().Get("code")
	if code == "" {
		code = request.FormValue("code")
	}
	var location ResolvedLocation
	var err error
	if strings.TrimSpace(code) != "" {
		location, err = s.resolver.Resolve(code)
	} else {
		location, err = s.defaultFeedLocationForID(firstNonBlank(request.URL.Query().Get("feed_id"), request.FormValue("feed_id")))
	}
	if err != nil {
		return ResolvedLocation{}, err
	}
	if lang := firstNonBlank(request.URL.Query().Get("lang"), request.FormValue("lang")); lang != "" {
		location.Language = lang
	}
	return location, nil
}

func (s *Service) defaultFeedLocation() (ResolvedLocation, error) {
	return s.defaultFeedLocationForID("")
}

func (s *Service) defaultFeedLocationForID(feedID string) (ResolvedLocation, error) {
	for _, feed := range s.cfg.Feeds {
		if strings.TrimSpace(feedID) != "" && !strings.EqualFold(feed.ID, feedID) {
			continue
		}
		if !xmlBool(feed.EnabledRaw, true) {
			continue
		}
		return ResolvedLocation{
			Code:     "",
			Source:   "feed",
			Name:     fallbackText(feedDisplayName(feed), feed.ID),
			FeedID:   feed.ID,
			Language: s.resolver.feedLanguage(feed.ID),
			Timezone: strings.TrimSpace(feed.Timezone),
		}, nil
	}
	if strings.TrimSpace(feedID) != "" {
		return ResolvedLocation{}, fmt.Errorf("feed %q is not configured or enabled", feedID)
	}
	return ResolvedLocation{}, fmt.Errorf("no enabled IVR feed is configured")
}

func (s *Service) prewarm(ctx context.Context, codes []string, force bool) {
	for _, code := range codes {
		if ctx.Err() != nil {
			return
		}
		if strings.TrimSpace(code) == "" {
			continue
		}
		if _, err := s.productForCode(ctx, code, nil, force); err != nil {
			log.Printf("IVR prewarm failed for %s: %v", code, err)
		}
	}
}

func writeJSON(writer http.ResponseWriter, value any) {
	writer.Header().Set("Content-Type", "application/json")
	encoder := json.NewEncoder(writer)
	encoder.SetIndent("", "  ")
	_ = encoder.Encode(value)
}

func packagesFromRequest(request *http.Request) []string {
	raw := request.URL.Query().Get("packages")
	if raw == "" {
		raw = request.FormValue("packages")
	}
	return splitCSV(raw)
}

func splitCSV(raw string) []string {
	fields := strings.Split(raw, ",")
	out := make([]string, 0, len(fields))
	for _, field := range fields {
		field = strings.TrimSpace(field)
		if field != "" {
			out = append(out, field)
		}
	}
	return out
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func promptValuesFromRequest(request *http.Request) map[string]string {
	values := map[string]string{}
	for _, key := range []string{"location", "code", "feed_id", "lang"} {
		if value := firstNonBlank(request.URL.Query().Get(key), request.FormValue(key)); value != "" {
			values[key] = value
		}
	}
	return values
}

func feedDisplayName(feed feedXML) string {
	transmitter := stationTransmitter(feed)
	return firstNonBlank(transmitter.SiteName, transmitter.Callsign, feed.ID)
}

func stationTransmitter(feed feedXML) transmitterXML {
	transmitters := transmitterList(feed)
	for _, transmitter := range transmitters {
		if transmitter.isRelationship("replaces") && strings.TrimSpace(transmitter.Callsign) != "" {
			return transmitter
		}
	}
	for _, transmitter := range transmitters {
		if transmitter.isRelationship("replaces") {
			return transmitter
		}
	}
	for _, transmitter := range transmitters {
		if transmitter.isRelationship("primary") {
			return transmitter
		}
	}
	if len(transmitters) > 0 {
		return transmitters[0]
	}
	return transmitterXML{}
}

func transmitterList(feed feedXML) []transmitterXML {
	out := make([]transmitterXML, 0, len(feed.Transmitter.Transmitters))
	for _, transmitter := range feed.Transmitter.Transmitters {
		if transmitter.empty() {
			continue
		}
		out = append(out, transmitter)
	}
	return out
}

func (t transmitterXML) empty() bool {
	return strings.TrimSpace(t.SiteName) == "" &&
		strings.TrimSpace(t.Callsign) == "" &&
		strings.TrimSpace(t.Relationship) == "" &&
		strings.TrimSpace(t.HostName) == "" &&
		strings.TrimSpace(t.FrequencyMHz.Value) == ""
}

func (t transmitterXML) isRelationship(relationship string) bool {
	current := strings.ToLower(strings.TrimSpace(t.Relationship))
	if current == "" {
		current = "unknown"
	}
	if current == "secondary/repeater" {
		current = "secondary"
	}
	wanted := strings.ToLower(strings.TrimSpace(relationship))
	return current == wanted || (wanted == "repeater" && current == "secondary")
}

func promptURL(request *http.Request, menuID string, lineKey string, values map[string]string) string {
	params := map[string]string{
		"line": lineKey,
	}
	if menuID != "" {
		params["menu"] = menuID
	}
	for key, value := range values {
		if strings.TrimSpace(value) != "" {
			params[key] = value
		}
	}
	return twimlURL(request, "/ivr/v1/prompt", params)
}

func twimlURL(request *http.Request, path string, params map[string]string) string {
	values := url.Values{}
	for key, value := range params {
		if strings.TrimSpace(value) != "" {
			values.Set(key, value)
		}
	}
	base := externalBaseURL(request) + path
	if encoded := values.Encode(); encoded != "" {
		base += "?" + encoded
	}
	return base
}

func externalBaseURL(request *http.Request) string {
	scheme := firstNonBlank(request.Header.Get("X-Forwarded-Proto"), request.URL.Scheme)
	if scheme == "" {
		scheme = "http"
		if request.TLS != nil {
			scheme = "https"
		}
	}
	host := firstNonBlank(request.Header.Get("X-Forwarded-Host"), request.Host)
	return strings.TrimRight(scheme+"://"+host, "/")
}

func twimlGather(action string, numDigits string, finishOnKey string, timeout time.Duration, plays []string, fallback []string) string {
	seconds := int(timeout.Seconds())
	if seconds <= 0 {
		seconds = 8
	}
	var builder strings.Builder
	builder.WriteString(`<Gather input="dtmf" action="`)
	builder.WriteString(html.EscapeString(action))
	builder.WriteString(`" timeout="`)
	builder.WriteString(fmt.Sprint(seconds))
	builder.WriteString(`"`)
	if numDigits != "" {
		builder.WriteString(` numDigits="`)
		builder.WriteString(html.EscapeString(numDigits))
		builder.WriteString(`"`)
	}
	if finishOnKey != "" {
		builder.WriteString(` finishOnKey="`)
		builder.WriteString(html.EscapeString(finishOnKey))
		builder.WriteString(`"`)
	}
	builder.WriteString(`>`)
	for _, play := range plays {
		builder.WriteString(twimlPlay(play))
	}
	builder.WriteString(`</Gather>`)
	for _, item := range fallback {
		builder.WriteString(item)
	}
	return builder.String()
}

func twimlPlay(url string) string {
	if strings.TrimSpace(url) == "" {
		return ""
	}
	return "<Play>" + html.EscapeString(strings.TrimSpace(url)) + "</Play>"
}

func twimlRedirect(url string) string {
	if strings.TrimSpace(url) == "" {
		return ""
	}
	return "<Redirect>" + html.EscapeString(strings.TrimSpace(url)) + "</Redirect>"
}

func writeTwiML(writer http.ResponseWriter, body string) {
	writer.Header().Set("Content-Type", "text/xml; charset=utf-8")
	_, _ = writer.Write([]byte(`<?xml version="1.0" encoding="UTF-8"?><Response>` + body + `</Response>`))
}

func sleepOrDone(ctx context.Context, duration time.Duration) {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}
