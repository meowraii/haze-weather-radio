package webgateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

// Server is the first-pass Go HTTP/WebSocket gateway for Haze.
type Server struct {
	startedAt  time.Time
	config     Config
	configPath string
	webroot    string
	surface    WebSurface
	auth       *AuthManager
	receiver   *ReceiverManager
	media      *MediaHub
	bannerHub  *BannerHub
	breakIn    *OperatorBreakInManager
}

// WebSurface controls which routes a gateway instance exposes.
type WebSurface string

const (
	SurfaceCombined WebSurface = "combined"
	SurfacePublic   WebSurface = "public"
	SurfaceAdmin    WebSurface = "admin"
)

const maxWebSocketMessageBytes = 1 << 20

// NewServer creates a web gateway server.
func NewServer(config Config, webroot string) *Server {
	return NewServerWithConfigPath(config, "config.yaml", webroot)
}

// NewServerWithConfigPath creates a web gateway server with a source config path.
func NewServerWithConfigPath(config Config, configPath string, webroot string) *Server {
	return NewServerWithSurface(config, configPath, webroot, string(SurfaceCombined))
}

// NewServerWithSurface creates a gateway server constrained to one HTTP surface.
func NewServerWithSurface(config Config, configPath string, webroot string, surface string) *Server {
	hostBridgeAddr := os.Getenv("HAZE_HOST_BRIDGE_ADDR")
	mediaBridgeAddr := firstNonBlank(os.Getenv("HAZE_MEDIA_BRIDGE_ADDR"), hostBridgeAddr)
	mediaHub := NewMediaHub(mediaBridgeAddr)
	bannerHub := NewBannerHub(configPath, hostBridgeAddr)
	return &Server{
		startedAt:  time.Now().UTC(),
		config:     config,
		configPath: filepath.Clean(configPath),
		webroot:    webroot,
		surface:    normalizeSurface(surface),
		auth:       NewAuthManager(config),
		receiver:   NewReceiverManager(config, configPath, mediaHub),
		media:      mediaHub,
		bannerHub:  bannerHub,
		breakIn:    NewOperatorBreakInManager(),
	}
}

// Handler builds the HTTP route tree.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/public/v1/health", s.health)
	if s.surface.allowsPublic() {
		mux.HandleFunc("/", s.publicIndex)
		mux.HandleFunc("/feeds", s.publicIndex)
		mux.HandleFunc("/listen", s.publicIndex)
		mux.HandleFunc("/alerts", s.publicIndex)
		mux.HandleFunc("/alerts/archive", s.publicIndex)
		mux.HandleFunc("/api/public/v1/panel/ws", s.websocket)
		mux.HandleFunc("/api/public/v1/feed/audio", s.publicFeedAudio)
	}
	if s.surface.allowsAdmin() {
		if !s.surface.allowsPublic() {
			mux.HandleFunc("/", s.adminRoot)
		}
		mux.HandleFunc("/admin", s.admin)
		mux.HandleFunc("/banner", s.banner)
		mux.HandleFunc("/api/v1/banner/stream", s.bannerStream)
		mux.HandleFunc("/api/v1/banner/audio", s.bannerAudio)
		mux.HandleFunc("/api/v1/banner/webrtc/offer", s.bannerWebRTCOffer)
		mux.HandleFunc("/api/v1/alerts/archive/cap.xml", s.alertsArchiveCAPXML)
		mux.HandleFunc("/api/v1/wx-on-demand/generate", s.wxOnDemandGenerate)
		mux.HandleFunc("/api/v1/wx-on-demand/packages", s.wxOnDemandPackages)
		mux.HandleFunc("/api/v1/wx-on-demand/readers", s.wxOnDemandReaders)
		mux.HandleFunc("/login", s.login)
		mux.HandleFunc("/api/v1/panel/ws", s.websocket)
	}
	if s.surface.allowsAdmin() && s.receiver != nil && s.receiver.Enabled() {
		base := s.receiver.BasePath()
		mux.HandleFunc(base+"/session", s.receiver.HandleSession)
		mux.HandleFunc(base+"/ws", s.receiver.HandleWebSocket)
	}
	mux.Handle("/assets/", noStore(http.StripPrefix("/assets/", http.FileServer(http.Dir(s.webroot)))))
	return s.withSecurityHeaders(mux)
}

func (s *Server) publicIndex(writer http.ResponseWriter, request *http.Request) {
	if request.URL.Path != "/" && request.URL.Path != "/feeds" && request.URL.Path != "/listen" && request.URL.Path != "/alerts" && request.URL.Path != "/alerts/archive" {
		http.NotFound(writer, request)
		return
	}
	if (request.URL.Path == "/feeds" || request.URL.Path == "/listen") && !s.publicFeedsAvailable() {
		http.NotFound(writer, request)
		return
	}
	if publicAlertsPath(request.URL.Path) && publicAlertsArchiveAccess(s.config) != "public" {
		http.NotFound(writer, request)
		return
	}
	s.serveHTML(writer, request, "index.html")
}

func publicAlertsPath(path string) bool {
	return path == "/alerts" || path == "/alerts/archive"
}

func (s *Server) adminRoot(writer http.ResponseWriter, request *http.Request) {
	if request.URL.Path != "/" {
		http.NotFound(writer, request)
		return
	}
	http.Redirect(writer, request, "/admin", http.StatusSeeOther)
}

func (s *Server) admin(writer http.ResponseWriter, request *http.Request) {
	if token := strings.TrimSpace(request.URL.Query().Get("token")); token != "" && s.auth.ValidToken(token) {
		s.auth.SetCookie(writer, token)
		cleanURL := *request.URL
		query := cleanURL.Query()
		query.Del("token")
		cleanURL.RawQuery = query.Encode()
		http.Redirect(writer, request, cleanURL.RequestURI(), http.StatusSeeOther)
		return
	}
	if !s.auth.Authenticated(request) {
		target := "/login?next=" + request.URL.EscapedPath()
		http.Redirect(writer, request, target, http.StatusSeeOther)
		return
	}
	s.serveHTML(writer, request, "admin.html")
}

func (s *Server) banner(writer http.ResponseWriter, request *http.Request) {
	if !s.auth.Authenticated(request) {
		target := "/login?next=" + request.URL.EscapedPath()
		http.Redirect(writer, request, target, http.StatusSeeOther)
		return
	}
	s.serveHTML(writer, request, "banner.html")
}

func (s *Server) login(writer http.ResponseWriter, request *http.Request) {
	s.serveHTML(writer, request, "login.html")
}

func (s *Server) serveHTML(writer http.ResponseWriter, request *http.Request, name string) {
	writer.Header().Set("Cache-Control", "no-store")
	path := filepath.Join(s.webroot, filepath.Clean(name))
	http.ServeFile(writer, request, path)
}

func (s *Server) health(writer http.ResponseWriter, request *http.Request) {
	webrtcPeers := s.media.WebRTCPeerSnapshots()
	webrtcSources := s.media.WebRTCFrameSourceSnapshots()
	writeJSON(writer, map[string]any{
		"ok":                  true,
		"service":             "haze-web",
		"started_at":          s.startedAt,
		"capabilities":        WebRTCAudioCapabilities(),
		"webrtc_peers":        webrtcPeers,
		"webrtc_peer_count":   len(webrtcPeers),
		"webrtc_sources":      webrtcSources,
		"webrtc_source_count": len(webrtcSources),
	})
}

func (s *Server) websocket(writer http.ResponseWriter, request *http.Request) {
	connection, err := websocket.Accept(writer, request, &websocket.AcceptOptions{
		OriginPatterns: sameOriginPatterns(request),
	})
	if err != nil {
		return
	}
	defer connection.CloseNow()
	connection.SetReadLimit(maxWebSocketMessageBytes)

	ctx, cancel := context.WithCancel(request.Context())
	defer cancel()
	session := &wsSession{
		conn:       connection,
		auth:       s.auth,
		request:    request,
		config:     s.config,
		configPath: s.configPath,
		startedAt:  s.startedAt,
		media:      s.media,
		server:     s,
		breakIns:   map[string]struct{}{},
	}
	defer session.cancelOwnedOperatorBreakIns()
	_ = session.send(ctx, "hello", map[string]any{
		"service": "haze-web",
	})
	publicSocket := strings.HasPrefix(request.URL.Path, "/api/public/")
	if publicSocket {
		if state, err := session.publicState(); err == nil {
			_ = session.send(ctx, "public_state", state)
			session.lastStateSignature = stateSignature(state)
		}
	} else if s.auth.Authenticated(request) {
		if state, err := session.panelState(); err == nil {
			_ = session.send(ctx, "admin_state", state)
			session.lastStateSignature = stateSignature(state)
		}
	}

	type readResult struct {
		data []byte
		err  error
	}
	readDone := make(chan readResult, 1)
	go func() {
		for {
			_, data, err := connection.Read(ctx)
			select {
			case readDone <- readResult{data: data, err: err}:
			case <-ctx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case result := <-readDone:
			if result.err != nil {
				return
			}
			if err := session.handle(ctx, result.data); err != nil {
				return
			}
		case <-ticker.C:
			if publicSocket {
				if state, err := session.publicState(); err == nil {
					if session.shouldSendState(state) {
						_ = session.send(ctx, "public_state", state)
						continue
					}
				}
			} else if s.auth.Authenticated(request) {
				if state, err := session.panelState(); err == nil {
					if session.shouldSendState(state) {
						_ = session.send(ctx, "admin_state", state)
						continue
					}
				}
			}
			_ = session.send(ctx, "heartbeat", map[string]any{
				"timestamp": time.Now().UTC(),
			})
		}
	}
}

type wsSession struct {
	mu                 sync.Mutex
	conn               *websocket.Conn
	auth               *AuthManager
	request            *http.Request
	config             Config
	configPath         string
	startedAt          time.Time
	lastStateSignature string
	media              *MediaHub
	server             *Server
	breakIns           map[string]struct{}
}

func (s *wsSession) send(ctx context.Context, messageType string, data map[string]any) error {
	return s.sendEnvelope(ctx, map[string]any{
		"type":      messageType,
		"timestamp": time.Now().UTC(),
		"data":      data,
	})
}

func (s *wsSession) sendEnvelope(ctx context.Context, payload map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return s.conn.Write(ctx, websocket.MessageText, raw)
}

func (s *wsSession) handle(ctx context.Context, raw []byte) error {
	var message map[string]any
	if err := json.Unmarshal(raw, &message); err != nil {
		return s.reply(ctx, message, "error", map[string]any{"detail": "invalid json"})
	}
	msgType := stringValue(message, "type")
	switch msgType {
	case "auth_check":
		return s.reply(ctx, message, "auth_state", map[string]any{
			"authenticated": s.auth.Authenticated(s.request),
			"auth_enabled":  s.auth.Enabled(),
			"auth_ready":    s.auth.Configured(),
			"site_name":     siteName(s.config),
			"on_air_name":   displayText(s.config.Operator.OnAirName),
			"version":       s.config.Version,
			"git_commit":    "unknown",
		})
	case "login":
		token, err := s.auth.Login(stringValue(message, "password"))
		if err != nil {
			return s.reply(ctx, message, "auth_error", map[string]any{"detail": err.Error()})
		}
		return s.reply(ctx, message, "auth_ok", map[string]any{"token": token})
	case "logout":
		s.auth.Logout(tokenFromRequest(s.request))
		return s.reply(ctx, message, "logout_ok", map[string]any{})
	case "ping":
		return s.reply(ctx, message, "pong", map[string]any{})
	case "webrtc_offer":
		return s.handleWebRTCOffer(ctx, message)
	case "command":
		if !s.auth.Authenticated(s.request) {
			return s.reply(ctx, message, "auth_error", map[string]any{"detail": "not authenticated"})
		}
		result, err := s.handleCommand(stringValue(message, "command"), mapValue(message, "payload"))
		if err != nil {
			return s.reply(ctx, message, "command_error", map[string]any{"detail": err.Error()})
		}
		return s.reply(ctx, message, "command_result", map[string]any{"result": result})
	default:
		if !s.auth.Authenticated(s.request) {
			return s.reply(ctx, message, "auth_error", map[string]any{"detail": "not authenticated"})
		}
		return s.reply(ctx, message, "error", map[string]any{"detail": "unsupported message type"})
	}
}

func (s *wsSession) handleWebRTCOffer(ctx context.Context, message map[string]any) error {
	feedID := strings.TrimSpace(stringValue(message, "feed_id"))
	if feedID == "" {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "feed_id is required"})
	}
	if !s.media.Available() || !s.config.Webpanel.Public.Feeds.WebRTC.Enabled {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "feed streaming is not available"})
	}
	if strings.HasPrefix(s.request.URL.Path, "/api/public/") {
		access := publicFeedAccess(s.config)
		if access == "disabled" {
			return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "public feeds are disabled"})
		}
		if access == "auth_required" && !s.auth.Authenticated(s.request) {
			return s.reply(ctx, message, "auth_error", map[string]any{"detail": "not authenticated"})
		}
	} else if !s.auth.Authenticated(s.request) {
		return s.reply(ctx, message, "auth_error", map[string]any{"detail": "not authenticated"})
	}
	if !s.feedWebRTCEnabled(feedID) {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "feed streaming is not configured or disabled"})
	}
	answer, err := s.media.AnswerWithOptions(ctx, feedID, stringValue(message, "sdp"), WebRTCAnswerOptions{
		DisableG722:    boolValue(message, "disable_g722"),
		RequireOpus:    boolValue(message, "require_opus"),
		PreferredCodec: firstNonBlank(stringValue(message, "codec"), stringValue(message, "preferred_codec")),
	})
	if err != nil {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": err.Error()})
	}
	return s.reply(ctx, message, "webrtc_answer", map[string]any{
		"feed_id":      feedID,
		"sdp":          answer.SDP,
		"sdp_type":     "answer",
		"media_recent": answer.MediaRecent,
		"codec":        answer.Codec.String(),
		"payload_type": answer.PayloadType,
	})
}

func (s *wsSession) feedWebRTCEnabled(feedID string) bool {
	feeds, err := loadFeedSummaries(s.configPath)
	if err != nil {
		return false
	}
	for _, feed := range feeds {
		if stringValue(feed, "id") != feedID {
			continue
		}
		enabled, _ := feed["enabled"].(bool)
		webrtc, _ := feed["webrtc_enabled"].(bool)
		return enabled && webrtc
	}
	return false
}

func (s *wsSession) handleCommand(command string, payload map[string]any) (any, error) {
	switch command {
	case "daemon.settings.get":
		return daemonSettingsPayload(s.configPath)
	case "daemon.settings.save":
		settings, ok := payload["settings"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("settings payload is required")
		}
		return writeDaemonSettings(s.configPath, settings)
	case "daemon.service.control":
		return s.publishServiceControl(payload)
	case "dictionary.get":
		return loadDictionaryPayload(s.configPath)
	case "dictionary.save":
		return writeDictionaryPayload(s.configPath, payload)
	case "bulletins.get":
		return loadBulletinsPayload(s.configPath)
	case "bulletins.save":
		return saveBulletinsPayload(s.configPath, payload)
	case "bulletins.import":
		return importBulletinsPayload(s.configPath, payload)
	case "bulletins.export":
		return exportBulletinsPayload(s.configPath, payload)
	case "bulletins.upload_audio":
		return uploadBulletinAudio(s.configPath, payload)
	case "state":
		return s.panelState()
	case "wx.packages":
		packageIDs, err := loadPackageIDs(s.configPath)
		if err != nil {
			return nil, err
		}
		return map[string]any{"packages": packageIDs}, nil
	case "wx.readers":
		readers, err := loadReaderCatalog(s.configPath)
		if err != nil {
			return nil, err
		}
		return map[string]any{"readers": readers}, nil
	case "playlist.state":
		return playlistStatePayload(s.configPath)
	case "playlist.control":
		return s.publishPlaylistCommand("playlist.control", payload)
	case "playlist.insert":
		if strings.EqualFold(stringValue(payload, "kind"), "same") {
			result, err := s.airSame(payload)
			if err != nil {
				return nil, err
			}
			return map[string]any{"accepted": true, "same": result}, nil
		}
		return s.publishPlaylistCommand("playlist.insert", payload)
	case "alerts.archive.get":
		return alertsArchivePayload(s.configPath)
	case "alerts.archive.action":
		return handleAlertsArchiveAction(s.configPath, payload)
	case "same.event_codes":
		return loadSAMEMapping(s.configPath)
	case "same.location_names":
		return loadLocationNames(s.configPath)
	case "automations.get", "same.templates.get":
		return loadAlertTemplates(s.configPath)
	case "automations.put", "same.templates.put":
		content, _ := payload["content"].(string)
		if content == "" {
			return nil, fmt.Errorf("automation content is required")
		}
		return writeAlertTemplates(s.configPath, content)
	case "same.intro":
		return sameIntroPayload(s.configPath, payload)
	case "same.test":
		return s.generateSameTest(payload)
	case "same.generate":
		return s.generateSame(payload)
	case "same.air":
		return s.airSame(payload)
	case "alert.broadcast":
		return s.broadcastAlert(payload)
	case "operator_breakin.prerolls":
		return s.listOperatorBreakInPrerolls()
	case "operator_breakin.upload_preroll":
		return s.uploadOperatorBreakInPreroll(payload)
	case "operator_breakin.generate_tone":
		return s.generateOperatorBreakInTone(payload)
	case "operator_breakin.start":
		return s.startOperatorBreakIn(payload)
	case "operator_breakin.chunk":
		return s.appendOperatorBreakInChunk(payload)
	case "operator_breakin.finish":
		return s.finishOperatorBreakIn(payload)
	case "operator_breakin.url":
		return s.queueOperatorBreakInURL(payload)
	case "operator_breakin.cancel":
		return s.cancelOperatorBreakIn(payload)
	case "same.upload_audio":
		return nil, fmt.Errorf("SAME media upload is not available in this gateway yet")
	case "wx.generate":
		return s.generateWx(payload)
	case "health":
		return map[string]any{
			"ok":            true,
			"service":       "haze-web",
			"started_at":    s.startedAt,
			"auth_required": s.auth != nil && s.auth.Enabled(),
			"wx_base":       "/api/v1/wx-on-demand",
			"capabilities": map[string]any{
				"same_queue":     true,
				"same_broadcast": false,
				"raw_udp_output": true,
				"wx_generate":    strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR")) != "",
			},
		}, nil
	default:
		return nil, fmt.Errorf("command %q is not implemented by the Go gateway yet", command)
	}
}

func (s *wsSession) publishPlaylistCommand(eventType string, payload map[string]any) (any, error) {
	feedID := strings.TrimSpace(stringValue(payload, "feed_id"))
	if feedID == "" {
		return nil, fmt.Errorf("feed_id is required")
	}
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil, fmt.Errorf("event bridge is not available")
	}
	before, _ := playlistStatePayload(s.configPath)
	previousUpdatedAt := playlistFeedUpdatedAt(before, feedID)
	publisher := events.NewHostBridgePublisher(bridgeAddr)
	defer publisher.Close()
	if err := publisher.Publish(events.Event{
		Type:    eventType,
		Source:  "haze-web",
		Subject: feedID,
		Data:    payload,
	}); err != nil {
		return nil, err
	}
	playlist, settled := waitForPlaylistStateChange(s.configPath, feedID, previousUpdatedAt, 1200*time.Millisecond)
	result := map[string]any{
		"accepted": true,
		"event":    eventType,
		"settled":  settled,
	}
	if playlist != nil {
		result["playlist"] = playlist
	}
	return result, nil
}

func (s *wsSession) publishServiceControl(payload map[string]any) (any, error) {
	serviceID := strings.TrimSpace(firstNonBlank(stringValue(payload, "service_id"), stringValue(payload, "id")))
	action := strings.ToLower(strings.TrimSpace(stringValue(payload, "action")))
	if serviceID == "" {
		return nil, fmt.Errorf("service_id is required")
	}
	if action != "start" && action != "stop" && action != "restart" {
		return nil, fmt.Errorf("unsupported service action %q", action)
	}
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil, fmt.Errorf("event bridge is not available")
	}
	publisher := events.NewHostBridgePublisher(bridgeAddr)
	defer publisher.Close()
	if err := publisher.Publish(events.Event{
		Type:    "service.control",
		Source:  "haze-web",
		Subject: serviceID,
		Data: map[string]any{
			"service_id": serviceID,
			"action":     action,
		},
	}); err != nil {
		return nil, err
	}
	return map[string]any{
		"accepted":   true,
		"event":      "service.control",
		"service_id": serviceID,
		"action":     action,
	}, nil
}

func (s *wsSession) panelState() (map[string]any, error) {
	return panelStatePayload(s.config, s.configPath, s.startedAt, s.request, s.media.Available())
}

func (s *wsSession) publicState() (map[string]any, error) {
	return publicStatePayload(s.config, s.configPath, s.startedAt, s.request, s.auth, s.media.Available())
}

func (s *Server) publicFeedsAvailable() bool {
	feeds, err := loadFeedSummaries(s.configPath)
	if err != nil {
		return false
	}
	return publicWebRTCAvailable(s.config, feeds)
}

func (s *wsSession) shouldSendState(state map[string]any) bool {
	signature := stateSignature(state)
	if signature == "" || signature == s.lastStateSignature {
		return false
	}
	s.lastStateSignature = signature
	return true
}

func (s *wsSession) reply(ctx context.Context, request map[string]any, messageType string, fields map[string]any) error {
	payload := map[string]any{
		"type":      messageType,
		"timestamp": time.Now().UTC(),
	}
	if requestID := stringValue(request, "request_id"); requestID != "" {
		payload["reply_to"] = requestID
	}
	for key, value := range fields {
		payload[key] = value
	}
	return s.sendEnvelope(ctx, payload)
}

func writeJSON(writer http.ResponseWriter, value any) {
	writer.Header().Set("Content-Type", "application/json")
	encoder := json.NewEncoder(writer)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(value); err != nil {
		http.Error(writer, fmt.Sprintf("json encode failed: %v", err), http.StatusInternalServerError)
	}
}

func (s *Server) withSecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		writer.Header().Set("X-Frame-Options", "SAMEORIGIN")
		writer.Header().Set("Referrer-Policy", "no-referrer")
		writer.Header().Set("Permissions-Policy", "camera=(), microphone=(self), geolocation=(), payment=()")
		writer.Header().Set("Content-Security-Policy", contentSecurityPolicy(request.URL.Path))
		if s.config.Webpanel.TLS.HSTS && requestIsHTTPS(request) {
			writer.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		}
		next.ServeHTTP(writer, request)
	})
}

func contentSecurityPolicy(path string) string {
	scriptSrc := "script-src 'self' https://unpkg.com"
	if path == "/banner" {
		scriptSrc = "script-src 'self' 'unsafe-inline'"
	}
	return strings.Join([]string{
		"default-src 'self'",
		scriptSrc,
		"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
		"font-src 'self' https://fonts.gstatic.com",
		"img-src 'self' data:",
		"connect-src 'self' ws: wss: stun: stuns: turn: turns:",
		"media-src 'self' blob: http: https:",
		"object-src 'none'",
		"base-uri 'self'",
		"frame-ancestors 'self'",
		"form-action 'self'",
	}, "; ")
}

func noStore(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Cache-Control", "no-store")
		next.ServeHTTP(writer, request)
	})
}

func sameOriginPatterns(request *http.Request) []string {
	host := request.Host
	if host == "" {
		return nil
	}
	scheme := "http"
	if requestIsHTTPS(request) {
		scheme = "https"
	}
	return []string{scheme + "://" + host}
}

func siteName(config Config) string {
	if config.Webpanel.Public.SiteName != "" {
		return config.Webpanel.Public.SiteName
	}
	return "Haze Weather Radio"
}

// NormalizeSurface converts a CLI/config surface value into a known surface.
func NormalizeSurface(surface string) WebSurface {
	return normalizeSurface(surface)
}

func normalizeSurface(surface string) WebSurface {
	switch strings.ToLower(strings.TrimSpace(surface)) {
	case string(SurfacePublic):
		return SurfacePublic
	case string(SurfaceAdmin):
		return SurfaceAdmin
	default:
		return SurfaceCombined
	}
}

func (s WebSurface) allowsPublic() bool {
	return s == SurfacePublic || s == SurfaceCombined
}

func (s WebSurface) allowsAdmin() bool {
	return s == SurfaceAdmin || s == SurfaceCombined
}

func stringValue(message map[string]any, key string) string {
	value, _ := message[key].(string)
	return value
}

func boolValue(message map[string]any, key string) bool {
	switch value := message[key].(type) {
	case bool:
		return value
	case string:
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "1", "true", "yes", "on", "enabled":
			return true
		}
	}
	return false
}

func mapValue(message map[string]any, key string) map[string]any {
	value, _ := message[key].(map[string]any)
	if value == nil {
		return map[string]any{}
	}
	return value
}
