package webgateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path"
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
	mediaServiceURL := mediaServiceBaseURL(config)
	mediaHub := NewMediaHub(mediaBridgeAddr)
	mediaHub.SetHTTPSource(mediaServiceURL)
	bannerHub := NewBannerHub(configPath, hostBridgeAddr)
	server := &Server{
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
	if server.surface.allowsAdmin() {
		server.startBannerStatePublisher(hostBridgeAddr)
	}
	return server
}

// Handler builds the HTTP route tree.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	if s.surface.allowsPublic() {
		mux.HandleFunc("/api/public/v1/health", s.publicHealth)
		mux.HandleFunc("/", s.publicIndex)
		mux.HandleFunc("/feeds", s.publicIndex)
		mux.HandleFunc("/listen", s.publicIndex)
		mux.HandleFunc("/alerts", s.publicIndex)
		mux.HandleFunc("/alerts/archive", s.publicIndex)
		mux.HandleFunc("/api/public/v1/panel/ws", s.websocket)
		mux.HandleFunc("/api/public/v1/feed/audio", s.publicFeedAudio)
		mux.HandleFunc("/api/public/v1/alerts/archive/cap.xml", s.publicAlertsArchiveCAPXML)
	}
	if s.surface.allowsAdmin() {
		if !s.surface.allowsPublic() {
			mux.HandleFunc("/", s.adminRoot)
		}
		mux.HandleFunc("/admin", s.admin)
		mux.HandleFunc("/banner", s.banner)
		mux.HandleFunc("/api/v1/banner/current", s.bannerCurrent)
		mux.HandleFunc("/api/v1/banner/stream", s.bannerStream)
		mux.HandleFunc("/api/v1/banner/audio", s.bannerAudio)
		mux.HandleFunc("/api/v1/banner/webrtc/offer", s.bannerWebRTCOffer)
		mux.HandleFunc("/api/v1/cgen/preview", s.cgenPreview)
		mux.HandleFunc("/api/v1/alerts/archive/cap.xml", s.alertsArchiveCAPXML)
		mux.HandleFunc("/api/v1/alert/audio", s.alertAudioUpload)
		mux.HandleFunc("/api/v1/wx-on-demand/generate", s.wxOnDemandGenerate)
		mux.HandleFunc("/api/v1/wx-on-demand/packages", s.wxOnDemandPackages)
		mux.HandleFunc("/api/v1/wx-on-demand/readers", s.wxOnDemandReaders)
		mux.HandleFunc("/api/v1/health", s.adminHealth)
		mux.HandleFunc("/login", s.login)
		mux.HandleFunc("/api/v1/auth/check", s.authCheckAPI)
		mux.HandleFunc("/api/v1/auth/login", s.loginAPI)
		mux.HandleFunc("/api/v1/panel/ws", s.websocket)
	}
	if s.surface.allowsAdmin() && s.receiver != nil && s.receiver.Enabled() {
		base := s.receiver.BasePath()
		mux.HandleFunc(base+"/pair/challenge", s.receiver.HandlePairChallenge)
		mux.HandleFunc(base+"/pair/complete", s.receiver.HandlePairComplete)
		mux.HandleFunc(base+"/session", s.receiver.HandleSession)
		mux.HandleFunc(base+"/ws", s.receiver.HandleWebSocket)
	}
	mux.HandleFunc("/assets/", s.staticAsset)
	return s.withSecurityHeaders(mux)
}

func (s *Server) publicIndex(writer http.ResponseWriter, request *http.Request) {
	if request.URL.Path != "/" && request.URL.Path != "/feeds" && request.URL.Path != "/listen" && request.URL.Path != "/alerts" && request.URL.Path != "/alerts/archive" {
		http.NotFound(writer, request)
		return
	}
	if !requestMethodGETOrHEAD(writer, request) {
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
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
	http.Redirect(writer, request, "/admin", http.StatusSeeOther)
}

func (s *Server) admin(writer http.ResponseWriter, request *http.Request) {
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
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
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
	s.serveHTML(writer, request, "banner.html")
}

func (s *Server) login(writer http.ResponseWriter, request *http.Request) {
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
	s.serveHTML(writer, request, "login.html")
}

func (s *Server) loginAPI(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		writer.Header().Set("Allow", "POST")
		http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writer.Header().Set("Cache-Control", "no-store")
	writer.Header().Set("Content-Type", "application/json")
	request.Body = http.MaxBytesReader(writer, request.Body, 8*1024)
	var payload struct {
		Password string `json:"password"`
	}
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		http.Error(writer, `{"type":"auth_error","detail":"invalid login request"}`, http.StatusBadRequest)
		return
	}
	token, err := s.auth.Login(payload.Password)
	if err != nil {
		writer.WriteHeader(http.StatusUnauthorized)
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"type":   "auth_error",
			"detail": err.Error(),
		})
		return
	}
	s.auth.SetCookie(writer, token)
	_ = json.NewEncoder(writer).Encode(map[string]any{
		"type":  "auth_ok",
		"token": token,
	})
}

func (s *Server) authCheckAPI(writer http.ResponseWriter, request *http.Request) {
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
	writer.Header().Set("Cache-Control", "no-store")
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(map[string]any{
		"type":          "auth_state",
		"authenticated": s.auth.Authenticated(request),
		"auth_enabled":  s.auth.Enabled(),
		"auth_ready":    s.auth.Configured(),
		"site_name":     siteName(s.config),
		"on_air_name":   displayText(s.config.Operator.OnAirName),
		"version":       s.config.Version,
		"git_commit":    "unknown",
	})
}

func (s *Server) serveHTML(writer http.ResponseWriter, request *http.Request, name string) {
	writer.Header().Set("Cache-Control", "no-store")
	path := filepath.Join(s.webroot, filepath.Clean(name))
	http.ServeFile(writer, request, path)
}

func (s *Server) staticAsset(writer http.ResponseWriter, request *http.Request) {
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
	assetPath := strings.TrimPrefix(request.URL.Path, "/assets/")
	clean := path.Clean("/" + assetPath)
	if clean == "/" || strings.HasSuffix(clean, "/") {
		http.NotFound(writer, request)
		return
	}
	ext := strings.ToLower(path.Ext(clean))
	if ext == "" || ext == ".html" || ext == ".htm" || ext == ".map" || assetPathHasHiddenSegment(clean) {
		http.NotFound(writer, request)
		return
	}
	assetFile, ok := s.staticAssetPath(clean)
	if !ok {
		http.NotFound(writer, request)
		return
	}
	if ext == ".webmanifest" {
		writer.Header().Set("Content-Type", "application/manifest+json")
	}
	writer.Header().Set("Cache-Control", staticAssetCacheControl(ext))
	http.ServeFile(writer, request, assetFile)
}

func assetPathHasHiddenSegment(cleanURLPath string) bool {
	for _, segment := range strings.Split(strings.Trim(cleanURLPath, "/"), "/") {
		if strings.HasPrefix(segment, ".") {
			return true
		}
	}
	return false
}

func staticAssetCacheControl(ext string) string {
	switch strings.ToLower(ext) {
	case ".gif", ".png", ".jpg", ".jpeg", ".webp", ".svg", ".ico", ".woff", ".woff2":
		return "public, max-age=86400"
	case ".js", ".css", ".webmanifest", ".json":
		return "public, max-age=3600, must-revalidate"
	default:
		return "public, max-age=3600"
	}
}

func (s *Server) staticAssetPath(cleanURLPath string) (string, bool) {
	localPath := filepath.FromSlash(strings.TrimPrefix(cleanURLPath, "/"))
	if localPath == "" || filepath.IsAbs(localPath) || filepath.VolumeName(localPath) != "" {
		return "", false
	}
	root, err := filepath.Abs(s.webroot)
	if err != nil {
		return "", false
	}
	target, err := filepath.Abs(filepath.Join(root, localPath))
	if err != nil {
		return "", false
	}
	rel, err := filepath.Rel(root, target)
	if err != nil || rel == "." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) || rel == ".." || filepath.IsAbs(rel) {
		return "", false
	}
	return target, true
}

func (s *Server) publicHealth(writer http.ResponseWriter, request *http.Request) {
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
	payload := map[string]any{
		"ok":           true,
		"service":      "haze-web",
		"started_at":   s.startedAt,
		"capabilities": WebRTCAudioCapabilities(),
	}
	if health, ok := mediaServiceHealth(request.Context(), s.config); ok {
		payload["media_service"] = publicMediaServiceHealth(health)
	}
	writeJSON(writer, payload)
}

func (s *Server) adminHealth(writer http.ResponseWriter, request *http.Request) {
	if !requestMethodGETOrHEAD(writer, request) {
		return
	}
	if !s.auth.Authenticated(request) {
		http.Error(writer, "unauthorized", http.StatusUnauthorized)
		return
	}
	webrtcPeers := s.media.WebRTCPeerSnapshots()
	webrtcSources := s.media.WebRTCFrameSourceSnapshots()
	payload := map[string]any{
		"ok":                  true,
		"service":             "haze-web",
		"started_at":          s.startedAt,
		"capabilities":        WebRTCAudioCapabilities(),
		"webrtc_peers":        webrtcPeers,
		"webrtc_peer_count":   len(webrtcPeers),
		"webrtc_sources":      webrtcSources,
		"webrtc_source_count": len(webrtcSources),
	}
	if s.receiver != nil {
		receivers := s.receiver.ActiveSnapshots()
		payload["receiver_connections"] = receivers
		payload["receiver_connection_count"] = len(receivers)
	}
	if health, ok := mediaServiceHealth(request.Context(), s.config); ok {
		payload["media_service"] = health
	}
	writeJSON(writer, payload)
}

func publicMediaServiceHealth(health map[string]any) map[string]any {
	out := map[string]any{
		"ok":                  boolValue(health, "ok"),
		"service":             stringValue(health, "service"),
		"backend":             stringValue(health, "backend"),
		"gstreamer_available": boolValue(health, "gstreamer_available"),
	}
	if capabilities, _ := health["capabilities"].(map[string]any); capabilities != nil {
		out["capabilities"] = map[string]any{
			"http_audio":         boolValue(capabilities, "http_audio"),
			"encoded_http_audio": boolValue(capabilities, "encoded_http_audio"),
			"webrtc":             boolValue(capabilities, "webrtc"),
			"webrtc_reason":      stringValue(capabilities, "webrtc_reason"),
		}
	}
	feeds, _ := health["feeds"].([]any)
	allAudioOK := true
	warningCount := 0
	feedCount := 0
	for _, item := range feeds {
		feed, _ := item.(map[string]any)
		if feed == nil {
			continue
		}
		feedCount++
		if !boolValue(feed, "audio_ok") {
			allAudioOK = false
		}
		if warnings, _ := feed["audio_warnings"].([]any); warnings != nil {
			warningCount += len(warnings)
		}
	}
	out["feed_count"] = feedCount
	out["audio_ok"] = allAudioOK
	out["audio_warning_count"] = warningCount
	return out
}

func (s *Server) websocket(writer http.ResponseWriter, request *http.Request) {
	connection, err := websocket.Accept(writer, request, &websocket.AcceptOptions{
		OriginPatterns: allOriginPatterns(),
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
	publicStateCache   map[string]any
	publicStateCacheAt time.Time
}

const (
	webRTCOfferMaxFeedIDLength = 128
	webRTCOfferMaxSDPLength    = 256 * 1024
	webRTCOfferMaxCodecLength  = 32
	publicAlertStateCacheTTL   = 15 * time.Second
)

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
	if strings.HasPrefix(s.request.URL.Path, "/api/public/") {
		switch msgType {
		case "ping":
			return s.reply(ctx, message, "pong", map[string]any{})
		case "webrtc_offer":
			return s.handleWebRTCOffer(ctx, message)
		default:
			return s.reply(ctx, message, "error", map[string]any{"detail": "unsupported public message type"})
		}
	}
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
	if len(feedID) > webRTCOfferMaxFeedIDLength {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "feed_id is too long"})
	}
	if !validPublicAudioFeedID(feedID) {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "feed_id is invalid"})
	}
	offerSDP := normalizeWebRTCOfferSDP(firstNonBlank(
		stringValue(message, "sdp"),
		stringValue(mapValue(message, "data"), "sdp"),
	))
	if offerSDP == "" {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "sdp is required"})
	}
	if len(offerSDP) > webRTCOfferMaxSDPLength {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "sdp is too long"})
	}
	preferredCodec := firstNonBlank(stringValue(message, "codec"), stringValue(message, "preferred_codec"))
	if len(strings.TrimSpace(preferredCodec)) > webRTCOfferMaxCodecLength {
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "codec is too long"})
	}
	legacyMediaAvailable := s.media != nil && s.media.Available()
	mediaServiceConfigured := mediaServiceBaseURL(s.config) != ""
	if (!legacyMediaAvailable && !mediaServiceConfigured) || !s.config.Webpanel.Public.Feeds.WebRTC.Enabled {
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
	if answer, ok := s.server.mediaServiceWebRTCAnswer(ctx, map[string]any{
		"feed_id":         feedID,
		"sdp":             offerSDP,
		"disable_g722":    boolValue(message, "disable_g722"),
		"require_opus":    boolValue(message, "require_opus"),
		"preferred_codec": preferredCodec,
		"client_ip":       clientIPForMediaRequest(s.request),
		"remote_addr":     s.request.RemoteAddr,
	}); ok {
		answer["feed_id"] = feedID
		if _, ok := answer["sdp_type"]; !ok {
			answer["sdp_type"] = "answer"
		}
		return s.reply(ctx, message, "webrtc_answer", answer)
	}
	if !legacyMediaAvailable {
		if mediaServiceConfigured {
			return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "haze-media WebRTC service is unavailable and legacy media bridge is not available"})
		}
		return s.reply(ctx, message, "webrtc_error", map[string]any{"detail": "media bridge is not available"})
	}
	answer, err := s.media.AnswerWithOptions(ctx, feedID, offerSDP, WebRTCAnswerOptions{
		DisableG722:    boolValue(message, "disable_g722"),
		RequireOpus:    boolValue(message, "require_opus"),
		PreferredCodec: preferredCodec,
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
	return feedAudioOutputEnabled(s.configPath, feedID)
}

func normalizeWebRTCOfferSDP(sdp string) string {
	sdp = strings.TrimSpace(sdp)
	if sdp == "" {
		return ""
	}
	sdp = strings.ReplaceAll(sdp, "\r\n", "\n")
	sdp = strings.ReplaceAll(sdp, "\r", "\n")
	lines := strings.Split(sdp, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimRight(line, " \t")
		if line == "" {
			continue
		}
		out = append(out, line)
	}
	if len(out) == 0 {
		return ""
	}
	return strings.Join(out, "\r\n") + "\r\n"
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
	case "tts.get":
		return loadTTSPayload(s.configPath)
	case "tts.save":
		return saveTTSPayload(s.configPath, payload)
	case "tts.preview":
		return s.previewTTS(payload)
	case "feeds.get":
		return loadFeedsControlPayload(s.configPath)
	case "feeds.save":
		return saveFeedsControlPayload(s.configPath, payload)
	case "feeds.control":
		return s.publishFeedControl(payload)
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
	case "cgen.get":
		return loadCgenPayload(s.configPath)
	case "cgen.catalog":
		return cgenCatalogPayload(s.configPath)
	case "cgen.save":
		result, err := saveCgenPayload(s.configPath, payload)
		if err != nil {
			return nil, err
		}
		_ = s.publishCgenConfigUpdated(result)
		return result, nil
	case "cgen.action":
		result, err := cgenActionPayload(s.configPath, payload)
		if err != nil {
			return nil, err
		}
		_ = s.publishCgenControl(payload, result)
		return result, nil
	case "state":
		return s.panelState()
	case "wx.packages":
		packageIDs, err := loadWxOnDemandPackageIDs(s.configPath)
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
	case "alert.preview":
		return s.previewAlert(payload)
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

func (s *wsSession) publishCgenConfigUpdated(payload map[string]any) error {
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil
	}
	publisher := events.NewHostBridgePublisher(bridgeAddr)
	defer publisher.Close()
	return publisher.Publish(events.Event{
		Type:   "cgen.config.updated",
		Source: "haze-web",
		Data:   payload,
	})
}

func (s *wsSession) publishCgenControl(request map[string]any, result map[string]any) error {
	bridgeAddr := strings.TrimSpace(os.Getenv("HAZE_HOST_BRIDGE_ADDR"))
	if bridgeAddr == "" {
		return nil
	}
	feedID := strings.TrimSpace(firstNonBlank(stringValue(request, "feed_id"), stringValue(request, "id")))
	if feedID == "" {
		return nil
	}
	feed, ok := cgenPayloadFeed(result, feedID)
	if !ok {
		return nil
	}
	data := cloneMap(feed)
	data["feed_id"] = feedID
	data["action"] = strings.ToLower(strings.TrimSpace(stringValue(request, "action")))
	publisher := events.NewHostBridgePublisher(bridgeAddr)
	defer publisher.Close()
	return publisher.Publish(events.Event{
		Type:    "cgen.control",
		Source:  "haze-web",
		Subject: feedID,
		Data:    data,
	})
}

func cgenPayloadFeed(payload map[string]any, feedID string) (map[string]any, bool) {
	feeds, ok := payload["feeds"].([]map[string]any)
	if !ok {
		return nil, false
	}
	for _, feed := range feeds {
		if strings.EqualFold(strings.TrimSpace(stringValue(feed, "id")), feedID) {
			return feed, true
		}
	}
	return nil, false
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

func (s *wsSession) publishFeedControl(payload map[string]any) (any, error) {
	action := strings.ToLower(strings.TrimSpace(stringValue(payload, "action")))
	switch action {
	case "pause":
		payload["action"] = "pause"
	case "unpause", "resume":
		payload["action"] = "resume"
	case "stop":
		payload["action"] = "flush_stop"
	case "restart", "start":
		payload["action"] = "flush_restart"
	default:
		return nil, fmt.Errorf("unsupported feed action %q", action)
	}
	return s.publishPlaylistCommand("playlist.control", payload)
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
	if s.publicAlertStateCacheable() && s.publicStateCache != nil && time.Since(s.publicStateCacheAt) < publicAlertStateCacheTTL {
		return s.publicStateCache, nil
	}
	state, err := publicStatePayload(s.config, s.configPath, s.startedAt, s.request, s.auth, s.media.Available())
	if err != nil {
		return nil, err
	}
	if s.publicAlertStateCacheable() {
		s.publicStateCache = state
		s.publicStateCacheAt = time.Now()
	}
	return state, nil
}

func (s *wsSession) publicAlertStateCacheable() bool {
	return publicRequestWantsAlerts(s.request) && !publicRequestWantsFeeds(s.request)
}

func (s *Server) publicFeedsAvailable() bool {
	feeds, err := loadBasicFeedSummaries(s.configPath)
	if err != nil {
		return false
	}
	return publicFeedPagesAvailable(s.config, feeds)
}

func publicFeedPagesAvailable(config Config, feeds []map[string]any) bool {
	if publicFeedAccess(config) == "disabled" {
		return false
	}
	for _, feed := range feeds {
		if enabled, _ := feed["enabled"].(bool); !enabled {
			continue
		}
		if httpStream, _ := feed["http_stream_enabled"].(bool); httpStream {
			return true
		}
		if webrtc, _ := feed["webrtc_enabled"].(bool); webrtc && config.Webpanel.Public.Feeds.WebRTC.Enabled {
			return true
		}
	}
	return false
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
	if writer.Header().Get("Cache-Control") == "" {
		writer.Header().Set("Cache-Control", "no-store")
	}
	if err := json.NewEncoder(writer).Encode(value); err != nil {
		http.Error(writer, "json encode failed", http.StatusInternalServerError)
	}
}

func (s *Server) withSecurityHeaders(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		writer.Header().Set("X-Frame-Options", "SAMEORIGIN")
		writer.Header().Set("Referrer-Policy", "no-referrer")
		writer.Header().Set("Permissions-Policy", "camera=(), microphone=(self), geolocation=(), payment=()")
		writer.Header().Set("Content-Security-Policy", contentSecurityPolicy(request.URL.Path))
		if publicNoStorePath(request.URL.Path) && writer.Header().Get("Cache-Control") == "" {
			writer.Header().Set("Cache-Control", "no-store")
		}
		if s.config.Webpanel.TLS.HSTS && requestIsHTTPS(request) {
			writer.Header().Set("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
		}
		next.ServeHTTP(writer, request)
	})
}

func requestMethodGETOrHEAD(writer http.ResponseWriter, request *http.Request) bool {
	if request.Method == http.MethodGet || request.Method == http.MethodHead {
		return true
	}
	writer.Header().Set("Allow", "GET, HEAD")
	http.Error(writer, "method not allowed", http.StatusMethodNotAllowed)
	return false
}

func contentSecurityPolicy(path string) string {
	scriptSrc := "script-src 'self' https://unpkg.com"
	styleSrc := "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com"
	if publicSecurityPolicyPath(path) {
		scriptSrc = "script-src 'self'"
		styleSrc = "style-src 'self' https://fonts.googleapis.com"
	} else if path == "/banner" {
		scriptSrc = "script-src 'self' 'unsafe-inline'"
	}
	return strings.Join([]string{
		"default-src 'self'",
		scriptSrc,
		styleSrc,
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

func publicHTMLPath(path string) bool {
	return path == "/" || path == "/feeds" || path == "/listen" || publicAlertsPath(path)
}

func publicSecurityPolicyPath(path string) bool {
	return publicHTMLPath(path) || strings.HasPrefix(path, "/api/public/") || strings.HasPrefix(path, "/assets/")
}

func publicNoStorePath(path string) bool {
	return publicHTMLPath(path) || strings.HasPrefix(path, "/api/public/")
}

func allOriginPatterns() []string {
	return []string{"*"}
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
