package webgateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

func TestServerIPUsesNonLoopbackLocalAddress(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "http://127.0.0.1:8086/", nil)
	request = request.WithContext(context.WithValue(request.Context(), http.LocalAddrContextKey, &net.TCPAddr{
		IP:   net.ParseIP("192.168.50.10"),
		Port: 8086,
	}))

	if got := serverIP(request); got != "192.168.50.10" {
		t.Fatalf("serverIP = %q", got)
	}
}

func TestServerIPIgnoresLoopbackLocalAddress(t *testing.T) {
	request := httptest.NewRequest(http.MethodGet, "http://203.0.113.10:8086/", nil)
	request = request.WithContext(context.WithValue(request.Context(), http.LocalAddrContextKey, &net.TCPAddr{
		IP:   net.ParseIP("127.0.0.1"),
		Port: 8086,
	}))

	if got := serverIP(request); got != "203.0.113.10" {
		t.Fatalf("serverIP = %q", got)
	}
}

func TestHealth(t *testing.T) {
	server := NewServer(Config{}, ".")
	request := httptest.NewRequest(http.MethodGet, "/api/public/v1/health", nil)
	response := httptest.NewRecorder()

	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	var payload map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	if payload["service"] != "haze-web" {
		t.Fatalf("service = %v", payload["service"])
	}
	capabilities, ok := payload["capabilities"].(map[string]any)
	if !ok {
		t.Fatalf("missing capabilities: %#v", payload)
	}
	if _, ok := capabilities["webrtc_opus"].(bool); !ok {
		t.Fatalf("missing webrtc_opus capability: %#v", capabilities)
	}
	if strings.TrimSpace(fmt.Sprint(capabilities["webrtc_default_codec"])) == "" {
		t.Fatalf("missing webrtc_default_codec capability: %#v", capabilities)
	}
	if _, ok := payload["webrtc_peer_count"].(float64); !ok {
		t.Fatalf("missing webrtc_peer_count: %#v", payload)
	}
	if _, ok := payload["webrtc_peers"].([]any); !ok {
		t.Fatalf("missing webrtc_peers: %#v", payload)
	}
	if _, ok := payload["webrtc_source_count"].(float64); !ok {
		t.Fatalf("missing webrtc_source_count: %#v", payload)
	}
	if _, ok := payload["webrtc_sources"].([]any); !ok {
		t.Fatalf("missing webrtc_sources: %#v", payload)
	}
}

func TestAdminRedirectsWithoutSession(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	config := authEnabledConfig()
	server := NewServer(config, ".")
	request := httptest.NewRequest(http.MethodGet, "/admin", nil)
	response := httptest.NewRecorder()

	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusSeeOther {
		t.Fatalf("status = %d", response.Code)
	}
	if location := response.Header().Get("Location"); !strings.HasPrefix(location, "/login?next=") {
		t.Fatalf("location = %q", location)
	}
}

func TestBannerDoesNotRequireSession(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "banner.html"), "<!doctype html><title>banner</title>")
	server := NewServerWithSurface(authEnabledConfig(), "config.yaml", dir, "admin")
	request := httptest.NewRequest(http.MethodGet, "/banner?feed=CAP-IT-ALL", nil)
	response := httptest.NewRecorder()

	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	if !strings.Contains(response.Body.String(), "banner") {
		t.Fatalf("body = %q", response.Body.String())
	}
}

func TestBannerStreamDoesNotRequireSession(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "")
	server := NewServerWithSurface(authEnabledConfig(), configPath, dir, "admin")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	request := httptest.NewRequest(http.MethodGet, "/api/v1/banner/stream?feed=CAP-IT-ALL", nil).WithContext(ctx)
	response := httptest.NewRecorder()

	server.Handler().ServeHTTP(response, request)

	if response.Code == http.StatusUnauthorized || response.Code == http.StatusSeeOther {
		t.Fatalf("status = %d", response.Code)
	}
	if !strings.Contains(response.Body.String(), "event: banner") {
		t.Fatalf("body = %q", response.Body.String())
	}
}

func TestBannerCurrentDoesNotRequireSession(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	mustWrite(t, configPath, "")
	server := NewServerWithSurface(authEnabledConfig(), configPath, dir, "admin")
	request := httptest.NewRequest(http.MethodGet, "/api/v1/banner/current?feed=CAP-IT-ALL", nil)
	response := httptest.NewRecorder()

	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	if !strings.Contains(response.Body.String(), `"feed_id": "CAP-IT-ALL"`) {
		t.Fatalf("body = %q", response.Body.String())
	}
}

func TestPublicSurfaceDoesNotExposeAdminRoutes(t *testing.T) {
	server := NewServerWithSurface(Config{}, "config.yaml", ".", "public")

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/admin", nil)
	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestCombinedSurfaceServesPublicAlertsArchiveAliasWhenEnabled(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "index.html"), "<!doctype html><title>public</title>")
	config := Config{}
	config.Webpanel.Public.AlertsArchive.Access = "public"
	server := NewServerWithSurface(config, "config.yaml", dir, "combined")

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/alerts/archive", nil)
	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestCombinedSurfaceHidesPublicAlertsArchiveWhenDisabled(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "index.html"), "<!doctype html><title>public</title>")
	server := NewServerWithSurface(Config{}, "config.yaml", dir, "combined")

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/alerts/archive", nil)
	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestAdminSurfaceDoesNotExposePublicFeedSocket(t *testing.T) {
	server := NewServerWithSurface(Config{}, "config.yaml", ".", "admin")

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/feeds", nil)
	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestPublicListenPageServedWhenFeedsAvailable(t *testing.T) {
	dir := t.TempDir()
	writePublicFixture(t, dir, "public")
	mustWrite(t, filepath.Join(dir, "index.html"), "<!doctype html><title>public listener</title>")
	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, dir)

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/listen?feed=sk-0001&codec=opus", nil)
	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	if !strings.Contains(response.Body.String(), "public listener") {
		t.Fatalf("body = %q", response.Body.String())
	}
}

func TestPublicAboutPageIsRemoved(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "index.html"), "<!doctype html><title>about</title>")
	server := NewServerWithSurface(Config{}, "config.yaml", dir, "public")

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/about", nil)
	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestAdminAllowsValidSessionCookie(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	config := authEnabledConfig()
	server := NewServer(config, ".")
	token, err := server.auth.Login("secret")
	if err != nil {
		t.Fatalf("login: %v", err)
	}
	request := httptest.NewRequest(http.MethodGet, "/admin", nil)
	request.AddCookie(&http.Cookie{Name: sessionCookieName, Value: token})
	response := httptest.NewRecorder()

	server.Handler().ServeHTTP(response, request)

	if response.Code == http.StatusSeeOther || response.Code == http.StatusUnauthorized {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestWebSocketLoginReturnsToken(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	server := NewServer(authEnabledConfig(), ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	ctx := context.Background()
	wsURL := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/v1/panel/ws?mode=control"
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()

	readType(t, ctx, conn, "hello")
	if err := conn.Write(ctx, websocket.MessageText, []byte(`{"type":"login","request_id":"r1","password":"secret"}`)); err != nil {
		t.Fatalf("write: %v", err)
	}
	payload := readType(t, ctx, conn, "auth_ok")
	if payload["reply_to"] != "r1" {
		t.Fatalf("reply_to = %v", payload["reply_to"])
	}
	if payload["token"] == "" {
		t.Fatal("token was empty")
	}
}

func TestWebSocketLoginTokenOpensAdminAndAuthenticatedStream(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "admin.html"), "<!doctype html><title>admin</title>")
	mustWrite(t, filepath.Join(dir, "login.html"), "<!doctype html><title>login</title>")
	server := NewServer(authEnabledConfig(), dir)
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	ctx := context.Background()
	loginWS := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/v1/panel/ws?mode=control"
	loginConn, _, err := websocket.Dial(ctx, loginWS, nil)
	if err != nil {
		t.Fatalf("login dial: %v", err)
	}
	readType(t, ctx, loginConn, "hello")
	writeWS(t, ctx, loginConn, map[string]any{
		"type":       "login",
		"request_id": "login1",
		"password":   "secret",
	})
	loginReply := readType(t, ctx, loginConn, "auth_ok")
	loginConn.CloseNow()
	token, _ := loginReply["token"].(string)
	if token == "" {
		t.Fatal("token was empty")
	}

	client := &http.Client{
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	adminURL := httpServer.URL + "/admin?token=" + url.QueryEscape(token)
	response, err := client.Get(adminURL)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusSeeOther {
		t.Fatalf("admin status = %d", response.StatusCode)
	}
	if response.Header.Get("Location") != "/admin" {
		t.Fatalf("admin location = %q", response.Header.Get("Location"))
	}
	cookies := response.Cookies()
	if len(cookies) == 0 || cookies[0].Name != sessionCookieName {
		t.Fatalf("admin did not set session cookie: %#v", response.Cookies())
	}
	if !cookies[0].HttpOnly || cookies[0].SameSite != http.SameSiteStrictMode {
		t.Fatalf("admin session cookie was not hardened: %#v", cookies[0])
	}

	adminRequest, err := http.NewRequest(http.MethodGet, httpServer.URL+"/admin", nil)
	if err != nil {
		t.Fatal(err)
	}
	adminRequest.AddCookie(cookies[0])
	adminResponse, err := http.DefaultClient.Do(adminRequest)
	if err != nil {
		t.Fatal(err)
	}
	defer adminResponse.Body.Close()
	if adminResponse.StatusCode != http.StatusOK {
		t.Fatalf("admin follow-up status = %d", adminResponse.StatusCode)
	}

	controlWS := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/v1/panel/ws?mode=control&token=" + url.QueryEscape(token)
	controlConn, _, err := websocket.Dial(ctx, controlWS, nil)
	if err != nil {
		t.Fatalf("control dial: %v", err)
	}
	defer controlConn.CloseNow()
	readType(t, ctx, controlConn, "hello")
	writeWS(t, ctx, controlConn, map[string]any{
		"type":       "auth_check",
		"request_id": "auth1",
	})
	authReply := readType(t, ctx, controlConn, "auth_state")
	if authReply["authenticated"] != true {
		t.Fatalf("auth_state = %#v", authReply)
	}
}

func TestWebSocketDaemonSettingsCommandRoundTrip(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	if err := os.MkdirAll(filepath.Join(dir, "managed"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(configPath, []byte(`version: test
daemon_settings_file: runtime/state/daemonSettings.json
services:
  go:
    enabled: true
    web_gateway:
      enabled: true
      addr: "127.0.0.1:8081"
    cap_ingest:
      enabled: true
      source_id: go-cap
      source: naads
      interval: 30s
      timeout: 15s
    tts:
      enabled: true
      readers: managed/configs/readers.xml
      provider: auto
      language: en-CA
      out_dir: runtime/audio/tts
      timeout: 60s
  daemon:
    enabled: true
    scheduler:
      enabled: true
    playlist:
      enabled: true
      interval_ms: 750
webpanel:
  public:
    enabled: true
    feeds:
      access: disabled
      webrtc:
        enabled: true
  admin:
    enabled: true
  receiver:
    enabled: false
`), 0o600); err != nil {
		t.Fatal(err)
	}

	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	token, err := server.auth.Login("secret")
	if err != nil {
		t.Fatal(err)
	}
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	ctx := context.Background()
	wsURL := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/v1/panel/ws?mode=control&token=" + token
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	readType(t, ctx, conn, "hello")

	writeWS(t, ctx, conn, map[string]any{
		"type":       "command",
		"request_id": "get1",
		"command":    "daemon.settings.get",
		"payload":    map[string]any{},
	})
	getReply := readType(t, ctx, conn, "command_result")
	getResult := getReply["result"].(map[string]any)
	effective := getResult["effective"].(map[string]any)
	services := effective["services"].(map[string]any)
	goServices := services["go"].(map[string]any)
	capIngest := goServices["cap_ingest"].(map[string]any)
	capIngest["enabled"] = false

	writeWS(t, ctx, conn, map[string]any{
		"type":       "command",
		"request_id": "save1",
		"command":    "daemon.settings.save",
		"payload": map[string]any{
			"settings": effective,
		},
	})
	saveReply := readType(t, ctx, conn, "command_result")
	saveResult := saveReply["result"].(map[string]any)
	if saveResult["pending_restart"] != true {
		t.Fatalf("pending_restart = %v", saveResult["pending_restart"])
	}
	raw, err := os.ReadFile(filepath.Join(dir, "runtime", "state", "daemonSettings.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"cap_ingest"`) || !strings.Contains(string(raw), `"enabled": false`) {
		t.Fatalf("unexpected overlay: %s", raw)
	}
}

func TestWebSocketStateCommandIncludesConfiguredFeeds(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	t.Setenv("SAME_ID", "WXR123")
	dir := t.TempDir()
	writePanelFixture(t, dir)

	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	token, err := server.auth.Login("secret")
	if err != nil {
		t.Fatal(err)
	}
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	ctx := context.Background()
	wsURL := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/v1/panel/ws?mode=control&token=" + token
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	readType(t, ctx, conn, "hello")

	writeWS(t, ctx, conn, map[string]any{
		"type":       "command",
		"request_id": "state1",
		"command":    "state",
		"payload":    map[string]any{},
	})
	reply := readType(t, ctx, conn, "command_result")
	result := reply["result"].(map[string]any)
	summary := result["summary"].(map[string]any)
	if summary["feed_count"].(float64) != 1 {
		t.Fatalf("feed_count = %v", summary["feed_count"])
	}
	feeds := summary["feeds"].([]any)
	feed := feeds[0].(map[string]any)
	if feed["id"] != "sk-0001" || feed["name"] != "Saskatoon" {
		t.Fatalf("feed = %#v", feed)
	}
	codes := feed["clc_codes"].([]any)
	if len(codes) != 2 {
		t.Fatalf("clc_codes = %#v", codes)
	}
	sameLocations := feed["same_locations"].([]any)
	if len(sameLocations) != 2 || sameLocations[0] != "065500" || sameLocations[1] != "065522" {
		t.Fatalf("same_locations = %#v", sameLocations)
	}
	regions := feed["coverage_regions"].([]any)
	if len(regions) != 1 {
		t.Fatalf("coverage_regions = %#v", regions)
	}
	region := regions[0].(map[string]any)
	if region["name"] != "Outlook - Watrous - Hanley - Imperial - Dinsmore" {
		t.Fatalf("region = %#v", region)
	}
	subregions := region["subregions"].([]any)
	if len(subregions) != 1 || subregions[0].(map[string]any)["name"] != "R.M. of Rudy including Outlook and Glenside" {
		t.Fatalf("subregions = %#v", subregions)
	}
	configView := result["config"].(map[string]any)
	same := configView["same"].(map[string]any)
	if same["sender"] != "WXR123" {
		t.Fatalf("sender = %v", same["sender"])
	}
}

func TestStateSignatureIgnoresVolatilePanelFields(t *testing.T) {
	base := map[string]any{
		"summary": map[string]any{
			"uptime_seconds": float64(1),
			"feeds": []any{
				map[string]any{
					"id": "sk-0001",
					"runtime": map[string]any{
						"now_playing": "Idle",
					},
				},
			},
		},
		"events": []any{
			map[string]any{
				"kind":      "gateway",
				"timestamp": "2026-06-15T00:00:00Z",
				"message":   "state",
			},
		},
		"last_connected": map[string]any{
			"ip": "127.0.0.1",
			"at": "2026-06-15T00:00:00Z",
		},
	}
	updatedClockOnly := cloneMap(base)
	updatedClockOnly["summary"].(map[string]any)["uptime_seconds"] = float64(90)
	updatedClockOnly["events"].([]any)[0].(map[string]any)["timestamp"] = "2026-06-15T00:01:00Z"
	updatedClockOnly["last_connected"].(map[string]any)["at"] = "2026-06-15T00:01:00Z"

	if stateSignature(base) != stateSignature(updatedClockOnly) {
		t.Fatal("clock-only changes should not force a panel state broadcast")
	}

	changedFeed := cloneMap(base)
	feeds := changedFeed["summary"].(map[string]any)["feeds"].([]any)
	feeds[0].(map[string]any)["runtime"].(map[string]any)["now_playing"] = "Date and Time"
	if stateSignature(base) == stateSignature(changedFeed) {
		t.Fatal("feed runtime changes must force a panel state broadcast")
	}
}

func TestAllLocationFeedSameLocationsIncludeNationalCanadaAndUS(t *testing.T) {
	clcNames := map[string]string{
		"065522": "City of Saskatoon",
	}
	nwsNames := map[string]string{
		"013121": "Fulton, GA",
	}
	var feed feedXML
	feed.Alerts.CapCP.EnabledRaw = "true"
	feed.Alerts.NWSCAP.EnabledRaw = "true"

	locations := feedSameLocations(feed, clcNames, nwsNames)
	if len(locations) != 3 {
		t.Fatalf("same locations = %#v", locations)
	}
	if locations[0] != "000000" {
		t.Fatalf("national SAME code should be first: %#v", locations)
	}
	if !containsString(locations, "065522") || !containsString(locations, "013121") {
		t.Fatalf("all-location feed should include Canada CLC and US SAME/FIPS codes: %#v", locations)
	}
}

func TestPublicWebSocketSendsPublicState(t *testing.T) {
	dir := t.TempDir()
	writePublicFixture(t, dir, "public")
	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	ctx := context.Background()
	wsURL := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/public/v1/panel/ws?feeds=1"
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	readType(t, ctx, conn, "hello")
	state := readType(t, ctx, conn, "public_state")
	summary := state["data"].(map[string]any)["summary"].(map[string]any)
	if summary["feeds_access"] != "public" || summary["webrtc_enabled"] != true || summary["media_available"] != false {
		t.Fatalf("summary = %#v", summary)
	}
	for _, key := range []string{"ip_address", "version", "git_commit", "os", "architecture"} {
		if strings.TrimSpace(fmt.Sprint(summary[key])) == "" {
			t.Fatalf("missing public summary %s in %#v", key, summary)
		}
	}
	capabilities := summary["capabilities"].(map[string]any)
	if _, ok := capabilities["webrtc_opus"].(bool); !ok {
		t.Fatalf("missing public webrtc_opus capability: %#v", capabilities)
	}
	if strings.TrimSpace(fmt.Sprint(capabilities["webrtc_default_codec"])) == "" {
		t.Fatalf("missing public webrtc_default_codec capability: %#v", capabilities)
	}
	feeds := summary["feeds"].([]any)
	if len(feeds) != 1 {
		t.Fatalf("feeds = %#v", feeds)
	}
	feed := feeds[0].(map[string]any)
	if feed["id"] != "sk-0001" {
		t.Fatalf("feed = %#v", feed)
	}
	if _, ok := feed["clc_codes"]; ok {
		t.Fatalf("public feed leaked admin field: %#v", feed)
	}
}

func TestPublicWebSocketHidesAuthRequiredFeedsWithoutToken(t *testing.T) {
	dir := t.TempDir()
	writePublicFixture(t, dir, "auth_required")
	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	ctx := context.Background()
	wsURL := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/public/v1/panel/ws?feeds=1"
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	readType(t, ctx, conn, "hello")
	state := readType(t, ctx, conn, "public_state")
	summary := state["data"].(map[string]any)["summary"].(map[string]any)
	if summary["feeds_access"] != "auth_required" {
		t.Fatalf("summary = %#v", summary)
	}
	if feeds := summary["feeds"].([]any); len(feeds) != 0 {
		t.Fatalf("auth-required feeds leaked without token: %#v", feeds)
	}
}

func TestWebSocketCatalogCommands(t *testing.T) {
	t.Setenv("ADMIN_PASSWD", "secret")
	dir := t.TempDir()
	writePanelFixture(t, dir)

	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	token, err := server.auth.Login("secret")
	if err != nil {
		t.Fatal(err)
	}
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	ctx := context.Background()
	wsURL := "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/api/v1/panel/ws?mode=control&token=" + token
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.CloseNow()
	readType(t, ctx, conn, "hello")

	writeWS(t, ctx, conn, map[string]any{"type": "command", "request_id": "pkg", "command": "wx.packages"})
	packagesReply := readType(t, ctx, conn, "command_result")
	packages := packagesReply["result"].(map[string]any)["packages"].([]any)
	if len(packages) != 1 || packages[0] != "forecast" {
		t.Fatalf("packages = %#v", packages)
	}

	writeWS(t, ctx, conn, map[string]any{"type": "command", "request_id": "readers", "command": "wx.readers"})
	readersReply := readType(t, ctx, conn, "command_result")
	readers := readersReply["result"].(map[string]any)["readers"].([]any)
	if len(readers) != 1 || readers[0].(map[string]any)["id"] != "00" {
		t.Fatalf("readers = %#v", readers)
	}

	writeWS(t, ctx, conn, map[string]any{"type": "command", "request_id": "codes", "command": "same.event_codes"})
	codesReply := readType(t, ctx, conn, "command_result")
	eas := codesReply["result"].(map[string]any)["eas"].(map[string]any)
	if eas["RWT"] != "Required Weekly Test" {
		t.Fatalf("eas = %#v", eas)
	}

	writeWS(t, ctx, conn, map[string]any{"type": "command", "request_id": "locs", "command": "same.location_names"})
	locationsReply := readType(t, ctx, conn, "command_result")
	locations := locationsReply["result"].(map[string]any)
	if locations["065522"] != "R.M. of Rudy including Outlook and Glenside" {
		t.Fatalf("locations = %#v", locations)
	}

	writeWS(t, ctx, conn, map[string]any{"type": "command", "request_id": "tpl", "command": "same.templates.get"})
	templatesReply := readType(t, ctx, conn, "command_result")
	templates := templatesReply["result"].(map[string]any)
	rwt, ok := templates["RWT"].(map[string]any)
	if !ok {
		t.Fatalf("templates = %#v", templates)
	}
	same := rwt["same"].(map[string]any)
	templateLocations := same["locations"].([]any)
	if len(templateLocations) != 1 || templateLocations[0].(map[string]any)["id"] != "065522" {
		t.Fatalf("template locations = %#v", same["locations"])
	}
}

func TestWxGeneratePayloadResolvesLocationHints(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")

	request, err := parseWxGeneratePayload(configPath, map[string]any{
		"feed_id":   "sk-0001",
		"locations": "sk-40",
		"packages":  []any{"forecast"},
		"format":    "json",
	})
	if err != nil {
		t.Fatal(err)
	}
	if request.ForecastID != "sk-40" || request.StationID != "sk-40" {
		t.Fatalf("request = %#v", request)
	}

	derived, err := parseWxGeneratePayload(configPath, map[string]any{
		"feed_id":   "sk-0001",
		"locations": "640",
		"packages":  []any{"forecast"},
		"format":    "json",
	})
	if err != nil {
		t.Fatal(err)
	}
	if derived.Code != "06040" || derived.ForecastID != "sk-40" || derived.StationID != "sk-40" || derived.Province != "SK" {
		t.Fatalf("derived = %#v", derived)
	}
}

func TestReceiverSessionAndWebSocketUseFeedID(t *testing.T) {
	dir := t.TempDir()
	writeReceiverFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	sessionStatus, session := postReceiverJSON(t, httpServer.URL+"/api/receiver/v1/session", map[string]any{
		"feed_id":           "sk-0001",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
	})
	if sessionStatus != http.StatusOK {
		t.Fatalf("session status = %d payload=%#v", sessionStatus, session)
	}
	wsURL := stringMapValue(session, "ws_url")
	if wsURL == "" || !strings.Contains(wsURL, "feed_id=sk-0001") {
		t.Fatalf("session = %#v", session)
	}

	conn, _, err := websocket.Dial(context.Background(), wsURL, nil)
	if err != nil {
		t.Fatalf("receiver ws dial: %v", err)
	}
	ready := readType(t, context.Background(), conn, "receiver_ready")
	if ready["feed_id"] != "sk-0001" {
		t.Fatalf("ready = %#v", ready)
	}
	transmitter := ready["transmitter"].(map[string]any)
	if transmitter["site_name"] != "Saskatoon" || transmitter["callsign"] != "XLF322" {
		t.Fatalf("transmitter = %#v", transmitter)
	}
	conn.CloseNow()
}

func TestReceiverRejectsUnknownFeed(t *testing.T) {
	dir := t.TempDir()
	writeReceiverFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")
	config, err := LoadConfig(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server := NewServerWithConfigPath(config, configPath, ".")
	httpServer := httptest.NewServer(server.Handler())
	defer httpServer.Close()

	status, _ := postReceiverJSON(t, httpServer.URL+"/api/receiver/v1/session", map[string]any{
		"feed_id":           "sk-9999",
		"receiver_id":       "rx-1",
		"receiver_hostname": "pi-one",
	})
	if status == http.StatusOK {
		t.Fatal("unknown receiver feed was accepted")
	}
}

func TestReceiverRoutesDisabledByDefault(t *testing.T) {
	server := NewServer(Config{}, ".")
	request := httptest.NewRequest(http.MethodPost, "/api/receiver/v1/session", strings.NewReader(`{}`))
	response := httptest.NewRecorder()

	server.Handler().ServeHTTP(response, request)

	if response.Code != http.StatusNotFound {
		t.Fatalf("status = %d", response.Code)
	}
}

func TestBuildSameRequestFromPanelPayload(t *testing.T) {
	t.Setenv("SAME_ID", "WXR123")
	dir := t.TempDir()
	writePanelFixture(t, dir)

	request, err := buildSameRequest(filepath.Join(dir, "config.yaml"), map[string]any{
		"originator":       "wxr",
		"event":            "rwt",
		"locations":        []any{"065522", "bad", "065500"},
		"duration_hours":   float64(1),
		"duration_minutes": float64(5),
		"tone_type":        "npas",
		"feed_id":          "sk-0001",
	})
	if err != nil {
		t.Fatal(err)
	}
	if request.Originator != "WXR" || request.Event != "RWT" {
		t.Fatalf("codes = %#v", request)
	}
	if request.Duration != "0105" {
		t.Fatalf("duration = %q", request.Duration)
	}
	if strings.Join(request.Locations, ",") != "065500,065522" {
		t.Fatalf("locations = %#v", request.Locations)
	}
	if request.Callsign != "WXR123" {
		t.Fatalf("callsign = %q", request.Callsign)
	}
	if request.Tone != "NPAS" {
		t.Fatalf("tone = %q", request.Tone)
	}
}

func TestBuildSameRequestExpandsForecastRegionSubregions(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "feeds.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<feeds>
  <feed id="sk-0001" enabled="true" timezone="America/Regina">
    <languages><lang code="en-CA"/></languages>
    <locations>
      <coverage>
        <region id="065400"/>
      </coverage>
    </locations>
    <transmitter_metadata><transmitter><site_name>Saskatoon</site_name><callsign>XLF322</callsign></transmitter></transmitter_metadata>
  </feed>
</feeds>
`)
	mustWrite(t, filepath.Join(dir, "managed", "csv", "CLC_Base_Zone.csv"), "CLC,FEATURE_ID,NAME,NOM\n065413,fixture,R.M. of Great Bend including Radisson and Borden,\n065421,fixture,R.M. of Laird including Waldheim Hepburn and Laird,\n065432,fixture,R.M. of Biggar including Biggar,\n")

	request, err := buildSameRequest(filepath.Join(dir, "config.yaml"), map[string]any{
		"originator": "WXR",
		"event":      "RWT",
		"feed_id":    "sk-0001",
		"locations":  []any{"065400"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(request.Locations, ",") != "065400,065413,065421,065432" {
		t.Fatalf("locations = %#v", request.Locations)
	}
}

func TestBuildSameRequestRequiresLocation(t *testing.T) {
	_, err := buildSameRequest("config.yaml", map[string]any{
		"originator": "WXR",
		"event":      "RWT",
		"locations":  []any{"bad"},
	})

	if err == nil {
		t.Fatal("expected missing location error")
	}
}

func TestSameIntroUsesSAMEToTextManualLead(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	result, err := sameIntroPayload(filepath.Join(dir, "config.yaml"), map[string]any{
		"originator":       "WXR",
		"event":            "RWT",
		"locations":        []any{"065522"},
		"duration_hours":   float64(0),
		"duration_minutes": float64(15),
		"callsign":         "XLF322",
	})
	if err != nil {
		t.Fatal(err)
	}
	intro, _ := result["intro"].(string)
	if !strings.Contains(intro, "Environment Canada has issued") || !strings.Contains(intro, "XLF322") {
		t.Fatalf("intro = %q", intro)
	}
}

func TestBroadcastAlertDataCanDisableSame(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	session := wsSession{configPath: filepath.Join(dir, "config.yaml")}
	data := session.broadcastAlertData(map[string]any{
		"originator":               "WXR",
		"event":                    "RWT",
		"locations":                []any{"065522"},
		"include_same":             false,
		"prepend_same_translation": true,
		"voice_message":            "This is only a drill.",
		"duration_hours":           float64(0),
		"duration_minutes":         float64(15),
	}, []string{"sk-0001"}, "manual-test", false)

	if data["include_same"] != false {
		t.Fatalf("include_same = %#v", data["include_same"])
	}
	text, _ := data["alert_text"].(string)
	if !strings.Contains(text, "This is only a drill.") || !strings.Contains(text, "has issued") {
		t.Fatalf("alert_text = %q", text)
	}
}

func TestBroadcastAlertDataDoesNotPrependSameTranslationWhenDisabled(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	session := wsSession{configPath: filepath.Join(dir, "config.yaml")}
	data := session.broadcastAlertData(map[string]any{
		"originator":               "WXR",
		"event":                    "RWT",
		"locations":                []any{"065522"},
		"include_same":             false,
		"prepend_same_translation": false,
		"voice_message":            "This is only a drill.",
		"duration_hours":           float64(0),
		"duration_minutes":         float64(15),
	}, []string{"sk-0001"}, "manual-test", false)

	text, _ := data["alert_text"].(string)
	if text != "This is only a drill." {
		t.Fatalf("alert_text = %q", text)
	}
	bannerText, _ := data["banner_text"].(string)
	if !strings.Contains(bannerText, "Environment Canada has issued") || !strings.Contains(bannerText, "This is only a drill.") {
		t.Fatalf("banner_text = %q", bannerText)
	}
}

func TestBroadcastAlertDataDoesNotUseSameIntroFallbackWhenSameEnabled(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	session := wsSession{configPath: filepath.Join(dir, "config.yaml")}
	data := session.broadcastAlertData(map[string]any{
		"originator":               "WXR",
		"event":                    "RWT",
		"locations":                []any{"065522"},
		"include_same":             true,
		"prepend_same_translation": false,
		"description":              "This is only a drill.",
		"instruction":              "No action is required.",
		"duration_hours":           float64(0),
		"duration_minutes":         float64(15),
	}, []string{"sk-0001"}, "manual-test", true)

	text, _ := data["alert_text"].(string)
	if strings.Contains(text, "has issued") {
		t.Fatalf("alert_text should not include SAME intro when prepend is disabled: %q", text)
	}
	for _, wanted := range []string{"This is only a drill.", "No action is required."} {
		if !strings.Contains(text, wanted) {
			t.Fatalf("alert_text missing %q in %q", wanted, text)
		}
	}
	bannerText, _ := data["banner_text"].(string)
	if !strings.Contains(bannerText, "Environment Canada has issued") {
		t.Fatalf("banner_text should still include SAME intro: %q", bannerText)
	}
}

func TestPersistSameQueueItemCreatesManifestAndStateDepth(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)
	configPath := filepath.Join(dir, "config.yaml")

	item, err := persistSameQueueItem(configPath, sameGenerateRequest{
		Originator: "WXR",
		Event:      "RWT",
		Locations:  []string{"065522"},
		Duration:   "0015",
		Callsign:   "XLF322",
		Tone:       "WXR",
	}, []string{"sk-0001"}, map[string]any{
		"header":       "ZCZC-WXR-RWT-065522+0015-1661200-XLF322  -",
		"format":       "raw",
		"sample_rate":  float64(48000),
		"channels":     float64(1),
		"audio_base64": "AQIDBA==",
	}, "This is only a drill.")
	if err != nil {
		t.Fatal(err)
	}
	if item.Status != "pending" || item.AudioBytes != 4 {
		t.Fatalf("item = %#v", item)
	}
	if item.BannerText != "This is only a drill." {
		t.Fatalf("banner text = %q", item.BannerText)
	}
	if len(item.Outputs) != 1 || item.Outputs[0].Type != "udp" || item.Outputs[0].Address != "127.0.0.1:8898" {
		t.Fatalf("outputs = %#v", item.Outputs)
	}
	if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(item.AudioPath))); err != nil {
		t.Fatalf("audio missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(item.ManifestPath))); err != nil {
		t.Fatalf("manifest missing: %v", err)
	}

	feeds, err := loadFeedSummaries(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if feeds[0]["alert_queue_depth"].(int) != 1 {
		t.Fatalf("queue depth = %v", feeds[0]["alert_queue_depth"])
	}
	runtime := feeds[0]["runtime"].(map[string]any)
	if runtime["last_alert_event"] != "RWT" || runtime["last_alert_severity"] != "Pending" {
		t.Fatalf("runtime = %#v", runtime)
	}
}

func TestAlertQueueStateCountsOnlyActiveItems(t *testing.T) {
	now := time.Now().UTC()
	depth, recent, latest := alertQueueState([]sameQueueItem{
		{ID: "played", Status: "played", FeedIDs: []string{"sk-0001"}, Event: "RWT", CreatedAt: now.Add(-time.Minute)},
		{ID: "failed", Status: "failed", FeedIDs: []string{"sk-0001"}, Event: "RWT", LastError: "unsupported", CreatedAt: now},
		{ID: "pending", Status: "pending", FeedIDs: []string{"sk-0001"}, Event: "RWT", CreatedAt: now.Add(-2 * time.Minute)},
	}, "sk-0001")

	if depth != 1 {
		t.Fatalf("depth = %d", depth)
	}
	if latest == nil || latest.ID != "failed" {
		t.Fatalf("latest = %#v", latest)
	}
	if len(recent) != 3 || recent[0]["last_error"] != "unsupported" {
		t.Fatalf("recent = %#v", recent)
	}
}

func TestTargetFeedIDsRejectsUnknownFeed(t *testing.T) {
	dir := t.TempDir()
	writePanelFixture(t, dir)

	_, err := targetFeedIDs(filepath.Join(dir, "config.yaml"), map[string]any{
		"feed_ids": []any{"missing"},
	})

	if err == nil {
		t.Fatal("expected unknown feed error")
	}
}

func TestAlertTargetFeedIDsIncludesAllLocationCatchalls(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "config.yaml"), `version: test
feeds_file: managed/configs/feeds.xml
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "feeds.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<feeds>
  <feed id="sk-0001" enabled="true">
    <locations>
      <coverage><region id="065500"><subregion id="065522"/></region></coverage>
    </locations>
  </feed>
  <feed id="CAP-IT-ALL" enabled="true">
    <alerts>
      <cap_cp enabled="true"/>
      <nws_cap enabled="true"/>
    </alerts>
    <locations><coverage/></locations>
  </feed>
</feeds>
`)

	targets, err := targetFeedIDs(filepath.Join(dir, "config.yaml"), map[string]any{
		"feed_ids": []any{"sk-0001"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(targets, ","); got != "sk-0001" {
		t.Fatalf("targetFeedIDs = %q", got)
	}

	targets, err = alertTargetFeedIDs(filepath.Join(dir, "config.yaml"), map[string]any{
		"feed_ids": []any{"sk-0001"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(targets, ","); got != "sk-0001,CAP-IT-ALL" {
		t.Fatalf("alertTargetFeedIDs = %q", got)
	}
}

func TestSameGeneratorIntegration(t *testing.T) {
	generator := os.Getenv("HAZE_SAME_GENERATOR")
	if generator == "" {
		t.Skip("HAZE_SAME_GENERATOR is not set")
	}
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "config.yaml"), "version: test\n")

	result, err := runSameGenerator(filepath.Join(dir, "config.yaml"), sameGenerateRequest{
		Originator: "WXR",
		Event:      "RWT",
		Locations:  []string{"065522"},
		Duration:   "0015",
		Callsign:   "XLF322",
		Tone:       "WXR",
	})
	if err != nil {
		t.Fatal(err)
	}
	header, _ := result["header"].(string)
	if !strings.HasPrefix(header, "ZCZC-WXR-RWT-065522+0015-") || !strings.HasSuffix(header, "-XLF322  -") {
		t.Fatalf("header = %v", header)
	}
	if result["format"] != "raw" || result["sample_rate"].(float64) != 48000 {
		t.Fatalf("result = %#v", result)
	}
	if len(result["audio_base64"].(string)) < 1000 {
		t.Fatalf("audio payload too small: %d", len(result["audio_base64"].(string)))
	}
}

func TestAirSameIntegrationQueuesGeneratorOutput(t *testing.T) {
	generator := os.Getenv("HAZE_SAME_GENERATOR")
	if generator == "" {
		t.Skip("HAZE_SAME_GENERATOR is not set")
	}
	dir := t.TempDir()
	writePanelFixture(t, dir)
	session := wsSession{configPath: filepath.Join(dir, "config.yaml")}

	result, err := session.airSame(map[string]any{
		"originator":       "WXR",
		"event":            "RWT",
		"locations":        []any{"065522"},
		"duration_hours":   float64(0),
		"duration_minutes": float64(15),
		"tone_type":        "WXR",
		"feed_id":          "sk-0001",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result["queued"] != true {
		t.Fatalf("result = %#v", result)
	}
	if result["feed_id"] != "sk-0001" {
		t.Fatalf("feed_id = %v", result["feed_id"])
	}
	if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(result["manifest_path"].(string)))); err != nil {
		t.Fatalf("manifest missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(result["audio_path"].(string)))); err != nil {
		t.Fatalf("audio missing: %v", err)
	}
	feeds, err := loadFeedSummaries(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if feeds[0]["alert_queue_depth"].(int) != 1 {
		t.Fatalf("queue depth = %v", feeds[0]["alert_queue_depth"])
	}
}

func authEnabledConfig() Config {
	var config Config
	enabled := true
	config.Webpanel.Authentication.Enabled = &enabled
	config.Webpanel.Authentication.SessionTTLSeconds = 60
	return config
}

func readType(t *testing.T, ctx context.Context, conn *websocket.Conn, wanted string) map[string]any {
	t.Helper()
	for range 5 {
		_, raw, err := conn.Read(ctx)
		if err != nil {
			t.Fatalf("read: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(raw, &payload); err != nil {
			t.Fatalf("json: %v", err)
		}
		if payload["type"] == wanted {
			return payload
		}
	}
	t.Fatalf("did not receive %s", wanted)
	return nil
}

func writeWS(t *testing.T, ctx context.Context, conn *websocket.Conn, payload map[string]any) {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if err := conn.Write(ctx, websocket.MessageText, raw); err != nil {
		t.Fatalf("write: %v", err)
	}
}

func writePanelFixture(t *testing.T, dir string) {
	t.Helper()
	mustWrite(t, filepath.Join(dir, "config.yaml"), `version: test
feeds_file: managed/configs/feeds.xml
outputs_file: managed/configs/output.xml
logging:
  file:
    main_path: logs/haze.log
webpanel:
  authentication:
    enabled: true
    session_ttl_seconds: 60
services:
  go:
    tts:
      readers: managed/configs/readers.xml
operator:
  operator_name:
    - text: "@operator"
  on_air_name:
    - text: "Canada RadioMET"
same: {}
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "feeds.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<feeds>
  <feed id="sk-0001" enabled="true" timezone="America/Regina">
    <languages><lang code="en-CA"/></languages>
    <locations>
      <coverage>
        <region id="065500"><subregion id="065522"/></region>
      </coverage>
      <observationLocations><location id="sk-40"/></observationLocations>
    </locations>
    <transmitter_metadata>
      <transmitter>
        <site_name>Saskatoon</site_name>
        <callsign>XLF322</callsign>
        <frequency_mhz>162.550</frequency_mhz>
      </transmitter>
    </transmitter_metadata>
  </feed>
</feeds>
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "output.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<outputs>
  <feed id="sk-0001">
    <udp enabled="true">
      <ip>127.0.0.1</ip>
      <port>8898</port>
      <format>raw</format>
      <acodec>pcm_s16le</acodec>
    </udp>
  </feed>
</outputs>
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "packages.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<Packages>
  <defaults><enabled>true</enabled></defaults>
  <package id="forecast" enabled="true"/>
  <package id="disabled_item" enabled="false"/>
</Packages>
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "readers.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<Readers>
  <reader id="00" provider="piper">
    <gender>male</gender>
    <language>en-CA</language>
    <voice_id>en_CA-test-medium</voice_id>
  </reader>
</Readers>
`)
	mustWrite(t, filepath.Join(dir, "managed", "sameMapping.json"), `{"eas":{"RWT":"Required Weekly Test"},"naadsToEas":{"test":"RWT"}}`)
	mustWrite(t, filepath.Join(dir, "managed", "csv", "CAP-CP_Geocodes.csv"), "NAME,NOM,CAPCPGCODE\nSaskatoon,,065522\n")
	mustWrite(t, filepath.Join(dir, "managed", "csv", "CLC_Base_Zone.csv"), "CLC,FEATURE_ID,NAME,NOM\n065522,fixture,R.M. of Rudy including Outlook and Glenside,\n")
	mustWrite(t, filepath.Join(dir, "managed", "csv", "FORECAST_LOCATIONS.csv"), "CODE,NAME,NOM,PROGRAMS\n065500,Outlook - Watrous - Hanley - Imperial - Dinsmore,,Public\n")
	mustWrite(t, filepath.Join(dir, "managed", "configs", "alertTemplates.xml"), `<?xml version="1.0" encoding="utf-8"?>
<templates>
  <template>
    <name>Weekly Test</name>
    <description>Routine test</description>
    <automated><enabled>true</enabled></automated>
    <same><enabled>true</enabled><event>RWT</event><locations><location id="065522" source="eccc"/></locations><duration hr="0" min="15"/><sender_id/></same>
    <content attention_tone="WXR"><lang code="en"><text>Test text</text><file/></lang></content>
  </template>
</templates>
`)
	mustWrite(t, filepath.Join(dir, "logs", "haze.log"), "line one\nline two\n")
}

func writeReceiverFixture(t *testing.T, dir string) {
	t.Helper()
	mustWrite(t, filepath.Join(dir, "config.yaml"), `version: test
feeds_file: managed/configs/feeds.xml
playout:
  sample_rate: 48000
  channels: 1
webpanel:
  receiver:
    enabled: true
    base_path: /api/receiver/v1
    require_tls: false
    challenge_ttl_seconds: 60
    cookie_ttl_seconds: 30
    credential_ttl_seconds: 3600
    credentials_path: runtime/state/receiver_credentials.json
    transmitter_defaults:
      bandwidth_khz: 12.5
      deviation_hz: 5000
      preemphasis: none
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "feeds.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<feeds>
  <feed id="sk-0001" enabled="true" timezone="America/Regina">
    <transmitter_metadata>
      <transmitter>
        <site_name>Saskatoon</site_name>
        <callsign>XLF322</callsign>
        <frequency_mhz>162.550</frequency_mhz>
      </transmitter>
    </transmitter_metadata>
  </feed>
</feeds>
`)
}

func writePublicFixture(t *testing.T, dir string, access string) {
	t.Helper()
	mustWrite(t, filepath.Join(dir, "config.yaml"), fmt.Sprintf(`version: test
feeds_file: managed/configs/feeds.xml
outputs_file: managed/configs/output.xml
webpanel:
  public:
    site_name: Test Haze
    feeds:
      access: %s
      webrtc:
        enabled: true
  admin:
    host: 127.0.0.1
    port: 8086
  authentication:
    enabled: true
    session_ttl_seconds: 60
operator:
  operator_name:
    - text: "@operator"
  on_air_name:
    - text: "Canada RadioMET"
`, access))
	mustWrite(t, filepath.Join(dir, "managed", "configs", "feeds.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<feeds>
  <feed id="sk-0001" enabled="true" timezone="America/Regina">
    <languages><lang code="en-CA"/></languages>
    <locations>
      <coverage>
        <region id="065500"><subregion id="065522"/></region>
      </coverage>
      <observationLocations><location id="sk-40"/></observationLocations>
    </locations>
    <transmitter_metadata>
      <transmitter>
        <site_name>Saskatoon</site_name>
        <callsign>XLF322</callsign>
        <frequency_mhz>162.550</frequency_mhz>
      </transmitter>
    </transmitter_metadata>
  </feed>
</feeds>
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "output.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<outputs>
  <feed id="sk-0001"><webrtc enabled="true"/></feed>
</outputs>
`)
}

func receiverChallengeForTest(t *testing.T, baseURL string, feedID string, receiverID string, hostname string, nonce string) map[string]any {
	t.Helper()
	status, payload := postReceiverJSON(t, baseURL+"/api/receiver/v1/pair/challenge", map[string]any{
		"feed_id":           feedID,
		"receiver_id":       receiverID,
		"receiver_hostname": hostname,
		"nonce":             nonce,
	})
	if status != http.StatusOK {
		t.Fatalf("challenge status = %d payload=%#v", status, payload)
	}
	return payload
}

func postReceiverJSON(t *testing.T, url string, payload map[string]any) (int, map[string]any) {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.Post(url, "application/json", strings.NewReader(string(raw)))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var decoded map[string]any
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil && response.Body != nil {
		decoded = map[string]any{"decode_error": err.Error()}
	}
	return response.StatusCode, decoded
}

func stringMapValue(values map[string]any, key string) string {
	value, _ := values[key].(string)
	return value
}

func mustWrite(t *testing.T, path string, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
