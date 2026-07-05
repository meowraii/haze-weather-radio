package webgateway

import (
	"bytes"
	"context"
	"encoding/json"
	"log"
	"net"
	"net/http"
	"strings"
	"time"
)

const mediaServiceWebRTCOfferTimeout = 1500 * time.Millisecond

func clientIPForMediaRequest(request *http.Request) string {
	if request == nil {
		return ""
	}
	for _, raw := range []string{
		strings.TrimSpace(strings.Split(request.Header.Get("X-Forwarded-For"), ",")[0]),
		request.Header.Get("X-Real-IP"),
		request.RemoteAddr,
	} {
		ip := strings.TrimSpace(raw)
		if host, _, err := net.SplitHostPort(ip); err == nil {
			ip = host
		}
		ip = strings.Trim(ip, "[]")
		parsed := net.ParseIP(ip)
		if parsed == nil || parsed.IsUnspecified() || parsed.IsLoopback() {
			continue
		}
		return parsed.String()
	}
	return ""
}

func (s *Server) mediaServiceWebRTCAnswer(ctx context.Context, payload map[string]any) (map[string]any, bool) {
	baseURL := mediaServiceBaseURL(s.config)
	return mediaServiceWebRTCAnswerFromBase(ctx, baseURL, payload)
}

func mediaServiceWebRTCAnswerFromBase(ctx context.Context, baseURL string, payload map[string]any) (map[string]any, bool) {
	if baseURL == "" {
		return nil, false
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, false
	}
	offerCtx, cancel := context.WithTimeout(ctx, mediaServiceWebRTCOfferTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(offerCtx, http.MethodPost, baseURL+"/api/v1/webrtc/offer", bytes.NewReader(raw))
	if err != nil {
		return nil, false
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		log.Printf("haze-media WebRTC unavailable: %v", err)
		return nil, false
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusNotImplemented || response.StatusCode == http.StatusNotFound || response.StatusCode == http.StatusServiceUnavailable {
		return nil, false
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		log.Printf("haze-media WebRTC returned %s", response.Status)
		return nil, false
	}
	var answer map[string]any
	if err := json.NewDecoder(response.Body).Decode(&answer); err != nil {
		log.Printf("haze-media WebRTC answer was invalid: %v", err)
		return nil, false
	}
	if strings.TrimSpace(stringValue(answer, "sdp")) == "" {
		return nil, false
	}
	return answer, true
}

func mediaServiceWebRTCAvailable(ctx context.Context, baseURL string) (bool, bool) {
	health, ok := mediaServiceHealthFromBaseURL(ctx, baseURL)
	if !ok {
		return false, false
	}
	capabilities, _ := health["capabilities"].(map[string]any)
	value, ok := capabilities["webrtc"].(bool)
	return value, ok
}

func mediaServiceHealth(ctx context.Context, config Config) (map[string]any, bool) {
	baseURL := mediaServiceBaseURL(config)
	if baseURL == "" {
		return nil, false
	}
	return mediaServiceHealthFromBaseURL(ctx, baseURL)
}

func mediaServiceHealthFromBaseURL(ctx context.Context, baseURL string) (map[string]any, bool) {
	checkCtx, cancel := context.WithTimeout(ctx, 750*time.Millisecond)
	defer cancel()
	request, err := http.NewRequestWithContext(checkCtx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/api/v1/health", nil)
	if err != nil {
		return nil, false
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return nil, false
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, false
	}
	var health map[string]any
	if err := json.NewDecoder(response.Body).Decode(&health); err != nil {
		return nil, false
	}
	return health, true
}
