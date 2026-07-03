package webgateway

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/coder/websocket"
)

const defaultReceiverBasePath = "/api/receiver/v1"

type ReceiverManager struct {
	config     Config
	configPath string
	media      *MediaHub
	mu         sync.Mutex
	challenges map[string]receiverChallenge
	cookies    map[string]receiverCookie
	active     map[string]receiverActiveConnection
	nextActive uint64
}

type receiverChallenge struct {
	ID               string
	FeedID           string
	ReceiverID       string
	ReceiverHostname string
	ReceiverNonce    string
	ServerNonce      string
	ExpiresAt        time.Time
}

type receiverCookie struct {
	Digest           string
	FeedID           string
	ReceiverID       string
	ReceiverHostname string
	ExpiresAt        time.Time
}

type receiverActiveConnection struct {
	ID               string
	FeedID           string
	ReceiverID       string
	ReceiverHostname string
	RemoteAddr       string
	ConnectedAt      time.Time
	LastSeenAt       time.Time
	LastMessageType  string
	Transport        string
}

type receiverStore struct {
	Credentials                    map[string]receiverCredential     `json:"credentials"`
	ConsumedPairingTokens          map[string]receiverConsumedToken  `json:"consumed_pairing_tokens"`
	LegacyConsumedPairTokenDigests map[string]receiverLegacyTokenUse `json:"consumed_pair_token_digests,omitempty"`
	LastUpdatedAt                  time.Time                         `json:"last_updated_at"`
}

type receiverCredential struct {
	ID                     string    `json:"id"`
	Secret                 string    `json:"secret"`
	LegacyCredentialSecret string    `json:"credential_secret,omitempty"`
	FeedID                 string    `json:"feed_id"`
	ReceiverID             string    `json:"receiver_id"`
	ReceiverHostname       string    `json:"receiver_hostname"`
	ServerHostname         string    `json:"server_hostname"`
	CreatedAt              time.Time `json:"created_at"`
	ExpiresAt              time.Time `json:"expires_at"`
}

type receiverConsumedToken struct {
	TokenID          string    `json:"token_id"`
	FeedID           string    `json:"feed_id"`
	ReceiverID       string    `json:"receiver_id"`
	ReceiverHostname string    `json:"receiver_hostname"`
	ConsumedAt       time.Time `json:"consumed_at"`
}

type receiverLegacyTokenUse struct {
	TokenID      string    `json:"token_id"`
	FeedID       string    `json:"feed_id"`
	CredentialID string    `json:"credential_id"`
	ConsumedAt   time.Time `json:"consumed_at"`
}

type receiverPairChallengeRequest struct {
	FeedID           string `json:"feed_id"`
	ReceiverID       string `json:"receiver_id"`
	ReceiverHostname string `json:"receiver_hostname"`
	Nonce            string `json:"nonce"`
}

type receiverPairCompleteRequest struct {
	ChallengeID      string `json:"challenge_id"`
	FeedID           string `json:"feed_id"`
	ReceiverID       string `json:"receiver_id"`
	ReceiverHostname string `json:"receiver_hostname"`
	Nonce            string `json:"nonce"`
	Proof            string `json:"proof"`
}

type receiverSessionRequest struct {
	FeedID           string `json:"feed_id"`
	ReceiverID       string `json:"receiver_id"`
	ReceiverHostname string `json:"receiver_hostname"`
	CredentialID     string `json:"credential_id"`
	Nonce            string `json:"nonce"`
	Proof            string `json:"proof"`
}

func NewReceiverManager(config Config, configPath string, media *MediaHub) *ReceiverManager {
	return &ReceiverManager{
		config:     config,
		configPath: filepath.Clean(configPath),
		media:      media,
		challenges: map[string]receiverChallenge{},
		cookies:    map[string]receiverCookie{},
		active:     map[string]receiverActiveConnection{},
	}
}

func (m *ReceiverManager) Enabled() bool {
	return m != nil && m.config.Webpanel.Receiver.Enabled
}

func (m *ReceiverManager) BasePath() string {
	if m == nil {
		return defaultReceiverBasePath
	}
	base := strings.TrimSpace(m.config.Webpanel.Receiver.BasePath)
	if base == "" {
		base = defaultReceiverBasePath
	}
	base = "/" + strings.Trim(base, "/")
	if base == "/" {
		return defaultReceiverBasePath
	}
	return base
}

func (m *ReceiverManager) HandlePairChallenge(writer http.ResponseWriter, request *http.Request) {
	if !m.receiverRequestAllowed(writer, request, http.MethodPost) {
		return
	}
	var payload receiverPairChallengeRequest
	if !decodeReceiverJSON(writer, request, &payload) {
		return
	}
	payload.FeedID = strings.TrimSpace(payload.FeedID)
	payload.ReceiverID = strings.TrimSpace(payload.ReceiverID)
	payload.ReceiverHostname = strings.TrimSpace(payload.ReceiverHostname)
	payload.Nonce = strings.TrimSpace(payload.Nonce)
	if payload.FeedID == "" || payload.ReceiverID == "" || payload.ReceiverHostname == "" || payload.Nonce == "" {
		receiverError(writer, http.StatusBadRequest, "feed_id, receiver_id, receiver_hostname, and nonce are required")
		return
	}
	if _, err := m.receiverTransmitter(payload.FeedID); err != nil {
		receiverError(writer, http.StatusForbidden, err.Error())
		return
	}
	challengeID, err := randomToken()
	if err != nil {
		receiverError(writer, http.StatusInternalServerError, "could not create receiver challenge")
		return
	}
	serverNonce, err := randomToken()
	if err != nil {
		receiverError(writer, http.StatusInternalServerError, "could not create receiver challenge")
		return
	}
	expiresAt := time.Now().UTC().Add(m.challengeTTL())
	challenge := receiverChallenge{
		ID:               challengeID,
		FeedID:           payload.FeedID,
		ReceiverID:       payload.ReceiverID,
		ReceiverHostname: payload.ReceiverHostname,
		ReceiverNonce:    payload.Nonce,
		ServerNonce:      serverNonce,
		ExpiresAt:        expiresAt,
	}
	m.mu.Lock()
	m.pruneLocked(time.Now().UTC())
	m.challenges[challengeID] = challenge
	m.mu.Unlock()
	writeJSON(writer, map[string]any{
		"challenge_id": challengeID,
		"server_nonce": serverNonce,
		"expires_at":   expiresAt,
	})
}

func (m *ReceiverManager) HandlePairComplete(writer http.ResponseWriter, request *http.Request) {
	if !m.receiverRequestAllowed(writer, request, http.MethodPost) {
		return
	}
	var payload receiverPairCompleteRequest
	if !decodeReceiverJSON(writer, request, &payload) {
		return
	}
	payload = cleanPairCompleteRequest(payload)
	now := time.Now().UTC()
	challenge, ok := m.takeChallenge(payload.ChallengeID, now)
	if !ok {
		receiverError(writer, http.StatusForbidden, "receiver challenge is invalid or expired")
		return
	}
	if !challengeMatchesComplete(challenge, payload) {
		receiverError(writer, http.StatusForbidden, "receiver challenge fields do not match")
		return
	}
	token, tokenSecret, err := m.matchPairingToken(payload, challenge)
	if err != nil {
		receiverError(writer, http.StatusForbidden, err.Error())
		return
	}
	expected := receiverHMACHex(tokenSecret, receiverProofMessage("pair-v1", map[string]string{
		"challenge_id":      challenge.ID,
		"feed_id":           challenge.FeedID,
		"receiver_id":       challenge.ReceiverID,
		"receiver_hostname": challenge.ReceiverHostname,
		"receiver_nonce":    challenge.ReceiverNonce,
		"server_nonce":      challenge.ServerNonce,
	}))
	if subtle.ConstantTimeCompare([]byte(expected), []byte(strings.ToLower(payload.Proof))) != 1 {
		receiverError(writer, http.StatusForbidden, "receiver pairing proof is invalid")
		return
	}

	credential, cookie, err := m.createCredentialAndCookie(token.ID, challenge, now, request)
	if err != nil {
		receiverError(writer, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(writer, map[string]any{
		"credential_id":     credential.ID,
		"credential_secret": credential.Secret,
		"cookie":            cookie,
		"ws_url":            m.receiverWSURL(request),
		"expires_at":        now.Add(m.cookieTTL()),
	})
}

func (m *ReceiverManager) HandleSession(writer http.ResponseWriter, request *http.Request) {
	if !m.receiverRequestAllowed(writer, request, http.MethodPost) {
		return
	}
	var payload receiverSessionRequest
	if !decodeReceiverJSON(writer, request, &payload) {
		return
	}
	payload = cleanSessionRequest(payload)
	if payload.FeedID == "" {
		receiverError(writer, http.StatusBadRequest, "feed_id is required")
		return
	}
	if _, err := m.receiverTransmitter(payload.FeedID); err != nil {
		receiverError(writer, http.StatusForbidden, err.Error())
		return
	}
	wsURL := m.receiverWSURL(request)
	separator := "?"
	if strings.Contains(wsURL, "?") {
		separator = "&"
	}
	writeJSON(writer, map[string]any{
		"feed_id": payload.FeedID,
		"ws_url":  wsURL + separator + "feed_id=" + payload.FeedID,
	})
}

func (m *ReceiverManager) HandleWebSocket(writer http.ResponseWriter, request *http.Request) {
	if !m.Enabled() {
		http.NotFound(writer, request)
		return
	}
	feedID := strings.TrimSpace(request.URL.Query().Get("feed_id"))
	cookieValue := receiverCookieFromRequest(request)
	cookie, cookieOK := m.consumeCookie(cookieValue)
	if cookieOK {
		if feedID == "" {
			feedID = cookie.FeedID
		}
	}
	if feedID == "" {
		receiverError(writer, http.StatusBadRequest, "feed_id is required")
		return
	}
	transmitter, err := m.receiverTransmitter(feedID)
	if err != nil {
		receiverError(writer, http.StatusForbidden, err.Error())
		return
	}
	connection, err := websocket.Accept(writer, request, &websocket.AcceptOptions{
		OriginPatterns: sameOriginPatterns(request),
	})
	if err != nil {
		return
	}
	defer connection.CloseNow()
	ctx := request.Context()
	activeID := m.registerActiveReceiver(feedID, cookie, cookieOK, request.RemoteAddr)
	defer m.unregisterActiveReceiver(activeID)
	_ = writeReceiverWS(ctx, connection, map[string]any{
		"type":        "receiver_ready",
		"timestamp":   time.Now().UTC(),
		"feed_id":     feedID,
		"transmitter": transmitter,
	})
	for {
		_, raw, err := connection.Read(ctx)
		if err != nil {
			return
		}
		var message map[string]any
		if err := json.Unmarshal(raw, &message); err != nil {
			_ = writeReceiverWS(ctx, connection, map[string]any{"type": "receiver_error", "detail": "invalid json"})
			continue
		}
		messageType := stringValue(message, "type")
		m.updateActiveReceiver(activeID, messageType, "")
		if messageType == "webrtc_offer" {
			m.updateActiveReceiver(activeID, messageType, "webrtc")
			offerSDP := normalizeWebRTCOfferSDP(stringValue(message, "sdp"))
			if offerSDP == "" {
				_ = writeReceiverWS(ctx, connection, map[string]any{
					"type":   "webrtc_error",
					"detail": "sdp is required",
				})
				continue
			}
			mediaServiceURL := mediaServiceBaseURL(m.config)
			legacyMediaAvailable := m.media != nil && m.media.Available()
			if !legacyMediaAvailable && mediaServiceURL == "" {
				_ = writeReceiverWS(ctx, connection, map[string]any{
					"type":   "webrtc_error",
					"detail": "receiver media bridge is not available",
				})
				continue
			}
			preferredCodec := firstNonBlank(stringValue(message, "codec"), stringValue(message, "preferred_codec"))
			if answer, ok := mediaServiceWebRTCAnswerFromBase(ctx, mediaServiceURL, map[string]any{
				"feed_id":         feedID,
				"sdp":             offerSDP,
				"disable_g722":    boolValue(message, "disable_g722"),
				"require_opus":    boolValue(message, "require_opus"),
				"preferred_codec": preferredCodec,
			}); ok {
				answer["type"] = "webrtc_answer"
				answer["timestamp"] = time.Now().UTC()
				answer["feed_id"] = feedID
				if _, ok := answer["sdp_type"]; !ok {
					answer["sdp_type"] = "answer"
				}
				_ = writeReceiverWS(ctx, connection, answer)
				continue
			}
			if !legacyMediaAvailable {
				detail := "receiver media bridge is not available"
				if mediaServiceURL != "" {
					detail = "haze-media WebRTC service is unavailable and legacy media bridge is not available"
				}
				_ = writeReceiverWS(ctx, connection, map[string]any{
					"type":   "webrtc_error",
					"detail": detail,
				})
				continue
			}
			answer, err := m.media.AnswerWithOptions(ctx, feedID, offerSDP, WebRTCAnswerOptions{
				DisableG722:    boolValue(message, "disable_g722"),
				RequireOpus:    boolValue(message, "require_opus"),
				PreferredCodec: preferredCodec,
			})
			if err != nil {
				_ = writeReceiverWS(ctx, connection, map[string]any{
					"type":   "webrtc_error",
					"detail": err.Error(),
				})
				continue
			}
			_ = writeReceiverWS(ctx, connection, map[string]any{
				"type":         "webrtc_answer",
				"timestamp":    time.Now().UTC(),
				"feed_id":      feedID,
				"sdp":          answer.SDP,
				"sdp_type":     "answer",
				"media_recent": m.media.HasRecentPCM(feedID, 5*time.Second),
				"codec":        answer.Codec.String(),
				"payload_type": answer.PayloadType,
			})
			continue
		}
	}
}

func (m *ReceiverManager) registerActiveReceiver(feedID string, cookie receiverCookie, hasCookie bool, remoteAddr string) string {
	now := time.Now().UTC()
	m.mu.Lock()
	defer m.mu.Unlock()
	m.nextActive++
	id := fmt.Sprintf("%d", m.nextActive)
	active := receiverActiveConnection{
		ID:          id,
		FeedID:      strings.TrimSpace(feedID),
		RemoteAddr:  strings.TrimSpace(remoteAddr),
		ConnectedAt: now,
		LastSeenAt:  now,
		Transport:   "control",
	}
	if hasCookie {
		active.ReceiverID = strings.TrimSpace(cookie.ReceiverID)
		active.ReceiverHostname = strings.TrimSpace(cookie.ReceiverHostname)
	}
	m.active[id] = active
	return id
}

func (m *ReceiverManager) updateActiveReceiver(id string, messageType string, transport string) {
	if id == "" {
		return
	}
	now := time.Now().UTC()
	m.mu.Lock()
	defer m.mu.Unlock()
	active, ok := m.active[id]
	if !ok {
		return
	}
	active.LastSeenAt = now
	if messageType = strings.TrimSpace(messageType); messageType != "" {
		active.LastMessageType = messageType
	}
	if transport = strings.TrimSpace(transport); transport != "" {
		active.Transport = transport
	}
	m.active[id] = active
}

func (m *ReceiverManager) unregisterActiveReceiver(id string) {
	if id == "" {
		return
	}
	m.mu.Lock()
	delete(m.active, id)
	m.mu.Unlock()
}

func (m *ReceiverManager) ActiveSnapshots() []map[string]any {
	if m == nil {
		return nil
	}
	now := time.Now().UTC()
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]map[string]any, 0, len(m.active))
	for _, active := range m.active {
		out = append(out, map[string]any{
			"id":                active.ID,
			"feed_id":           active.FeedID,
			"receiver_id":       active.ReceiverID,
			"receiver_hostname": active.ReceiverHostname,
			"remote_addr":       active.RemoteAddr,
			"connected_at":      active.ConnectedAt,
			"connected_ms":      now.Sub(active.ConnectedAt).Milliseconds(),
			"last_seen_at":      active.LastSeenAt,
			"last_seen_ms":      now.Sub(active.LastSeenAt).Milliseconds(),
			"last_message_type": active.LastMessageType,
			"transport":         active.Transport,
		})
	}
	sort.Slice(out, func(i, j int) bool {
		return fmt.Sprint(out[i]["connected_at"]) < fmt.Sprint(out[j]["connected_at"])
	})
	return out
}

func (m *ReceiverManager) receiverRequestAllowed(writer http.ResponseWriter, request *http.Request, method string) bool {
	if !m.Enabled() {
		http.NotFound(writer, request)
		return false
	}
	if request.Method != method {
		writer.Header().Set("Allow", method)
		receiverError(writer, http.StatusMethodNotAllowed, "method not allowed")
		return false
	}
	return true
}

func (m *ReceiverManager) takeChallenge(id string, now time.Time) (receiverChallenge, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pruneLocked(now)
	challenge, ok := m.challenges[id]
	if ok {
		delete(m.challenges, id)
	}
	return challenge, ok
}

func (m *ReceiverManager) matchPairingToken(payload receiverPairCompleteRequest, challenge receiverChallenge) (receiverPairingToken, string, error) {
	store, err := m.loadStore()
	if err != nil {
		return receiverPairingToken{}, "", fmt.Errorf("could not read receiver credentials")
	}
	for _, token := range m.receiverPairingTokens() {
		if !token.enabled || !token.allowsFeed(challenge.FeedID) {
			continue
		}
		if _, consumed := store.ConsumedPairingTokens[token.ID]; consumed {
			continue
		}
		secret := token.secret()
		if secret == "" {
			continue
		}
		expected := receiverHMACHex(secret, receiverProofMessage("pair-v1", map[string]string{
			"challenge_id":      challenge.ID,
			"feed_id":           challenge.FeedID,
			"receiver_id":       challenge.ReceiverID,
			"receiver_hostname": challenge.ReceiverHostname,
			"receiver_nonce":    challenge.ReceiverNonce,
			"server_nonce":      challenge.ServerNonce,
		}))
		if subtle.ConstantTimeCompare([]byte(expected), []byte(strings.ToLower(payload.Proof))) == 1 {
			return token, secret, nil
		}
	}
	return receiverPairingToken{}, "", fmt.Errorf("receiver pairing token is invalid, exhausted, or out of scope")
}

func (m *ReceiverManager) createCredentialAndCookie(tokenID string, challenge receiverChallenge, now time.Time, request *http.Request) (receiverCredential, string, error) {
	store, err := m.loadStore()
	if err != nil {
		return receiverCredential{}, "", fmt.Errorf("could not read receiver credentials")
	}
	credentialID, err := randomToken()
	if err != nil {
		return receiverCredential{}, "", err
	}
	credentialSecret, err := randomToken()
	if err != nil {
		return receiverCredential{}, "", err
	}
	serverHostname, _ := os.Hostname()
	credential := receiverCredential{
		ID:               credentialID,
		Secret:           credentialSecret,
		FeedID:           challenge.FeedID,
		ReceiverID:       challenge.ReceiverID,
		ReceiverHostname: challenge.ReceiverHostname,
		ServerHostname:   serverHostname,
		CreatedAt:        now,
		ExpiresAt:        now.Add(m.credentialTTL()),
	}
	if store.Credentials == nil {
		store.Credentials = map[string]receiverCredential{}
	}
	if store.ConsumedPairingTokens == nil {
		store.ConsumedPairingTokens = map[string]receiverConsumedToken{}
	}
	store.Credentials[credential.ID] = credential
	store.ConsumedPairingTokens[tokenID] = receiverConsumedToken{
		TokenID:          tokenID,
		FeedID:           challenge.FeedID,
		ReceiverID:       challenge.ReceiverID,
		ReceiverHostname: challenge.ReceiverHostname,
		ConsumedAt:       now,
	}
	store.LastUpdatedAt = now
	if err := m.saveStore(store); err != nil {
		return receiverCredential{}, "", err
	}
	cookie, err := m.issueCookie(receiverCookie{
		FeedID:           challenge.FeedID,
		ReceiverID:       challenge.ReceiverID,
		ReceiverHostname: challenge.ReceiverHostname,
		ExpiresAt:        now.Add(m.cookieTTL()),
	}, challenge.ReceiverHostname)
	return credential, cookie, err
}

func (m *ReceiverManager) issueCookie(cookie receiverCookie, receiverHostname string) (string, error) {
	value, err := receiverUUIDCookie(receiverHostname)
	if err != nil {
		return "", err
	}
	cookie.Digest = digestString(value)
	m.mu.Lock()
	m.cookies[cookie.Digest] = cookie
	m.mu.Unlock()
	return value, nil
}

func (m *ReceiverManager) consumeCookie(value string) (receiverCookie, bool) {
	if value == "" {
		return receiverCookie{}, false
	}
	now := time.Now().UTC()
	digest := digestString(value)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.pruneLocked(now)
	cookie, ok := m.cookies[digest]
	if ok {
		delete(m.cookies, digest)
	}
	return cookie, ok
}

func (m *ReceiverManager) pruneLocked(now time.Time) {
	for id, challenge := range m.challenges {
		if !challenge.ExpiresAt.After(now) {
			delete(m.challenges, id)
		}
	}
	for digest, cookie := range m.cookies {
		if !cookie.ExpiresAt.After(now) {
			delete(m.cookies, digest)
		}
	}
}

func (m *ReceiverManager) loadStore() (receiverStore, error) {
	path := m.credentialsPath()
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return receiverStore{
				Credentials:           map[string]receiverCredential{},
				ConsumedPairingTokens: map[string]receiverConsumedToken{},
			}, nil
		}
		return receiverStore{}, err
	}
	var store receiverStore
	if err := json.Unmarshal(raw, &store); err != nil {
		return receiverStore{}, err
	}
	if store.Credentials == nil {
		store.Credentials = map[string]receiverCredential{}
	}
	if store.ConsumedPairingTokens == nil {
		store.ConsumedPairingTokens = map[string]receiverConsumedToken{}
	}
	normalizeReceiverStore(&store)
	return store, nil
}

func normalizeReceiverStore(store *receiverStore) {
	if store == nil {
		return
	}
	if store.Credentials == nil {
		store.Credentials = map[string]receiverCredential{}
	}
	for id, credential := range store.Credentials {
		if credential.ID == "" {
			credential.ID = id
		}
		if credential.Secret == "" {
			credential.Secret = strings.TrimSpace(credential.LegacyCredentialSecret)
		}
		store.Credentials[id] = credential
	}
	if store.ConsumedPairingTokens == nil {
		store.ConsumedPairingTokens = map[string]receiverConsumedToken{}
	}
	for _, legacy := range store.LegacyConsumedPairTokenDigests {
		tokenID := strings.TrimSpace(legacy.TokenID)
		if tokenID == "" {
			continue
		}
		if _, exists := store.ConsumedPairingTokens[tokenID]; exists {
			continue
		}
		credential := store.Credentials[legacy.CredentialID]
		store.ConsumedPairingTokens[tokenID] = receiverConsumedToken{
			TokenID:          tokenID,
			FeedID:           fallbackText(legacy.FeedID, credential.FeedID),
			ReceiverID:       credential.ReceiverID,
			ReceiverHostname: credential.ReceiverHostname,
			ConsumedAt:       legacy.ConsumedAt,
		}
	}
}

func (m *ReceiverManager) saveStore(store receiverStore) error {
	path := m.credentialsPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.MarshalIndent(store, "", "  ")
	if err != nil {
		return err
	}
	return writeFileAtomic(path, append(raw, '\n'), 0o600)
}

func (m *ReceiverManager) credentialsPath() string {
	configured := strings.TrimSpace(m.config.Webpanel.Receiver.CredentialsPath)
	if configured == "" {
		configured = "runtime/state/receiver_credentials.json"
	}
	return resolveConfigPath(m.configPath, configured)
}

func (m *ReceiverManager) challengeTTL() time.Duration {
	seconds := m.config.Webpanel.Receiver.ChallengeTTLSeconds
	if seconds <= 0 {
		seconds = 60
	}
	return time.Duration(seconds) * time.Second
}

func (m *ReceiverManager) cookieTTL() time.Duration {
	seconds := m.config.Webpanel.Receiver.CookieTTLSeconds
	if seconds <= 0 {
		seconds = 30
	}
	return time.Duration(seconds) * time.Second
}

func (m *ReceiverManager) credentialTTL() time.Duration {
	seconds := m.config.Webpanel.Receiver.CredentialTTLSeconds
	if seconds <= 0 {
		seconds = 365 * 24 * 60 * 60
	}
	return time.Duration(seconds) * time.Second
}

func (m *ReceiverManager) receiverWSURL(request *http.Request) string {
	scheme := "ws"
	if requestIsSecure(request) {
		scheme = "wss"
	}
	return scheme + "://" + request.Host + m.BasePath() + "/ws"
}

type receiverPairingToken struct {
	ID       string
	enabled  bool
	Token    string
	TokenEnv string
	FeedIDs  []string
}

func (m *ReceiverManager) receiverPairingTokens() []receiverPairingToken {
	tokens := make([]receiverPairingToken, 0, len(m.config.Webpanel.Receiver.PairingTokens))
	for index, raw := range m.config.Webpanel.Receiver.PairingTokens {
		enabled := true
		if raw.Enabled != nil {
			enabled = *raw.Enabled
		}
		id := strings.TrimSpace(raw.ID)
		if id == "" {
			id = fmt.Sprintf("pairing-token-%d", index+1)
		}
		tokens = append(tokens, receiverPairingToken{
			ID:       id,
			enabled:  enabled,
			Token:    strings.TrimSpace(raw.Token),
			TokenEnv: strings.TrimSpace(raw.TokenEnv),
			FeedIDs:  uniqueStrings(raw.FeedIDs),
		})
	}
	return tokens
}

func (token receiverPairingToken) allowsFeed(feedID string) bool {
	if len(token.FeedIDs) == 0 {
		return true
	}
	for _, candidate := range token.FeedIDs {
		if candidate == feedID {
			return true
		}
	}
	return false
}

func (token receiverPairingToken) secret() string {
	if token.TokenEnv != "" {
		return strings.TrimSpace(os.Getenv(token.TokenEnv))
	}
	return strings.TrimSpace(token.Token)
}

func (m *ReceiverManager) receiverTransmitter(feedID string) (map[string]any, error) {
	root, err := loadYAMLMap(m.configPath)
	if err != nil {
		return nil, err
	}
	feeds, err := loadFeedsXML(m.configPath, root)
	if err != nil {
		return nil, err
	}
	for _, feed := range feeds.Feeds {
		if strings.TrimSpace(feed.ID) != feedID {
			continue
		}
		if !xmlBool(feed.EnabledRaw, true) {
			return nil, fmt.Errorf("feed %s is disabled", feedID)
		}
		transmitter := receiverPreferredTransmitter(feed)
		frequency, err := strconv.ParseFloat(strings.TrimSpace(transmitter.frequencyText()), 64)
		if err != nil || frequency <= 0 {
			return nil, fmt.Errorf("feed %s has no usable transmitter frequency", feedID)
		}
		bandwidth := m.config.Webpanel.Receiver.TransmitterDefaults.BandwidthKHz
		if bandwidth <= 0 {
			bandwidth = 12.5
		}
		deviation := m.config.Webpanel.Receiver.TransmitterDefaults.DeviationHz
		if deviation <= 0 {
			deviation = 5000
		}
		preemphasis := strings.TrimSpace(m.config.Webpanel.Receiver.TransmitterDefaults.Preemphasis)
		if preemphasis == "" {
			preemphasis = "none"
		}
		if transmitter.isRelationship("fm") {
			if m.config.Webpanel.Receiver.TransmitterDefaults.BandwidthKHz <= 0 {
				bandwidth = 200
			}
			if m.config.Webpanel.Receiver.TransmitterDefaults.DeviationHz <= 0 {
				deviation = 75000
			}
			if strings.TrimSpace(m.config.Webpanel.Receiver.TransmitterDefaults.Preemphasis) == "" {
				preemphasis = "75"
			}
		}
		return map[string]any{
			"feed_id":       feedID,
			"site_name":     fallbackText(transmitter.SiteName, feedID),
			"callsign":      strings.TrimSpace(transmitter.Callsign),
			"relationship":  transmitter.relationship(),
			"frequency_mhz": frequency,
			"gpclk":         transmitter.gpclk(),
			"gpio":          transmitter.gpio(),
			"transmitters":  transmitterPayloads(feed),
			"bandwidth_khz": bandwidth,
			"deviation_hz":  deviation,
			"preemphasis":   preemphasis,
			"sample_rate":   numberAt(root, []string{"playout", "sample_rate"}, 48000),
			"channels":      numberAt(root, []string{"playout", "channels"}, 1),
		}, nil
	}
	return nil, fmt.Errorf("feed %s is not configured", feedID)
}

func cleanPairCompleteRequest(payload receiverPairCompleteRequest) receiverPairCompleteRequest {
	payload.ChallengeID = strings.TrimSpace(payload.ChallengeID)
	payload.FeedID = strings.TrimSpace(payload.FeedID)
	payload.ReceiverID = strings.TrimSpace(payload.ReceiverID)
	payload.ReceiverHostname = strings.TrimSpace(payload.ReceiverHostname)
	payload.Nonce = strings.TrimSpace(payload.Nonce)
	payload.Proof = strings.ToLower(strings.TrimSpace(payload.Proof))
	return payload
}

func cleanSessionRequest(payload receiverSessionRequest) receiverSessionRequest {
	payload.FeedID = strings.TrimSpace(payload.FeedID)
	payload.ReceiverID = strings.TrimSpace(payload.ReceiverID)
	payload.ReceiverHostname = strings.TrimSpace(payload.ReceiverHostname)
	payload.CredentialID = strings.TrimSpace(payload.CredentialID)
	payload.Nonce = strings.TrimSpace(payload.Nonce)
	payload.Proof = strings.ToLower(strings.TrimSpace(payload.Proof))
	return payload
}

func challengeMatchesComplete(challenge receiverChallenge, payload receiverPairCompleteRequest) bool {
	return challenge.FeedID == payload.FeedID &&
		challenge.ReceiverID == payload.ReceiverID &&
		challenge.ReceiverHostname == payload.ReceiverHostname &&
		challenge.ReceiverNonce == payload.Nonce
}

func receiverProofMessage(kind string, values map[string]string) []byte {
	payload := make(map[string]string, len(values)+1)
	payload["kind"] = kind
	for key, value := range values {
		payload[key] = value
	}
	keys := make([]string, 0, len(payload))
	for key := range payload {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	var builder strings.Builder
	builder.WriteByte('{')
	for index, key := range keys {
		if index > 0 {
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.Quote(key))
		builder.WriteByte(':')
		builder.WriteString(strconv.Quote(payload[key]))
	}
	builder.WriteByte('}')
	return []byte(builder.String())
}

func receiverHMACHex(secret string, message []byte) string {
	mac := hmac.New(sha256.New, []byte(secret))
	_, _ = mac.Write(message)
	return hex.EncodeToString(mac.Sum(nil))
}

func receiverUUIDCookie(receiverHostname string) (string, error) {
	var raw [16]byte
	nowMillis := uint64(time.Now().UTC().UnixMilli())
	raw[0] = byte(nowMillis >> 40)
	raw[1] = byte(nowMillis >> 32)
	raw[2] = byte(nowMillis >> 24)
	raw[3] = byte(nowMillis >> 16)
	raw[4] = byte(nowMillis >> 8)
	raw[5] = byte(nowMillis)
	receiverHash := sha256.Sum256([]byte(receiverHostname))
	serverName, _ := os.Hostname()
	serverHash := sha256.Sum256([]byte(serverName))
	raw[6] = 0x80 | (receiverHash[0] & 0x0f)
	raw[7] = receiverHash[1]
	raw[8] = 0x80 | (serverHash[0] & 0x3f)
	raw[9] = serverHash[1]
	if _, err := rand.Read(raw[10:]); err != nil {
		return "", err
	}
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		binary.BigEndian.Uint32(raw[0:4]),
		binary.BigEndian.Uint16(raw[4:6]),
		binary.BigEndian.Uint16(raw[6:8]),
		binary.BigEndian.Uint16(raw[8:10]),
		raw[10:16],
	), nil
}

func digestString(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func decodeReceiverJSON(writer http.ResponseWriter, request *http.Request, target any) bool {
	defer request.Body.Close()
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		receiverError(writer, http.StatusBadRequest, "invalid receiver JSON")
		return false
	}
	return true
}

func receiverError(writer http.ResponseWriter, status int, detail string) {
	writer.Header().Set("Content-Type", "application/json")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(map[string]any{"detail": detail})
}

func receiverCookieFromRequest(request *http.Request) string {
	header := strings.TrimSpace(request.Header.Get("Authorization"))
	const prefix = "HazeReceiverCookie "
	if strings.HasPrefix(strings.ToLower(header), strings.ToLower(prefix)) {
		return strings.TrimSpace(header[len(prefix):])
	}
	return ""
}

func requestIsSecure(request *http.Request) bool {
	if request.TLS != nil {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(request.Header.Get("X-Forwarded-Proto")), "https")
}

func writeReceiverWS(ctx context.Context, connection *websocket.Conn, payload map[string]any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return connection.Write(ctx, websocket.MessageText, raw)
}

func numberAt(source map[string]any, path []string, fallback int) int {
	value, ok := valueAt(source, path)
	if !ok || value == nil {
		return fallback
	}
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case string:
		return parseIntText(typed, fallback)
	default:
		return fallback
	}
}
