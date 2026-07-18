package webgateway

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

var errSessionNotFound = errors.New("session not found")

// ActiveSession is the server-side lease for one PASETO session.
type ActiveSession struct {
	ID          string    `json:"id"`
	UserID      string    `json:"user_id"`
	Username    string    `json:"username"`
	IP          string    `json:"ip"`
	UserAgent   string    `json:"user_agent"`
	Persistent  bool      `json:"persistent"`
	AuthVersion int64     `json:"auth_version"`
	CreatedAt   time.Time `json:"created_at"`
	LastSeenAt  time.Time `json:"last_seen_at"`
	ExpiresAt   time.Time `json:"expires_at"`
}

type sessionRegistry interface {
	Put(ctx context.Context, rawSessionID string, session ActiveSession) error
	Get(ctx context.Context, rawSessionID string) (ActiveSession, error)
	Touch(ctx context.Context, rawSessionID string, lastSeen time.Time) error
	Delete(ctx context.Context, rawSessionID string) error
	DeleteDigest(ctx context.Context, digest string) error
	DeleteUser(ctx context.Context, userID string) error
	ListUser(ctx context.Context, userID string) ([]ActiveSession, error)
	Close() error
}

type memorySessionRegistry struct {
	mu       sync.Mutex
	sessions map[string]ActiveSession
}

func newMemorySessionRegistry() *memorySessionRegistry {
	return &memorySessionRegistry{sessions: map[string]ActiveSession{}}
}

func (r *memorySessionRegistry) Put(_ context.Context, rawSessionID string, session ActiveSession) error {
	digest := sessionIDDigest(rawSessionID)
	if digest == "" {
		return fmt.Errorf("session ID is required")
	}
	session.ID = digest
	r.mu.Lock()
	r.cleanupLocked(time.Now().UTC())
	r.sessions[digest] = session
	r.mu.Unlock()
	return nil
}

func (r *memorySessionRegistry) Get(_ context.Context, rawSessionID string) (ActiveSession, error) {
	digest := sessionIDDigest(rawSessionID)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cleanupLocked(time.Now().UTC())
	session, ok := r.sessions[digest]
	if !ok {
		return ActiveSession{}, errSessionNotFound
	}
	return session, nil
}

func (r *memorySessionRegistry) Touch(_ context.Context, rawSessionID string, lastSeen time.Time) error {
	digest := sessionIDDigest(rawSessionID)
	r.mu.Lock()
	defer r.mu.Unlock()
	session, ok := r.sessions[digest]
	if !ok || !session.ExpiresAt.After(lastSeen) {
		delete(r.sessions, digest)
		return errSessionNotFound
	}
	session.LastSeenAt = lastSeen.UTC()
	r.sessions[digest] = session
	return nil
}

func (r *memorySessionRegistry) Delete(_ context.Context, rawSessionID string) error {
	return r.DeleteDigest(context.Background(), sessionIDDigest(rawSessionID))
}

func (r *memorySessionRegistry) DeleteDigest(_ context.Context, digest string) error {
	r.mu.Lock()
	delete(r.sessions, strings.TrimSpace(digest))
	r.mu.Unlock()
	return nil
}

func (r *memorySessionRegistry) DeleteUser(_ context.Context, userID string) error {
	r.mu.Lock()
	for digest, session := range r.sessions {
		if session.UserID == userID {
			delete(r.sessions, digest)
		}
	}
	r.mu.Unlock()
	return nil
}

func (r *memorySessionRegistry) ListUser(_ context.Context, userID string) ([]ActiveSession, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cleanupLocked(time.Now().UTC())
	out := []ActiveSession{}
	for _, session := range r.sessions {
		if session.UserID == userID {
			out = append(out, session)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].LastSeenAt.After(out[j].LastSeenAt) })
	return out, nil
}

func (r *memorySessionRegistry) cleanupLocked(now time.Time) {
	for digest, session := range r.sessions {
		if !session.ExpiresAt.After(now) {
			delete(r.sessions, digest)
		}
	}
}

func (r *memorySessionRegistry) Close() error { return nil }

type redisSessionRegistry struct {
	client *redis.Client
	prefix string
}

func newRedisSessionRegistry(ctx context.Context, rawURL string, prefix string) (*redisSessionRegistry, error) {
	options, err := redis.ParseURL(strings.TrimSpace(rawURL))
	if err != nil {
		return nil, fmt.Errorf("parse Redis URL: %w", err)
	}
	options.DialTimeout = 2 * time.Second
	options.ReadTimeout = 2 * time.Second
	options.WriteTimeout = 2 * time.Second
	client := redis.NewClient(options)
	checkCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	if err := client.Ping(checkCtx).Err(); err != nil {
		_ = client.Close()
		return nil, fmt.Errorf("connect session Redis: %w", err)
	}
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		prefix = "haze:auth"
	}
	return &redisSessionRegistry{client: client, prefix: prefix}, nil
}

func (r *redisSessionRegistry) sessionKey(digest string) string {
	return r.prefix + ":session:" + digest
}

func (r *redisSessionRegistry) userKey(userID string) string {
	return r.prefix + ":user:" + userID + ":sessions"
}

func (r *redisSessionRegistry) rateKey(scope string, subject string) string {
	return "{haze-auth-rates}:" + r.prefix + ":rate:" + scope + ":" + sessionIDDigest(subject)
}

func (r *redisSessionRegistry) rateGroupKey(scope string, group string) string {
	return "{haze-auth-rates}:" + r.prefix + ":rate:group:" + scope + ":" + sessionIDDigest(group)
}

// AllowRate applies a Redis-backed sliding-window limit. The Lua script keeps
// the prune, count, and admission steps atomic across gateway processes.
func (r *redisSessionRegistry) AllowRate(ctx context.Context, scope string, subject string, member string, limit int, window time.Duration, now time.Time) (bool, error) {
	if limit <= 0 || window <= 0 {
		return true, nil
	}
	if strings.TrimSpace(subject) == "" || strings.TrimSpace(member) == "" {
		return false, fmt.Errorf("rate-limit subject and member are required")
	}
	const slidingWindowScript = `
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
if redis.call('ZCARD', KEYS[1]) >= limit then
  redis.call('PEXPIRE', KEYS[1], window)
  return 0
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('PEXPIRE', KEYS[1], window)
return 1`
	allowed, err := r.client.Eval(
		ctx,
		slidingWindowScript,
		[]string{r.rateKey(scope, subject)},
		now.UTC().UnixMilli(),
		window.Milliseconds(),
		limit,
		member,
	).Int()
	if err != nil {
		return false, fmt.Errorf("apply Redis rate limit: %w", err)
	}
	return allowed == 1, nil
}

// AllowGroupedRate records the admitted rate key in a short-lived group index.
// Administrators can then clear every username/IP pair for an unlocked account
// without storing plaintext usernames or scanning the Redis keyspace.
func (r *redisSessionRegistry) AllowGroupedRate(ctx context.Context, scope string, subject string, group string, member string, limit int, window time.Duration, now time.Time) (bool, error) {
	if limit <= 0 || window <= 0 {
		return true, nil
	}
	if strings.TrimSpace(subject) == "" || strings.TrimSpace(group) == "" || strings.TrimSpace(member) == "" {
		return false, fmt.Errorf("grouped rate-limit subject, group, and member are required")
	}
	const groupedSlidingWindowScript = `
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - window)
redis.call('SADD', KEYS[2], KEYS[1])
redis.call('PEXPIRE', KEYS[2], window)
if redis.call('ZCARD', KEYS[1]) >= limit then
  redis.call('PEXPIRE', KEYS[1], window)
  return 0
end
redis.call('ZADD', KEYS[1], now, ARGV[4])
redis.call('PEXPIRE', KEYS[1], window)
return 1`
	allowed, err := r.client.Eval(
		ctx,
		groupedSlidingWindowScript,
		[]string{r.rateKey(scope, subject), r.rateGroupKey(scope, group)},
		now.UTC().UnixMilli(),
		window.Milliseconds(),
		limit,
		member,
	).Int()
	if err != nil {
		return false, fmt.Errorf("apply grouped Redis rate limit: %w", err)
	}
	return allowed == 1, nil
}

func (r *redisSessionRegistry) ResetRate(ctx context.Context, scope string, subject string) error {
	if strings.TrimSpace(subject) == "" {
		return nil
	}
	if err := r.client.Del(ctx, r.rateKey(scope, subject)).Err(); err != nil {
		return fmt.Errorf("reset Redis rate limit: %w", err)
	}
	return nil
}

func (r *redisSessionRegistry) ResetRateGroup(ctx context.Context, scope string, group string) error {
	if strings.TrimSpace(group) == "" {
		return nil
	}
	const resetRateGroupScript = `
local members = redis.call('SMEMBERS', KEYS[1])
for _, key in ipairs(members) do
  redis.call('DEL', key)
end
redis.call('DEL', KEYS[1])
return #members`
	if err := r.client.Eval(ctx, resetRateGroupScript, []string{r.rateGroupKey(scope, group)}).Err(); err != nil {
		return fmt.Errorf("reset Redis rate-limit group: %w", err)
	}
	return nil
}

func (r *redisSessionRegistry) Put(ctx context.Context, rawSessionID string, session ActiveSession) error {
	digest := sessionIDDigest(rawSessionID)
	if digest == "" {
		return fmt.Errorf("session ID is required")
	}
	session.ID = digest
	ttl := time.Until(session.ExpiresAt)
	if ttl <= 0 {
		return fmt.Errorf("session is already expired")
	}
	raw, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("encode session: %w", err)
	}
	pipe := r.client.TxPipeline()
	pipe.Set(ctx, r.sessionKey(digest), raw, ttl)
	pipe.SAdd(ctx, r.userKey(session.UserID), digest)
	pipe.ExpireNX(ctx, r.userKey(session.UserID), ttl)
	pipe.ExpireGT(ctx, r.userKey(session.UserID), ttl)
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("store session lease: %w", err)
	}
	return nil
}

func (r *redisSessionRegistry) Get(ctx context.Context, rawSessionID string) (ActiveSession, error) {
	digest := sessionIDDigest(rawSessionID)
	raw, err := r.client.Get(ctx, r.sessionKey(digest)).Bytes()
	if errors.Is(err, redis.Nil) {
		return ActiveSession{}, errSessionNotFound
	}
	if err != nil {
		return ActiveSession{}, fmt.Errorf("load session lease: %w", err)
	}
	var session ActiveSession
	if err := json.Unmarshal(raw, &session); err != nil {
		return ActiveSession{}, fmt.Errorf("decode session lease: %w", err)
	}
	if !session.ExpiresAt.After(time.Now().UTC()) {
		_ = r.DeleteDigest(ctx, digest)
		return ActiveSession{}, errSessionNotFound
	}
	return session, nil
}

func (r *redisSessionRegistry) Touch(ctx context.Context, rawSessionID string, lastSeen time.Time) error {
	session, err := r.Get(ctx, rawSessionID)
	if err != nil {
		return err
	}
	session.LastSeenAt = lastSeen.UTC()
	raw, err := json.Marshal(session)
	if err != nil {
		return fmt.Errorf("encode touched session: %w", err)
	}
	// Preserve the existing lease TTL and update only while the session key
	// still exists. This makes a concurrent revocation win and prevents Touch
	// from recreating a deleted session.
	const touchScript = `
local ttl = redis.call('PTTL', KEYS[1])
if ttl <= 0 then
  return 0
end
redis.call('SET', KEYS[1], ARGV[1], 'PX', ttl, 'XX')
return 1`
	updated, err := r.client.Eval(ctx, touchScript, []string{r.sessionKey(sessionIDDigest(rawSessionID))}, raw).Int()
	if err != nil {
		return fmt.Errorf("touch session lease: %w", err)
	}
	if updated != 1 {
		return errSessionNotFound
	}
	return nil
}

func (r *redisSessionRegistry) Delete(ctx context.Context, rawSessionID string) error {
	return r.DeleteDigest(ctx, sessionIDDigest(rawSessionID))
}

func (r *redisSessionRegistry) DeleteDigest(ctx context.Context, digest string) error {
	digest = strings.TrimSpace(digest)
	if digest == "" {
		return nil
	}
	raw, err := r.client.Get(ctx, r.sessionKey(digest)).Bytes()
	if err != nil && !errors.Is(err, redis.Nil) {
		return fmt.Errorf("load session before revoke: %w", err)
	}
	var session ActiveSession
	_ = json.Unmarshal(raw, &session)
	pipe := r.client.TxPipeline()
	pipe.Del(ctx, r.sessionKey(digest))
	if session.UserID != "" {
		pipe.SRem(ctx, r.userKey(session.UserID), digest)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("revoke session: %w", err)
	}
	return nil
}

func (r *redisSessionRegistry) DeleteUser(ctx context.Context, userID string) error {
	digests, err := r.client.SMembers(ctx, r.userKey(userID)).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		return fmt.Errorf("list user sessions for revoke: %w", err)
	}
	keys := make([]string, 0, len(digests)+1)
	for _, digest := range digests {
		keys = append(keys, r.sessionKey(digest))
	}
	keys = append(keys, r.userKey(userID))
	if err := r.client.Del(ctx, keys...).Err(); err != nil {
		return fmt.Errorf("revoke user sessions: %w", err)
	}
	return nil
}

func (r *redisSessionRegistry) ListUser(ctx context.Context, userID string) ([]ActiveSession, error) {
	digests, err := r.client.SMembers(ctx, r.userKey(userID)).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		return nil, fmt.Errorf("list user sessions: %w", err)
	}
	out := []ActiveSession{}
	stale := []any{}
	for _, digest := range digests {
		raw, err := r.client.Get(ctx, r.sessionKey(digest)).Bytes()
		if errors.Is(err, redis.Nil) {
			stale = append(stale, digest)
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("load user session: %w", err)
		}
		var session ActiveSession
		if err := json.Unmarshal(raw, &session); err != nil {
			return nil, fmt.Errorf("decode user session: %w", err)
		}
		out = append(out, session)
	}
	if len(stale) > 0 {
		_ = r.client.SRem(ctx, r.userKey(userID), stale...).Err()
	}
	sort.Slice(out, func(i, j int) bool { return out[i].LastSeenAt.After(out[j].LastSeenAt) })
	return out, nil
}

func (r *redisSessionRegistry) Close() error {
	if r == nil || r.client == nil {
		return nil
	}
	return r.client.Close()
}

func sessionIDDigest(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(raw))
	return base64.RawURLEncoding.EncodeToString(sum[:])
}
