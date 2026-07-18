package webgateway

import (
	"bufio"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const auditKeyID = "haze-audit-v1"

var auditCategories = map[string]struct{}{
	"access":   {},
	"alerts":   {},
	"webpanel": {},
}

// AuditEvent is the canonical payload protected by the per-account HMAC chain.
type AuditEvent struct {
	Timestamp     time.Time      `json:"timestamp"`
	Event         string         `json:"event"`
	ActorID       string         `json:"actor_id,omitempty"`
	ActorUsername string         `json:"actor_username"`
	SessionID     string         `json:"session_id,omitempty"`
	IP            string         `json:"ip,omitempty"`
	UserAgent     string         `json:"user_agent,omitempty"`
	Severity      string         `json:"severity,omitempty"`
	Details       map[string]any `json:"details,omitempty"`
}

type auditLine struct {
	Index        uint64          `json:"index"`
	KeyID        string          `json:"key_id"`
	PreviousHMAC string          `json:"previous_hmac"`
	Payload      json.RawMessage `json:"payload"`
	HMAC         string          `json:"hmac"`
}

type auditChainHead struct {
	Index uint64 `json:"index"`
	HMAC  string `json:"hmac"`
}

type auditHeadsDocument struct {
	KeyID    string                    `json:"key_id"`
	Category string                    `json:"category"`
	Heads    map[string]auditChainHead `json:"heads"`
	HMAC     string                    `json:"hmac"`
}

type auditHeadsPayload struct {
	KeyID    string                    `json:"key_id"`
	Category string                    `json:"category"`
	Heads    map[string]auditChainHead `json:"heads"`
}

type auditLogger struct {
	baseDir string
	key     []byte
	mu      sync.Mutex
}

func newAuditLogger(baseDir string, key []byte) (*auditLogger, error) {
	if len(key) < 32 {
		return nil, fmt.Errorf("audit HMAC key must contain at least 32 bytes")
	}
	baseDir = filepath.Clean(baseDir)
	for category := range auditCategories {
		if err := os.MkdirAll(filepath.Join(baseDir, category), 0o700); err != nil {
			return nil, fmt.Errorf("create %s audit directory: %w", category, err)
		}
	}
	logger := &auditLogger{baseDir: baseDir, key: append([]byte{}, key...)}
	if err := logger.VerifyAll(); err != nil {
		return nil, err
	}
	return logger, nil
}

func (l *auditLogger) Append(category string, event AuditEvent) error {
	if l == nil {
		return fmt.Errorf("audit logger is unavailable")
	}
	category = strings.ToLower(strings.TrimSpace(category))
	if _, ok := auditCategories[category]; !ok {
		return fmt.Errorf("unsupported audit category %q", category)
	}
	username := strings.TrimSpace(event.ActorUsername)
	if !validUsername.MatchString(username) {
		username = "unknown"
	}
	event.ActorUsername = username
	event.Event = strings.ToUpper(strings.TrimSpace(event.Event))
	if event.Event == "" {
		return fmt.Errorf("audit event type is required")
	}
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	} else {
		event.Timestamp = event.Timestamp.UTC()
	}
	payload, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("encode audit event: %w", err)
	}

	l.mu.Lock()
	defer l.mu.Unlock()
	path := l.logPath(category, username)
	head, err := l.verifyFileLocked(path)
	if err != nil {
		return err
	}
	heads, err := l.readHeadsLocked(category)
	if err != nil {
		return err
	}
	if recorded, ok := heads[username]; ok && (recorded.Index != head.Index || !hmac.Equal(decodeAuditMAC(recorded.HMAC), decodeAuditMAC(head.HMAC))) {
		return fmt.Errorf("CRITICAL audit chain head mismatch for %s/%s", category, username)
	}
	previous := decodeAuditMAC(head.HMAC)
	index := head.Index + 1
	current := l.sign(index, previous, payload)
	line := auditLine{
		Index:        index,
		KeyID:        auditKeyID,
		PreviousHMAC: base64.RawURLEncoding.EncodeToString(previous),
		Payload:      payload,
		HMAC:         base64.RawURLEncoding.EncodeToString(current),
	}
	raw, err := json.Marshal(line)
	if err != nil {
		return fmt.Errorf("encode audit chain line: %w", err)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open audit log: %w", err)
	}
	if _, err := file.Write(append(raw, '\n')); err != nil {
		_ = file.Close()
		return fmt.Errorf("append audit log: %w", err)
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return fmt.Errorf("sync audit log: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close audit log: %w", err)
	}
	heads[username] = auditChainHead{Index: index, HMAC: line.HMAC}
	if err := l.writeHeadsLocked(category, heads); err != nil {
		return err
	}
	return nil
}

func (l *auditLogger) VerifyAll() error {
	if l == nil {
		return fmt.Errorf("audit logger is unavailable")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	for category := range auditCategories {
		heads, err := l.readHeadsLocked(category)
		if err != nil {
			return err
		}
		entries, err := os.ReadDir(filepath.Join(l.baseDir, category))
		if err != nil {
			return fmt.Errorf("read audit directory: %w", err)
		}
		seen := map[string]struct{}{}
		for _, entry := range entries {
			if entry.IsDir() || !strings.HasSuffix(strings.ToLower(entry.Name()), ".log") {
				continue
			}
			username := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
			head, err := l.verifyFileLocked(filepath.Join(l.baseDir, category, entry.Name()))
			if err != nil {
				return err
			}
			recorded, ok := heads[username]
			if !ok || recorded.Index != head.Index || !hmac.Equal(decodeAuditMAC(recorded.HMAC), decodeAuditMAC(head.HMAC)) {
				return fmt.Errorf("CRITICAL audit integrity signature mismatch for %s/%s", category, username)
			}
			seen[username] = struct{}{}
		}
		for username, head := range heads {
			if head.Index == 0 {
				continue
			}
			if _, ok := seen[username]; !ok {
				return fmt.Errorf("CRITICAL audit log missing for signed chain %s/%s", category, username)
			}
		}
	}
	return nil
}

func (l *auditLogger) RenameAccount(oldUsername string, newUsername string) error {
	if !validUsername.MatchString(oldUsername) || !validUsername.MatchString(newUsername) {
		return fmt.Errorf("valid old and new usernames are required")
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	type renamePlan struct {
		category     string
		oldPath      string
		newPath      string
		oldHeads     map[string]auditChainHead
		newHeads     map[string]auditChainHead
		moveFile     bool
		fileMoved    bool
		headsWritten bool
	}
	plans := make([]renamePlan, 0, len(auditCategories))
	for _, category := range []string{"access", "alerts", "webpanel"} {
		oldPath := l.logPath(category, oldUsername)
		newPath := l.logPath(category, newUsername)
		_, oldErr := os.Stat(oldPath)
		oldExists := oldErr == nil
		if oldErr != nil && !errors.Is(oldErr, os.ErrNotExist) {
			return fmt.Errorf("inspect existing audit log: %w", oldErr)
		}
		if _, err := os.Stat(newPath); err == nil {
			return fmt.Errorf("audit log already exists for renamed account %s", newUsername)
		} else if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("inspect renamed audit log destination: %w", err)
		}
		heads, err := l.readHeadsLocked(category)
		if err != nil {
			return err
		}
		if _, exists := heads[newUsername]; exists {
			return fmt.Errorf("audit chain already exists for renamed account %s", newUsername)
		}
		head, hasHead := heads[oldUsername]
		if oldExists != hasHead {
			return fmt.Errorf("CRITICAL audit log and signed chain disagree for %s/%s", category, oldUsername)
		}
		if oldExists {
			verified, err := l.verifyFileLocked(oldPath)
			if err != nil {
				return err
			}
			if verified.Index != head.Index || !hmac.Equal(decodeAuditMAC(verified.HMAC), decodeAuditMAC(head.HMAC)) {
				return fmt.Errorf("CRITICAL audit chain head mismatch for %s/%s", category, oldUsername)
			}
		}
		updated := cloneAuditHeads(heads)
		if hasHead {
			delete(updated, oldUsername)
			updated[newUsername] = head
		}
		plans = append(plans, renamePlan{
			category: category, oldPath: oldPath, newPath: newPath,
			oldHeads: cloneAuditHeads(heads), newHeads: updated, moveFile: oldExists,
		})
	}
	rollback := func(last int) error {
		var rollbackErr error
		for index := last; index >= 0; index-- {
			plan := &plans[index]
			if plan.headsWritten {
				if err := l.writeHeadsLocked(plan.category, plan.oldHeads); err != nil && rollbackErr == nil {
					rollbackErr = err
				}
			}
			if plan.fileMoved {
				if err := os.Rename(plan.newPath, plan.oldPath); err != nil && rollbackErr == nil {
					rollbackErr = err
				}
			}
		}
		return rollbackErr
	}
	for index := range plans {
		plan := &plans[index]
		if plan.moveFile {
			if err := os.Rename(plan.oldPath, plan.newPath); err != nil {
				if rollbackErr := rollback(index - 1); rollbackErr != nil {
					return fmt.Errorf("CRITICAL audit rename failed and rollback failed: %v; rollback: %w", err, rollbackErr)
				}
				return fmt.Errorf("rename audit log: %w", err)
			}
			plan.fileMoved = true
		}
		if len(plan.newHeads) > 0 || len(plan.oldHeads) > 0 {
			if err := l.writeHeadsLocked(plan.category, plan.newHeads); err != nil {
				if rollbackErr := rollback(index); rollbackErr != nil {
					return fmt.Errorf("CRITICAL audit rename checkpoint failed and rollback failed: %v; rollback: %w", err, rollbackErr)
				}
				return err
			}
			plan.headsWritten = true
		}
	}
	return nil
}

func cloneAuditHeads(source map[string]auditChainHead) map[string]auditChainHead {
	out := make(map[string]auditChainHead, len(source))
	for username, head := range source {
		out[username] = head
	}
	return out
}

func (l *auditLogger) logPath(category string, username string) string {
	return filepath.Join(l.baseDir, category, username+".log")
}

func (l *auditLogger) verifyFileLocked(path string) (auditChainHead, error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return auditChainHead{}, nil
	}
	if err != nil {
		return auditChainHead{}, fmt.Errorf("open audit log for verification: %w", err)
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var (
		expectedIndex uint64 = 1
		previous             = []byte{}
	)
	for {
		raw, readErr := reader.ReadBytes('\n')
		if len(raw) > 0 {
			raw = []byte(strings.TrimSpace(string(raw)))
			if len(raw) == 0 {
				if readErr == nil {
					continue
				}
			} else {
				var line auditLine
				if err := json.Unmarshal(raw, &line); err != nil {
					return auditChainHead{}, fmt.Errorf("CRITICAL invalid audit JSON in %s: %w", path, err)
				}
				if line.Index != expectedIndex || line.KeyID != auditKeyID {
					return auditChainHead{}, fmt.Errorf("CRITICAL audit sequence mismatch in %s at index %d", path, expectedIndex)
				}
				if !hmac.Equal(decodeAuditMAC(line.PreviousHMAC), previous) {
					return auditChainHead{}, fmt.Errorf("CRITICAL previous audit HMAC mismatch in %s at index %d", path, expectedIndex)
				}
				expected := l.sign(line.Index, previous, line.Payload)
				actual := decodeAuditMAC(line.HMAC)
				if !hmac.Equal(expected, actual) {
					return auditChainHead{}, fmt.Errorf("CRITICAL audit HMAC mismatch in %s at index %d", path, expectedIndex)
				}
				previous = actual
				expectedIndex++
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return auditChainHead{}, fmt.Errorf("read audit log: %w", readErr)
		}
	}
	return auditChainHead{
		Index: expectedIndex - 1,
		HMAC:  base64.RawURLEncoding.EncodeToString(previous),
	}, nil
}

func (l *auditLogger) sign(index uint64, previous []byte, payload []byte) []byte {
	mac := hmac.New(sha256.New, l.key)
	_, _ = mac.Write([]byte("haze-audit-chain-v1\x00"))
	_, _ = mac.Write([]byte(auditKeyID))
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], index)
	_, _ = mac.Write(encoded[:])
	_, _ = mac.Write(previous)
	binary.BigEndian.PutUint64(encoded[:], uint64(len(payload)))
	_, _ = mac.Write(encoded[:])
	_, _ = mac.Write(payload)
	return mac.Sum(nil)
}

func (l *auditLogger) readHeadsLocked(category string) (map[string]auditChainHead, error) {
	path := filepath.Join(l.baseDir, category, "integrity.sig")
	raw, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]auditChainHead{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read audit integrity signature: %w", err)
	}
	var document auditHeadsDocument
	if err := json.Unmarshal(raw, &document); err != nil {
		return nil, fmt.Errorf("CRITICAL invalid audit integrity signature: %w", err)
	}
	if document.KeyID != auditKeyID || document.Category != category || document.Heads == nil {
		return nil, fmt.Errorf("CRITICAL invalid audit integrity signature metadata for %s", category)
	}
	payload := auditHeadsPayload{KeyID: document.KeyID, Category: document.Category, Heads: document.Heads}
	expected, err := l.signHeads(payload)
	if err != nil {
		return nil, err
	}
	actual := decodeAuditMAC(document.HMAC)
	if len(actual) != sha256.Size || !hmac.Equal(expected, actual) {
		return nil, fmt.Errorf("CRITICAL audit integrity signature HMAC mismatch for %s", category)
	}
	return document.Heads, nil
}

func (l *auditLogger) writeHeadsLocked(category string, heads map[string]auditChainHead) error {
	if heads == nil {
		heads = map[string]auditChainHead{}
	}
	payload := auditHeadsPayload{KeyID: auditKeyID, Category: category, Heads: heads}
	signature, err := l.signHeads(payload)
	if err != nil {
		return err
	}
	document := auditHeadsDocument{
		KeyID: payload.KeyID, Category: payload.Category, Heads: payload.Heads,
		HMAC: base64.RawURLEncoding.EncodeToString(signature),
	}
	raw, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return fmt.Errorf("encode audit integrity signature: %w", err)
	}
	path := filepath.Join(l.baseDir, category, "integrity.sig")
	if err := writeFileAtomicMode(path, append(raw, '\n'), 0o600); err != nil {
		return fmt.Errorf("write audit integrity signature: %w", err)
	}
	return nil
}

func (l *auditLogger) signHeads(payload auditHeadsPayload) ([]byte, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode audit heads payload: %w", err)
	}
	mac := hmac.New(sha256.New, l.key)
	_, _ = mac.Write([]byte("haze-audit-heads-v1\x00"))
	_, _ = mac.Write(raw)
	return mac.Sum(nil), nil
}

func decodeAuditMAC(value string) []byte {
	raw, _ := base64.RawURLEncoding.DecodeString(strings.TrimSpace(value))
	return raw
}

func writeFileAtomicMode(path string, raw []byte, mode os.FileMode) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), ".audit-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(mode); err != nil {
		_ = tmp.Close()
		return err
	}
	if _, err := tmp.Write(raw); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return replaceAuditFileAtomically(tmpPath, path)
}
