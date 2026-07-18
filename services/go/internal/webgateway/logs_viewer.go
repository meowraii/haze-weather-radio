package webgateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	maxLogViewerFiles     = 500
	maxLogViewerReadBytes = 128 * 1024
)

type logViewerFile struct {
	Name       string    `json:"name"`
	Size       int64     `json:"size"`
	ModifiedAt time.Time `json:"modified_at"`
}

func logsViewerRoot(configPath string) string {
	if workingDirectory, err := os.Getwd(); err == nil {
		candidate := filepath.Join(workingDirectory, "logs")
		if info, statErr := os.Stat(candidate); statErr == nil && info.IsDir() {
			return filepath.Clean(candidate)
		}
	}
	return resolveConfigPath(configPath, "logs")
}

func listLogViewerFiles(root string) (map[string]any, error) {
	root = filepath.Clean(root)
	files := make([]logViewerFile, 0)
	err := filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			if path == root {
				return walkErr
			}
			return nil
		}
		if path == root {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		if len(files) >= maxLogViewerFiles {
			return fs.SkipAll
		}
		if !logViewerFilename(entry.Name()) {
			return nil
		}
		info, err := entry.Info()
		if err != nil || !info.Mode().IsRegular() {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil || relative == "." || relativeEscapesRoot(relative) {
			return nil
		}
		files = append(files, logViewerFile{
			Name:       filepath.ToSlash(relative),
			Size:       info.Size(),
			ModifiedAt: info.ModTime().UTC(),
		})
		return nil
	})
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return map[string]any{"files": []logViewerFile{}}, nil
		}
		return nil, fmt.Errorf("list logs: %w", err)
	}
	sort.SliceStable(files, func(left, right int) bool {
		if files[left].ModifiedAt.Equal(files[right].ModifiedAt) {
			return files[left].Name < files[right].Name
		}
		return files[left].ModifiedAt.After(files[right].ModifiedAt)
	})
	return map[string]any{"files": files}, nil
}

func tailLogViewerFile(root string, payload map[string]any) (map[string]any, error) {
	name := strings.TrimSpace(stringValue(payload, "file"))
	path, err := secureLogViewerPath(root, name)
	if err != nil {
		return nil, err
	}
	file, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, fmt.Errorf("log file is no longer available")
		}
		return nil, fmt.Errorf("open log file: %w", err)
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("inspect log file: %w", err)
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("requested log is not a regular file")
	}

	offset, hasOffset, err := logViewerOffset(payload)
	if err != nil {
		return nil, err
	}
	reset := hasOffset && offset > info.Size()
	initial := !hasOffset || reset
	start := offset
	if initial {
		start = info.Size() - maxLogViewerReadBytes
		if start < 0 {
			start = 0
		}
	}
	available := info.Size() - start
	if available < 0 {
		available = 0
	}
	readBytes := minInt64(available, maxLogViewerReadBytes)
	raw := make([]byte, readBytes)
	read := 0
	if readBytes > 0 {
		read, err = file.ReadAt(raw, start)
		if err != nil && !errors.Is(err, io.EOF) {
			return nil, fmt.Errorf("read log file: %w", err)
		}
		raw = raw[:read]
	}
	nextOffset := start + int64(read)
	if initial && start > 0 {
		if newline := bytesIndexByte(raw, '\n'); newline >= 0 {
			raw = raw[newline+1:]
		}
	}
	return map[string]any{
		"file":        name,
		"content":     strings.ToValidUTF8(string(raw), "�"),
		"next_offset": nextOffset,
		"size":        info.Size(),
		"modified_at": info.ModTime().UTC(),
		"reset":       reset,
		"truncated":   initial && start > 0,
		"caught_up":   nextOffset >= info.Size(),
	}, nil
}

func secureLogViewerPath(root string, name string) (string, error) {
	if name == "" || strings.ContainsRune(name, '\x00') || strings.Contains(name, "\\") {
		return "", fmt.Errorf("log file is required")
	}
	normalized := filepath.ToSlash(name)
	clean := filepath.ToSlash(filepath.Clean(filepath.FromSlash(normalized)))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, "../") || filepath.IsAbs(filepath.FromSlash(normalized)) {
		return "", fmt.Errorf("invalid log file")
	}
	if !logViewerFilename(filepath.Base(filepath.FromSlash(clean))) {
		return "", fmt.Errorf("requested file is not a log")
	}
	root = filepath.Clean(root)
	candidate := filepath.Join(root, filepath.FromSlash(clean))
	relative, err := filepath.Rel(root, candidate)
	if err != nil || relativeEscapesRoot(relative) {
		return "", fmt.Errorf("invalid log file")
	}
	current := root
	for _, part := range strings.Split(filepath.Clean(relative), string(filepath.Separator)) {
		if part == "" || part == "." {
			continue
		}
		current = filepath.Join(current, part)
		info, err := os.Lstat(current)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return "", fmt.Errorf("log file is no longer available")
			}
			return "", fmt.Errorf("inspect log path: %w", err)
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return "", fmt.Errorf("symbolic links are not available in the log viewer")
		}
	}
	return candidate, nil
}

func logViewerFilename(name string) bool {
	lower := strings.ToLower(strings.TrimSpace(name))
	if lower == "" || strings.HasSuffix(lower, ".gz") || strings.HasSuffix(lower, ".zip") || strings.HasSuffix(lower, ".xz") {
		return false
	}
	return strings.HasSuffix(lower, ".log") || strings.Contains(lower, ".log.")
}

func relativeEscapesRoot(relative string) bool {
	return relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) || filepath.IsAbs(relative)
}

func logViewerOffset(payload map[string]any) (int64, bool, error) {
	raw, ok := payload["offset"]
	if !ok || raw == nil || strings.TrimSpace(fmt.Sprint(raw)) == "" {
		return 0, false, nil
	}
	var offset int64
	var err error
	switch value := raw.(type) {
	case float64:
		offset = int64(value)
		if float64(offset) != value {
			err = fmt.Errorf("offset must be an integer")
		}
	case float32:
		offset = int64(value)
		if float32(offset) != value {
			err = fmt.Errorf("offset must be an integer")
		}
	case int:
		offset = int64(value)
	case int64:
		offset = value
	case json.Number:
		offset, err = value.Int64()
	default:
		offset, err = strconv.ParseInt(strings.TrimSpace(fmt.Sprint(value)), 10, 64)
	}
	if err != nil || offset < 0 {
		return 0, false, fmt.Errorf("offset must be a non-negative integer")
	}
	return offset, true, nil
}

func minInt64(left int64, right int) int64 {
	if left < int64(right) {
		return left
	}
	return int64(right)
}
