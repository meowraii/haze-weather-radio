package webgateway

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestListLogViewerFilesIncludesLogsAndLegacyRotations(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "haze.2026-07-17.log"), "current")
	mustWrite(t, filepath.Join(root, "haze.log.2026-07-16"), "legacy")
	mustWrite(t, filepath.Join(root, "access", "admin.log"), "audit")
	mustWrite(t, filepath.Join(root, "access", "integrity.sig"), "signature")
	mustWrite(t, filepath.Join(root, "notes.txt"), "not a log")

	payload, err := listLogViewerFiles(root)
	if err != nil {
		t.Fatal(err)
	}
	files, ok := payload["files"].([]logViewerFile)
	if !ok {
		t.Fatalf("files payload type = %T", payload["files"])
	}
	names := map[string]bool{}
	for _, file := range files {
		names[file.Name] = true
	}
	for _, expected := range []string{"haze.2026-07-17.log", "haze.log.2026-07-16", "access/admin.log"} {
		if !names[expected] {
			t.Fatalf("missing log %q from %#v", expected, names)
		}
	}
	if names["access/integrity.sig"] || names["notes.txt"] {
		t.Fatalf("non-log files were exposed: %#v", names)
	}
}

func TestListLogViewerFilesSkipsSymlinks(t *testing.T) {
	root := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.log")
	mustWrite(t, outside, "secret")
	link := filepath.Join(root, "linked.log")
	if err := os.Symlink(outside, link); err != nil {
		t.Skipf("symlinks are not available: %v", err)
	}

	payload, err := listLogViewerFiles(root)
	if err != nil {
		t.Fatal(err)
	}
	files := payload["files"].([]logViewerFile)
	if len(files) != 0 {
		t.Fatalf("symlink was exposed: %#v", files)
	}
	if _, err := tailLogViewerFile(root, map[string]any{"file": "linked.log"}); err == nil {
		t.Fatal("symlink tail was allowed")
	}
}

func TestTailLogViewerFileIsBoundedAndSupportsOffsets(t *testing.T) {
	root := t.TempDir()
	var content strings.Builder
	for content.Len() < maxLogViewerReadBytes*2 {
		content.WriteString("2026-07-17 test log line\n")
	}
	mustWrite(t, filepath.Join(root, "haze.2026-07-17.log"), content.String())

	initial, err := tailLogViewerFile(root, map[string]any{"file": "haze.2026-07-17.log"})
	if err != nil {
		t.Fatal(err)
	}
	if !initial["truncated"].(bool) || !initial["caught_up"].(bool) {
		t.Fatalf("unexpected initial tail state: %#v", initial)
	}
	if len(initial["content"].(string)) > maxLogViewerReadBytes {
		t.Fatalf("initial content exceeded cap: %d", len(initial["content"].(string)))
	}
	next := initial["next_offset"].(int64)
	mustAppend(t, filepath.Join(root, "haze.2026-07-17.log"), "after-tail\n")

	delta, err := tailLogViewerFile(root, map[string]any{
		"file":   "haze.2026-07-17.log",
		"offset": next,
	})
	if err != nil {
		t.Fatal(err)
	}
	if delta["content"] != "after-tail\n" || !delta["caught_up"].(bool) {
		t.Fatalf("unexpected delta: %#v", delta)
	}
}

func TestTailLogViewerFileRejectsTraversalAndInvalidOffsets(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "valid.log"), "ok\n")
	for _, name := range []string{"../secret.log", `..\secret.log`, "/tmp/secret.log", "integrity.sig"} {
		if _, err := tailLogViewerFile(root, map[string]any{"file": name}); err == nil {
			t.Fatalf("unsafe log name %q was allowed", name)
		}
	}
	if _, err := tailLogViewerFile(root, map[string]any{"file": "valid.log", "offset": -1}); err == nil {
		t.Fatal("negative offset was allowed")
	}
}

func TestLogViewerCommandsRequireLogPermission(t *testing.T) {
	for _, command := range []string{"logs.list", "logs.tail"} {
		if nonAdminCommandAllowed(command, nil, Account{}) {
			t.Fatalf("%s was allowed without log permission", command)
		}
		if !nonAdminCommandAllowed(command, nil, Account{CanViewLogs: true}) {
			t.Fatalf("%s was denied with log permission", command)
		}
	}
}

func mustAppend(t *testing.T, path string, content string) {
	t.Helper()
	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := file.WriteString(content); err != nil {
		file.Close()
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
