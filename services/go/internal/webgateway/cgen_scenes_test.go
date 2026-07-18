package webgateway

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
)

func TestCgenScenesCustomLifecycle(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	raw := cgenSceneTestXML("Weather_Wall", "Weather Wall", `<node id="root"/>`)
	created, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               raw,
		"expected_revision": "",
	})
	if err != nil {
		t.Fatalf("save new scene: %v", err)
	}
	revision := stringValue(created, "revision")
	if !validCgenSceneRevision(revision) {
		t.Fatalf("save returned invalid revision %q", revision)
	}
	if got := stringValue(created, "changed_scene_id"); got != "Weather_Wall" {
		t.Fatalf("changed_scene_id = %q", got)
	}

	scenePath := filepath.Join(filepath.Dir(configPath), "managed", "cgen", "scenes", "Weather_Wall.xml")
	info, err := os.Stat(scenePath)
	if err != nil {
		t.Fatalf("stat saved scene: %v", err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
		t.Fatalf("scene permissions = %o, want 600", info.Mode().Perm())
	}

	got, err := getCgenScenePayload(configPath, map[string]any{"id": "Weather_Wall"})
	if err != nil {
		t.Fatalf("get scene: %v", err)
	}
	scene, ok := got["scene"].(map[string]any)
	if !ok {
		t.Fatalf("get scene payload = %#v", got)
	}
	if stringValue(scene, "xml") != raw || stringValue(scene, "revision") != revision {
		t.Fatalf("unexpected scene payload %#v", scene)
	}

	listed, err := listCgenScenesPayload(configPath)
	if err != nil {
		t.Fatalf("list scenes: %v", err)
	}
	rows, ok := listed["scenes"].([]map[string]any)
	if !ok || len(rows) != 1 || stringValue(rows[0], "id") != "Weather_Wall" {
		t.Fatalf("unexpected scene list %#v", listed)
	}

	updatedRaw := cgenSceneTestXML("Weather_Wall", "Weather Wall", `<node id="root"><text>updated</text></node>`)
	updated, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               updatedRaw,
		"expected_revision": revision,
	})
	if err != nil {
		t.Fatalf("update scene: %v", err)
	}
	updatedRevision := stringValue(updated, "revision")
	if updatedRevision == revision || !validCgenSceneRevision(updatedRevision) {
		t.Fatalf("updated revision = %q", updatedRevision)
	}
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               raw,
		"expected_revision": revision,
	}); err == nil || !strings.Contains(err.Error(), "revision conflict") {
		t.Fatalf("stale update error = %v", err)
	}

	deleted, err := deleteCgenScenePayload(configPath, map[string]any{
		"filename":          "Weather_Wall.xml",
		"expected_revision": updatedRevision,
	})
	if err != nil {
		t.Fatalf("delete scene: %v", err)
	}
	if !boolValue(deleted, "deleted") || stringValue(deleted, "changed_scene_id") != "Weather_Wall" {
		t.Fatalf("unexpected delete payload %#v", deleted)
	}
	if !validCgenSceneRevision(stringValue(deleted, "revision")) {
		t.Fatalf("delete collection revision = %q", stringValue(deleted, "revision"))
	}
	if _, err := os.Stat(scenePath); !os.IsNotExist(err) {
		t.Fatalf("deleted scene still exists, stat error = %v", err)
	}
	assertNoCgenSceneTemps(t, filepath.Dir(scenePath))
}

func TestCgenScenesCustomRename(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	created, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Original", "Original Scene", ""),
		"expected_revision": "",
	})
	if err != nil {
		t.Fatalf("create original: %v", err)
	}
	renamed, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Renamed", "Renamed Scene", ""),
		"original_id":       "Original",
		"expected_revision": stringValue(created, "revision"),
	})
	if err != nil {
		t.Fatalf("rename custom scene: %v", err)
	}
	if stringValue(renamed, "changed_scene_id") != "Renamed" {
		t.Fatalf("rename result = %#v", renamed)
	}
	if _, err := getCgenScenePayload(configPath, map[string]any{"id": "Original"}); err == nil {
		t.Fatal("original scene remained after rename")
	}
	if _, err := getCgenScenePayload(configPath, map[string]any{"id": "Renamed"}); err != nil {
		t.Fatalf("get renamed scene: %v", err)
	}
}

func TestCgenScenesRejectPortableFilenameCollision(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	created, err := saveCgenScenePayload(configPath, map[string]any{
		"filename":          "Weather.xml",
		"xml":               cgenSceneTestXML("Weather", "Weather", ""),
		"expected_revision": "",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"filename":          "weather.xml",
		"xml":               cgenSceneTestXML("Another", "Another", ""),
		"expected_revision": "",
	}); err == nil || !strings.Contains(err.Error(), "already in use") {
		t.Fatalf("case-insensitive filename collision error = %v", err)
	}
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"filename":          "weather.xml",
		"xml":               cgenSceneTestXML("Weather", "Weather", `<node id="updated"/>`),
		"expected_revision": stringValue(created, "revision"),
	}); err != nil {
		t.Fatalf("case-only filename update: %v", err)
	}
	if _, err := getCgenScenePayload(configPath, map[string]any{"id": "Weather"}); err != nil {
		t.Fatalf("scene disappeared after case-only filename update: %v", err)
	}
}

func TestCgenScenesProtectedRules(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Program_Passthrough", "Program_Passthrough", ""),
		"expected_revision": "",
	}); err == nil || !strings.Contains(err.Error(), "locked") {
		t.Fatalf("Program_Passthrough save error = %v", err)
	}

	crawl, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Standard_Crawl", "Standard_Crawl", `<node id="root"/>`),
		"expected_revision": "",
	})
	if err != nil {
		t.Fatalf("create protected crawl: %v", err)
	}
	if stringValue(crawl, "filename") != "crawl.xml" {
		t.Fatalf("protected filename = %q", stringValue(crawl, "filename"))
	}
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Standard_Crawl", "Renamed Crawl", ""),
		"expected_revision": stringValue(crawl, "revision"),
	}); err == nil || !strings.Contains(err.Error(), "cannot be renamed") {
		t.Fatalf("protected name change error = %v", err)
	}
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Renamed_Crawl", "Renamed Crawl", ""),
		"original_id":       "Standard_Crawl",
		"expected_revision": stringValue(crawl, "revision"),
	}); err == nil || !strings.Contains(err.Error(), "cannot be renamed") {
		t.Fatalf("protected id change error = %v", err)
	}
	if _, err := deleteCgenScenePayload(configPath, map[string]any{
		"id":                "Standard_Crawl",
		"expected_revision": stringValue(crawl, "revision"),
	}); err == nil || !strings.Contains(err.Error(), "cannot be deleted") {
		t.Fatalf("protected delete error = %v", err)
	}
}

func TestCgenScenesRejectInvalidInputs(t *testing.T) {
	absFilename := filepath.Join(t.TempDir(), "outside.xml")
	validXML := cgenSceneTestXML("Valid_Scene", "Valid Scene", "")
	tests := []struct {
		name    string
		payload map[string]any
		want    string
	}{
		{name: "missing revision", payload: map[string]any{"xml": validXML}, want: "expected_revision"},
		{name: "non-string revision", payload: map[string]any{"xml": validXML, "expected_revision": 42}, want: "must be a string"},
		{name: "invalid revision", payload: map[string]any{"xml": validXML, "expected_revision": "nope"}, want: "invalid"},
		{name: "oversize", payload: map[string]any{"xml": strings.Repeat("x", cgenSceneMaxXMLBytes+1), "expected_revision": ""}, want: "size limit"},
		{name: "invalid xml", payload: map[string]any{"xml": `<scene`, "expected_revision": ""}, want: "invalid"},
		{name: "wrong root", payload: map[string]any{"xml": `<layout schema_version="1" id="Safe" name="Safe"/>`, "expected_revision": ""}, want: "scene root"},
		{name: "namespace", payload: map[string]any{"xml": `<scene xmlns="urn:test" schema_version="1" id="Safe" name="Safe"/>`, "expected_revision": ""}, want: "unqualified"},
		{name: "directive", payload: map[string]any{"xml": `<!DOCTYPE scene><scene schema_version="1" id="Safe" name="Safe"/>`, "expected_revision": ""}, want: "directives"},
		{name: "duplicate attributes", payload: map[string]any{"xml": `<scene schema_version="1" id="Safe" id="Other" name="Safe"/>`, "expected_revision": ""}, want: "duplicate attributes"},
		{name: "schema", payload: map[string]any{"xml": `<scene schema_version="2" id="Safe" name="Safe"/>`, "expected_revision": ""}, want: "unsupported"},
		{name: "unsafe id", payload: map[string]any{"xml": `<scene schema_version="1" id="../Safe" name="Safe"/>`, "expected_revision": ""}, want: "id is invalid"},
		{name: "protected id case spoof", payload: map[string]any{"xml": `<scene schema_version="1" id="standard_crawl" name="Safe"/>`, "expected_revision": ""}, want: "id is invalid"},
		{name: "control name", payload: map[string]any{"xml": "<scene schema_version=\"1\" id=\"Safe\" name=\"Bad&#xA;Name\"/>", "expected_revision": ""}, want: "name is invalid"},
		{name: "duplicate node ids", payload: map[string]any{"xml": cgenSceneTestXML("Safe", "Safe", `<node id="root"><node id="root"/></node>`), "expected_revision": ""}, want: "duplicate node ids"},
		{name: "multiple root nodes", payload: map[string]any{"xml": cgenSceneTestXML("Safe", "Safe", `<node id="one"/><node id="two"/>`), "expected_revision": ""}, want: "exactly one root node"},
		{name: "asset traversal", payload: map[string]any{"xml": cgenSceneTestXML("Safe", "Safe", `<node id="root"><image asset="../logo.png"/></node>`), "expected_revision": ""}, want: "image asset id"},
		{name: "id mismatch", payload: map[string]any{"id": "Other", "xml": validXML, "expected_revision": ""}, want: "does not match"},
		{name: "traversal filename", payload: map[string]any{"filename": "../outside.xml", "xml": validXML, "expected_revision": ""}, want: "filename is invalid"},
		{name: "absolute filename", payload: map[string]any{"filename": absFilename, "xml": validXML, "expected_revision": ""}, want: "filename is invalid"},
		{name: "reserved filename", payload: map[string]any{"filename": "standby.xml", "xml": validXML, "expected_revision": ""}, want: "reserved"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			_, err := saveCgenScenePayload(configPath, test.payload)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("error = %v, want substring %q", err, test.want)
			}
			if strings.Contains(err.Error(), filepath.Dir(configPath)) || strings.Contains(err.Error(), absFilename) {
				t.Fatalf("error exposed a filesystem path: %v", err)
			}
		})
	}
}

func TestCgenScenesRejectExcessiveNodeDepth(t *testing.T) {
	var body strings.Builder
	for index := 0; index <= cgenSceneMaxNodeDepth; index++ {
		body.WriteString(`<node id="node_`)
		body.WriteString(strconv.Itoa(index))
		body.WriteString(`">`)
	}
	for index := 0; index <= cgenSceneMaxNodeDepth; index++ {
		body.WriteString(`</node>`)
	}
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	_, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Deep", "Deep", body.String()),
		"expected_revision": "",
	})
	if err == nil || !strings.Contains(err.Error(), "node depth limit") {
		t.Fatalf("depth error = %v", err)
	}
}

func TestCgenScenesRejectSymlinkEscape(t *testing.T) {
	base := t.TempDir()
	configPath := filepath.Join(base, "config.yaml")
	scenesDir := filepath.Join(base, "managed", "cgen", "scenes")
	if err := os.MkdirAll(scenesDir, 0o700); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(t.TempDir(), "outside.xml")
	if err := os.WriteFile(outside, []byte(cgenSceneTestXML("Linked", "Linked", "")), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(scenesDir, "Linked.xml")); err != nil {
		t.Skipf("symlinks are unavailable: %v", err)
	}
	if _, err := listCgenScenesPayload(configPath); err == nil || !strings.Contains(err.Error(), "unsafe") {
		t.Fatalf("list symlink error = %v", err)
	}
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Linked", "Linked", ""),
		"expected_revision": "",
	}); err == nil {
		t.Fatal("save accepted a symlink scene target")
	}
	raw, err := os.ReadFile(outside)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != cgenSceneTestXML("Linked", "Linked", "") {
		t.Fatal("outside symlink target was modified")
	}
}

func TestCgenScenesRejectEscapingSceneDirectorySymlink(t *testing.T) {
	base := t.TempDir()
	configPath := filepath.Join(base, "config.yaml")
	cgenDir := filepath.Join(base, "managed", "cgen")
	if err := os.MkdirAll(cgenDir, 0o700); err != nil {
		t.Fatal(err)
	}
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(cgenDir, "scenes")); err != nil {
		t.Skipf("symlinks are unavailable: %v", err)
	}
	if _, err := listCgenScenesPayload(configPath); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("list escaping directory error = %v", err)
	}
	if _, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Outside", "Outside", ""),
		"expected_revision": "",
	}); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("save escaping directory error = %v", err)
	}
}

func TestCgenScenesRevisionCheckIsSerialized(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	created, err := saveCgenScenePayload(configPath, map[string]any{
		"xml":               cgenSceneTestXML("Concurrent", "Concurrent", ""),
		"expected_revision": "",
	})
	if err != nil {
		t.Fatal(err)
	}
	revision := stringValue(created, "revision")
	start := make(chan struct{})
	errorsByWriter := make(chan error, 2)
	var wait sync.WaitGroup
	for _, content := range []string{"first", "second"} {
		content := content
		wait.Add(1)
		go func() {
			defer wait.Done()
			<-start
			_, err := saveCgenScenePayload(configPath, map[string]any{
				"xml":               cgenSceneTestXML("Concurrent", "Concurrent", "<text>"+content+"</text>"),
				"expected_revision": revision,
			})
			errorsByWriter <- err
		}()
	}
	close(start)
	wait.Wait()
	close(errorsByWriter)
	successes := 0
	conflicts := 0
	for err := range errorsByWriter {
		if err == nil {
			successes++
		} else if strings.Contains(err.Error(), "revision conflict") {
			conflicts++
		}
	}
	if successes != 1 || conflicts != 1 {
		t.Fatalf("successes = %d, conflicts = %d", successes, conflicts)
	}
}

func TestCgenSceneCommandsUseWebSocketCommandPath(t *testing.T) {
	t.Setenv("HAZE_HOST_BRIDGE_ADDR", "")
	session := &wsSession{configPath: filepath.Join(t.TempDir(), "config.yaml")}
	createdAny, err := session.handleCommand("cgen.scenes.save", map[string]any{
		"xml":               cgenSceneTestXML("Command_Scene", "Command Scene", ""),
		"expected_revision": "",
	})
	if err != nil {
		t.Fatalf("save command: %v", err)
	}
	created, ok := createdAny.(map[string]any)
	if !ok {
		t.Fatalf("save result type = %T", createdAny)
	}
	if _, err := session.handleCommand("cgen.scenes.list", map[string]any{}); err != nil {
		t.Fatalf("list command: %v", err)
	}
	if _, err := session.handleCommand("cgen.scenes.get", map[string]any{"id": "Command_Scene"}); err != nil {
		t.Fatalf("get command: %v", err)
	}
	if _, err := session.handleCommand("cgen.scenes.delete", map[string]any{
		"id":                "Command_Scene",
		"expected_revision": stringValue(created, "revision"),
	}); err != nil {
		t.Fatalf("delete command: %v", err)
	}
}

func cgenSceneTestXML(id string, name string, body string) string {
	trimmed := strings.TrimSpace(body)
	if !strings.HasPrefix(trimmed, "<node") {
		body = `<node id="root">` + body + `</node>`
	}
	return `<scene schema_version="1" id="` + id + `" name="` + name + `">` + body + `</scene>`
}

func assertNoCgenSceneTemps(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".scene-") && strings.HasSuffix(entry.Name(), ".tmp") {
			t.Fatalf("temporary scene file was not cleaned up: %s", entry.Name())
		}
	}
}
