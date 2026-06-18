package webgateway

import (
	"path/filepath"
	"testing"
)

func TestPlaylistStatePayloadReadsRuntimePlaylistDirectory(t *testing.T) {
	dir := t.TempDir()
	writePlaylistGatewayFixture(t, dir)
	mustWrite(t, filepath.Join(dir, "runtime", "playlists", "sk-0001.json"), `{
  "feed_id": "sk-0001",
  "feed_name": "Saskatoon",
  "mode": "running",
  "current": null,
  "next": null,
  "queue": [],
  "updated_at": "2026-06-17T01:02:03Z"
}`)

	payload, err := playlistStatePayload(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}

	feed := playlistFeedState(payload, "sk-0001")
	if feed == nil {
		t.Fatal("missing feed state")
	}
	if mode, _ := feed["mode"].(string); mode != "running" {
		t.Fatalf("mode = %q, want running", mode)
	}
	if updated := playlistFeedUpdatedAt(payload, "sk-0001"); updated != "2026-06-17T01:02:03Z" {
		t.Fatalf("updated_at = %q", updated)
	}
}

func TestPlaylistStatePayloadIgnoresManagedRuntimePlaylistDirectory(t *testing.T) {
	dir := t.TempDir()
	writePlaylistGatewayFixture(t, dir)
	mustWrite(t, filepath.Join(dir, "managed", "runtime", "playlists", "sk-0001.json"), `{
  "feed_id": "sk-0001",
  "mode": "stale",
  "queue": [],
  "updated_at": "2026-06-17T00:00:00Z"
}`)

	payload, err := playlistStatePayload(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}

	feed := playlistFeedState(payload, "sk-0001")
	if feed == nil {
		t.Fatal("missing feed state")
	}
	if mode, _ := feed["mode"].(string); mode != "unknown" {
		t.Fatalf("mode = %q, want default unknown", mode)
	}
	if updated := playlistFeedUpdatedAt(payload, "sk-0001"); updated != "" {
		t.Fatalf("updated_at = %q, want empty", updated)
	}
}

func writePlaylistGatewayFixture(t *testing.T, dir string) {
	t.Helper()
	mustWrite(t, filepath.Join(dir, "config.yaml"), `feeds_file: managed/configs/feeds.xml
outputs_file: managed/configs/output.xml
`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "feeds.xml"), `<feeds>
  <feed id="sk-0001" enabled="true" timezone="America/Regina">
    <transmitter_metadata>
      <transmitter>
        <site_name>Saskatoon</site_name>
      </transmitter>
    </transmitter_metadata>
    <playout routine="true" same="true"/>
  </feed>
</feeds>`)
	mustWrite(t, filepath.Join(dir, "managed", "configs", "output.xml"), `<outputs/>`)
}
