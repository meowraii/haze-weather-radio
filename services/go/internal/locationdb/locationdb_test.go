package locationdb

import (
	"database/sql"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

func TestLoadPathReadsPlacesAndLinks(t *testing.T) {
	path := filepath.Join(t.TempDir(), "alert_location_map.sqlite")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec(`
		CREATE TABLE places (
			source TEXT NOT NULL,
			code TEXT NOT NULL,
			name TEXT NOT NULL,
			name_fr TEXT NOT NULL,
			region TEXT NOT NULL,
			country TEXT NOT NULL,
			kind TEXT NOT NULL,
			lat REAL,
			lon REAL,
			attrs_json TEXT NOT NULL,
			PRIMARY KEY (source, code)
		);
		CREATE TABLE links (
			link_type TEXT NOT NULL,
			from_source TEXT NOT NULL,
			from_code TEXT NOT NULL,
			to_source TEXT NOT NULL,
			to_code TEXT NOT NULL,
			score REAL NOT NULL,
			confidence TEXT NOT NULL,
			distance_km REAL,
			method TEXT NOT NULL,
			components_json TEXT NOT NULL
		);
		INSERT INTO places VALUES
			('nws_marine_same', '073531', 'Chesapeake Bay from Pooles Island to Sandy Point MD', '', 'AN', 'US', 'NWS marine SAME zone', 39.1806, -76.3446, '{"zone_ugc":"ANZ531"}'),
			('nws_marine_zone', 'ANZ531', 'Chesapeake Bay from Pooles Island to Sandy Point MD', '', 'AN', 'US', 'NWS marine forecast zone', 39.1806, -76.3446, '{"same":"073531"}');
		INSERT INTO links VALUES
			('nws_marine_same_to_zone', 'nws_marine_same', '073531', 'nws_marine_zone', 'ANZ531', 1.0, 'exact', 0.0, 'fixture', '{}');
	`)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}

	snap, ok := LoadPath(path)
	if !ok {
		t.Fatal("LoadPath returned !ok")
	}
	place, ok := snap.Place("nws_marine_same", "073531")
	if !ok || place.Name != "Chesapeake Bay from Pooles Island to Sandy Point MD" {
		t.Fatalf("place = %#v, ok=%v", place, ok)
	}
	if labels := snap.Labels(); labels["ANZ531"] != "Chesapeake Bay from Pooles Island to Sandy Point MD" {
		t.Fatalf("labels = %#v", labels)
	}
	if len(snap.Links) != 1 || snap.Links[0].FromCode != "073531" || snap.Links[0].ToCode != "ANZ531" {
		t.Fatalf("links = %#v", snap.Links)
	}
}
