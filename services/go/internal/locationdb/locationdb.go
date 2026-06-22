package locationdb

import (
	"database/sql"
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

const DefaultRelPath = "managed/alert_location_map.sqlite"

type Place struct {
	Source  string
	Code    string
	Name    string
	NameFR  string
	Region  string
	Country string
	Kind    string
	Lat     float64
	Lon     float64
	Attrs   map[string]any
}

type Link struct {
	Type       string
	FromSource string
	FromCode   string
	ToSource   string
	ToCode     string
	Score      float64
	Confidence string
	Method     string
	Components map[string]any
}

type Snapshot struct {
	Places []Place
	Links  []Link

	bySourceCode map[string]map[string]Place
}

type cachedSnapshot struct {
	mtime time.Time
	size  int64
	snap  Snapshot
}

var cache = struct {
	sync.Mutex
	byPath map[string]cachedSnapshot
}{byPath: map[string]cachedSnapshot{}}

func Path(baseDir string) string {
	baseDir = strings.TrimSpace(baseDir)
	if baseDir == "" {
		baseDir = "."
	}
	return filepath.Clean(filepath.Join(baseDir, DefaultRelPath))
}

func Load(baseDir string) (Snapshot, bool) {
	return LoadPath(Path(baseDir))
}

func LoadPath(path string) (Snapshot, bool) {
	path = filepath.Clean(path)
	stat, err := os.Stat(path)
	if err != nil || stat.IsDir() {
		return Snapshot{}, false
	}
	cache.Lock()
	if cached, ok := cache.byPath[path]; ok && cached.mtime.Equal(stat.ModTime()) && cached.size == stat.Size() {
		cache.Unlock()
		return cached.snap, true
	}
	cache.Unlock()

	snap, err := readSnapshot(path)
	if err != nil {
		return Snapshot{}, false
	}
	cache.Lock()
	cache.byPath[path] = cachedSnapshot{mtime: stat.ModTime(), size: stat.Size(), snap: snap}
	cache.Unlock()
	return snap, true
}

func (s Snapshot) Place(source string, code string) (Place, bool) {
	if len(s.bySourceCode) == 0 {
		s.bySourceCode = indexPlaces(s.Places)
	}
	source = strings.ToLower(strings.TrimSpace(source))
	code = strings.ToUpper(strings.TrimSpace(code))
	if byCode := s.bySourceCode[source]; byCode != nil {
		place, ok := byCode[code]
		return place, ok
	}
	return Place{}, false
}

func (s Snapshot) PlacesBySource(source string) []Place {
	source = strings.ToLower(strings.TrimSpace(source))
	out := []Place{}
	for _, place := range s.Places {
		if strings.EqualFold(place.Source, source) {
			out = append(out, place)
		}
	}
	return out
}

func (s Snapshot) Labels() map[string]string {
	out := map[string]string{}
	for _, source := range []string{"forecast", "clc", "sgc", "nws_same", "nws_zone", "nws_marine_same", "nws_marine_zone"} {
		for _, place := range s.PlacesBySource(source) {
			if place.Code != "" && place.Name != "" {
				out[place.Code] = place.Name
			}
		}
	}
	return out
}

func readSnapshot(path string) (Snapshot, error) {
	db, err := sql.Open("sqlite", sqliteReadOnlyDSN(path))
	if err != nil {
		return Snapshot{}, err
	}
	defer db.Close()
	db.SetMaxOpenConns(1)
	db.SetMaxIdleConns(1)
	_, _ = db.Exec("PRAGMA query_only=ON")

	places, err := readPlaces(db)
	if err != nil {
		return Snapshot{}, err
	}
	links, err := readLinks(db)
	if err != nil {
		return Snapshot{}, err
	}
	return Snapshot{
		Places:       places,
		Links:        links,
		bySourceCode: indexPlaces(places),
	}, nil
}

func sqliteReadOnlyDSN(path string) string {
	values := url.Values{}
	values.Set("mode", "ro")
	values.Add("_pragma", "query_only(ON)")
	values.Add("_pragma", "temp_store(MEMORY)")
	separator := "?"
	if strings.Contains(path, "?") {
		separator = "&"
	}
	return path + separator + values.Encode()
}

func readPlaces(db *sql.DB) ([]Place, error) {
	rows, err := db.Query(`
		SELECT source, code, name, name_fr, region, country, kind,
		       COALESCE(lat, 0), COALESCE(lon, 0), attrs_json
		FROM places
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Place{}
	for rows.Next() {
		var place Place
		var attrsRaw string
		if err := rows.Scan(&place.Source, &place.Code, &place.Name, &place.NameFR, &place.Region, &place.Country, &place.Kind, &place.Lat, &place.Lon, &attrsRaw); err != nil {
			return nil, err
		}
		place.Source = strings.ToLower(strings.TrimSpace(place.Source))
		place.Code = strings.ToUpper(strings.TrimSpace(place.Code))
		place.Attrs = map[string]any{}
		_ = json.Unmarshal([]byte(attrsRaw), &place.Attrs)
		out = append(out, place)
	}
	return out, rows.Err()
}

func readLinks(db *sql.DB) ([]Link, error) {
	rows, err := db.Query(`
		SELECT link_type, from_source, from_code, to_source, to_code,
		       score, confidence, method, components_json
		FROM links
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Link{}
	for rows.Next() {
		var link Link
		var componentsRaw string
		if err := rows.Scan(&link.Type, &link.FromSource, &link.FromCode, &link.ToSource, &link.ToCode, &link.Score, &link.Confidence, &link.Method, &componentsRaw); err != nil {
			return nil, err
		}
		link.Type = strings.ToLower(strings.TrimSpace(link.Type))
		link.FromSource = strings.ToLower(strings.TrimSpace(link.FromSource))
		link.ToSource = strings.ToLower(strings.TrimSpace(link.ToSource))
		link.FromCode = strings.ToUpper(strings.TrimSpace(link.FromCode))
		link.ToCode = strings.ToUpper(strings.TrimSpace(link.ToCode))
		link.Components = map[string]any{}
		_ = json.Unmarshal([]byte(componentsRaw), &link.Components)
		out = append(out, link)
	}
	return out, rows.Err()
}

func indexPlaces(places []Place) map[string]map[string]Place {
	out := map[string]map[string]Place{}
	for _, place := range places {
		source := strings.ToLower(strings.TrimSpace(place.Source))
		code := strings.ToUpper(strings.TrimSpace(place.Code))
		if source == "" || code == "" {
			continue
		}
		if out[source] == nil {
			out[source] = map[string]Place{}
		}
		out[source][code] = place
	}
	return out
}
