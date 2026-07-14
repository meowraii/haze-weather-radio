package productrender

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capmodel"
	"github.com/meowraii/haze-weather-radio/services/go/internal/datastore"
)

func TestRenderableCAPEntryWithoutExpiryAgesOut(t *testing.T) {
	now := time.Date(2026, 7, 4, 18, 0, 0, 0, time.UTC)
	entry := capRegistryEntry{
		ID:        "urn:test",
		UpdatedAt: now.Add(-alertRegistryNoExpiryMaxAge - time.Minute),
		Alert: capmodel.Alert{
			Identifier:  "urn:test",
			MessageType: "Alert",
			Infos: []capmodel.AlertInfo{{
				Event: "Special Weather Statement",
			}},
		},
	}

	if isRenderableCAPEntry(entry, now) {
		t.Fatal("CAP entry without expiry stayed renderable past fallback age")
	}
}

func TestRenderableCAPEntryWithoutExpiryStaysRenderableWithinFallbackAge(t *testing.T) {
	now := time.Date(2026, 7, 4, 18, 0, 0, 0, time.UTC)
	entry := capRegistryEntry{
		ID:        "urn:test",
		UpdatedAt: now.Add(-time.Hour),
		Alert: capmodel.Alert{
			Identifier:  "urn:test",
			MessageType: "Alert",
			Infos: []capmodel.AlertInfo{{
				Event: "Special Weather Statement",
			}},
		},
	}

	if !isRenderableCAPEntry(entry, now) {
		t.Fatal("CAP entry without expiry aged out too early")
	}
}

func TestCAPRegistryTracksLocationDeltasAndTonesOnlyAddedCoverage(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	cfg.Feeds[0].Locations.Coverage.Regions = []coverageRegionXML{
		{
			ID:     "065500",
			Source: "eccc",
			Subregions: []coverageSubregionXML{
				{ID: "065514"},
				{ID: "065522"},
			},
		},
	}
	service := &Service{cfg: cfg}
	now := time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC)

	initial := parseTestAlert(t, testWatchCAP("urn:test:registry:initial", "065514"))
	initialUpdates, err := service.recordCAPAlert(initial, now)
	if err != nil {
		t.Fatal(err)
	}
	if len(initialUpdates) != 1 || !initialUpdates[0].Broadcast || initialUpdates[0].Change != "registered" {
		t.Fatalf("initial update = %#v", initialUpdates)
	}

	expandedRaw := capWithReferences(
		capAsUpdate(testWatchCAP("urn:test:registry:expanded", "065514,065522")),
		"cap-pac@canada.ca,urn:test:registry:initial,2026-06-15T15:58:00-06:00",
	)
	expanded := parseTestAlert(t, expandedRaw)
	expandedUpdates, err := service.recordCAPAlert(expanded, now.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(expandedUpdates) != 1 || !expandedUpdates[0].Broadcast || expandedUpdates[0].Change != "expanded" {
		t.Fatalf("expanded update = %#v", expandedUpdates)
	}
	assertCAPLocations(t, expandedUpdates[0].AddedLocations, "065522")
	assertCAPLocations(t, expandedUpdates[0].RetainedLocations, "065514")
	if locations, _ := expandedUpdates[0].SAME["same_locations"].([]string); !containsString(locations, "065522") || containsString(locations, "065514") {
		t.Fatalf("expanded SAME locations = %#v", expandedUpdates[0].SAME["same_locations"])
	}

	sameRaw := capWithReferences(
		capAsUpdate(testWatchCAP("urn:test:registry:same", "065514,065522")),
		"cap-pac@canada.ca,urn:test:registry:expanded,2026-06-15T15:58:00-06:00",
	)
	same := parseTestAlert(t, sameRaw)
	sameUpdates, err := service.recordCAPAlert(same, now.Add(2*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(sameUpdates) != 1 || sameUpdates[0].Broadcast || sameUpdates[0].Change != "updated" {
		t.Fatalf("same-location update = %#v", sameUpdates)
	}
	assertCAPLocations(t, sameUpdates[0].RetainedLocations, "065514", "065522")

	contractedRaw := capWithReferences(
		capAsUpdate(testWatchCAP("urn:test:registry:contracted", "065522")),
		"cap-pac@canada.ca,urn:test:registry:same,2026-06-15T15:58:00-06:00",
	)
	contracted := parseTestAlert(t, contractedRaw)
	contractedUpdates, err := service.recordCAPAlert(contracted, now.Add(3*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(contractedUpdates) != 1 || contractedUpdates[0].Broadcast || contractedUpdates[0].Change != "contracted" {
		t.Fatalf("contracted update = %#v", contractedUpdates)
	}
	assertCAPLocations(t, contractedUpdates[0].RemovedLocations, "065514")
	assertCAPLocations(t, contractedUpdates[0].RetainedLocations, "065522")

	ancestorRaw := capWithReferences(
		capAsUpdate(testWatchCAP("urn:test:registry:older-ancestor", "065522")),
		"cap-pac@canada.ca,urn:test:registry:initial,2026-06-15T15:58:00-06:00",
	)
	ancestor := parseTestAlert(t, ancestorRaw)
	ancestorUpdates, err := service.recordCAPAlert(ancestor, now.Add(4*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(ancestorUpdates) != 1 || ancestorUpdates[0].Broadcast || ancestorUpdates[0].Change != "updated" {
		t.Fatalf("older-ancestor update = %#v", ancestorUpdates)
	}

	rows, err := cfg.Store.ListCAPArchives(context.Background(), "accepted", time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0].AlertID != ancestor.Identifier {
		t.Fatalf("accepted rows = %#v", rows)
	}
	assertCAPLocations(t, metadataStringList(rows[0].Metadata, "active_locations"), "065522")
	lineage := metadataStringList(rows[0].Metadata, "alert_lineage")
	for _, identifier := range []string{initial.Identifier, expanded.Identifier, same.Identifier, contracted.Identifier, ancestor.Identifier} {
		if !containsString(lineage, identifier) {
			t.Fatalf("lineage %#v is missing %q", lineage, identifier)
		}
	}
}

func TestCAPRegistryPartiallyCancelsOneLocationAndPersistsRemainder(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	cfg.Feeds[0].Locations.Coverage.Regions = []coverageRegionXML{
		{
			ID:     "065500",
			Source: "eccc",
			Subregions: []coverageSubregionXML{
				{ID: "065514"},
				{ID: "065522"},
			},
		},
	}
	service := &Service{cfg: cfg}
	now := time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC)

	initial := parseTestAlert(t, testWatchCAP("urn:test:registry:cancel-base", "065514,065522"))
	if _, err := service.recordCAPAlert(initial, now); err != nil {
		t.Fatal(err)
	}
	initialEntries := loadActiveCAPEntries(cfg.Store, cfg.Feeds[0].ID, now)
	if len(initialEntries) != 1 {
		t.Fatalf("initial registry = %#v", initialEntries)
	}
	assertCAPLocations(t, initialEntries[0].Locations, "065514", "065522")
	if containsString(initialEntries[0].Locations, "065500") {
		t.Fatalf("feed region replaced actual locations: %#v", initialEntries[0].Locations)
	}
	partialRaw := capWithReferences(
		capAsCancellation(testWatchCAP("urn:test:registry:cancel-one", "065514")),
		"cap-pac@canada.ca,urn:test:registry:cancel-base,2026-06-15T15:58:00-06:00",
	)
	partial := parseTestAlert(t, partialRaw)
	updates, err := service.recordCAPAlert(partial, now.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(updates) != 1 || !updates[0].Cancelled || updates[0].Broadcast || updates[0].Change != "partially_cancelled" {
		t.Fatalf("partial cancellation = %#v", updates)
	}
	if len(updates[0].CancelledIDs) != 0 {
		t.Fatalf("partial cancellation removed the whole alert: %#v", updates[0].CancelledIDs)
	}
	assertCAPLocations(t, updates[0].RemovedLocations, "065514")
	if include, _ := updates[0].SAME["include_same"].(bool); include {
		t.Fatalf("cancellation requested SAME: %#v", updates[0].SAME)
	}

	cfg.Store.Close()
	restartedStore, err := datastore.OpenSQLite(context.Background(), datastore.SQLiteConfig{}, dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(restartedStore.Close)
	cfg.Store = restartedStore
	service.cfg.Store = restartedStore
	reloaded := loadActiveCAPEntries(restartedStore, cfg.Feeds[0].ID, now.Add(2*time.Minute))
	if len(reloaded) != 1 {
		t.Fatalf("reloaded registry = %#v", reloaded)
	}
	assertCAPLocations(t, reloaded[0].Locations, "065522")

	product, err := newRenderer(cfg).Render(renderRequest{FeedID: cfg.Feeds[0].ID, PackageID: "alerts"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(product.Text, "Fertile Valley") || !strings.Contains(product.Text, "Rudy") {
		t.Fatalf("partially cancelled product rendered the wrong locations:\n%s", product.Text)
	}

	finalRaw := capWithReferences(
		capAsCancellation(testWatchCAP("urn:test:registry:cancel-last", "065522")),
		"cap-pac@canada.ca,urn:test:registry:cancel-base,2026-06-15T15:58:00-06:00",
	)
	final := parseTestAlert(t, finalRaw)
	finalUpdates, err := service.recordCAPAlert(final, now.Add(3*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(finalUpdates) != 1 || finalUpdates[0].Change != "cancelled" || finalUpdates[0].Broadcast {
		t.Fatalf("final cancellation = %#v", finalUpdates)
	}
	if !containsString(finalUpdates[0].CancelledIDs, initial.Identifier) {
		t.Fatalf("final cancellation IDs = %#v", finalUpdates[0].CancelledIDs)
	}
	if remaining := loadActiveCAPEntries(cfg.Store, cfg.Feeds[0].ID, now.Add(4*time.Minute)); len(remaining) != 0 {
		t.Fatalf("registry remained active after final cancellation: %#v", remaining)
	}
}

func TestCAPRegistrationScansArchiveOnceAcrossFeeds(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	second := cfg.Feeds[0]
	second.ID = "sk-0002"
	cfg.Feeds = append(cfg.Feeds, second)
	counting := &capArchiveCountingStore{Store: cfg.Store, calls: map[string]int{}}
	cfg.Store = counting
	service := &Service{cfg: cfg}
	alert := parseTestAlert(t, testCAP("urn:test:registry:scan-count", "Alert", "active", "2099-06-15T21:30:00-06:00", false))

	if _, err := service.recordCAPAlert(alert, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC)); err != nil {
		t.Fatal(err)
	}
	if counting.calls["accepted"] != 1 || counting.calls["expired"] != 1 {
		t.Fatalf("CAP archive list calls = %#v, want one accepted and one expired scan", counting.calls)
	}
}

func TestCAPRegistryIgnoresExactDuplicate(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	cfg := loadFixtureConfig(t, dir)
	counting := &capArchiveCountingStore{Store: cfg.Store, calls: map[string]int{}}
	cfg.Store = counting
	service := &Service{cfg: cfg}
	alert := parseTestAlert(t, testCAP("urn:test:registry:duplicate", "Alert", "active", "2099-06-15T21:30:00-06:00", false))

	first, err := service.recordCAPAlert(alert, time.Date(2026, 6, 15, 22, 10, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 1 {
		t.Fatalf("first update = %#v", first)
	}
	stored := counting.stores
	second, err := service.recordCAPAlert(alert, time.Date(2026, 6, 15, 22, 11, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(second) != 0 {
		t.Fatalf("exact duplicate produced work: %#v", second)
	}
	if counting.stores != stored {
		t.Fatalf("exact duplicate stored %d additional archive records", counting.stores-stored)
	}
}

func TestCAPRegistryPreservesProviderLocationAndMatchesAliasCancellation(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)
	mustWrite(t, filepath.Join(dir, "managed", "csv", "CAP-CP_Geocodes.csv"), `NAME,NOM,CAPCPGCODE,LAT_DD,LON_DD,CGNDBKEY,PROVINCE_C,COUNTRY_C
Wabamun,,4811045,53.56186389990,-114.47830913600,,AB,CA
`)
	mustWrite(t, filepath.Join(dir, "managed", "csv", "CLC_Base_Zone.csv"), `CLC,UUID,English,French,X1,X2,LAT_DD,LON_DD,X3,X4,X5,PROVINCE_C,COUNTRY_C
076232,fixture,Parkland Co. near Wabamun Carvel and Keephills,Parkland, , ,53.48353858000,-114.38112697000, , , ,AB,CA
`)
	cfg := loadFixtureConfig(t, dir)
	cfg.Feeds[0].Locations.Coverage.Regions = []coverageRegionXML{{ID: "076232", Source: "eccc"}}
	service := &Service{cfg: cfg}
	now := time.Date(2026, 6, 22, 16, 0, 0, 0, time.UTC)

	active := parseTestAlert(t, testWatchCAP("urn:test:registry:provider-location", "4811045"))
	if _, err := service.recordCAPAlert(active, now); err != nil {
		t.Fatal(err)
	}
	entries := loadActiveCAPEntries(cfg.Store, cfg.Feeds[0].ID, now)
	if len(entries) != 1 {
		t.Fatalf("provider-location registry = %#v", entries)
	}
	assertCAPLocations(t, entries[0].Locations, "4811045")
	if containsString(entries[0].Locations, "076232") {
		t.Fatalf("configured feed region replaced provider location: %#v", entries[0].Locations)
	}

	cancelRaw := capWithReferences(
		capAsCancellation(testWatchCAP("urn:test:registry:provider-location-cancel", "076232")),
		"cap-pac@canada.ca,urn:test:registry:provider-location,2026-06-22T10:00:00-06:00",
	)
	cancel := parseTestAlert(t, cancelRaw)
	updates, err := service.recordCAPAlert(cancel, now.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if len(updates) != 1 || updates[0].Change != "cancelled" || updates[0].Broadcast {
		t.Fatalf("provider-location alias cancellation = %#v", updates)
	}
	assertCAPLocations(t, updates[0].RemovedLocations, "4811045")
	if remaining := loadActiveCAPEntries(cfg.Store, cfg.Feeds[0].ID, now.Add(2*time.Minute)); len(remaining) != 0 {
		t.Fatalf("provider-location alias cancellation left active entries: %#v", remaining)
	}
}

type capArchiveCountingStore struct {
	datastore.Store
	calls  map[string]int
	stores int
}

func (s *capArchiveCountingStore) StoreCAPArchive(ctx context.Context, record datastore.CAPArchiveRecord) error {
	s.stores++
	return s.Store.StoreCAPArchive(ctx, record)
}

func (s *capArchiveCountingStore) ListCAPArchives(ctx context.Context, bucket string, since time.Time) ([]datastore.StoredCAPArchive, error) {
	s.calls[bucket]++
	return s.Store.ListCAPArchives(ctx, bucket, since)
}

func capAsCancellation(raw string) string {
	raw = strings.Replace(raw, "<msgType>Alert</msgType>", "<msgType>Cancel</msgType>", 1)
	raw = strings.Replace(raw, "<value>active</value>", "<value>ended</value>", 1)
	raw = strings.Replace(raw, "<responseType>Monitor</responseType>", "<responseType>AllClear</responseType>", 1)
	return strings.Replace(raw, " - in effect</headline>", " - ended</headline>", 1)
}

func assertCAPLocations(t *testing.T, got []string, want ...string) {
	t.Helper()
	got = sortedUniqueCAPLocations(got)
	want = sortedUniqueCAPLocations(want)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("locations = %#v, want %#v", got, want)
	}
}
