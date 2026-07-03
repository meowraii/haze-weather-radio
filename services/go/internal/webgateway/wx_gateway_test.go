package webgateway

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestLoadWxOnDemandPackageIDsFiltersRadioOnlyPackages(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "config.yaml"), "packages_file: managed/configs/packages.xml\n")
	mustWrite(t, filepath.Join(dir, "managed", "configs", "packages.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<Packages>
  <package id="station_id" enabled="true"/>
  <package id="date_time" enabled="true"/>
  <package id="current_conditions" enabled="true"/>
  <package id="user_bulletin" enabled="true"/>
  <package id="disabled_item" enabled="false"/>
</Packages>
`)

	got, err := loadWxOnDemandPackageIDs(filepath.Join(dir, "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"date_time", "current_conditions"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("packages = %#v, want %#v", got, want)
	}
}

func TestParseWxGeneratePayloadAllowsFeedlessRequest(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, filepath.Join(dir, "config.yaml"), "packages_file: managed/configs/packages.xml\n")
	mustWrite(t, filepath.Join(dir, "managed", "configs", "packages.xml"), `<?xml version="1.0" encoding="UTF-8"?>
<Packages>
  <package id="forecast" enabled="true"/>
</Packages>
`)

	payload, err := parseWxGeneratePayload(filepath.Join(dir, "config.yaml"), map[string]any{
		"locations": "06099",
		"packages":  []any{"forecast"},
		"format":    "json",
	})
	if err != nil {
		t.Fatal(err)
	}
	if payload.FeedID != "" {
		t.Fatalf("FeedID = %q", payload.FeedID)
	}
}
