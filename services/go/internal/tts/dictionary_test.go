package tts

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNormalizeTextExpandsISOTimestampInRequestedTimezone(t *testing.T) {
	got := NormalizeText(
		"Issued 2026-06-16T05:08:16.568075Z.",
		Dictionary{},
		"America/Regina",
	)

	want := "Issued 11:08 PM Central Standard Time."
	if got != want {
		t.Fatalf("normalized = %q, want %q", got, want)
	}
}

func TestLoadDictionaryAppliesLanguageWildcardAndWordBoundaries(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dictionary.json")
	raw := `{
  "en-*": {
    "XLF322": "X L F three twenty two",
    "N": "north",
    "km/h": "kilometres per hour"
  }
}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	dictionary, err := LoadDictionary(path, "en-CA")
	if err != nil {
		t.Fatal(err)
	}

	got := NormalizeText("XLF322 winds N at 20 km/h near North Battleford.", dictionary, "UTC")
	want := "X L F three twenty two winds north at 20 kilometres per hour near North Battleford."
	if got != want {
		t.Fatalf("normalized = %q, want %q", got, want)
	}
}

func TestNormalizeTextSpellsUnknownAcronyms(t *testing.T) {
	got := NormalizeText("Conditions at Scott AAFC with SAME CAP data.", Dictionary{}, "UTC")
	want := "Conditions at Scott A A F C with S A M E C A P data."
	if got != want {
		t.Fatalf("normalized = %q, want %q", got, want)
	}
}

func TestNormalizeTextDoesNotSpellOrdinaryUppercaseWords(t *testing.T) {
	got := NormalizeText("SEVERE THUNDERSTORM WATCH for CITY OF SASKATOON at 3 PM.", Dictionary{}, "UTC")
	want := "SEVERE THUNDERSTORM WATCH for CITY OF SASKATOON at 3 PM."
	if got != want {
		t.Fatalf("normalized = %q, want %q", got, want)
	}
}

func TestNormalizeTextLetsDictionaryOverrideAcronyms(t *testing.T) {
	dictionary := Dictionary{entries: []dictionaryEntry{
		{Pattern: "AAFC", Replacement: "Agriculture and Agri-Food Canada"},
	}}

	got := NormalizeText("Conditions at Scott AAFC.", dictionary, "UTC")
	want := "Conditions at Scott Agriculture and Agri-Food Canada."
	if got != want {
		t.Fatalf("normalized = %q, want %q", got, want)
	}
}
