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

	want := "Issued 11 oh 8 P.M. Central Standard Time."
	if got != want {
		t.Fatalf("normalized = %q, want %q", got, want)
	}
}

func TestNormalizeTextExpandsClockTimesForSpeech(t *testing.T) {
	got := NormalizeText(
		"Valid from 9:00 PM through 10:05 PM. Updated at 3 PM and again at 7 a.m.",
		Dictionary{},
		"UTC",
	)
	want := "Valid from 9 o'clock P.M. through 10 oh 5 P.M. Updated at 3 P.M. and again at 7 A.M."
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
    "km/h": "kilometers per hour",
    "kilometres": "kilometers"
  }
}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	dictionary, err := LoadDictionary(path, "en-US")
	if err != nil {
		t.Fatal(err)
	}

	got := NormalizeText("XLF322 winds N at 20 km/h near North Battleford. Visibility 10 kilometres.", dictionary, "UTC")
	want := "X L F three twenty two winds north at 20 kilometers per hour near North Battleford. Visibility 10 kilometers."
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
	want := "SEVERE THUNDERSTORM WATCH for CITY OF SASKATOON at 3 P.M."
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
