package tts

import (
	"encoding/json"
	"errors"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"
)

var (
	isoTimestampPattern = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})\b`)
	acronymPattern      = regexp.MustCompile(`\b[A-Z]{2,8}\b`)
)

var acronymAlwaysSpell = map[string]struct{}{
	"AAFC":  {},
	"ACME":  {},
	"API":   {},
	"CAP":   {},
	"CLC":   {},
	"CPU":   {},
	"DTMF":  {},
	"GPU":   {},
	"HTTP":  {},
	"HTTPS": {},
	"IPAWS": {},
	"IVR":   {},
	"NAADS": {},
	"RTP":   {},
	"SAME":  {},
	"SAPI":  {},
	"SIP":   {},
	"TCP":   {},
	"TTS":   {},
	"UDP":   {},
	"XML":   {},
}

var acronymWordSkips = map[string]struct{}{
	"AM":           {},
	"PM":           {},
	"ADVISORY":     {},
	"ALERT":        {},
	"ALL":          {},
	"AND":          {},
	"AREA":         {},
	"AREAS":        {},
	"AS":           {},
	"AT":           {},
	"BY":           {},
	"CITY":         {},
	"COLD":         {},
	"CONTINUED":    {},
	"COUNTY":       {},
	"DAY":          {},
	"EAST":         {},
	"ENDED":        {},
	"FIRE":         {},
	"FOG":          {},
	"FOR":          {},
	"FORT":         {},
	"FROM":         {},
	"HAIL":         {},
	"HAS":          {},
	"HEAT":         {},
	"HIGH":         {},
	"ICE":          {},
	"IN":           {},
	"IS":           {},
	"LAKE":         {},
	"LOW":          {},
	"NEAR":         {},
	"NO":           {},
	"OF":           {},
	"ON":           {},
	"OR":           {},
	"PARK":         {},
	"RAIN":         {},
	"REGION":       {},
	"RIVER":        {},
	"SEVERE":       {},
	"SNOW":         {},
	"SPECIAL":      {},
	"STORM":        {},
	"THE":          {},
	"THIS":         {},
	"THUNDERSTORM": {},
	"TIME":         {},
	"TO":           {},
	"TORNADO":      {},
	"TOWN":         {},
	"UPDATE":       {},
	"UPDATED":      {},
	"WARNINGS":     {},
	"WARNING":      {},
	"WAS":          {},
	"WATCH":        {},
	"WATCHES":      {},
	"WEATHER":      {},
	"WEST":         {},
	"WIND":         {},
	"WINTER":       {},
	"WITH":         {},
}

// Dictionary normalizes text before it is handed to a speech engine.
type Dictionary struct {
	entries []dictionaryEntry
}

type dictionaryEntry struct {
	Pattern     string
	Replacement string
}

// LoadDictionary loads managed/dictionary.json for the requested language.
func LoadDictionary(path string, language string) (Dictionary, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return Dictionary{}, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Dictionary{}, nil
		}
		return Dictionary{}, err
	}
	var parsed map[string]map[string]string
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return Dictionary{}, err
	}

	merged := map[string]string{}
	for _, key := range dictionaryLayers(parsed, language) {
		for from, to := range parsed[key] {
			from = strings.TrimSpace(from)
			if from == "" {
				continue
			}
			merged[from] = strings.TrimSpace(to)
		}
	}

	entries := make([]dictionaryEntry, 0, len(merged))
	for from, to := range merged {
		if to == "" {
			continue
		}
		entries = append(entries, dictionaryEntry{Pattern: from, Replacement: to})
	}
	sort.SliceStable(entries, func(i, j int) bool {
		if len(entries[i].Pattern) == len(entries[j].Pattern) {
			return entries[i].Pattern < entries[j].Pattern
		}
		return len(entries[i].Pattern) > len(entries[j].Pattern)
	})
	return Dictionary{entries: entries}, nil
}

// NormalizeText expands timestamps and configured pronunciation entries.
func NormalizeText(text string, dictionary Dictionary, timezone string) string {
	text = expandISOTimestamps(text, timezone)
	for _, entry := range dictionary.entries {
		text = replaceDictionaryEntry(text, entry)
	}
	text = expandAcronyms(text)
	return normalizeWhitespace(text)
}

func dictionaryLayers(parsed map[string]map[string]string, language string) []string {
	lang := NormalizeLanguage(language)
	family := lang
	if idx := strings.Index(family, "-"); idx > 0 {
		family = family[:idx]
	}
	candidates := []string{"*", family + "-*", family, lang}
	seen := map[string]bool{}
	out := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		if candidate == "-" || seen[candidate] {
			continue
		}
		if _, ok := parsed[candidate]; ok {
			out = append(out, candidate)
			seen[candidate] = true
		}
	}
	return out
}

func expandISOTimestamps(text string, timezone string) string {
	if strings.TrimSpace(text) == "" {
		return text
	}
	loc := time.Local
	if zone := strings.TrimSpace(timezone); zone != "" && !strings.EqualFold(zone, "local") {
		if loaded, err := time.LoadLocation(zone); err == nil {
			loc = loaded
		}
	}
	return isoTimestampPattern.ReplaceAllStringFunc(text, func(raw string) string {
		parsed, err := parseTimestamp(raw)
		if err != nil {
			return raw
		}
		return spokenClockTime(parsed.In(loc))
	})
}

func parseTimestamp(raw string) (time.Time, error) {
	if parsed, err := time.Parse(time.RFC3339Nano, raw); err == nil {
		return parsed, nil
	}
	for _, layout := range []string{
		"2006-01-02T15:04:05-0700",
		"2006-01-02T15:04:05.999999999-0700",
	} {
		if parsed, err := time.Parse(layout, raw); err == nil {
			return parsed, nil
		}
	}
	return time.Time{}, errors.New("unsupported timestamp")
}

func expandAcronyms(text string) string {
	if strings.TrimSpace(text) == "" {
		return text
	}
	return acronymPattern.ReplaceAllStringFunc(text, func(token string) string {
		if !shouldSpellAcronym(token) {
			return token
		}
		return spellAcronym(token)
	})
}

func shouldSpellAcronym(token string) bool {
	if _, ok := acronymWordSkips[token]; ok {
		return false
	}
	if _, ok := acronymAlwaysSpell[token]; ok {
		return true
	}
	if len(token) <= 4 {
		return true
	}
	return !containsAcronymVowel(token)
}

func spellAcronym(token string) string {
	var builder strings.Builder
	for i, ch := range token {
		if i > 0 {
			builder.WriteByte(' ')
		}
		builder.WriteRune(ch)
	}
	return builder.String()
}

func containsAcronymVowel(token string) bool {
	for _, ch := range token {
		switch ch {
		case 'A', 'E', 'I', 'O', 'U', 'Y':
			return true
		}
	}
	return false
}

func spokenClockTime(value time.Time) string {
	format := "3:04 PM"
	if value.Minute() == 0 {
		format = "3 PM"
	}
	zone := timezoneName(value.Format("MST"))
	if zone == "" {
		return value.Format(format)
	}
	return strings.TrimSpace(value.Format(format) + " " + zone)
}

func timezoneName(abbrev string) string {
	switch abbrev {
	case "CST":
		return "Central Standard Time"
	case "CDT":
		return "Central Daylight Time"
	case "MST":
		return "Mountain Standard Time"
	case "MDT":
		return "Mountain Daylight Time"
	case "EST":
		return "Eastern Standard Time"
	case "EDT":
		return "Eastern Daylight Time"
	case "PST":
		return "Pacific Standard Time"
	case "PDT":
		return "Pacific Daylight Time"
	case "AST":
		return "Atlantic Standard Time"
	case "ADT":
		return "Atlantic Daylight Time"
	case "NST":
		return "Newfoundland Standard Time"
	case "NDT":
		return "Newfoundland Daylight Time"
	case "UTC":
		return "Coordinated Universal Time"
	default:
		return abbrev
	}
}

func replaceDictionaryEntry(text string, entry dictionaryEntry) string {
	if text == "" || entry.Pattern == "" || !strings.Contains(text, entry.Pattern) {
		return text
	}
	needLeftBoundary := edgeIsWord(entry.Pattern, true)
	needRightBoundary := edgeIsWord(entry.Pattern, false)
	var builder strings.Builder
	searchFrom := 0
	for {
		index := strings.Index(text[searchFrom:], entry.Pattern)
		if index < 0 {
			break
		}
		index += searchFrom
		end := index + len(entry.Pattern)
		if hasDictionaryBoundaries(text, index, end, needLeftBoundary, needRightBoundary) {
			builder.WriteString(text[searchFrom:index])
			builder.WriteString(entry.Replacement)
			searchFrom = end
			continue
		}
		builder.WriteString(text[searchFrom : index+1])
		searchFrom = index + 1
	}
	if searchFrom == 0 {
		return text
	}
	builder.WriteString(text[searchFrom:])
	return builder.String()
}

func edgeIsWord(value string, first bool) bool {
	runes := []rune(value)
	if len(runes) == 0 {
		return false
	}
	ch := runes[len(runes)-1]
	if first {
		ch = runes[0]
	}
	return isWordRune(ch)
}

func hasDictionaryBoundaries(text string, start int, end int, left bool, right bool) bool {
	if left && start > 0 {
		if isWordRune(rune(text[start-1])) {
			return false
		}
	}
	if right && end < len(text) {
		if isWordRune(rune(text[end])) {
			return false
		}
	}
	return true
}

func isWordRune(ch rune) bool {
	return unicode.IsLetter(ch) || unicode.IsDigit(ch) || ch == '_'
}

func normalizeWhitespace(text string) string {
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	for i := range lines {
		lines[i] = strings.Join(strings.Fields(lines[i]), " ")
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}
