package ivr

import (
	"fmt"
	"sort"
	"strings"
)

// ResolvedLocation is the canonical IVR lookup result for caller-entered digits.
type ResolvedLocation struct {
	Input     string `json:"input"`
	Code      string `json:"code"`
	Source    string `json:"source"`
	Name      string `json:"name"`
	Province  string `json:"province,omitempty"`
	FeedID    string `json:"feed_id"`
	Language  string `json:"language"`
	Timezone  string `json:"timezone,omitempty"`
	Forecast  string `json:"forecast_id,omitempty"`
	StationID string `json:"station_id,omitempty"`
}

type Resolver struct {
	cfg loadedConfig
}

func NewResolver(cfg loadedConfig) *Resolver {
	return &Resolver{cfg: cfg}
}

func (r *Resolver) Resolve(input string) (ResolvedLocation, error) {
	code := normalizeCallerCode(input)
	if code == "" {
		return ResolvedLocation{}, fmt.Errorf("enter a location code followed by pound")
	}
	candidates := r.candidates(code)
	if len(candidates) == 0 {
		return ResolvedLocation{}, fmt.Errorf("no weather location matched %q", input)
	}
	sort.SliceStable(candidates, func(left, right int) bool {
		if candidates[left].FeedID != "" && candidates[right].FeedID == "" {
			return true
		}
		if candidates[left].FeedID == "" && candidates[right].FeedID != "" {
			return false
		}
		return candidates[left].Name < candidates[right].Name
	})
	if candidates[0].FeedID == "" {
		return ResolvedLocation{}, fmt.Errorf("%s is known, but no Haze feed currently covers it", candidates[0].Name)
	}
	return candidates[0], nil
}

func (r *Resolver) candidates(code string) []ResolvedLocation {
	seen := map[string]struct{}{}
	var out []ResolvedLocation
	add := func(record locationRecord) {
		if record.Code == "" {
			return
		}
		record = r.attachFeed(record)
		key := strings.Join([]string{record.Code, record.Source, record.FeedID, record.Forecast, record.StationID}, "|")
		if _, ok := seen[key]; ok {
			return
		}
		seen[key] = struct{}{}
		out = append(out, ResolvedLocation{
			Input:     code,
			Code:      record.Code,
			Source:    record.Source,
			Name:      fallbackText(record.Name, record.Code),
			Province:  record.Province,
			FeedID:    record.FeedID,
			Language:  r.feedLanguage(record.FeedID),
			Timezone:  r.feedTimezone(record.FeedID),
			Forecast:  record.Forecast,
			StationID: record.StationID,
		})
	}

	if record, ok := r.helloWeather(code); ok {
		add(record)
	}
	if record, ok := r.cfg.ForecastLocations[leftPad6(code)]; ok {
		add(record)
	}
	if record, ok := r.cfg.CLCs[leftPad6(code)]; ok {
		add(record)
	}
	if record, ok := r.cfg.Geocodes[code]; ok {
		add(record)
	}
	if record, ok := r.cfg.NWS[strings.ToUpper(code)]; ok {
		add(record)
	}
	if record, ok := r.cfg.NWS[leftPad6(code)]; ok {
		add(record)
	}
	for _, feed := range r.cfg.Feeds {
		if !xmlBool(feed.EnabledRaw, true) {
			continue
		}
		for _, loc := range feed.Locations.ObservationLocations.Locations {
			if strings.EqualFold(loc.ID, code) || strings.EqualFold(t9(loc.ID), code) {
				add(locationRecord{Code: loc.ID, Source: "station", Name: fallbackText(loc.NameOverride, loc.ID), FeedID: feed.ID, StationID: loc.ID})
			}
		}
	}
	return out
}

func (r *Resolver) helloWeather(code string) (locationRecord, bool) {
	if len(code) != 5 && len(code) != 6 {
		return locationRecord{}, false
	}
	prefix := code[:3]
	city := strings.TrimLeft(code[3:], "0")
	if city == "" {
		city = "0"
	}
	province := map[string]string{
		"010": "NL", "020": "PE", "030": "NS", "040": "NB", "050": "QC",
		"060": "SK", "061": "AB", "062": "MB", "063": "ON", "064": "BC",
		"070": "YT", "080": "NT", "090": "NU",
	}[prefix]
	if province == "" {
		return locationRecord{}, false
	}
	forecast := strings.ToLower(province) + "-" + city
	name := forecast
	for _, record := range r.cfg.ForecastLocations {
		if strings.EqualFold(record.Forecast, forecast) {
			name = record.Name
			break
		}
	}
	return locationRecord{Code: code, Source: "hello_weather", Name: name, Province: province, Forecast: forecast}, true
}

func (r *Resolver) attachFeed(record locationRecord) locationRecord {
	if record.FeedID != "" {
		return record
	}
	for _, feed := range r.cfg.Feeds {
		if !xmlBool(feed.EnabledRaw, true) {
			continue
		}
		for _, region := range feed.Locations.Coverage.Regions {
			if regionMatchesRecord(region, record) {
				record.FeedID = feed.ID
				return record
			}
			for _, subregion := range region.Subregions {
				if sameCode(subregion.ID, record.Code) {
					record.FeedID = feed.ID
					return record
				}
			}
		}
		for _, loc := range feed.Locations.ObservationLocations.Locations {
			if sameCode(loc.ID, record.Forecast) || sameCode(loc.ID, record.StationID) || sameCode(loc.ID, record.Code) {
				record.FeedID = feed.ID
				if record.StationID == "" {
					record.StationID = loc.ID
				}
				return record
			}
		}
	}
	return record
}

func regionMatchesRecord(region coverageRegionXML, record locationRecord) bool {
	return sameCode(region.ID, record.Code) ||
		sameCode(region.DeriveForecast, record.Forecast) ||
		sameCode(region.DeriveForecast, record.Code)
}

func sameCode(left string, right string) bool {
	return strings.EqualFold(strings.TrimSpace(left), strings.TrimSpace(right)) && strings.TrimSpace(left) != ""
}

func (r *Resolver) feedLanguage(feedID string) string {
	for _, feed := range r.cfg.Feeds {
		if !strings.EqualFold(feed.ID, feedID) {
			continue
		}
		for _, lang := range feed.Languages.Langs {
			if strings.TrimSpace(lang.Code) != "" {
				return strings.TrimSpace(lang.Code)
			}
		}
	}
	return r.cfg.IVR.DefaultLanguage
}

func (r *Resolver) feedTimezone(feedID string) string {
	for _, feed := range r.cfg.Feeds {
		if strings.EqualFold(feed.ID, feedID) {
			return strings.TrimSpace(feed.Timezone)
		}
	}
	return ""
}

func normalizeCallerCode(input string) string {
	input = strings.TrimSpace(input)
	var builder strings.Builder
	for _, char := range input {
		if char >= '0' && char <= '9' {
			builder.WriteRune(char)
			continue
		}
		if char >= 'A' && char <= 'Z' || char >= 'a' && char <= 'z' {
			builder.WriteRune(char)
		}
	}
	return builder.String()
}

func leftPad6(code string) string {
	code = strings.TrimSpace(code)
	if len(code) >= 6 {
		return code
	}
	return strings.Repeat("0", 6-len(code)) + code
}

func t9(value string) string {
	var builder strings.Builder
	for _, char := range strings.ToUpper(value) {
		switch {
		case char >= '0' && char <= '9':
			builder.WriteRune(char)
		case strings.ContainsRune("ABC", char):
			builder.WriteByte('2')
		case strings.ContainsRune("DEF", char):
			builder.WriteByte('3')
		case strings.ContainsRune("GHI", char):
			builder.WriteByte('4')
		case strings.ContainsRune("JKL", char):
			builder.WriteByte('5')
		case strings.ContainsRune("MNO", char):
			builder.WriteByte('6')
		case strings.ContainsRune("PQRS", char):
			builder.WriteByte('7')
		case strings.ContainsRune("TUV", char):
			builder.WriteByte('8')
		case strings.ContainsRune("WXYZ", char):
			builder.WriteByte('9')
		}
	}
	return builder.String()
}
