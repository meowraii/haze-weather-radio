package productrender

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type liveObservationFile struct {
	Source     string `json:"source"`
	ObservedAt string `json:"observed_at"`
	Station    any    `json:"station"`
	StationID  string `json:"station_id"`
	Properties struct {
		Temp      *float64 `json:"temp"`
		Condition any      `json:"condition"`
		Wind      struct {
			Speed     *float64 `json:"speed"`
			Direction string   `json:"direction"`
			Gust      *float64 `json:"gust"`
		} `json:"wind"`
		Humidity   *float64 `json:"humidity"`
		Dewpoint   *float64 `json:"dewpoint"`
		Visibility *float64 `json:"visibility"`
		Pressure   struct {
			Value    *float64 `json:"value"`
			Tendency any      `json:"tendency"`
		} `json:"pressure"`
	} `json:"properties"`
}

type liveForecastFile struct {
	Source         string `json:"source"`
	ForecastRegion string `json:"forecast_region"`
	Name           any    `json:"name"`
	IssuedAt       string `json:"issued_at"`
	UpdatedAt      string `json:"updated_at"`
	PublishedAt    string `json:"published_at"`
	LastUpdated    string `json:"last_updated"`
	Forecast       []struct {
		Period      any `json:"period"`
		TextSummary any `json:"textSummary"`
	} `json:"forecast"`
}

type liveAirQualityFile struct {
	Source     string `json:"source"`
	Location   any    `json:"location"`
	ObservedAt string `json:"observed_at"`
	AQHI       any    `json:"aqhi"`
	Forecast   struct {
		PublishedAt string `json:"published_at"`
		Periods     []struct {
			Period      any `json:"period"`
			AQHI        any `json:"aqhi"`
			AQHIInSmoke any `json:"aqhi_insmoke"`
		} `json:"periods"`
	} `json:"forecast"`
	SpecialNotes any `json:"special_notes"`
}

type liveClimateFile struct {
	Source       string `json:"source"`
	Name         any    `json:"name"`
	LastUpdated  string `json:"last_updated"`
	Observations struct {
		Station       any      `json:"station"`
		Date          string   `json:"date"`
		High          *float64 `json:"high"`
		Low           *float64 `json:"low"`
		Mean          *float64 `json:"mean"`
		Precipitation *float64 `json:"precipitation"`
		Rain          *float64 `json:"rain"`
		Snowfall      *float64 `json:"snowfall"`
	} `json:"observations"`
	Normals struct {
		TextSummary any `json:"textSummary"`
		Temperature struct {
			High *float64 `json:"high"`
			Low  *float64 `json:"low"`
		} `json:"temperature"`
	} `json:"normals"`
	Records map[string]struct {
		Value *float64 `json:"value"`
		Year  any      `json:"year"`
	} `json:"records"`
	Astronomy struct {
		Sunrise  string `json:"sunrise"`
		Sunset   string `json:"sunset"`
		Timezone string `json:"timezone"`
	} `json:"astronomy"`
}

func (r renderer) loadLiveObservationSnapshot(feed feedXML, snapshot *observationSnapshot) (string, bool) {
	lang := feedLanguage(feed)
	var inputs []string
	var observations []observation
	for _, loc := range feed.Locations.ObservationLocations.Locations {
		obs, path, ok := r.liveObservation(loc, lang)
		if !ok {
			continue
		}
		inputs = append(inputs, path)
		observations = append(observations, obs)
	}
	if len(observations) == 0 {
		return "", false
	}
	*snapshot = observationSnapshot{
		ReportedAt:       newestObservationTime(observations),
		Primary:          observations[0],
		Observations:     observations,
		AreaObservations: observations[1:],
	}
	return strings.Join(inputs, ";"), true
}

func (r renderer) liveObservation(loc locationXML, lang string) (observation, string, bool) {
	var raw liveObservationFile
	if input, ok := r.loadStoreObservation(loc, &raw); ok {
		return observationFromLiveFile(loc, lang, raw), input, true
	}
	return observation{}, "", false
}

func observationFromLiveFile(loc locationXML, lang string, raw liveObservationFile) observation {
	name := fallbackText(loc.NameOverride, localizedString(raw.Station, lang))
	if name == "" {
		name = strings.TrimSpace(loc.ID)
	}
	id := fallbackText(raw.StationID, loc.ID)
	return observation{
		ID:               id,
		Source:           fallbackText(raw.Source, loc.Source),
		LocationName:     name,
		Condition:        localizedString(raw.Properties.Condition, lang),
		TemperatureC:     raw.Properties.Temp,
		DewpointC:        raw.Properties.Dewpoint,
		HumidityPercent:  raw.Properties.Humidity,
		WindDirection:    raw.Properties.Wind.Direction,
		WindSpeedKMH:     raw.Properties.Wind.Speed,
		WindGustKMH:      raw.Properties.Wind.Gust,
		VisibilityKM:     raw.Properties.Visibility,
		PressureKPA:      raw.Properties.Pressure.Value,
		PressureTendency: localizedString(raw.Properties.Pressure.Tendency, lang),
		ObservedAt:       raw.ObservedAt,
	}
}

func (r renderer) loadLiveForecastSnapshot(feed feedXML, snapshot *forecastSnapshot) (string, bool) {
	lang := feedLanguage(feed)
	var paths []string
	var regions []forecastRegion
	var issuedAt string
	for _, region := range feed.Locations.Coverage.Regions {
		forecastID := fallbackText(region.DeriveForecast, region.ID)
		var raw liveForecastFile
		input := ""
		if dbInput, ok := r.loadStoreForecast(region, forecastID, &raw); ok {
			input = dbInput
		}
		if input == "" {
			continue
		}
		periods := make([]forecastPeriod, 0, len(raw.Forecast))
		for _, item := range raw.Forecast {
			name := localizedString(item.Period, lang)
			text := localizedString(item.TextSummary, lang)
			if name == "" && text == "" {
				continue
			}
			periods = append(periods, forecastPeriod{Name: name, Text: text})
		}
		if len(periods) == 0 {
			continue
		}
		name := r.forecastRegionDisplayName(region, raw.Name, lang)
		paths = append(paths, input)
		regions = append(regions, forecastRegion{Name: name, Periods: periods})
		if issuedAt == "" {
			issuedAt = firstNonBlank(raw.IssuedAt, raw.UpdatedAt, raw.PublishedAt, raw.LastUpdated)
		}
	}
	if len(regions) == 0 {
		return "", false
	}
	*snapshot = forecastSnapshot{IssuedAt: issuedAt, Regions: regions}
	return strings.Join(paths, ";"), true
}

func (r renderer) forecastRegionDisplayName(region coverageRegionXML, rawName any, lang string) string {
	langShort := strings.ToLower(strings.TrimSpace(lang))
	if len(langShort) > 2 {
		langShort = langShort[:2]
	}
	if names, ok := r.cfg.ForecastNames[forecastRegionBaseCode(region.ID)]; ok {
		if langShort == "fr" {
			if names.French != "" {
				return pauseForecastRegionName(names.French, langShort)
			}
		}
		if names.English != "" {
			return pauseForecastRegionName(names.English, "en")
		}
	}
	if name := localizedString(rawName, lang); name != "" {
		return pauseForecastRegionName(name, langShort)
	}
	return pauseForecastRegionName(fallbackText(region.Name, region.ID), langShort)
}

func (r renderer) loadLiveAirQualitySnapshot(feed feedXML, snapshot *airQualitySnapshot) (string, bool) {
	lang := feedLanguage(feed)
	for _, loc := range feed.Locations.AirQualityLocations.Locations {
		var raw liveAirQualityFile
		input, ok := r.loadStoreProductPayload("air_quality", loc.Source, loc.ID, &raw)
		if !ok {
			continue
		}
		aqhi, ok := numberFromAny(raw.AQHI)
		value := aqhiValueText(raw.AQHI, lang)
		periods := make([]airQualityPeriodSnapshot, 0, len(raw.Forecast.Periods))
		for _, period := range raw.Forecast.Periods {
			name := localizedString(period.Period, lang)
			periodAQHI := aqhiValueText(period.AQHI, lang)
			if name == "" || periodAQHI == "" {
				continue
			}
			periodRisk := ""
			if numeric, hasNumeric := numberFromAny(period.AQHI); hasNumeric {
				periodRisk = aqhiRisk(numeric)
			}
			periods = append(periods, airQualityPeriodSnapshot{
				Name:        name,
				AQHI:        periodAQHI,
				AQHIInSmoke: aqhiValueText(period.AQHIInSmoke, lang),
				Risk:        periodRisk,
			})
		}
		specialNotes := localizedString(raw.SpecialNotes, lang)
		if !ok && value == "" && len(periods) == 0 && specialNotes == "" {
			continue
		}
		location := fallbackText(loc.NameOverride, localizedString(raw.Location, lang))
		if location == "" {
			location = loc.ID
		}
		*snapshot = airQualitySnapshot{
			ReportedAt:   raw.ObservedAt,
			Location:     location,
			AQHI:         value,
			Risk:         aqhiRisk(aqhi),
			Forecast:     forecastFromAQHIPeriods(periods),
			Periods:      periods,
			SpecialNotes: specialNotes,
		}
		return input, true
	}
	return "", false
}

func (r renderer) loadLiveClimateSnapshot(feed feedXML, snapshot *climateSnapshot) (string, bool) {
	lang := feedLanguage(feed)
	for _, loc := range feed.Locations.ClimateLocations.Locations {
		var raw liveClimateFile
		input, ok := r.loadStoreProductPayload("climate_summary", loc.Source, loc.ID, &raw)
		if !ok {
			continue
		}
		location := fallbackText(loc.NameOverride, localizedString(raw.Name, lang))
		if location == "" {
			location = loc.ID
		}
		lines := climateLines(raw, lang, feed.Timezone)
		if len(lines) == 0 {
			continue
		}
		*snapshot = climateSnapshot{
			ReportedAt: raw.LastUpdated,
			Location:   titleText(location),
			Summary:    lines,
		}
		return input, true
	}
	return "", false
}

func (r renderer) loadLiveBulletinSnapshot(feed feedXML, snapshot *bulletinSnapshot) (string, bool) {
	path := resolvePath(r.cfg.BaseDir, filepath.Join("managed", "userbulletins.json"))
	var raw []struct {
		Enabled   bool                         `json:"enabled"`
		DateStart *string                      `json:"dateStart"`
		DateEnd   *string                      `json:"dateEnd"`
		Text      map[string]map[string]string `json:"text"`
	}
	if !readManagedJSON(path, &raw) {
		return "", false
	}
	now := time.Now()
	lang := feedLanguage(feed)
	lines := []string{}
	for _, bulletin := range raw {
		if !bulletin.Enabled || !bulletinActive(bulletin.DateStart, bulletin.DateEnd, now) {
			continue
		}
		block := localizedBulletinBlock(bulletin.Text, lang)
		for _, key := range []string{"header", "body", "footer"} {
			if text := strings.TrimSpace(block[key]); text != "" {
				lines = append(lines, text)
			}
		}
	}
	if len(lines) == 0 {
		return "", false
	}
	*snapshot = bulletinSnapshot{Title: "User Bulletin", Lines: lines}
	return path, true
}

func (r renderer) loadStoreObservation(loc locationXML, target *liveObservationFile) (string, bool) {
	if r.cfg.Store == nil {
		return "", false
	}
	source := canonicalSource(loc.Source)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	ok, err := r.cfg.Store.ObservationPayload(ctx, source, loc.ID, target)
	if err != nil || !ok {
		return "", false
	}
	return fmt.Sprintf("store:observations.current/%s/%s", source, loc.ID), true
}

func (r renderer) loadStoreForecast(region coverageRegionXML, forecastID string, target *liveForecastFile) (string, bool) {
	if r.cfg.Store == nil {
		return "", false
	}
	source := canonicalSource(region.Source)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	ok, err := r.cfg.Store.ForecastPayload(ctx, source, forecastID, target)
	if err != nil || !ok {
		return "", false
	}
	return fmt.Sprintf("store:forecasts.current/%s/%s", source, forecastID), true
}

func (r renderer) loadStoreProductPayload(kind string, sourceRaw string, id string, target any) (string, bool) {
	if r.cfg.Store == nil {
		return "", false
	}
	source := canonicalSource(sourceRaw)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	ok, err := r.cfg.Store.ProductPayload(ctx, kind, source, id, target)
	if err != nil || !ok {
		return "", false
	}
	return fmt.Sprintf("store:products.payloads/%s/%s/%s", kind, source, id), true
}

func canonicalSource(raw string) string {
	value := strings.ToLower(strings.TrimSpace(raw))
	if value == "" {
		return "eccc"
	}
	if value == "weather.com" || value == "weatherdotcom" {
		return "twc"
	}
	return value
}

func readManagedJSON(path string, target any) bool {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return false
	}
	return json.Unmarshal(raw, target) == nil
}

func newestObservationTime(observations []observation) string {
	var newest time.Time
	var rawNewest string
	for _, obs := range observations {
		raw := strings.TrimSpace(obs.ObservedAt)
		if raw == "" {
			continue
		}
		parsed, err := parseLooseTime(raw)
		if err != nil {
			if rawNewest == "" {
				rawNewest = raw
			}
			continue
		}
		if newest.IsZero() || parsed.After(newest) {
			newest = parsed
			rawNewest = raw
		}
	}
	return rawNewest
}

func parseLooseTime(raw string) (time.Time, error) {
	raw = strings.TrimSpace(raw)
	for _, layout := range []string{time.RFC3339, "2006-01-02T15:04:05-0700", "2006-01-02 15:04:05", "2006-01-02T15:04:05"} {
		if parsed, err := time.Parse(layout, raw); err == nil {
			return parsed, nil
		}
	}
	return time.Time{}, fmt.Errorf("unsupported time %q", raw)
}

func localizedString(value any, lang string) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(typed)
	case map[string]string:
		return localizedStringMap(typed, lang)
	case map[string]any:
		values := make(map[string]string, len(typed))
		for key, item := range typed {
			values[key] = localizedString(item, lang)
		}
		return localizedStringMap(values, lang)
	case float64:
		return rounded(typed)
	case int:
		return strconv.Itoa(typed)
	case json.Number:
		return typed.String()
	default:
		return strings.TrimSpace(fmt.Sprint(value))
	}
}

func localizedStringMap(values map[string]string, lang string) string {
	lang = strings.ToLower(strings.ReplaceAll(strings.TrimSpace(lang), "_", "-"))
	short := lang
	if idx := strings.Index(short, "-"); idx > 0 {
		short = short[:idx]
	}
	for _, key := range []string{lang, short, "en-ca", "en", "fr-ca", "fr"} {
		if text := strings.TrimSpace(values[key]); text != "" {
			return text
		}
		for rawKey, text := range values {
			if strings.EqualFold(rawKey, key) && strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
	}
	for _, text := range values {
		if strings.TrimSpace(text) != "" {
			return strings.TrimSpace(text)
		}
	}
	return ""
}

func numberFromAny(value any) (float64, bool) {
	switch typed := value.(type) {
	case nil:
		return 0, false
	case float64:
		return typed, true
	case int:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	case string:
		parsed, err := strconv.ParseFloat(strings.TrimSpace(typed), 64)
		return parsed, err == nil
	default:
		return 0, false
	}
}

func aqhiValueText(value any, lang string) string {
	if numeric, ok := numberFromAny(value); ok {
		return rounded(numeric)
	}
	return localizedString(value, lang)
}

func aqhiRisk(value float64) string {
	switch {
	case value <= 0:
		return ""
	case value <= 3:
		return "low"
	case value <= 6:
		return "moderate"
	case value <= 10:
		return "high"
	default:
		return "very high"
	}
}

func forecastFromAQHIPeriods(periods []airQualityPeriodSnapshot) string {
	if len(periods) == 0 {
		return ""
	}
	parts := make([]string, 0, len(periods))
	for _, period := range periods {
		if period.Name == "" || period.AQHI == "" {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s, %s", period.Name, period.AQHI))
	}
	return forecastSentence(parts)
}

func forecastSentence(parts []string) string {
	if len(parts) == 0 {
		return ""
	}
	return "The forecast air quality health index is " + strings.Join(parts, "; ") + "."
}

func climateLines(raw liveClimateFile, lang string, timezone string) []string {
	lines := []string{}
	station := localizedString(raw.Observations.Station, lang)
	if raw.Observations.High != nil || raw.Observations.Low != nil || raw.Observations.Precipitation != nil {
		parts := []string{}
		if station != "" {
			parts = append(parts, "At "+titleText(station))
		}
		if raw.Observations.High != nil {
			parts = append(parts, "the high was "+degrees(*raw.Observations.High))
		}
		if raw.Observations.Low != nil {
			parts = append(parts, "the low was "+degrees(*raw.Observations.Low))
		}
		if raw.Observations.Precipitation != nil {
			parts = append(parts, "precipitation was "+oneDecimal(*raw.Observations.Precipitation)+" millimetres")
		}
		lines = append(lines, sentence(strings.Join(parts, ", ")))
	}
	if normal := localizedString(raw.Normals.TextSummary, lang); normal != "" {
		lines = append(lines, "The normal temperatures are "+strings.TrimSuffix(normal, ".")+".")
	} else if raw.Normals.Temperature.Low != nil || raw.Normals.Temperature.High != nil {
		parts := []string{}
		if raw.Normals.Temperature.Low != nil {
			parts = append(parts, "low "+degreesBare(*raw.Normals.Temperature.Low))
		}
		if raw.Normals.Temperature.High != nil {
			parts = append(parts, "high "+degreesBare(*raw.Normals.Temperature.High))
		}
		lines = append(lines, "The normal temperatures are "+strings.Join(parts, ", ")+".")
	}
	for _, key := range []string{"high_max", "low_min", "precipitation", "snowfall"} {
		if text := climateRecordLine(key, raw.Records[key]); text != "" {
			lines = append(lines, text)
		}
	}
	if raw.Astronomy.Sunrise != "" || raw.Astronomy.Sunset != "" {
		parts := []string{}
		if sunrise := clockTime(raw.Astronomy.Sunrise, firstNonBlank(raw.Astronomy.Timezone, timezone)); sunrise != "" {
			parts = append(parts, "sunrise is at "+sunrise)
		}
		if sunset := clockTime(raw.Astronomy.Sunset, firstNonBlank(raw.Astronomy.Timezone, timezone)); sunset != "" {
			parts = append(parts, "sunset is at "+sunset)
		}
		if len(parts) > 0 {
			lines = append(lines, sentence(strings.Join(parts, " and ")))
		}
	}
	return lines
}

func climateRecordLine(key string, record struct {
	Value *float64 `json:"value"`
	Year  any      `json:"year"`
}) string {
	if record.Value == nil {
		return ""
	}
	label := map[string]string{
		"high_max":      "record high",
		"low_min":       "record low",
		"precipitation": "record precipitation",
		"snowfall":      "record snowfall",
	}[key]
	if label == "" {
		return ""
	}
	unit := "degrees"
	if key == "precipitation" || key == "snowfall" {
		unit = "millimetres"
	}
	year := localizedString(record.Year, "en")
	if year != "" {
		return fmt.Sprintf("The %s is %s %s, set in %s.", label, oneDecimal(*record.Value), unit, year)
	}
	return fmt.Sprintf("The %s is %s %s.", label, oneDecimal(*record.Value), unit)
}

func clockTime(raw string, timezone string) string {
	parsed, err := parseLooseTime(raw)
	if err != nil {
		return ""
	}
	if loc, locErr := time.LoadLocation(fallbackText(timezone, "Local")); locErr == nil {
		parsed = parsed.In(loc)
	}
	return parsed.Format("3:04 PM")
}

func bulletinActive(start *string, end *string, now time.Time) bool {
	if start != nil && strings.TrimSpace(*start) != "" {
		if parsed, err := parseLooseTime(*start); err == nil && now.Before(parsed) {
			return false
		}
	}
	if end != nil && strings.TrimSpace(*end) != "" {
		if parsed, err := parseLooseTime(*end); err == nil && now.After(parsed) {
			return false
		}
	}
	return true
}

func localizedBulletinBlock(blocks map[string]map[string]string, lang string) map[string]string {
	if len(blocks) == 0 {
		return nil
	}
	lang = strings.ToLower(strings.ReplaceAll(strings.TrimSpace(lang), "_", "-"))
	short := lang
	if idx := strings.Index(short, "-"); idx > 0 {
		short = short[:idx]
	}
	for _, key := range []string{lang, short, "en-ca", "en", "fr-ca", "fr"} {
		for rawKey, block := range blocks {
			if strings.EqualFold(rawKey, key) {
				return block
			}
		}
	}
	for _, block := range blocks {
		return block
	}
	return nil
}

func cleanPlaintextProduct(raw string) string {
	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	paragraphs := []string{}
	current := []string{}
	flush := func() {
		if len(current) == 0 {
			return
		}
		paragraphs = append(paragraphs, strings.Join(current, " "))
		current = nil
	}
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			flush()
			continue
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, ":") {
			line = strings.Trim(line, ":")
			line = strings.ReplaceAll(line, ":", ". ")
		}
		current = append(current, line)
	}
	flush()
	return strings.Join(paragraphs, "\n\n")
}

func firstNonBlank(values ...string) string {
	for _, value := range values {
		if text := strings.TrimSpace(value); text != "" {
			return text
		}
	}
	return ""
}

func titleText(value string) string {
	value = strings.TrimSpace(value)
	if value == strings.ToUpper(value) {
		words := strings.Fields(strings.ToLower(value))
		for i, word := range words {
			if word == "" {
				continue
			}
			words[i] = strings.ToUpper(word[:1]) + word[1:]
		}
		return strings.Join(words, " ")
	}
	return value
}
