package productrender

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

var discussionHeadingPattern = regexp.MustCompile(`(?:^|\s)([A-Z][A-Z0-9 /&'\-]{1,48})\.\.\.`)
var discussionRangeUnitPattern = regexp.MustCompile(`(?i)\b(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*(KM/H|KPH|KT|KTS|MM|CM|M|KM)\b`)
var discussionNumberUnitPattern = regexp.MustCompile(`(?i)\b(\d+(?:\.\d+)?)\s*(KM/H|KPH|KT|KTS|HPA|KPA|MB|MM|CM|M|KM|C)\b`)
var discussionStandaloneReplacements = []struct {
	pattern     *regexp.Regexp
	replacement string
}{
	{regexp.MustCompile(`(?i)\bNIL\s+SIG\s+WX\b`), "no significant weather"},
	{regexp.MustCompile(`(?i)\bSIG\s+WX\b`), "significant weather"},
	{regexp.MustCompile(`(?i)\bNIL\b`), "no significant"},
	{regexp.MustCompile(`(?i)\bQPF\b`), "quantitative precipitation forecast"},
	{regexp.MustCompile(`(?i)\bTSTMS?\b`), "thunderstorms"},
	{regexp.MustCompile(`(?i)\bTSRA\b`), "thunderstorms with rain"},
	{regexp.MustCompile(`(?i)\bSFC\b`), "surface"},
	{regexp.MustCompile(`(?i)\bMSL\b`), "mean sea level"},
	{regexp.MustCompile(`(?i)\bAMSL\b`), "above mean sea level"},
	{regexp.MustCompile(`(?i)\bAGL\b`), "above ground level"},
	{regexp.MustCompile(`(?i)\bPOPS?\b`), "probability of precipitation"},
	{regexp.MustCompile(`(?i)\bNRN\b`), "northern"},
	{regexp.MustCompile(`(?i)\bSRN\b`), "southern"},
	{regexp.MustCompile(`(?i)\bERN\b`), "eastern"},
	{regexp.MustCompile(`(?i)\bWRN\b`), "western"},
	{regexp.MustCompile(`(?i)\bCNTRL\b`), "central"},
	{regexp.MustCompile(`(?i)\bAFTN\b`), "afternoon"},
	{regexp.MustCompile(`(?i)\bEVE\b`), "evening"},
	{regexp.MustCompile(`(?i)\bOVNGT\b`), "overnight"},
	{regexp.MustCompile(`(?i)\bTEMPS?\b`), "temperatures"},
	{regexp.MustCompile(`(?i)\bPRECIP\b`), "precipitation"},
	{regexp.MustCompile(`(?i)\bPCPN\b`), "precipitation"},
	{regexp.MustCompile(`(?i)\bFCST\b`), "forecast"},
	{regexp.MustCompile(`(?i)\bPROG\b`), "prognosis"},
	{regexp.MustCompile(`(?i)\bVFR\b`), "V F R"},
	{regexp.MustCompile(`(?i)\bIFR\b`), "I F R"},
	{regexp.MustCompile(`(?i)\bMVFR\b`), "M V F R"},
	{regexp.MustCompile(`(?i)\bECCC\b`), "Environment and Climate Change Canada"},
	{regexp.MustCompile(`(?i)\bSPC\b`), "Storm Prediction Centre"},
	{regexp.MustCompile(`(?i)\bSK\b`), "Saskatchewan"},
	{regexp.MustCompile(`(?i)\bSASK\b`), "Saskatchewan"},
	{regexp.MustCompile(`(?i)\bAB\b`), "Alberta"},
	{regexp.MustCompile(`(?i)\bALTA\b`), "Alberta"},
	{regexp.MustCompile(`(?i)\bMB\b`), "Manitoba"},
	{regexp.MustCompile(`(?i)\bMAN\b`), "Manitoba"},
}

type renderer struct {
	cfg loadedConfig
}

func newRenderer(cfg loadedConfig) renderer {
	return renderer{cfg: cfg}
}

func (r renderer) RenderWxOnDemand(request wxOnDemandRequest) (Product, error) {
	if strings.TrimSpace(request.FeedID) == "" {
		return Product{}, fmt.Errorf("feed_id is required")
	}
	packages := cleanStringList(request.Packages)
	if len(packages) == 0 {
		packages = []string{"current_conditions", "forecast"}
	}
	var products []Product
	var texts []string
	var titles []string
	var inputs []InputRef
	readerID := strings.TrimSpace(request.ReaderID)
	language := strings.TrimSpace(request.Language)
	for _, packageID := range packages {
		product, err := r.Render(renderRequest{
			RequestID: request.RequestID + "-" + safeID(packageID),
			FeedID:    request.FeedID,
			PackageID: packageID,
			Force:     request.Force,
		})
		if err != nil {
			return Product{}, err
		}
		products = append(products, product)
		if text := strings.TrimSpace(product.Text); text != "" {
			texts = append(texts, text)
		}
		if title := strings.TrimSpace(product.Title); title != "" {
			titles = append(titles, title)
		}
		inputs = append(inputs, product.Inputs...)
		if readerID == "" {
			readerID = product.ReaderID
		}
		if language == "" {
			language = product.Language
		}
	}
	if len(texts) == 0 {
		return Product{}, fmt.Errorf("no weather product text was generated")
	}
	now := time.Now().UTC()
	locationName := strings.TrimSpace(request.LocationName)
	if locationName == "" {
		locationName = request.Code
	}
	title := strings.Join(titles, " / ")
	if title == "" {
		title = "Weather On Demand"
	}
	segments := make([]Segment, 0, len(products))
	for _, product := range products {
		segments = append(segments, Segment{
			Kind:  "product",
			Label: product.PackageID,
			Text:  product.Text,
		})
	}
	return Product{
		ID:          safeID(fmt.Sprintf("wx-%s-%s-%d", request.FeedID, request.Code, now.UnixNano())),
		FeedID:      request.FeedID,
		PackageID:   "wx_on_demand",
		Title:       title,
		Text:        strings.Join(texts, "\n\n"),
		ReaderID:    readerID,
		Language:    fallbackText(language, "en-CA"),
		Segments:    segments,
		Inputs:      inputs,
		GeneratedAt: now,
		Metadata: map[string]string{
			"location_code": request.Code,
			"location_name": locationName,
			"source":        request.Source,
			"packages":      strings.Join(packages, ","),
		},
	}, nil
}

func (r renderer) Render(request renderRequest) (Product, error) {
	feed, ok := r.cfg.feedByID(request.FeedID)
	if !ok {
		return Product{}, fmt.Errorf("feed %q is not configured", request.FeedID)
	}
	if strings.TrimSpace(feed.ID) == "" || !xmlBool(feed.EnabledRaw, true) {
		return Product{}, fmt.Errorf("feed %q is disabled", request.FeedID)
	}
	if !xmlBool(feed.Playout.Routine, true) {
		return Product{}, fmt.Errorf("routine playout is disabled for feed %q", request.FeedID)
	}
	if !r.cfg.packageEnabled(request.PackageID) {
		return Product{}, fmt.Errorf("package %q is disabled", request.PackageID)
	}

	base := productBase(r.cfg, feed, request.PackageID)
	var product Product
	var err error
	switch strings.ToLower(strings.TrimSpace(request.PackageID)) {
	case "station_id":
		product = r.stationIDProduct(base, feed)
	case "date_time":
		product = r.dateTimeProduct(base, feed)
	case "current_conditions":
		product, err = r.currentConditionsProduct(base, feed)
	case "forecast":
		product, err = r.forecastProduct(base, feed)
	case "air_quality":
		product, err = r.airQualityProduct(base, feed)
	case "climate_summary":
		product, err = r.climateProduct(base, feed)
	case "user_bulletin":
		product, err = r.userBulletinProduct(base, feed)
	case "alerts":
		product, err = r.alertsProduct(base, feed)
	case "geophysical_alert":
		product, err = r.textStoreProduct(base, "Geophysical Alert", "nws", "wwv", cleanWWVProduct)
	case "eccc_discussion":
		product, err = r.discussionProduct(base)
	default:
		return Product{}, fmt.Errorf("package %q is not supported by product-render yet", request.PackageID)
	}
	if err != nil {
		return Product{}, err
	}
	product.Text = flattenSegments(product.Segments)
	product.GeneratedAt = time.Now().UTC()
	if product.ID == "" {
		product.ID = safeID(fmt.Sprintf("%s-%s-%d", product.FeedID, product.PackageID, product.GeneratedAt.UnixNano()))
	}
	return product, nil
}

func productBase(cfg loadedConfig, feed feedXML, pkgID string) Product {
	return Product{
		FeedID:    feed.ID,
		PackageID: pkgID,
		Title:     titleForPackage(pkgID),
		ReaderID:  cfg.readerID(pkgID),
		Language:  feedLanguage(feed),
		Metadata: map[string]string{
			"site_name": feedSiteName(feed),
			"callsign":  feedCallsign(feed),
		},
	}
}

func (r renderer) stationIDProduct(base Product, feed feedXML) Product {
	onAirName := displayText(r.cfg.Root.Operator.OnAirName)
	if onAirName == "" {
		onAirName = "Haze Weather Radio"
	}
	site := feedSiteName(feed)
	callsign := feedCallsign(feed)
	frequency := feedFrequencyMHz(feed)
	parts := []string{r.packageText(base.PackageID, "opener", base.Language, "You are listening to {on_air_name}.", map[string]string{
		"on_air_name": onAirName,
		"site":        site,
		"callsign":    callsign,
		"frequency":   frequency,
	})}
	if callsignSpoken := spokenCallsign(callsign); callsignSpoken != "" {
		parts = append(parts, fmt.Sprintf("Callsign %s.", callsignSpoken))
	}
	if frequency != "" {
		location := fallbackText(site, feed.ID)
		parts = append(parts, fmt.Sprintf("Broadcasting from %s on a frequency of %s megahertz.", location, frequency))
	} else if site != "" {
		parts = append(parts, fmt.Sprintf("Serving %s.", site))
	}
	if desc := feedDescription(feed, base.Language); desc != "" {
		parts = append(parts, desc)
	}
	if replacement := replacementStationStatement(feed); replacement != "" {
		parts = append(parts, replacement)
	}
	base.Title = "Station Identification"
	base.Segments = []Segment{{Kind: "static", Label: "station_id", Text: strings.Join(parts, " ")}}
	return base
}

func replacementStationStatement(feed feedXML) string {
	transmitter, ok := replacementTransmitter(feed)
	if !ok {
		return ""
	}
	callsign := strings.TrimSpace(transmitter.Callsign)
	site := strings.TrimSpace(transmitter.SiteName)
	if callsign == "" && site == "" {
		return ""
	}
	network := strings.TrimSpace(transmitter.Network.Name)
	if network == "" {
		network = "Weatheradio Canada"
	}
	parts := []string{"This station replaces former", network, "station"}
	if callsign != "" {
		parts = append(parts, callsign)
	}
	if site != "" {
		parts = append(parts, "in", site)
	}
	return strings.Join(parts, " ") + "."
}

func (r renderer) dateTimeProduct(base Product, feed feedXML) Product {
	now := time.Now()
	if loc, err := time.LoadLocation(fallbackText(feed.Timezone, "Local")); err == nil {
		now = now.In(loc)
	}
	base.Title = "Date and Time"
	base.Segments = []Segment{{Kind: "static", Label: "date_time", Text: dateTimeAnnouncement(now, base.Language)}}
	return base
}

func (r renderer) currentConditionsProduct(base Product, feed feedXML) (Product, error) {
	var snapshot observationSnapshot
	inputPath, ok := r.loadLiveObservationSnapshot(feed, &snapshot)
	if !ok {
		return Product{}, fmt.Errorf("current weather observations are unavailable for feed %s", feed.ID)
	}

	base.Title = "Current Conditions"
	base.Inputs = append(base.Inputs, InputRef{Type: inputTypeForPath(inputPath), ID: inputPath})
	callsign := fallbackText(feedCallsign(feed), feedSiteName(feed))
	reported := reportTime(snapshot.ReportedAt, feed.Timezone)
	source := spokenWeatherSource(firstNonBlank(snapshot.Primary.Source, sourceFromObservations(snapshot.Observations), sourceFromObservations(snapshot.AreaObservations)))
	segments := []Segment{}
	if text := r.packageText(base.PackageID, "opener", base.Language, "The current weather conditions. Issued by {source} at {time}.", map[string]string{
		"source":   fallbackText(source, "Environment and Climate Change Canada"),
		"time":     fallbackText(reported, "the latest report time"),
		"callsign": callsign,
		"site":     feedSiteName(feed),
	}); text != "" {
		segments = append(segments, Segment{Kind: "opener", Label: "main_opener", Text: text})
	}
	if text := r.packageText(base.PackageID, "report_time", base.Language, "", map[string]string{
		"time":     reported,
		"callsign": callsign,
		"site":     feedSiteName(feed),
	}); text != "" {
		segments = append(segments, Segment{Kind: "opener", Label: "report_time", Text: text})
	}
	if strings.TrimSpace(snapshot.Primary.LocationName) != "" {
		segments = append(segments, Segment{Kind: "package", Label: "primary_observation", Text: fullObservationText(snapshot.Primary)})
	}

	area := snapshot.AreaObservations
	if len(area) == 0 {
		area = snapshot.Observations
	}
	addedAreaOpener := false
	for _, obs := range area {
		if obs.LocationName == "" || sameObservation(obs, snapshot.Primary) {
			continue
		}
		if !addedAreaOpener {
			segments = append(segments, Segment{
				Kind:  "opener",
				Label: "area_observations",
				Text: r.packageText(base.PackageID, "area_opener", base.Language, "Elsewhere around the {callsign} listening area:", map[string]string{
					"callsign": callsign,
					"site":     feedSiteName(feed),
				}),
			})
			addedAreaOpener = true
		}
		if text := shortObservationText(obs); text != "" {
			segments = append(segments, Segment{Kind: "package", Label: "area_observation", Text: text})
		}
	}
	if repeat := repeatObservationText(snapshot.Primary); repeat != "" {
		segments = append(segments, Segment{Kind: "closure", Label: "repeat_primary", Text: repeat})
	}
	base.Segments = segments
	return base, nil
}

func (r renderer) discussionProduct(base Product) (Product, error) {
	locations := r.cfg.packageLocations(base.PackageID)
	product, err := r.textStoreProduct(base, "Weather Discussion", "eccc", "focn45.cwwg", func(raw string) string {
		return cleanDiscussionProduct(raw, locations)
	})
	if err != nil {
		return Product{}, err
	}
	opener := r.packageText(base.PackageID, "opener", base.Language, "Here is the latest significant weather discussion from Environment and Climate Change Canada for the {callsign} listening area.", map[string]string{
		"callsign": fallbackText(base.Metadata["callsign"], base.Metadata["site_name"]),
		"site":     base.Metadata["site_name"],
	})
	if opener != "" {
		product.Segments = append([]Segment{{Kind: "opener", Label: "discussion_opener", Text: opener}}, product.Segments...)
	}
	if len(locations.Mentions) > 0 {
		product.Metadata["location_filter"] = strings.Join(locations.Mentions, ", ")
	}
	return product, nil
}

func (r renderer) forecastProduct(base Product, feed feedXML) (Product, error) {
	var snapshot forecastSnapshot
	inputPath, ok := r.loadLiveForecastSnapshot(feed, &snapshot)
	if !ok {
		return Product{}, fmt.Errorf("forecast information is unavailable for feed %s", feed.ID)
	}
	base.Title = "Forecast"
	base.Inputs = append(base.Inputs, InputRef{Type: inputTypeForPath(inputPath), ID: inputPath})
	callsign := fallbackText(feedCallsign(feed), feedSiteName(feed))
	segments := []Segment{{Kind: "opener", Text: r.packageText(base.PackageID, "opener", base.Language, "Forecast for the {callsign} listening area:", map[string]string{
		"callsign": callsign,
		"site":     feedSiteName(feed),
	})}}
	if issued := reportTime(snapshot.IssuedAt, feed.Timezone); issued != "" {
		segments = append(segments, Segment{Kind: "opener", Label: "issued_at", Text: r.packageText(base.PackageID, "issued_at", base.Language, "Issued at {time}.", map[string]string{
			"time":     issued,
			"callsign": callsign,
			"site":     feedSiteName(feed),
		})})
	}
	for _, region := range snapshot.Regions {
		name := normalizeRegionTitle(region.Name)
		if name != "" {
			segments = append(segments, Segment{Kind: "opener", Label: "forecast_region", Text: r.packageText(base.PackageID, "region", base.Language, "For the {region}.", map[string]string{
				"region": name,
			})})
		}
		for _, period := range region.Periods {
			if period.Text == "" {
				continue
			}
			label := strings.TrimSpace(period.Name)
			segments = append(segments, Segment{Kind: "package", Label: strings.TrimSpace(strings.Join([]string{name, label}, ", ")), Text: periodForecastText(period)})
		}
	}
	for _, period := range snapshot.Periods {
		if period.Text == "" {
			continue
		}
		label := strings.TrimSpace(period.Name)
		segments = append(segments, Segment{Kind: "package", Label: label, Text: periodForecastText(period)})
	}
	if len(segments) == 1 {
		segments = append(segments, Segment{Kind: "package", Text: "No active forecast periods are available."})
	}
	base.Segments = segments
	return base, nil
}

func (r renderer) airQualityProduct(base Product, feed feedXML) (Product, error) {
	var snapshot airQualitySnapshot
	inputPath, ok := r.loadLiveAirQualitySnapshot(feed, &snapshot)
	if !ok {
		return Product{}, fmt.Errorf("air quality information is unavailable for feed %s", feed.ID)
	}
	base.Title = "Air Quality"
	base.Inputs = append(base.Inputs, InputRef{Type: inputTypeForPath(inputPath), ID: inputPath})
	location := fallbackText(fallbackText(snapshot.Location, feedSiteName(feed)), "this area")
	segments := []Segment{}
	if snapshot.AQHI != "" {
		segments = append(segments, Segment{Kind: "package", Label: "observed", Text: r.packageText(base.PackageID, "now_eccc", base.Language, "The air quality health index was observed at {name} and reported a value of {val} at {time}.", map[string]string{
			"name": location,
			"val":  snapshot.AQHI,
			"time": fallbackText(reportTime(snapshot.ReportedAt, feed.Timezone), "the latest report time"),
		})})
	}
	if narrative := r.airQualityRiskNarrative(base.Language, snapshot.AQHI, snapshot.Risk); narrative != "" {
		segments = append(segments, Segment{Kind: "package", Label: "risk", Text: narrative})
	}
	if note := sentence(snapshot.SpecialNotes); note != "" {
		segments = append(segments, Segment{Kind: "package", Label: "special_notes", Text: note})
	}
	segments = append(segments, r.airQualityForecastSegments(base, location, snapshot.Periods)...)
	if len(snapshot.Periods) == 0 && snapshot.Forecast != "" {
		segments = append(segments, Segment{Kind: "package", Label: "forecast", Text: sentence(snapshot.Forecast)})
	}
	if len(segments) == 0 {
		segments = append(segments, Segment{Kind: "package", Label: "unavailable", Text: r.packageText(base.PackageID, "unavailable_report", base.Language, "The air quality information for {name} was unavailable.", map[string]string{
			"name": location,
		})})
	}
	base.Segments = segments
	return base, nil
}

func (r renderer) airQualityForecastSegments(base Product, location string, periods []airQualityPeriodSnapshot) []Segment {
	segments := []Segment{}
	for index := 0; index < len(periods) && index < 2; index++ {
		period := periods[index]
		if period.Name == "" || period.AQHI == "" {
			continue
		}
		key := "forecast_eccc"
		fallback := "For {period_name}, the maximum air quality health index is forecast to be {val}, or {risk}."
		label := "forecast_period"
		if index == 0 {
			key = "forecast_opener_eccc"
			fallback = "The air quality health index forecast for {name} is {val} for {period_name} and is considered {risk}."
			label = "forecast_opener"
		}
		segments = append(segments, Segment{Kind: "package", Label: label, Text: r.packageText(base.PackageID, key, base.Language, fallback, map[string]string{
			"name":        location,
			"val":         period.AQHI,
			"period_name": period.Name,
			"risk":        airQualityRiskLabel(base.Language, period.AQHI, period.Risk),
		})})
		if period.AQHIInSmoke != "" {
			segments = append(segments, Segment{Kind: "package", Label: "forecast_smoke", Text: r.packageText(base.PackageID, "forecast_insmoke_eccc", base.Language, "The air quality health index is expected to be {val} in smoke.", map[string]string{
				"val": period.AQHIInSmoke,
			})})
		}
	}
	if len(periods) >= 4 && periods[2].AQHI != "" && periods[3].AQHI != "" {
		segments = append(segments, Segment{Kind: "package", Label: "forecast_trailing", Text: r.packageText(base.PackageID, "forecast_trailing_eccc", base.Language, "{period2_val} on {period2_name}, and lastly, {period3_val} on {period3_name}.", map[string]string{
			"period2_val":  periods[2].AQHI,
			"period2_name": periods[2].Name,
			"period3_val":  periods[3].AQHI,
			"period3_name": periods[3].Name,
		})})
	} else if len(periods) >= 3 && periods[2].AQHI != "" {
		period := periods[2]
		segments = append(segments, Segment{Kind: "package", Label: "forecast_period", Text: r.packageText(base.PackageID, "forecast_eccc", base.Language, "For {period_name}, the maximum air quality health index is forecast to be {val}, or {risk}.", map[string]string{
			"period_name": period.Name,
			"val":         period.AQHI,
			"risk":        airQualityRiskLabel(base.Language, period.AQHI, period.Risk),
		})})
		if period.AQHIInSmoke != "" {
			segments = append(segments, Segment{Kind: "package", Label: "forecast_smoke", Text: r.packageText(base.PackageID, "forecast_insmoke_eccc", base.Language, "The air quality health index is expected to be {val} in smoke.", map[string]string{
				"val": period.AQHIInSmoke,
			})})
		}
	}
	return segments
}

func (r renderer) airQualityRiskNarrative(lang string, aqhi string, risk string) string {
	key := airQualityRiskKey(aqhi, risk)
	if key == "" {
		return ""
	}
	fallbacks := map[string]string{
		"low":       "This is ideal air quality for outdoor activities for both at-risk and general populations.",
		"moderate":  "This is acceptable air quality for outdoor activities for most people. However, at-risk individuals should consider reducing or rescheduling strenuous outdoor activities if symptoms occur.",
		"high":      "This is unhealthy air quality for at-risk individuals, which include children, seniors, and those with pre-existing respiratory or heart conditions. At-risk individuals should reduce or reschedule strenuous outdoor activities. The general population is not likely to be affected.",
		"very_high": "This is hazardous air quality for most people. At-risk individuals should avoid strenuous outdoor activities. Everyone else should also reduce or reschedule strenuous outdoor activities.",
	}
	return r.packageText("air_quality", key+"_eccc", lang, fallbacks[key], nil)
}

func (r renderer) climateProduct(base Product, feed feedXML) (Product, error) {
	var snapshot climateSnapshot
	inputPath, ok := r.loadLiveClimateSnapshot(feed, &snapshot)
	if !ok {
		return Product{}, fmt.Errorf("climate summary information is unavailable for feed %s", feed.ID)
	}
	base.Title = "Climate Summary"
	base.Inputs = append(base.Inputs, InputRef{Type: inputTypeForPath(inputPath), ID: inputPath})
	location := fallbackText(snapshot.Location, feedSiteName(feed))
	segments := []Segment{{Kind: "opener", Text: r.packageText(base.PackageID, "opener", base.Language, "Climate summary for {location}.", map[string]string{
		"location": location,
		"site":     feedSiteName(feed),
	})}}
	for _, line := range snapshot.Summary {
		if text := sentence(strings.TrimSpace(line)); text != "" {
			segments = append(segments, Segment{Kind: "package", Text: text})
		}
	}
	base.Segments = segments
	return base, nil
}

func (r renderer) userBulletinProduct(base Product, feed feedXML) (Product, error) {
	var snapshot bulletinSnapshot
	inputPath, ok := r.loadLiveBulletinSnapshot(feed, &snapshot)
	if !ok {
		return Product{}, fmt.Errorf("no active user bulletins are available for feed %s", feed.ID)
	}
	base.Title = fallbackText(snapshot.Title, "User Bulletin")
	base.Inputs = append(base.Inputs, InputRef{Type: inputTypeForPath(inputPath), ID: inputPath})
	for _, line := range snapshot.Lines {
		if text := sentence(strings.TrimSpace(line)); text != "" {
			base.Segments = append(base.Segments, Segment{Kind: "package", Text: text})
		}
	}
	if len(base.Segments) == 0 {
		base.Segments = []Segment{{Kind: "package", Text: "There are no user bulletins at this time."}}
	}
	return base, nil
}

func (r renderer) textStoreProduct(base Product, title string, source string, id string, clean func(string) string) (Product, error) {
	if r.cfg.Store == nil {
		return Product{}, fmt.Errorf("%s is unavailable", strings.ToLower(title))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	body, ok, err := r.cfg.Store.TextProduct(ctx, source, id)
	if err != nil || !ok {
		return Product{}, fmt.Errorf("%s is unavailable", strings.ToLower(title))
	}
	if clean != nil {
		body = clean(body)
	}
	text := cleanPlaintextProduct(body)
	if text == "" {
		return Product{}, fmt.Errorf("%s is empty", strings.ToLower(title))
	}
	base.Title = title
	base.Inputs = append(base.Inputs, InputRef{Type: "store", ID: fmt.Sprintf("products.text/%s/%s", source, id)})
	base.Segments = []Segment{{Kind: "package", Text: text}}
	return base, nil
}

func cleanWWVProduct(raw string) string {
	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	out := make([]string, 0, len(lines))
	seenBody := false
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !seenBody {
			lower := strings.ToLower(trimmed)
			if trimmed == "" ||
				strings.HasPrefix(trimmed, ":") ||
				strings.HasPrefix(trimmed, "#") ||
				strings.Contains(lower, "geophysical alert message") ||
				strings.Contains(lower, "prepared by the us dept. of commerce") {
				continue
			}
		}
		if trimmed != "" {
			seenBody = true
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

type discussionSection struct {
	Heading string
	Body    string
}

func cleanDiscussionProduct(raw string, locations packageLocations) string {
	mentions := locations.Mentions
	if len(mentions) == 0 && strings.EqualFold(locations.StateProv, "SK") {
		mentions = []string{"SK", "Saskatchewan", "Southern SK", "Southern Saskatchewan"}
	}
	if len(mentions) == 0 {
		return cleanPlaintextProduct(raw)
	}
	text := discussionPlainText(raw)
	text = removeDiscussionAlertLead(text)
	patterns := mentionPatterns(mentions)
	sections := splitDiscussionSections(text)
	out := make([]string, 0, len(sections))
	for _, section := range sections {
		heading := spokenDiscussionHeading(section.Heading)
		body := strings.TrimSpace(section.Body)
		if body == "" {
			continue
		}
		if matchesAnyMention(section.Heading, patterns) {
			out = append(out, sentence(strings.TrimSpace(heading+". "+body)))
			continue
		}
		sentences := matchingSentences(body, patterns)
		if len(sentences) == 0 {
			continue
		}
		out = append(out, strings.Join(sentences, " "))
	}
	return normalizeDiscussionForSpeech(strings.Join(out, "\n\n"))
}

func discussionPlainText(raw string) string {
	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || strings.HasPrefix(line, ":") {
			continue
		}
		parts = append(parts, line)
	}
	return strings.Join(strings.Fields(strings.Join(parts, " ")), " ")
}

func removeDiscussionAlertLead(text string) string {
	upper := strings.ToUpper(text)
	start := strings.Index(upper, "ALERTS IN EFFECT")
	if start < 0 {
		return text
	}
	end := strings.Index(upper[start:], "OVERVIEW")
	if end < 0 {
		return strings.TrimSpace(text[:start])
	}
	end += start
	return strings.TrimSpace(text[:start] + text[end:])
}

func splitDiscussionSections(text string) []discussionSection {
	matches := discussionHeadingPattern.FindAllStringSubmatchIndex(text, -1)
	if len(matches) == 0 {
		return []discussionSection{{Body: strings.TrimSpace(text)}}
	}
	sections := make([]discussionSection, 0, len(matches)+1)
	if prefix := strings.TrimSpace(text[:matches[0][0]]); prefix != "" {
		sections = append(sections, discussionSection{Body: prefix})
	}
	for index, match := range matches {
		bodyStart := match[1]
		bodyEnd := len(text)
		if index+1 < len(matches) {
			bodyEnd = matches[index+1][0]
		}
		sections = append(sections, discussionSection{
			Heading: strings.TrimSpace(text[match[2]:match[3]]),
			Body:    strings.TrimSpace(text[bodyStart:bodyEnd]),
		})
	}
	return sections
}

func mentionPatterns(mentions []string) []*regexp.Regexp {
	patterns := make([]*regexp.Regexp, 0, len(mentions))
	for _, mention := range mentions {
		mention = strings.Join(strings.Fields(strings.TrimSpace(mention)), " ")
		if mention == "" {
			continue
		}
		words := strings.Fields(mention)
		for index, word := range words {
			words[index] = regexp.QuoteMeta(word)
		}
		pattern := strings.Join(words, `\s+`)
		patterns = append(patterns, regexp.MustCompile(`(?i)\b`+pattern+`\b`))
	}
	return patterns
}

func matchesAnyMention(text string, patterns []*regexp.Regexp) bool {
	for _, pattern := range patterns {
		if pattern.MatchString(text) {
			return true
		}
	}
	return false
}

func matchingSentences(text string, patterns []*regexp.Regexp) []string {
	chunks := splitSentences(text)
	out := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		if matchesAnyMention(chunk, patterns) {
			out = append(out, sentence(chunk))
		}
	}
	return out
}

func splitSentences(text string) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}
	var out []string
	start := 0
	for index, char := range text {
		if char != '.' && char != '!' && char != '?' {
			continue
		}
		end := index + len(string(char))
		out = append(out, strings.TrimSpace(text[start:end]))
		for end < len(text) && text[end] == ' ' {
			end++
		}
		start = end
	}
	if start < len(text) {
		out = append(out, strings.TrimSpace(text[start:]))
	}
	return out
}

func spokenDiscussionHeading(heading string) string {
	heading = strings.TrimSpace(heading)
	switch strings.ToUpper(heading) {
	case "SK":
		return "Saskatchewan"
	case "SOUTHERN SK":
		return "Southern Saskatchewan"
	default:
		titled := titleText(heading)
		return strings.ReplaceAll(titled, " Sk", " Saskatchewan")
	}
}

func normalizeDiscussionForSpeech(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	text = discussionRangeUnitPattern.ReplaceAllStringFunc(text, func(raw string) string {
		matches := discussionRangeUnitPattern.FindStringSubmatch(raw)
		if len(matches) != 4 {
			return raw
		}
		return matches[1] + " to " + matches[2] + " " + spokenDiscussionUnit(matches[3])
	})
	text = discussionNumberUnitPattern.ReplaceAllStringFunc(text, func(raw string) string {
		matches := discussionNumberUnitPattern.FindStringSubmatch(raw)
		if len(matches) != 3 {
			return raw
		}
		return matches[1] + " " + spokenDiscussionUnit(matches[2])
	})
	for _, replacement := range discussionStandaloneReplacements {
		text = replacement.pattern.ReplaceAllString(text, replacement.replacement)
	}
	return strings.Join(strings.Fields(text), " ")
}

func spokenDiscussionUnit(unit string) string {
	switch strings.ToUpper(strings.TrimSpace(unit)) {
	case "KM/H", "KPH":
		return "kilometres per hour"
	case "KT":
		return "knot"
	case "KTS":
		return "knots"
	case "HPA":
		return "hectopascals"
	case "KPA":
		return "kilopascals"
	case "MB":
		return "millibars"
	case "MM":
		return "millimetres"
	case "CM":
		return "centimetres"
	case "M":
		return "metres"
	case "KM":
		return "kilometres"
	case "C":
		return "degrees Celsius"
	default:
		return strings.ToLower(unit)
	}
}

func unavailableProduct(base Product, title string, text string) Product {
	base.Title = title
	base.Segments = []Segment{{Kind: "package", Label: "unavailable", Text: text}}
	return base
}

func (r renderer) packageText(pkgID string, key string, lang string, fallback string, values map[string]string) string {
	text := fallback
	pkgID = strings.ToLower(strings.TrimSpace(pkgID))
	key = strings.ToLower(strings.TrimSpace(key))
	if byKey, ok := r.cfg.ProductText[pkgID]; ok {
		if byLang, ok := byKey[key]; ok {
			if configured := localizedTextEntry(byLang, lang); configured != "" {
				text = configured
			}
		}
	}
	return renderTemplateText(text, values)
}

func localizedTextEntry(values map[string]string, lang string) string {
	if len(values) == 0 {
		return ""
	}
	lang = normalizeLangKey(lang)
	short := lang
	if idx := strings.Index(short, "-"); idx > 0 {
		short = short[:idx]
	}
	for _, key := range []string{lang, short, "en-ca", "en", "*"} {
		if text := strings.TrimSpace(values[key]); text != "" {
			return text
		}
		for rawKey, text := range values {
			if strings.EqualFold(rawKey, key) && strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
	}
	return ""
}

func renderTemplateText(text string, values map[string]string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	for key, value := range values {
		text = strings.ReplaceAll(text, "{"+key+"}", strings.TrimSpace(value))
	}
	return strings.Join(strings.Fields(text), " ")
}

func inputTypeForPath(path string) string {
	if strings.HasPrefix(path, "store:") {
		return "store"
	}
	if strings.Contains(filepath.ToSlash(path), "/managed/") {
		return "managed"
	}
	return "input"
}

func fullObservationText(obs observation) string {
	location := fallbackText(obs.LocationName, "the reporting station")
	parts := []string{fmt.Sprintf("The weather at %s", location)}
	if obs.Condition != "" {
		parts[0] += " was " + obs.Condition
	}
	if obs.TemperatureC != nil {
		parts = append(parts, fmt.Sprintf("The temperature was %s", degrees(*obs.TemperatureC)))
	}
	if obs.DewpointC != nil {
		parts = append(parts, fmt.Sprintf("dewpoint %s", degreesBare(*obs.DewpointC)))
	}
	if obs.HumidityPercent != nil {
		parts = append(parts, fmt.Sprintf("and the relative humidity was %s percent", rounded(*obs.HumidityPercent)))
	}
	if wind := windText(obs, true); wind != "" {
		parts = append(parts, wind)
	}
	if obs.VisibilityKM != nil {
		parts = append(parts, fmt.Sprintf("Visibility was up to %s kilometres", rounded(*obs.VisibilityKM)))
	}
	if obs.PressureKPA != nil {
		pressure := fmt.Sprintf("The pressure was %s kilopascals", oneDecimal(*obs.PressureKPA))
		if obs.PressureTendency != "" {
			pressure += " and " + strings.ToLower(obs.PressureTendency)
		}
		parts = append(parts, pressure)
	}
	return sentence(strings.Join(parts, ". "))
}

func shortObservationText(obs observation) string {
	if obs.LocationName == "" {
		return ""
	}
	parts := []string{obs.LocationName}
	if obs.TemperatureC != nil {
		parts = append(parts, degrees(*obs.TemperatureC))
	}
	if wind := windText(obs, false); wind != "" {
		parts = append(parts, wind)
	}
	return sentence(strings.Join(parts, ", "))
}

func repeatObservationText(obs observation) string {
	if obs.LocationName == "" {
		return ""
	}
	parts := []string{fmt.Sprintf("Again, at %s", obs.LocationName)}
	if obs.Condition != "" {
		parts = append(parts, "it was "+obs.Condition)
	}
	if obs.TemperatureC != nil {
		parts = append(parts, "with a temperature of "+degrees(*obs.TemperatureC))
	}
	return sentence(strings.Join(parts, " "))
}

func windText(obs observation, sentenceCase bool) string {
	if obs.WindDirection == "" && obs.WindSpeedKMH == nil {
		return ""
	}
	prefix := "winds were"
	if sentenceCase {
		prefix = "Winds were"
	}
	direction := readableDirection(obs.WindDirection)
	if obs.WindSpeedKMH == nil {
		return strings.TrimSpace(prefix + " " + direction)
	}
	text := strings.TrimSpace(fmt.Sprintf("%s %s at %s kilometres per hour", prefix, direction, rounded(*obs.WindSpeedKMH)))
	if obs.WindGustKMH != nil && *obs.WindGustKMH > *obs.WindSpeedKMH {
		text += fmt.Sprintf(" with gusts up to %s kilometres per hour", rounded(*obs.WindGustKMH))
	}
	return text
}

func sameObservation(left observation, right observation) bool {
	if left.ID != "" && strings.EqualFold(left.ID, right.ID) {
		return true
	}
	return left.LocationName != "" && strings.EqualFold(left.LocationName, right.LocationName)
}

func sourceFromObservations(observations []observation) string {
	for _, obs := range observations {
		if text := strings.TrimSpace(obs.Source); text != "" {
			return text
		}
	}
	return ""
}

func airQualityRiskKey(aqhi string, risk string) string {
	if numeric, ok := numberFromAny(strings.TrimSpace(aqhi)); ok {
		switch {
		case numeric <= 0:
			return ""
		case numeric <= 3:
			return "low"
		case numeric <= 6:
			return "moderate"
		case numeric <= 10:
			return "high"
		default:
			return "very_high"
		}
	}
	normalized := strings.ToLower(strings.TrimSpace(risk))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.ReplaceAll(normalized, " ", "_")
	switch normalized {
	case "low", "moderate", "high", "very_high":
		return normalized
	default:
		return ""
	}
}

func airQualityRiskLabel(lang string, aqhi string, risk string) string {
	key := airQualityRiskKey(aqhi, risk)
	labels := map[string]map[string]string{
		"en": {
			"low":       "Low",
			"moderate":  "Moderate",
			"high":      "High",
			"very_high": "Very High",
		},
		"fr": {
			"low":       "Faible",
			"moderate":  "Modere",
			"high":      "Eleve",
			"very_high": "Tres eleve",
		},
		"es": {
			"low":       "Bajo",
			"moderate":  "Moderado",
			"high":      "Alto",
			"very_high": "Muy alto",
		},
	}
	short := strings.ToLower(strings.TrimSpace(lang))
	if idx := strings.Index(short, "-"); idx > 0 {
		short = short[:idx]
	}
	if localized := labels[short][key]; localized != "" {
		return localized
	}
	if localized := labels["en"][key]; localized != "" {
		return localized
	}
	return strings.TrimSpace(risk)
}

func spokenWeatherSource(source string) string {
	switch strings.ToLower(strings.TrimSpace(source)) {
	case "eccc", "msc", "envcan", "environment canada":
		return "Environment and Climate Change Canada"
	case "noaa":
		return "the National Oceanic and Atmospheric Administration"
	case "nws":
		return "the National Weather Service"
	case "twc", "weather.com", "weatherdotcom":
		return "The Weather Channel"
	default:
		return strings.TrimSpace(source)
	}
}

func feedDescription(feed feedXML, lang string) string {
	lang = strings.ToLower(strings.TrimSpace(lang))
	short := lang
	if idx := strings.Index(short, "-"); idx > 0 {
		short = short[:idx]
	}
	for _, entry := range feed.Description.Langs {
		code := strings.ToLower(strings.TrimSpace(entry.Code))
		if code == lang || code == short || (code == "en-ca" && short == "en") {
			return strings.TrimSpace(strings.Join([]string{entry.Text, entry.Suffix}, " "))
		}
	}
	for _, entry := range feed.Description.Langs {
		if text := strings.TrimSpace(strings.Join([]string{entry.Text, entry.Suffix}, " ")); text != "" {
			return text
		}
	}
	return ""
}

func reportTime(raw string, timezone string) string {
	if strings.TrimSpace(raw) == "" {
		return ""
	}
	parsed, err := time.Parse(time.RFC3339, strings.TrimSpace(raw))
	if err != nil {
		for _, layout := range []string{"2006-01-02 15:04:05", "2006-01-02T15:04:05"} {
			parsed, err = time.Parse(layout, strings.TrimSpace(raw))
			if err == nil {
				break
			}
		}
	}
	if err != nil {
		return strings.TrimSpace(raw)
	}
	if loc, locErr := time.LoadLocation(fallbackText(timezone, "Local")); locErr == nil {
		parsed = parsed.In(loc)
	}
	return strings.TrimSpace(parsed.Format("3:04 PM ") + timezoneName(parsed.Format("MST")))
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
	default:
		return abbrev
	}
}

func dateTimeAnnouncement(now time.Time, lang string) string {
	short := strings.ToLower(strings.TrimSpace(lang))
	if idx := strings.Index(short, "-"); idx > 0 {
		short = short[:idx]
	}
	prefix := timeGreeting(now, short)
	timeText := spokenClockTime(now)
	switch short {
	case "fr":
		return strings.TrimSpace(prefix + " Il est actuellement " + timeText + ".")
	case "es":
		return strings.TrimSpace(prefix + " La hora actual es " + timeText + ".")
	default:
		return strings.TrimSpace(prefix + " The current time is " + timeText + ".")
	}
}

func timeGreeting(now time.Time, lang string) string {
	hour := now.Hour()
	period := "night"
	switch {
	case hour >= 5 && hour < 12:
		period = "morning"
	case hour >= 12 && hour < 17:
		period = "afternoon"
	case hour >= 17 && hour < 22:
		period = "evening"
	}
	switch lang {
	case "fr":
		switch period {
		case "morning":
			return "Bonjour."
		case "afternoon":
			return "Bon après-midi."
		case "evening":
			return "Bonsoir."
		default:
			return "Bonne nuit."
		}
	case "es":
		switch period {
		case "morning":
			return "Buenos días."
		case "afternoon":
			return "Buenas tardes."
		default:
			return "Buenas noches."
		}
	default:
		switch period {
		case "morning":
			return "Good morning."
		case "afternoon":
			return "Good afternoon."
		case "evening":
			return "Good evening."
		default:
			return "Good night."
		}
	}
}

func spokenClockTime(now time.Time) string {
	hour := numberToWords(now.Hour()%12, true)
	if now.Hour()%12 == 0 {
		hour = "twelve"
	}
	minute := now.Minute()
	ampm := "A.M."
	if now.Hour() >= 12 {
		ampm = "P.M."
	}
	tz := timezoneName(now.Format("MST"))
	if minute == 0 {
		return strings.TrimSpace(fmt.Sprintf("%s %s, %s", hour, ampm, tz))
	}
	minuteText := numberToWords(minute, false)
	if minute < 10 {
		minuteText = "oh " + minuteText
	}
	return strings.TrimSpace(fmt.Sprintf("%s %s %s, %s", hour, minuteText, ampm, tz))
}

func numberToWords(value int, hour bool) string {
	if hour && value == 0 {
		value = 12
	}
	ones := []string{"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"}
	if value >= 0 && value < len(ones) {
		return ones[value]
	}
	tens := []string{"", "", "twenty", "thirty", "forty", "fifty"}
	if value >= 20 && value < 60 {
		ten := value / 10
		one := value % 10
		if one == 0 {
			return tens[ten]
		}
		return strings.TrimSpace(tens[ten] + " " + ones[one])
	}
	return fmt.Sprintf("%d", value)
}

func spokenCallsign(callsign string) string {
	callsign = strings.TrimSpace(callsign)
	if callsign == "" {
		return ""
	}
	parts := make([]string, 0, len(callsign))
	for _, ch := range callsign {
		if ch == '-' || ch == ' ' || ch == '_' {
			continue
		}
		parts = append(parts, strings.ToUpper(string(ch)))
	}
	return strings.Join(parts, " ")
}

func normalizeRegionTitle(raw string) string {
	value := strings.TrimSpace(raw)
	value = strings.TrimSuffix(value, " Area")
	value = strings.TrimSuffix(value, " area")
	if value == "" {
		return ""
	}
	if strings.Contains(value, " - ") {
		return pauseForecastRegionName(value, "en")
	}
	if strings.Contains(strings.ToLower(value), "region") {
		return value
	}
	return value
}

func pauseForecastRegionName(value string, language string) string {
	cleaned := strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
	if cleaned == "" {
		return ""
	}
	parts := strings.Split(cleaned, " - ")
	if len(parts) <= 1 {
		return cleaned
	}
	cleanParts := make([]string, 0, len(parts))
	for index, part := range parts {
		part = strings.TrimSpace(strings.Trim(part, " ,."))
		part = strings.TrimSpace(strings.TrimSuffix(part, " region"))
		part = strings.TrimSpace(strings.TrimSuffix(part, " Region"))
		part = strings.TrimSpace(strings.TrimPrefix(part, "and "))
		if index == len(parts)-1 && strings.Contains(part, " and ") {
			for _, child := range strings.Split(part, " and ") {
				child = strings.TrimSpace(strings.Trim(child, " ,."))
				if child != "" {
					cleanParts = append(cleanParts, child)
				}
			}
			continue
		}
		if part != "" {
			cleanParts = append(cleanParts, part)
		}
	}
	if len(cleanParts) <= 1 {
		return cleaned
	}
	if language == "fr" {
		return strings.Join(cleanParts, ". ")
	}
	if len(cleanParts) == 2 {
		return cleanParts[0] + " and " + cleanParts[1] + " region"
	}
	return strings.Join(cleanParts[:len(cleanParts)-1], ", ") + ", and " + cleanParts[len(cleanParts)-1] + " region"
}

func forecastRegionBaseCode(region string) string {
	region = strings.TrimSpace(region)
	if before, _, ok := strings.Cut(region, "-"); ok {
		return strings.TrimSpace(before)
	}
	return region
}

func readableDirection(raw string) string {
	normalized := strings.ToUpper(strings.TrimSpace(raw))
	replacements := map[string]string{
		"N":   "north",
		"NE":  "north east",
		"E":   "east",
		"SE":  "south east",
		"S":   "south",
		"SW":  "south west",
		"W":   "west",
		"NW":  "north west",
		"NNE": "north north east",
		"ENE": "east north east",
		"ESE": "east south east",
		"SSE": "south south east",
		"SSW": "south south west",
		"WSW": "west south west",
		"WNW": "west north west",
		"NNW": "north north west",
	}
	if text, ok := replacements[normalized]; ok {
		return text
	}
	return strings.ToLower(strings.ReplaceAll(raw, "_", " "))
}

func degrees(value float64) string {
	return rounded(value) + " degrees"
}

func degreesBare(value float64) string {
	return rounded(value)
}

func rounded(value float64) string {
	if math.Abs(value-math.Round(value)) < 0.05 {
		return strconv.Itoa(int(math.Round(value)))
	}
	return oneDecimal(value)
}

func oneDecimal(value float64) string {
	return strconv.FormatFloat(math.Round(value*10)/10, 'f', 1, 64)
}

func periodForecastText(period forecastPeriod) string {
	name := strings.TrimSpace(period.Name)
	text := strings.TrimSpace(period.Text)
	if name == "" {
		return sentence(text)
	}
	if text == "" {
		return sentence(name)
	}
	return sentence(strings.TrimSpace(name + ". " + text))
}

func sentence(text string) string {
	text = strings.TrimSpace(text)
	if text == "" {
		return ""
	}
	last := text[len(text)-1]
	if last == '.' || last == '!' || last == '?' {
		return text
	}
	return text + "."
}

func flattenSegments(segments []Segment) string {
	parts := make([]string, 0, len(segments))
	for _, segment := range segments {
		if text := strings.TrimSpace(segment.Text); text != "" {
			parts = append(parts, text)
		}
	}
	return strings.Join(parts, "\n\n")
}

func titleForPackage(pkgID string) string {
	switch strings.ToLower(strings.TrimSpace(pkgID)) {
	case "date_time":
		return "Date and Time"
	case "station_id":
		return "Station Identification"
	case "current_conditions":
		return "Current Conditions"
	case "forecast":
		return "Forecast"
	case "air_quality":
		return "Air Quality"
	case "climate_summary":
		return "Climate Summary"
	case "geophysical_alert":
		return "Geophysical Alert"
	case "eccc_discussion":
		return "Weather Discussion"
	case "alerts":
		return "Alerts"
	case "user_bulletin":
		return "User Bulletin"
	default:
		words := strings.Fields(strings.ReplaceAll(pkgID, "_", " "))
		for i := range words {
			words[i] = strings.ToUpper(words[i][:1]) + strings.ToLower(words[i][1:])
		}
		return strings.Join(words, " ")
	}
}

func safeID(value string) string {
	var builder strings.Builder
	for _, ch := range value {
		if ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9' || ch == '-' || ch == '_' || ch == '.' {
			builder.WriteRune(ch)
		}
	}
	if builder.Len() == 0 {
		return "item"
	}
	return builder.String()
}
