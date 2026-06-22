package webgateway

import (
	"crypto/sha1"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/locationdb"
)

type feedsXML struct {
	Feeds []feedXML `xml:"feed"`
}

type feedXML struct {
	ID         string `xml:"id,attr"`
	EnabledRaw string `xml:"enabled,attr"`
	Timezone   string `xml:"timezone,attr"`
	Playout    struct {
		Routine string `xml:"routine,attr"`
		SAME    string `xml:"same,attr"`
	} `xml:"playout"`
	Alerts struct {
		CapCP struct {
			EnabledRaw string `xml:"enabled,attr"`
		} `xml:"cap_cp"`
		NWSCAP struct {
			EnabledRaw string `xml:"enabled,attr"`
		} `xml:"nws_cap"`
	} `xml:"alerts"`
	Languages struct {
		Langs []struct {
			Code string `xml:"code,attr"`
		} `xml:"lang"`
	} `xml:"languages"`
	Locations struct {
		Coverage struct {
			Regions []coverageRegionXML `xml:"region"`
		} `xml:"coverage"`
		ObservationLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"observationLocations"`
		AirQualityLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"airQualityLocations"`
		ClimateLocations struct {
			Locations []locationXML `xml:"location"`
		} `xml:"climateLocations"`
	} `xml:"locations"`
	Transmitter struct {
		Transmitters []transmitterXML `xml:"transmitter"`
	} `xml:"transmitter_metadata"`
}

type coverageRegionXML struct {
	ID             string                 `xml:"id,attr"`
	Source         string                 `xml:"source,attr"`
	Name           string                 `xml:"name,attr"`
	DeriveForecast string                 `xml:"derive_forecast,attr"`
	Subregions     []coverageSubregionXML `xml:"subregion"`
}

type coverageSubregionXML struct {
	ID string `xml:"id,attr"`
}

type locationXML struct {
	ID           string `xml:"id,attr"`
	Source       string `xml:"source,attr"`
	NameOverride string `xml:"name_override,attr"`
}

type transmitterXML struct {
	Network      transmitterNetworkXML   `xml:"network"`
	HostName     string                  `xml:"host_name"`
	SiteName     string                  `xml:"site_name"`
	Callsign     string                  `xml:"callsign"`
	Relationship string                  `xml:"relationship"`
	FrequencyMHz transmitterFrequencyXML `xml:"frequency_mhz"`
	RDS          transmitterRDSXML       `xml:"rds"`
}

type transmitterNetworkXML struct {
	Name           string `xml:"name"`
	Pronounciation string `xml:"pronounciation"`
	Pronunciation  string `xml:"pronunciation"`
}

type transmitterFrequencyXML struct {
	GPCLK string `xml:"gpclk,attr"`
	GPIO  string `xml:"gpio,attr"`
	Value string `xml:",chardata"`
}

type transmitterRDSXML struct {
	EnabledRaw string   `xml:"enabled,attr"`
	PI         string   `xml:"pi"`
	PS         string   `xml:"ps"`
	RT         string   `xml:"rt"`
	PTY        string   `xml:"pty"`
	TP         string   `xml:"tp"`
	AF         []string `xml:"af"`
}

type outputsXML struct {
	Feeds []outputFeedXML `xml:"feed"`
}

// BuildGitCommit is populated by release/dev build scripts with -ldflags.
var BuildGitCommit = "unknown"

type outputFeedXML struct {
	ID          string        `xml:"id,attr"`
	Stream      outputNodeXML `xml:"stream"`
	UDP         outputNodeXML `xml:"udp"`
	RTP         outputNodeXML `xml:"rtp"`
	RTMP        outputNodeXML `xml:"rtmp"`
	SRT         outputNodeXML `xml:"srt"`
	RTSP        outputNodeXML `xml:"rtsp"`
	AudioDevice outputNodeXML `xml:"audio_device"`
	File        outputNodeXML `xml:"file"`
	WebRTC      outputNodeXML `xml:"webrtc"`
}

type outputXML = outputFeedXML

type outputNodeXML struct {
	EnabledRaw string `xml:"enabled,attr"`
	Type       string `xml:"type"`
	Host       string `xml:"host"`
	IP         string `xml:"ip"`
	Port       string `xml:"port"`
	URL        string `xml:"url"`
	Path       string `xml:"path"`
	Format     string `xml:"format"`
	Acodec     string `xml:"acodec"`
}

type packageCatalogXML struct {
	Packages []struct {
		ID         string `xml:"id,attr"`
		EnabledRaw string `xml:"enabled,attr"`
	} `xml:"package"`
}

func panelStatePayload(config Config, configPath string, startedAt time.Time, request *http.Request, mediaAvailable bool) (map[string]any, error) {
	summary, err := summaryPayload(config, configPath, startedAt, request, mediaAvailable, true)
	if err != nil {
		return nil, err
	}
	root, _ := loadYAMLMap(configPath)
	return map[string]any{
		"summary":        summary,
		"events":         runtimeEvents(configPath),
		"logs":           logsPayload(configPath, request),
		"datapool":       datapoolPayload(configPath),
		"config":         adminConfigPayload(config, root),
		"last_connected": lastConnectedPayload(request),
	}, nil
}

func publicStatePayload(config Config, configPath string, startedAt time.Time, request *http.Request, auth *AuthManager, mediaAvailable bool) (map[string]any, error) {
	includeFeeds := publicFeedAccess(config) == "public" || (publicFeedAccess(config) == "auth_required" && auth != nil && auth.Authenticated(request))
	summary, err := summaryPayload(config, configPath, startedAt, request, mediaAvailable, false)
	if err != nil {
		return nil, err
	}
	if !includeFeeds {
		summary["feeds"] = []any{}
	}
	if publicAlertsArchiveAccess(config) == "public" {
		if archive, err := publicAlertsArchivePayload(configPath); err == nil {
			summary["alerts_archive_data"] = archive
			summary["alerts"] = archive
		}
	}
	return map[string]any{"summary": summary}, nil
}

func summaryPayload(config Config, configPath string, startedAt time.Time, request *http.Request, mediaAvailable bool, admin bool) (map[string]any, error) {
	feeds, err := loadFeedSummaries(configPath)
	if err != nil {
		return nil, err
	}
	enabled := 0
	for _, feed := range feeds {
		if value, _ := feed["enabled"].(bool); value {
			enabled++
		}
	}
	webrtcEnabled := publicWebRTCAvailable(config, feeds)
	summary := map[string]any{
		"name":               siteName(config),
		"hostname":           hostname(),
		"ip_address":         serverIP(request),
		"operator":           displayText(config.Operator.OperatorName),
		"on_air_name":        displayText(config.Operator.OnAirName),
		"version":            fallbackText(config.Version, "dev"),
		"git_commit":         gitCommit(),
		"os":                 runtime.GOOS,
		"architecture":       runtime.GOARCH,
		"feed_count":         len(feeds),
		"enabled_feed_count": enabled,
		"uptime_seconds":     time.Since(startedAt).Seconds(),
		"feeds":              feeds,
		"feeds_access":       publicFeedAccess(config),
		"webrtc_enabled":     webrtcEnabled,
		"media_available":    mediaAvailable,
		"alerts_archive":     publicAlertsArchiveAccess(config),
		"admin_url":          adminURL(config, request),
		"tls":                tlsStatePayload(config, request),
		"capabilities": map[string]any{
			"public_alerts": publicAlertsArchiveAccess(config) == "public",
			"webrtc":        webrtcEnabled,
		},
	}
	for key, value := range WebRTCAudioCapabilities() {
		summary["capabilities"].(map[string]any)[key] = value
	}
	if admin {
		summary["data_pool_key_count"] = len(readJSONMap(resolveConfigPath(configPath, "runtime/state/dataPool.json")))
		return summary, nil
	}
	summary["feeds"] = publicFeedSummaries(feeds)
	return summary, nil
}

func loadFeedSummaries(configPath string) ([]map[string]any, error) {
	root, err := loadYAMLMap(configPath)
	if err != nil {
		return nil, err
	}
	parsed, err := loadFeedsXML(configPath, root)
	if err != nil {
		return nil, err
	}
	outputs, _ := loadOutputsXML(configPath, root)
	forecastNames := loadForecastRegionNames(resolveConfigPath(configPath, "managed/csv/FORECAST_LOCATIONS.csv"))
	clcNames := loadCLCNames(resolveConfigPath(configPath, "managed/csv/CLC_Base_Zone.csv"))
	nwsFIPSNames := loadNWSFIPSNames(resolveConfigPath(configPath, "managed/csv/NWS_ZONE_COUNTY_CORRELATION.csv"))
	queueItems, _ := loadAlertQueueItems(configPath)
	out := make([]map[string]any, 0, len(parsed.Feeds))
	for _, feed := range parsed.Feeds {
		if strings.TrimSpace(feed.ID) == "" {
			continue
		}
		out = append(out, feedSummary(feed, outputs[feed.ID], forecastNames, clcNames, nwsFIPSNames, queueItems))
	}
	sort.SliceStable(out, func(i, j int) bool {
		return fmt.Sprint(out[i]["id"]) < fmt.Sprint(out[j]["id"])
	})
	return out, nil
}

func feedSummary(feed feedXML, outputs outputXML, forecastNames map[string]string, clcNames map[string]string, nwsFIPSNames map[string]string, queueItems []sameQueueItem) map[string]any {
	station := stationTransmitter(feed)
	regions := coverageRegionPayloads(feed, forecastNames, clcNames)
	clcCodes := feedCoverageCodes(feed, clcNames)
	allLocations := feedCoversAllLocations(feed)
	sameLocations := feedSameLocations(feed, clcNames, nwsFIPSNames)
	outputLabels := outputLabels(outputs)
	webrtcEnabled := xmlBool(outputs.WebRTC.EnabledRaw, false)
	queueDepth, recentQueue, latestQueue := alertQueueState(queueItems, feed.ID)
	runtime := map[string]any{
		"now_playing": "Idle",
	}
	if latestQueue != nil {
		runtime["last_alert_event"] = latestQueue.Event
		runtime["last_alert_severity"] = queueSeverity(latestQueue.Status)
		runtime["last_alert_status"] = latestQueue.Status
		runtime["last_alert_header"] = latestQueue.Header
	}
	return map[string]any{
		"id":                  strings.TrimSpace(feed.ID),
		"name":                fallbackText(station.SiteName, feed.ID),
		"enabled":             xmlBool(feed.EnabledRaw, true),
		"timezone":            fallbackText(feed.Timezone, "Local"),
		"languages":           feedLanguages(feed),
		"location_count":      feedLocationCount(feed),
		"clc_codes":           sortedKeys(clcCodes),
		"same_locations":      sameLocations,
		"same_all_locations":  allLocations,
		"coverage_regions":    regions,
		"transmitter":         transmitterPayload(station, feed),
		"transmitters":        transmitterPayloads(feed),
		"outputs":             outputLabels,
		"webrtc_enabled":      webrtcEnabled,
		"http_stream_enabled": webrtcEnabled,
		"alert_queue_depth":   queueDepth,
		"recent_alerts":       recentQueue,
		"playlist_items":      []any{},
		"runtime":             runtime,
	}
}

func publicFeedSummaries(feeds []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(feeds))
	for _, feed := range feeds {
		transmitter, _ := feed["transmitter"].(map[string]any)
		out = append(out, map[string]any{
			"id":                  feed["id"],
			"name":                feed["name"],
			"enabled":             feed["enabled"],
			"runtime":             feed["runtime"],
			"transmitter":         publicTransmitterPayload(transmitter),
			"transmitters":        publicTransmitters(feed["transmitters"]),
			"webrtc_enabled":      feed["webrtc_enabled"],
			"http_stream_enabled": feed["http_stream_enabled"],
		})
	}
	return out
}

func publicTransmitterPayload(transmitter map[string]any) map[string]any {
	if transmitter == nil {
		return map[string]any{}
	}
	out := map[string]any{
		"site_name":     transmitter["site_name"],
		"site_names":    transmitter["site_names"],
		"relationship":  transmitter["relationship"],
		"frequency_mhz": transmitter["frequency_mhz"],
		"gpclk":         transmitter["gpclk"],
		"gpio":          transmitter["gpio"],
		"rds":           transmitter["rds"],
	}
	if callsign := strings.TrimSpace(fmt.Sprint(transmitter["callsign"])); callsign != "" {
		out["callsign"] = callsign
	}
	return out
}

func publicTransmitters(value any) []map[string]any {
	items, _ := value.([]map[string]any)
	if items == nil {
		if raw, ok := value.([]any); ok {
			items = make([]map[string]any, 0, len(raw))
			for _, item := range raw {
				if mapped, ok := item.(map[string]any); ok {
					items = append(items, mapped)
				}
			}
		}
	}
	out := make([]map[string]any, 0, len(items))
	for _, item := range items {
		out = append(out, publicTransmitterPayload(item))
	}
	return out
}

func loadFeedsXML(configPath string, root map[string]any) (feedsXML, error) {
	path := textAt(root, []string{"feeds_file"}, "managed/configs/feeds.xml", 240)
	raw, err := os.ReadFile(resolveConfigPath(configPath, path))
	if err != nil {
		return feedsXML{}, err
	}
	var parsed feedsXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return feedsXML{}, fmt.Errorf("failed to parse feeds XML %s: %w", resolveConfigPath(configPath, path), err)
	}
	return parsed, nil
}

func loadOutputsXML(configPath string, root map[string]any) (map[string]outputXML, error) {
	path := textAt(root, []string{"outputs_file"}, "managed/configs/output.xml", 240)
	raw, err := os.ReadFile(resolveConfigPath(configPath, path))
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]outputXML{}, nil
		}
		return nil, err
	}
	var parsed outputsXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse outputs XML %s: %w", resolveConfigPath(configPath, path), err)
	}
	out := map[string]outputXML{}
	for _, feed := range parsed.Feeds {
		id := strings.TrimSpace(feed.ID)
		if id != "" {
			out[id] = outputXML(feed)
		}
	}
	return out, nil
}

func stationTransmitter(feed feedXML) transmitterXML {
	transmitters := transmitterList(feed)
	for _, transmitter := range transmitters {
		if transmitter.isRelationship("primary") {
			return transmitter
		}
	}
	for _, transmitter := range transmitters {
		if !transmitter.isRelationship("replaces") && !transmitter.isRelationship("ip") &&
			(strings.TrimSpace(transmitter.Callsign) != "" || strings.TrimSpace(transmitter.SiteName) != "") {
			return transmitter
		}
	}
	for _, transmitter := range transmitters {
		if strings.TrimSpace(transmitter.Callsign) != "" || strings.TrimSpace(transmitter.SiteName) != "" {
			return transmitter
		}
	}
	if len(transmitters) > 0 {
		return transmitters[0]
	}
	return transmitterXML{}
}

func receiverPreferredTransmitter(feed feedXML) transmitterXML {
	for _, transmitter := range transmitterList(feed) {
		if transmitter.isRelationship("fm") && transmitter.frequencyText() != "" {
			return transmitter
		}
	}
	return stationTransmitter(feed)
}

func transmitterList(feed feedXML) []transmitterXML {
	out := make([]transmitterXML, 0, len(feed.Transmitter.Transmitters))
	for _, transmitter := range feed.Transmitter.Transmitters {
		if transmitter.empty() {
			continue
		}
		out = append(out, transmitter)
	}
	return out
}

func transmitterPayloads(feed feedXML) []map[string]any {
	transmitters := transmitterList(feed)
	out := make([]map[string]any, 0, len(transmitters))
	siteNames := siteNamesForFeed(feed)
	for _, transmitter := range transmitters {
		payload := transmitterPayload(transmitter, feed)
		payload["site_names"] = siteNames
		out = append(out, payload)
	}
	return out
}

func transmitterPayload(transmitter transmitterXML, feed feedXML) map[string]any {
	frequency, _ := strconv.ParseFloat(transmitter.frequencyText(), 64)
	payload := map[string]any{
		"site_name":     fallbackText(transmitter.SiteName, feed.ID),
		"site_names":    siteNamesForFeed(feed),
		"callsign":      strings.TrimSpace(transmitter.Callsign),
		"relationship":  transmitter.relationship(),
		"host_name":     strings.TrimSpace(transmitter.HostName),
		"frequency_mhz": frequency,
		"gpclk":         transmitter.gpclk(),
		"gpio":          transmitter.gpio(),
		"rds":           transmitter.rdsPayload(feed),
	}
	if transmitter.frequencyText() == "" {
		delete(payload, "frequency_mhz")
	}
	return payload
}

func siteNamesForFeed(feed feedXML) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, transmitter := range transmitterList(feed) {
		if transmitter.isRelationship("ip") {
			continue
		}
		name := strings.TrimSpace(transmitter.SiteName)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	if len(out) == 0 {
		out = append(out, feed.ID)
	}
	return out
}

func (t transmitterXML) empty() bool {
	return strings.TrimSpace(t.SiteName) == "" &&
		strings.TrimSpace(t.Callsign) == "" &&
		strings.TrimSpace(t.Relationship) == "" &&
		strings.TrimSpace(t.HostName) == "" &&
		t.frequencyText() == ""
}

func (t transmitterXML) relationship() string {
	relationship := strings.ToLower(strings.TrimSpace(t.Relationship))
	if relationship == "secondary/repeater" {
		return "secondary"
	}
	if relationship == "" {
		return "unknown"
	}
	return relationship
}

func (t transmitterXML) isRelationship(relationship string) bool {
	current := t.relationship()
	wanted := strings.ToLower(strings.TrimSpace(relationship))
	return current == wanted || (wanted == "repeater" && current == "secondary")
}

func (t transmitterXML) frequencyText() string {
	return strings.TrimSpace(t.FrequencyMHz.Value)
}

func (t transmitterXML) gpclk() string {
	configured := strings.TrimSpace(t.FrequencyMHz.GPCLK)
	if configured != "" {
		return configured
	}
	if t.isRelationship("fm") {
		return "2"
	}
	return "0"
}

func (t transmitterXML) gpio() string {
	if configured := strings.TrimSpace(t.FrequencyMHz.GPIO); configured != "" {
		return configured
	}
	switch t.gpclk() {
	case "0":
		return "4"
	case "1":
		return "5"
	case "2":
		return "6"
	default:
		return ""
	}
}

func (t transmitterXML) rdsPayload(feed feedXML) map[string]any {
	enabled := xmlBool(t.RDS.EnabledRaw, t.isRelationship("fm"))
	pi := strings.ToUpper(strings.TrimSpace(t.RDS.PI))
	if pi == "" {
		pi = defaultRDSPI(feed.ID)
	}
	ps := shortRDSText(t.RDS.PS, "HAZE WX")
	site := fallbackText(t.SiteName, feed.ID)
	frequency := strings.TrimSpace(t.frequencyText())
	rt := strings.TrimSpace(t.RDS.RT)
	if rt == "" {
		parts := []string{"Haze Weather Radio", site}
		if callsign := strings.TrimSpace(t.Callsign); callsign != "" {
			parts = append(parts, callsign)
		}
		if frequency != "" {
			parts = append(parts, frequency+" MHz")
		}
		rt = strings.Join(parts, " - ")
	}
	return map[string]any{
		"enabled": enabled,
		"pi":      truncate(pi, 4),
		"ps":      ps,
		"rt":      truncate(rt, 64),
		"pty":     fallbackText(t.RDS.PTY, "3"),
		"tp":      fallbackText(t.RDS.TP, "0"),
		"af":      normalizedRDSAF(t.RDS.AF),
	}
}

func defaultRDSPI(feedID string) string {
	sum := sha1.Sum([]byte(fallbackText(feedID, "haze")))
	pi := strings.ToUpper(hex.EncodeToString(sum[:2]))
	if pi == "0000" {
		return "48A5"
	}
	return pi
}

func shortRDSText(value string, fallback string) string {
	text := strings.ToUpper(fallbackText(value, fallback))
	builder := strings.Builder{}
	for _, r := range text {
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == ' ' {
			builder.WriteRune(r)
		}
	}
	return truncate(strings.TrimSpace(builder.String()), 8)
}

func normalizedRDSAF(values []string) []string {
	out := []string{}
	for _, value := range values {
		text := strings.TrimSpace(value)
		if text == "" {
			continue
		}
		frequency, err := strconv.ParseFloat(text, 64)
		if err != nil || frequency < 87.6 || frequency > 107.9 {
			continue
		}
		out = append(out, strconv.FormatFloat(frequency, 'f', 1, 64))
	}
	return uniqueStrings(out)
}

func publicWebRTCAvailable(config Config, feeds []map[string]any) bool {
	if !config.Webpanel.Public.Feeds.WebRTC.Enabled {
		return false
	}
	if publicFeedAccess(config) == "disabled" {
		return false
	}
	for _, feed := range feeds {
		if enabled, _ := feed["enabled"].(bool); !enabled {
			continue
		}
		if webrtc, _ := feed["webrtc_enabled"].(bool); webrtc {
			return true
		}
	}
	return false
}

func publicFeedAccess(config Config) string {
	access := strings.ToLower(strings.TrimSpace(config.Webpanel.Public.Feeds.Access))
	switch access {
	case "public", "auth_required":
		return access
	default:
		return "disabled"
	}
}

func publicAlertsArchiveAccess(config Config) string {
	access := strings.ToLower(strings.TrimSpace(config.Webpanel.Public.AlertsArchive.Access))
	if access == "public" {
		return "public"
	}
	return "disabled"
}

func outputLabels(output outputXML) []string {
	items := []string{}
	addOutput := func(name string, node outputNodeXML) {
		if xmlBool(node.EnabledRaw, false) {
			items = append(items, name)
		}
	}
	addOutput("webrtc", output.WebRTC)
	addOutput("udp", output.UDP)
	addOutput("rtp", output.RTP)
	addOutput("stream", output.Stream)
	addOutput("rtmp", output.RTMP)
	addOutput("srt", output.SRT)
	addOutput("rtsp", output.RTSP)
	addOutput("audio_device", output.AudioDevice)
	addOutput("file", output.File)
	return items
}

func coverageRegionPayloads(feed feedXML, forecastNames map[string]string, clcNames map[string]string) []map[string]any {
	out := make([]map[string]any, 0, len(feed.Locations.Coverage.Regions))
	for _, region := range feed.Locations.Coverage.Regions {
		id := strings.TrimSpace(region.ID)
		if id == "" {
			continue
		}
		name := fallbackText(region.Name, fallbackText(forecastNames[id], fallbackText(clcNames[id], id)))
		subregionIDs := expandedCoverageSubregionIDs(region, clcNames)
		subregions := make([]map[string]any, 0, len(subregionIDs))
		for _, subID := range subregionIDs {
			if subID == "" {
				continue
			}
			subregions = append(subregions, map[string]any{
				"id":   subID,
				"name": fallbackText(clcNames[subID], subID),
			})
		}
		out = append(out, map[string]any{
			"id":              id,
			"name":            name,
			"source":          strings.TrimSpace(region.Source),
			"derive_forecast": strings.TrimSpace(region.DeriveForecast),
			"subregions":      subregions,
		})
	}
	return out
}

func feedCoverageCodes(feed feedXML, clcNames map[string]string) map[string]struct{} {
	codes := map[string]struct{}{}
	for _, region := range feed.Locations.Coverage.Regions {
		addCode(codes, region.ID)
		for _, subregion := range expandedCoverageSubregionIDs(region, clcNames) {
			addCode(codes, subregion)
		}
	}
	return codes
}

func feedSameLocations(feed feedXML, clcNames map[string]string, nwsFIPSNames map[string]string) []string {
	if feedCoversAllLocations(feed) {
		return allCanadaUSSameLocations(clcNames, nwsFIPSNames)
	}
	codes := map[string]struct{}{}
	for _, region := range feed.Locations.Coverage.Regions {
		addCode(codes, region.ID)
		for _, subregion := range expandedCoverageSubregionIDs(region, clcNames) {
			addCode(codes, subregion)
		}
	}
	return sortedKeys(codes)
}

func feedCoversAllLocations(feed feedXML) bool {
	if len(feed.Locations.Coverage.Regions) > 0 {
		return false
	}
	return xmlBool(feed.Alerts.CapCP.EnabledRaw, true) || xmlBool(feed.Alerts.NWSCAP.EnabledRaw, false)
}

func allCanadaUSSameLocations(clcNames map[string]string, nwsFIPSNames map[string]string) []string {
	codes := map[string]struct{}{"000000": {}}
	for code := range clcNames {
		addCode(codes, cleanLocationCode(code))
	}
	for code := range nwsFIPSNames {
		addCode(codes, cleanLocationCode(code))
	}
	out := sortedKeys(codes)
	if len(out) == 0 || out[0] == "000000" {
		return out
	}
	return append([]string{"000000"}, removeString(out, "000000")...)
}

func removeString(values []string, unwanted string) []string {
	out := values[:0]
	for _, value := range values {
		if value != unwanted {
			out = append(out, value)
		}
	}
	return out
}

func expandedCoverageSubregionIDs(region coverageRegionXML, clcNames map[string]string) []string {
	codes := map[string]struct{}{}
	for _, subregion := range region.Subregions {
		addCode(codes, subregion.ID)
	}
	for _, subregion := range alertWildcardSubregions(region.ID, clcNames) {
		addCode(codes, subregion)
	}
	return sortedKeys(codes)
}

func alertWildcardSubregions(raw string, clcNames map[string]string) []string {
	code := cleanLocationCode(raw)
	if len(code) != 6 || !strings.HasSuffix(code, "00") {
		return nil
	}
	prefix := code[:4]
	out := []string{}
	for candidate := range clcNames {
		candidateCode := cleanLocationCode(candidate)
		if candidateCode == "" || candidateCode == code {
			continue
		}
		if strings.HasPrefix(candidateCode, prefix) {
			out = append(out, candidateCode)
		}
	}
	sort.Strings(out)
	return out
}

func addCode(codes map[string]struct{}, raw string) {
	code := strings.TrimSpace(raw)
	if code != "" {
		codes[code] = struct{}{}
	}
}

func feedLanguages(feed feedXML) []string {
	out := []string{}
	for _, lang := range feed.Languages.Langs {
		if code := strings.TrimSpace(lang.Code); code != "" {
			out = append(out, code)
		}
	}
	if len(out) == 0 {
		out = append(out, "en-CA")
	}
	return uniqueStrings(out)
}

func feedLocationCount(feed feedXML) int {
	return len(feed.Locations.ObservationLocations.Locations) +
		len(feed.Locations.AirQualityLocations.Locations) +
		len(feed.Locations.ClimateLocations.Locations)
}

func loadForecastRegionNames(path string) map[string]string {
	if snap, ok := loadLocationSnapshotFromCSVPath(path); ok {
		out := map[string]string{}
		for _, place := range snap.PlacesBySource("forecast") {
			out[place.Code] = place.Name
		}
		if len(out) > 0 {
			return out
		}
	}
	return loadCSVNames(path, 0, 1)
}

func loadCLCNames(path string) map[string]string {
	if snap, ok := loadLocationSnapshotFromCSVPath(path); ok {
		out := map[string]string{}
		for _, place := range snap.PlacesBySource("clc") {
			out[place.Code] = place.Name
		}
		if len(out) > 0 {
			return out
		}
	}
	return loadCSVNames(path, 0, 2)
}

func loadNWSFIPSNames(path string) map[string]string {
	if snap, ok := loadLocationSnapshotFromCSVPath(path); ok {
		out := map[string]string{}
		for _, source := range []string{"nws_same", "nws_zone", "nws_marine_same", "nws_marine_zone"} {
			for _, place := range snap.PlacesBySource(source) {
				out[place.Code] = place.Name
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return map[string]string{}
	}
	out := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, "|")
		if len(parts) < 7 || strings.EqualFold(strings.TrimSpace(parts[0]), "STATE") {
			continue
		}
		code := cleanLocationCode(parts[6])
		if code == "" {
			continue
		}
		name := strings.TrimSpace(parts[5])
		if name == "" {
			name = strings.TrimSpace(parts[3])
		}
		if state := strings.TrimSpace(parts[0]); state != "" && name != "" {
			name = name + ", " + strings.ToUpper(state)
		}
		if name != "" {
			out[code] = name
		}
	}
	return out
}

func loadNWSMarineNames(path string) map[string]string {
	if snap, ok := loadLocationSnapshotFromCSVPath(path); ok {
		out := map[string]string{}
		for _, source := range []string{"nws_marine_same", "nws_marine_zone"} {
			for _, place := range snap.PlacesBySource(source) {
				out[place.Code] = place.Name
			}
		}
		if len(out) > 0 {
			return out
		}
	}
	file, err := os.Open(filepath.Clean(path))
	if err != nil {
		return map[string]string{}
	}
	defer file.Close()
	reader := csv.NewReader(file)
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil || len(rows) == 0 {
		return map[string]string{}
	}
	header := map[string]int{}
	for i, value := range rows[0] {
		header[strings.ToLower(strings.TrimSpace(value))] = i
	}
	sameIndex := csvHeaderIndex(header, "same_code", "same", "ssnum")
	zoneIndex := csvHeaderIndex(header, "zone_ugc", "zone", "ugc")
	nameIndex := csvHeaderIndex(header, "name", "zonename", "zone_name")
	if sameIndex < 0 || nameIndex < 0 {
		return map[string]string{}
	}
	out := map[string]string{}
	for _, row := range rows[1:] {
		name := csvCell(row, nameIndex)
		if name == "" {
			continue
		}
		for _, raw := range []string{csvCell(row, sameIndex), csvCell(row, zoneIndex)} {
			code := cleanLocationCode(raw)
			if code == "" {
				code = strings.ToUpper(strings.TrimSpace(raw))
			}
			if code != "" {
				out[code] = name
			}
		}
	}
	return out
}

func loadLocationSnapshotFromCSVPath(path string) (locationdb.Snapshot, bool) {
	path = filepath.Clean(path)
	baseDir := filepath.Dir(filepath.Dir(filepath.Dir(path)))
	if strings.TrimSpace(baseDir) == "." || strings.TrimSpace(baseDir) == "" {
		return locationdb.Snapshot{}, false
	}
	return locationdb.Load(baseDir)
}

func csvHeaderIndex(header map[string]int, keys ...string) int {
	for _, key := range keys {
		if index, ok := header[strings.ToLower(strings.TrimSpace(key))]; ok {
			return index
		}
	}
	return -1
}

func csvCell(row []string, index int) string {
	if index < 0 || index >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[index])
}

func loadCSVNames(path string, codeIndex int, nameIndex int) map[string]string {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return map[string]string{}
	}
	lines := strings.Split(string(raw), "\n")
	out := map[string]string{}
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := splitCSVLine(line)
		if i == 0 && len(parts) > codeIndex && strings.EqualFold(strings.TrimSpace(parts[codeIndex]), "code") {
			continue
		}
		if len(parts) <= codeIndex || len(parts) <= nameIndex {
			continue
		}
		code := strings.TrimSpace(parts[codeIndex])
		name := strings.TrimSpace(parts[nameIndex])
		if code != "" && name != "" {
			out[code] = name
		}
	}
	return out
}

func splitCSVLine(line string) []string {
	parts := []string{}
	var builder strings.Builder
	quoted := false
	for _, r := range line {
		switch r {
		case '"':
			quoted = !quoted
		case ',':
			if !quoted {
				parts = append(parts, builder.String())
				builder.Reset()
				continue
			}
			builder.WriteRune(r)
		default:
			builder.WriteRune(r)
		}
	}
	parts = append(parts, builder.String())
	return parts
}

func runtimeEvents(configPath string) []map[string]any {
	lines := logLines(resolveConfigPath(configPath, "logs/haze.log"), 12)
	out := make([]map[string]any, 0, len(lines))
	for _, line := range lines {
		out = append(out, map[string]any{
			"kind":      "log",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"message":   line,
		})
	}
	return out
}

func logsPayload(configPath string, request *http.Request) map[string]any {
	source := "app"
	if request != nil {
		source = fallbackText(request.URL.Query().Get("source"), "app")
	}
	path := resolveConfigPath(configPath, "logs/haze.log")
	return map[string]any{"source": source, "lines": logLines(path, 120)}
}

func logLines(path string, limit int) []string {
	raw, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return []string{}
	}
	lines := strings.Split(strings.ReplaceAll(string(raw), "\r\n", "\n"), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	if limit > 0 && len(lines) > limit {
		lines = lines[len(lines)-limit:]
	}
	return lines
}

func datapoolPayload(configPath string) map[string]any {
	for _, rel := range []string{"runtime/state/dataPool.json", "managed/runtime/dataPool.json"} {
		if payload := readJSONMap(resolveConfigPath(configPath, rel)); len(payload) > 0 {
			return payload
		}
	}
	return map[string]any{}
}

func adminConfigPayload(config Config, root map[string]any) map[string]any {
	return map[string]any{
		"version":  config.Version,
		"site":     siteName(config),
		"webpanel": daemonSettingsView(root)["webpanel"],
		"services": daemonSettingsView(root)["services"],
		"same": map[string]any{
			"sender": strings.TrimSpace(os.Getenv("SAME_ID")),
		},
	}
}

func lastConnectedPayload(request *http.Request) map[string]any {
	ip := "unknown"
	if request != nil {
		ip = strings.TrimSpace(request.Header.Get("X-Forwarded-For"))
		if strings.Contains(ip, ",") {
			ip = strings.TrimSpace(strings.Split(ip, ",")[0])
		}
		if ip == "" {
			ip = strings.TrimSpace(request.RemoteAddr)
			if host, _, err := strings.Cut(ip, ":"); err {
				ip = host
			}
		}
	}
	return map[string]any{"ip": fallbackText(ip, "unknown"), "at": time.Now().UTC().Format(time.RFC3339)}
}

func tlsStatePayload(config Config, request *http.Request) map[string]any {
	host := ""
	if request != nil {
		host = request.Host
	}
	https := request != nil && requestIsHTTPS(request)
	actualDomain := host != "" && !strings.HasPrefix(host, "127.") && !strings.HasPrefix(host, "localhost") && !strings.Contains(host, "[::1]")
	enabled := config.Webpanel.TLS.Enabled
	needsSetup := actualDomain && !https && !enabled
	message := "Panel is using local HTTP."
	if https {
		message = "HTTPS is active."
	} else if needsSetup {
		message = "This looks like a real domain; HTTPS can be enabled from TLS settings."
	}
	return map[string]any{
		"enabled":            enabled,
		"https":              https,
		"host":               host,
		"actual_domain":      actualDomain,
		"needs_setup":        needsSetup,
		"configured_domains": config.Webpanel.TLS.Domains,
		"message":            message,
	}
}

func adminURL(config Config, request *http.Request) string {
	host := ""
	if request != nil {
		host = request.Host
	}
	if config.Webpanel.Admin.Port > 0 {
		hostPart := config.Webpanel.Admin.Host
		if hostPart == "" || hostPart == "0.0.0.0" {
			if request != nil {
				hostPart = request.Host
				if strings.Contains(hostPart, ":") {
					hostPart, _, _ = strings.Cut(hostPart, ":")
				}
			}
			if hostPart == "" {
				hostPart = "localhost"
			}
		}
		host = fmt.Sprintf("%s:%d", hostPart, config.Webpanel.Admin.Port)
	}
	if host == "" {
		host = "localhost:8086"
	}
	return "http://" + host + "/admin"
}

func publicAlertsArchivePayload(configPath string) (map[string]any, error) {
	payload, err := alertsArchivePayload(configPath)
	if err != nil {
		return nil, err
	}
	byFeed := map[string]any{}
	switch accepted := payload["accepted_by_feed"].(type) {
	case map[string][]map[string]any:
		for feedID, records := range accepted {
			byFeed[feedID] = records
		}
	case map[string]any:
		for feedID, records := range accepted {
			byFeed[feedID] = records
		}
	case []map[string]any:
		for _, record := range accepted {
			feedID := fallbackText(fmt.Sprint(record["feed_id"]), "unknown")
			byFeed[feedID] = appendAny(byFeed[feedID], record)
		}
	case []any:
		for _, item := range accepted {
			record, ok := item.(map[string]any)
			if !ok {
				continue
			}
			feedID := fallbackText(fmt.Sprint(record["feed_id"]), "unknown")
			byFeed[feedID] = appendAny(byFeed[feedID], record)
		}
	}
	return map[string]any{
		"by_feed":  byFeed,
		"rejected": payload["rejected"],
		"expired":  payload["expired"],
	}, nil
}

func loadPackageIDs(configPath string) ([]string, error) {
	root, err := loadYAMLMap(configPath)
	if err != nil {
		return nil, err
	}
	path := textAt(root, []string{"packages_file"}, "managed/configs/packages.xml", 240)
	raw, err := os.ReadFile(resolveConfigPath(configPath, path))
	if err != nil {
		return nil, err
	}
	var parsed packageCatalogXML
	if err := xml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse packages XML: %w", err)
	}
	out := []string{}
	for _, item := range parsed.Packages {
		id := strings.TrimSpace(item.ID)
		if id == "" || !xmlBool(item.EnabledRaw, true) {
			continue
		}
		out = append(out, id)
	}
	return uniqueStrings(out), nil
}

func loadSAMEMapping(configPath string) (map[string]any, error) {
	for _, rel := range []string{"managed/sameMapping.json", "managed/same_mapping.json"} {
		path := resolveConfigPath(configPath, rel)
		raw, err := os.ReadFile(path)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		var payload map[string]any
		if err := jsonUnmarshalMap(raw, &payload); err != nil {
			return nil, err
		}
		return payload, nil
	}
	return map[string]any{"eas": map[string]any{}, "naadsToEas": map[string]any{}}, nil
}

func loadLocationNames(configPath string) (map[string]any, error) {
	names := map[string]any{}
	if snap, ok := locationdb.Load(filepath.Dir(filepath.Clean(configPath))); ok {
		for code, name := range snap.Labels() {
			names[code] = name
		}
		names["000000"] = "All areas"
		return names, nil
	}
	for code, name := range loadForecastRegionNames(resolveConfigPath(configPath, "managed/csv/FORECAST_LOCATIONS.csv")) {
		names[code] = name
	}
	for code, name := range loadCLCNames(resolveConfigPath(configPath, "managed/csv/CLC_Base_Zone.csv")) {
		if clean := cleanLocationCode(code); clean != "" {
			names[clean] = name
		}
	}
	for code, name := range loadNWSFIPSNames(resolveConfigPath(configPath, "managed/csv/NWS_ZONE_COUNTY_CORRELATION.csv")) {
		names[code] = name
	}
	for code, name := range loadNWSMarineNames(resolveConfigPath(configPath, "managed/csv/NWS_MARINE_ZONES.csv")) {
		names[code] = name
	}
	names["000000"] = "All areas"
	return names, nil
}

func cleanLocationCode(raw string) string {
	text := strings.ToUpper(strings.TrimSpace(raw))
	text = strings.Trim(text, "\"' ")
	builder := strings.Builder{}
	for _, r := range text {
		if r >= '0' && r <= '9' {
			builder.WriteRune(r)
		}
	}
	cleaned := builder.String()
	if len(cleaned) == 5 {
		return "0" + cleaned
	}
	if len(cleaned) != 6 {
		return ""
	}
	return cleaned
}

func resolveConfigPath(configPath string, configured string) string {
	configured = strings.TrimSpace(configured)
	if configured == "" {
		configured = "."
	}
	if filepath.IsAbs(configured) {
		return filepath.Clean(configured)
	}
	base := filepath.Dir(filepath.Clean(configPath))
	return filepath.Clean(filepath.Join(base, configured))
}

func fallbackText(value string, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return strings.TrimSpace(value)
	}
	return strings.TrimSpace(fallback)
}

func xmlBool(raw string, fallback bool) bool {
	text := strings.ToLower(strings.TrimSpace(raw))
	switch text {
	case "1", "true", "yes", "on", "enabled":
		return true
	case "0", "false", "no", "off", "disabled":
		return false
	default:
		return fallback
	}
}

func parseIntText(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil {
		return fallback
	}
	return value
}

func sortedKeys(values map[string]struct{}) []string {
	out := make([]string, 0, len(values))
	for key := range values {
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func uniqueStrings(values []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func hostname() string {
	name, err := os.Hostname()
	if err != nil || strings.TrimSpace(name) == "" {
		return "unknown"
	}
	return name
}

func requestIP(request *http.Request) string {
	ip := "unknown"
	if request != nil {
		ip = strings.TrimSpace(request.Header.Get("X-Forwarded-For"))
		if strings.Contains(ip, ",") {
			ip = strings.TrimSpace(strings.Split(ip, ",")[0])
		}
		if ip == "" {
			ip = strings.TrimSpace(request.Header.Get("X-Real-IP"))
		}
		if ip == "" {
			ip = strings.TrimSpace(request.RemoteAddr)
			if host, _, found := strings.Cut(ip, ":"); found {
				ip = host
			}
		}
	}
	return fallbackText(ip, "unknown")
}

func serverIP(request *http.Request) string {
	if request != nil {
		if addr, ok := request.Context().Value(http.LocalAddrContextKey).(net.Addr); ok && addr != nil {
			if ip := addrIP(addr.String()); usableServerIP(ip) {
				return ip
			}
		}
		if ip := addrIP(request.Host); usableServerIP(ip) {
			return ip
		}
	}
	if ip := interfaceServerIP(); ip != "" {
		return ip
	}
	return "unknown"
}

func addrIP(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if host, _, err := net.SplitHostPort(value); err == nil {
		value = host
	} else if strings.Count(value, ":") == 1 {
		if host, _, found := strings.Cut(value, ":"); found {
			value = host
		}
	}
	return strings.Trim(value, "[]")
}

func usableServerIP(value string) bool {
	ip := net.ParseIP(strings.TrimSpace(value))
	if ip == nil {
		return false
	}
	return !ip.IsLoopback() && !ip.IsUnspecified() && !ip.IsMulticast()
}

func interfaceServerIP() string {
	type candidate struct {
		ip    string
		score int
	}
	candidates := []candidate{}
	interfaces, err := net.Interfaces()
	if err != nil {
		return ""
	}
	for _, iface := range interfaces {
		if iface.Flags&net.FlagUp == 0 || iface.Flags&net.FlagLoopback != 0 {
			continue
		}
		addrs, err := iface.Addrs()
		if err != nil {
			continue
		}
		for _, addr := range addrs {
			ip := interfaceAddrIP(addr)
			if ip == nil || !usableServerIP(ip.String()) || ip.IsLinkLocalUnicast() {
				continue
			}
			score := 1
			if ip4 := ip.To4(); ip4 != nil {
				score = 3
				if ip4.IsPrivate() {
					score = 4
				}
			} else if ip.IsPrivate() {
				score = 2
			}
			candidates = append(candidates, candidate{ip: ip.String(), score: score})
		}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})
	if len(candidates) == 0 {
		return ""
	}
	return candidates[0].ip
}

func interfaceAddrIP(addr net.Addr) net.IP {
	switch value := addr.(type) {
	case *net.IPNet:
		return value.IP
	case *net.IPAddr:
		return value.IP
	default:
		return net.ParseIP(addrIP(value.String()))
	}
}

func gitCommit() string {
	for _, name := range []string{"HAZE_GIT_COMMIT", "GIT_COMMIT"} {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	if strings.TrimSpace(BuildGitCommit) != "" {
		return strings.TrimSpace(BuildGitCommit)
	}
	return "unknown"
}

func truncate(value string, limit int) string {
	value = strings.TrimSpace(value)
	if limit <= 0 || len(value) <= limit {
		return value
	}
	return value[:limit]
}

func queueSeverity(status string) string {
	status = strings.TrimSpace(status)
	if status == "" {
		return "Pending"
	}
	return strings.ToUpper(status[:1]) + strings.ToLower(status[1:])
}

func appendAny(value any, item any) []any {
	switch typed := value.(type) {
	case []any:
		return append(typed, item)
	case []map[string]any:
		out := make([]any, 0, len(typed)+1)
		for _, record := range typed {
			out = append(out, record)
		}
		return append(out, item)
	default:
		return []any{item}
	}
}

func jsonUnmarshalMap(raw []byte, target *map[string]any) error {
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.UseNumber()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}
	if *target == nil {
		*target = map[string]any{}
	}
	return nil
}
