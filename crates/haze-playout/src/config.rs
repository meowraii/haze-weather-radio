#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone)]
pub(crate) struct LoadedConfig {
    pub(crate) root: RootConfig,
    pub(crate) feeds: Vec<FeedConfig>,
    pub(crate) packages: BTreeMap<String, PackageConfig>,
    pub(crate) base_dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct RootConfig {
    #[serde(default = "default_feeds_file")]
    pub(crate) feeds_file: String,
    #[serde(default = "default_outputs_file")]
    pub(crate) outputs_file: String,
    #[serde(default)]
    pub(crate) operator: OperatorConfig,
    #[serde(default)]
    pub(crate) playout: PlayoutConfig,
    #[serde(default)]
    pub(crate) services: ServicesConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ServicesConfig {
    #[serde(default)]
    pub(crate) rust: RustServicesConfig,
    #[serde(default)]
    pub(crate) go: GoServicesConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RustServicesConfig {
    #[serde(default)]
    pub(crate) playout: RustPlayoutConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RustPlayoutConfig {
    #[serde(default)]
    pub(crate) enabled: bool,
    #[serde(default)]
    pub(crate) ffmpeg: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct GoServicesConfig {
    #[serde(default)]
    pub(crate) product_render: ServiceToggle,
    #[serde(default)]
    pub(crate) playlist: ServiceToggle,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ServiceToggle {
    #[serde(default)]
    pub(crate) enabled: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OperatorConfig {
    #[serde(default)]
    pub(crate) on_air_name: serde_yaml::Value,
    #[serde(default)]
    pub(crate) operator_name: serde_yaml::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PlayoutConfig {
    #[serde(default = "default_sample_rate")]
    pub(crate) sample_rate: u32,
    #[serde(default = "default_channels")]
    pub(crate) channels: u16,
    #[serde(default)]
    pub(crate) playlist_order: Vec<String>,
    #[serde(default)]
    pub(crate) pacing: PacingConfig,
}

impl Default for PlayoutConfig {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            channels: default_channels(),
            playlist_order: Vec::new(),
            pacing: PacingConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct PacingConfig {
    #[serde(default = "default_gap_seconds")]
    pub(crate) package_gap_s: f64,
}

impl Default for PacingConfig {
    fn default() -> Self {
        Self {
            package_gap_s: default_gap_seconds(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FeedsXml {
    #[serde(rename = "feed", default)]
    feeds: Vec<FeedConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct OutputsXml {
    #[serde(rename = "feed", default)]
    feeds: Vec<FeedOutputConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FeedOutputConfig {
    #[serde(rename = "@id", default)]
    id: String,
    #[serde(default)]
    webrtc: OutputNodeConfig,
    #[serde(default)]
    stream: OutputNodeConfig,
    #[serde(default)]
    udp: OutputNodeConfig,
    #[serde(default)]
    rtp: OutputNodeConfig,
    #[serde(default)]
    rtmp: OutputNodeConfig,
    #[serde(default)]
    srt: OutputNodeConfig,
    #[serde(default)]
    rtsp: OutputNodeConfig,
    #[serde(default)]
    audio_device: OutputNodeConfig,
    #[serde(default)]
    file: OutputNodeConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FeedConfig {
    #[serde(rename = "@id", default)]
    pub(crate) id: String,
    #[serde(rename = "@enabled", default)]
    pub(crate) enabled: Option<String>,
    #[serde(rename = "@timezone", default)]
    pub(crate) timezone: String,
    #[serde(default)]
    pub(crate) playout: FeedPlayoutConfig,
    #[serde(default)]
    pub(crate) alerts: Option<FeedAlertsConfig>,
    #[serde(default)]
    pub(crate) locations: FeedLocationsConfig,
    #[serde(default)]
    pub(crate) languages: LanguagesConfig,
    #[serde(default)]
    pub(crate) description: DescriptionConfig,
    #[serde(default)]
    pub(crate) transmitter_metadata: TransmitterMetadataConfig,
    #[serde(skip)]
    pub(crate) output: OutputConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FeedPlayoutConfig {
    #[serde(rename = "@routine", default)]
    pub(crate) routine: Option<String>,
    #[serde(rename = "@same", default)]
    pub(crate) same: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FeedAlertsConfig {
    #[serde(default)]
    pub(crate) cap_cp: FeedAlertProviderConfig,
    #[serde(default)]
    pub(crate) nws_cap: FeedAlertProviderConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FeedAlertProviderConfig {
    #[serde(rename = "@enabled", default)]
    pub(crate) enabled: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FeedLocationsConfig {
    #[serde(default)]
    pub(crate) coverage: FeedCoverageConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FeedCoverageConfig {
    #[serde(rename = "region", default)]
    pub(crate) regions: Vec<FeedCoverageRegionConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct FeedCoverageRegionConfig {
    #[serde(rename = "@id", default)]
    pub(crate) id: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct LanguagesConfig {
    #[serde(rename = "lang", default)]
    pub(crate) langs: Vec<LanguageConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct LanguageConfig {
    #[serde(rename = "@code", default)]
    pub(crate) code: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DescriptionConfig {
    #[serde(rename = "lang", default)]
    pub(crate) langs: Vec<DescriptionLangConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct DescriptionLangConfig {
    #[serde(rename = "@code", default)]
    pub(crate) code: String,
    #[serde(rename = "@text", default)]
    pub(crate) text: String,
    #[serde(rename = "@suffix", default)]
    pub(crate) suffix: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TransmitterMetadataConfig {
    #[serde(rename = "transmitter", default)]
    pub(crate) transmitters: Vec<TransmitterConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TransmitterConfig {
    #[serde(default)]
    pub(crate) site_name: String,
    #[serde(default)]
    pub(crate) callsign: String,
    #[serde(default)]
    pub(crate) relationship: String,
    #[serde(default)]
    pub(crate) host_name: String,
    #[serde(default)]
    pub(crate) frequency_mhz: TransmitterFrequencyConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TransmitterFrequencyConfig {
    #[serde(rename = "@gpclk", default)]
    pub(crate) gpclk: String,
    #[serde(rename = "@gpio", default)]
    pub(crate) gpio: String,
    #[serde(rename = "$text", default)]
    pub(crate) value: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OutputConfig {
    #[serde(default)]
    pub(crate) webrtc: OutputNodeConfig,
    #[serde(default)]
    pub(crate) stream: OutputNodeConfig,
    #[serde(default)]
    pub(crate) udp: OutputNodeConfig,
    #[serde(default)]
    pub(crate) rtp: OutputNodeConfig,
    #[serde(default)]
    pub(crate) rtmp: OutputNodeConfig,
    #[serde(default)]
    pub(crate) srt: OutputNodeConfig,
    #[serde(default)]
    pub(crate) rtsp: OutputNodeConfig,
    #[serde(default)]
    pub(crate) audio_device: OutputNodeConfig,
    #[serde(default)]
    pub(crate) file: OutputNodeConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct OutputNodeConfig {
    #[serde(rename = "@enabled", default)]
    pub(crate) enabled: Option<String>,
    #[serde(default)]
    pub(crate) r#type: String,
    #[serde(default)]
    pub(crate) ip: String,
    #[serde(default)]
    pub(crate) port: String,
    #[serde(default)]
    pub(crate) host: String,
    #[serde(default)]
    pub(crate) url: String,
    #[serde(default)]
    pub(crate) path: String,
    #[serde(default)]
    pub(crate) format: String,
    #[serde(default)]
    pub(crate) acodec: String,
    #[serde(default)]
    pub(crate) bitrate_kbps: String,
    #[serde(default)]
    pub(crate) device: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PackagesXml {
    #[serde(default)]
    defaults: PackageDefaultsXml,
    #[serde(rename = "package", default)]
    packages: Vec<PackageXml>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PackageDefaultsXml {
    #[serde(default)]
    enabled: String,
    #[serde(default)]
    reader_id: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PackageXml {
    #[serde(rename = "@id", default)]
    id: String,
    #[serde(rename = "@enabled", default)]
    enabled: Option<String>,
    #[serde(default)]
    reader_id: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct PackageConfig {
    pub(crate) enabled: bool,
    pub(crate) reader_id: String,
}

pub(crate) fn load_config(config_path: &Path) -> Result<LoadedConfig> {
    let config_path = dunce::simplified(config_path).to_path_buf();
    let raw = fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read config {}", config_path.display()))?;
    let mut root: RootConfig = serde_yaml::from_str(&raw)
        .with_context(|| format!("failed to parse config {}", config_path.display()))?;
    if root.playout.sample_rate == 0 {
        root.playout.sample_rate = default_sample_rate();
    }
    if root.playout.channels == 0 {
        root.playout.channels = default_channels();
    }
    if root.playout.playlist_order.is_empty() {
        root.playout.playlist_order = vec![
            "current_conditions".to_string(),
            "air_quality".to_string(),
            "forecast".to_string(),
            "climate_summary".to_string(),
            "geophysical_alert".to_string(),
            "user_bulletin".to_string(),
        ];
    }

    let base_dir = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let feeds_path = resolve_path(&base_dir, &root.feeds_file);
    let outputs_path = resolve_path(&base_dir, &root.outputs_file);
    let mut feeds = load_feeds(&feeds_path)?;
    let outputs = load_outputs(&outputs_path)?;
    for feed in &mut feeds {
        feed.output = outputs.get(feed.id.trim()).cloned().unwrap_or_default();
    }
    let packages = load_packages(&resolve_path(&base_dir, "managed/configs/packages.xml"))?;
    Ok(LoadedConfig {
        root,
        feeds,
        packages,
        base_dir,
    })
}

fn load_feeds(path: &Path) -> Result<Vec<FeedConfig>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read feeds XML {}", path.display()))?;
    let parsed: FeedsXml = quick_xml::de::from_str(&raw)
        .with_context(|| format!("failed to parse feeds XML {}", path.display()))?;
    Ok(parsed.feeds)
}

fn load_outputs(path: &Path) -> Result<BTreeMap<String, OutputConfig>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read outputs XML {}", path.display()))?;
    let parsed: OutputsXml = quick_xml::de::from_str(&raw)
        .with_context(|| format!("failed to parse outputs XML {}", path.display()))?;
    let mut outputs = BTreeMap::new();
    for feed in parsed.feeds {
        let id = feed.id.trim();
        if id.is_empty() {
            continue;
        }
        outputs.insert(
            id.to_string(),
            OutputConfig {
                webrtc: feed.webrtc,
                stream: feed.stream,
                udp: feed.udp,
                rtp: feed.rtp,
                rtmp: feed.rtmp,
                srt: feed.srt,
                rtsp: feed.rtsp,
                audio_device: feed.audio_device,
                file: feed.file,
            },
        );
    }
    Ok(outputs)
}

fn load_packages(path: &Path) -> Result<BTreeMap<String, PackageConfig>> {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(err) => return Err(err).with_context(|| format!("failed to read {}", path.display())),
    };
    let parsed: PackagesXml = quick_xml::de::from_str(&raw)
        .with_context(|| format!("failed to parse packages XML {}", path.display()))?;
    let default_enabled = xml_bool(&parsed.defaults.enabled, true);
    let default_reader = parsed.defaults.reader_id.trim().to_string();
    let mut packages = BTreeMap::new();
    for package in parsed.packages {
        let id = package.id.trim();
        if id.is_empty() {
            continue;
        }
        let reader = if package.reader_id.trim().is_empty() {
            default_reader.clone()
        } else {
            package.reader_id.trim().to_string()
        };
        packages.insert(
            id.to_string(),
            PackageConfig {
                enabled: package
                    .enabled
                    .as_deref()
                    .map(|raw| xml_bool(raw, default_enabled))
                    .unwrap_or(default_enabled),
                reader_id: reader,
            },
        );
    }
    Ok(packages)
}

impl LoadedConfig {
    pub(crate) fn enabled_feeds(&self) -> impl Iterator<Item = &FeedConfig> {
        self.feeds
            .iter()
            .filter(|feed| !feed.id.trim().is_empty() && feed.is_enabled())
    }

    pub(crate) fn package_enabled(&self, package_id: &str) -> bool {
        self.packages
            .get(package_id)
            .map(|package| package.enabled)
            .unwrap_or(true)
    }

    pub(crate) fn reader_id(&self, package_id: &str) -> String {
        self.packages
            .get(package_id)
            .map(|package| package.reader_id.trim().to_string())
            .unwrap_or_default()
    }

    pub(crate) fn routine_playlist_order(&self) -> Vec<String> {
        let order: Vec<String> = self
            .root
            .playout
            .playlist_order
            .iter()
            .map(|raw| raw.trim())
            .filter(|id| !id.is_empty() && *id != "alerts" && self.package_enabled(id))
            .map(str::to_string)
            .collect();
        if order.is_empty() {
            vec!["current_conditions".to_string(), "forecast".to_string()]
        } else {
            order
        }
    }
}

impl FeedConfig {
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
            .as_deref()
            .map(|raw| xml_bool(raw, true))
            .unwrap_or(true)
    }

    pub(crate) fn routine_enabled(&self) -> bool {
        self.playout
            .routine
            .as_deref()
            .map(|raw| xml_bool(raw, true))
            .unwrap_or(true)
    }

    pub(crate) fn same_enabled(&self) -> bool {
        self.playout
            .same
            .as_deref()
            .map(|raw| xml_bool(raw, true))
            .unwrap_or(true)
    }

    pub(crate) fn alert_covers_all_locations(&self) -> bool {
        if self
            .locations
            .coverage
            .regions
            .iter()
            .any(|region| !region.id.trim().is_empty())
        {
            return false;
        }
        let Some(alerts) = &self.alerts else {
            return false;
        };
        alerts
            .cap_cp
            .enabled
            .as_deref()
            .map(|raw| xml_bool(raw, true))
            .unwrap_or(true)
            || alerts
                .nws_cap
                .enabled
                .as_deref()
                .map(|raw| xml_bool(raw, false))
                .unwrap_or(false)
    }

    pub(crate) fn language(&self) -> String {
        self.languages
            .langs
            .iter()
            .find_map(|lang| {
                let code = lang.code.trim();
                (!code.is_empty()).then(|| code.to_string())
            })
            .unwrap_or_else(|| "en-CA".to_string())
    }

    pub(crate) fn site_name(&self) -> String {
        self.station_transmitter()
            .map(|transmitter| fallback_text(&transmitter.site_name, &self.id))
            .unwrap_or_else(|| self.id.clone())
    }

    pub(crate) fn station_callsign(&self) -> String {
        self.station_transmitter()
            .map(|transmitter| transmitter.callsign.trim().to_string())
            .unwrap_or_default()
    }

    pub(crate) fn station_frequency_mhz(&self) -> String {
        self.station_transmitter()
            .map(|transmitter| transmitter.frequency_mhz.value.trim().to_string())
            .unwrap_or_default()
    }

    fn station_transmitter(&self) -> Option<&TransmitterConfig> {
        let transmitters: Vec<&TransmitterConfig> = self
            .transmitter_metadata
            .transmitters
            .iter()
            .filter(|transmitter| !transmitter.is_empty())
            .collect();
        transmitters
            .iter()
            .copied()
            .find(|transmitter| transmitter.is_relationship("primary"))
            .or_else(|| {
                transmitters.iter().copied().find(|transmitter| {
                    !transmitter.is_relationship("replaces")
                        && !transmitter.is_relationship("ip")
                        && (!transmitter.callsign.trim().is_empty()
                            || !transmitter.site_name.trim().is_empty())
                })
            })
            .or_else(|| {
                transmitters.iter().copied().find(|transmitter| {
                    !transmitter.callsign.trim().is_empty()
                        || !transmitter.site_name.trim().is_empty()
                })
            })
            .or_else(|| transmitters.first().copied())
    }

    pub(crate) fn replacement_transmitter(&self) -> Option<&TransmitterConfig> {
        self.transmitter_metadata
            .transmitters
            .iter()
            .find(|transmitter| !transmitter.is_empty() && transmitter.is_relationship("replaces"))
    }
}

impl TransmitterConfig {
    fn is_empty(&self) -> bool {
        self.site_name.trim().is_empty()
            && self.callsign.trim().is_empty()
            && self.relationship.trim().is_empty()
            && self.host_name.trim().is_empty()
            && self.frequency_mhz.value.trim().is_empty()
    }

    fn relationship(&self) -> String {
        let relationship = self.relationship.trim().to_ascii_lowercase();
        if relationship == "secondary/repeater" {
            return "secondary".to_string();
        }
        if relationship.is_empty() {
            return "unknown".to_string();
        }
        relationship
    }

    fn is_relationship(&self, relationship: &str) -> bool {
        let current = self.relationship();
        let wanted = relationship.trim().to_ascii_lowercase();
        current == wanted || (wanted == "repeater" && current == "secondary")
    }
}

impl OutputNodeConfig {
    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled
            .as_deref()
            .map(|raw| xml_bool(raw, false))
            .unwrap_or(false)
    }
}

pub(crate) fn xml_bool(raw: &str, fallback: bool) -> bool {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => fallback,
        "1" | "true" | "yes" | "on" | "enabled" => true,
        "0" | "false" | "no" | "off" | "disabled" => false,
        _ => fallback,
    }
}

pub(crate) fn resolve_path(base: &Path, value: &str) -> PathBuf {
    let value = value.trim();
    if value.is_empty() {
        return base.to_path_buf();
    }
    let path = Path::new(value);
    if path.is_absolute() {
        dunce::simplified(path).to_path_buf()
    } else {
        dunce::simplified(&base.join(path)).to_path_buf()
    }
}

pub(crate) fn display_text(value: &serde_yaml::Value) -> String {
    match value {
        serde_yaml::Value::String(text) => text.trim().to_string(),
        serde_yaml::Value::Sequence(items) => items
            .iter()
            .map(display_text)
            .find(|text| !text.is_empty())
            .unwrap_or_default(),
        serde_yaml::Value::Mapping(map) => ["text", "name", "value", "address"]
            .iter()
            .find_map(|key| {
                map.get(serde_yaml::Value::String((*key).to_string()))
                    .map(display_text)
                    .filter(|text| !text.is_empty())
            })
            .unwrap_or_default(),
        _ => String::new(),
    }
}

pub(crate) fn fallback_text(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.trim().to_string()
    } else {
        value.to_string()
    }
}

fn default_feeds_file() -> String {
    "managed/configs/feeds.xml".to_string()
}

fn default_outputs_file() -> String {
    "managed/configs/output.xml".to_string()
}

fn default_sample_rate() -> u32 {
    48_000
}

fn default_channels() -> u16 {
    1
}

fn default_gap_seconds() -> f64 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_feed_attributes() {
        let raw = r#"
<feeds>
  <feed id="sk-0001" enabled="true" timezone="America/Regina">
    <playout routine="true" same="false"/>
    <languages><lang code="en-CA"/></languages>
    <transmitter_metadata>
      <transmitter><site_name>Saskatoon</site_name><relationship>primary</relationship><frequency_mhz>162.550</frequency_mhz></transmitter>
      <transmitter><site_name>Saskatoon</site_name><relationship>fm</relationship><frequency_mhz gpclk="2" gpio="6">87.900</frequency_mhz></transmitter>
      <transmitter><site_name>Saskatoon</site_name><callsign>XLF322</callsign><relationship>replaces</relationship><frequency_mhz>162.550</frequency_mhz></transmitter>
    </transmitter_metadata>
  </feed>
</feeds>
"#;
        let parsed: FeedsXml = quick_xml::de::from_str(raw).expect("feeds XML");
        assert_eq!(parsed.feeds.len(), 1);
        let feed = &parsed.feeds[0];
        assert_eq!(feed.id, "sk-0001");
        assert!(feed.is_enabled());
        assert!(feed.routine_enabled());
        assert!(!feed.same_enabled());
        assert_eq!(feed.language(), "en-CA");
        assert_eq!(feed.site_name(), "Saskatoon");
        assert_eq!(feed.station_callsign(), "");
        assert_eq!(feed.station_frequency_mhz(), "162.550");
        assert!(!feed.alert_covers_all_locations());
        assert_eq!(feed.transmitter_metadata.transmitters.len(), 3);
        assert_eq!(
            feed.replacement_transmitter()
                .map(|transmitter| transmitter.callsign.as_str()),
            Some("XLF322")
        );
        assert_eq!(
            feed.transmitter_metadata.transmitters[1]
                .frequency_mhz
                .gpclk,
            "2"
        );
        assert_eq!(
            feed.transmitter_metadata.transmitters[1].frequency_mhz.gpio,
            "6"
        );
    }

    #[test]
    fn parses_all_location_alert_relay_feed() {
        let raw = r#"
<feeds>
  <feed id="CAP-IT-ALL" enabled="true">
    <playout routine="false" same="true"/>
    <alerts>
      <cap_cp enabled="true"/>
      <nws_cap enabled="true"/>
    </alerts>
    <locations><coverage/></locations>
  </feed>
</feeds>
"#;
        let parsed: FeedsXml = quick_xml::de::from_str(raw).expect("feeds XML");
        let feed = &parsed.feeds[0];

        assert!(feed.same_enabled());
        assert!(feed.alert_covers_all_locations());
    }

    #[test]
    fn parses_outputs_by_feed_id() {
        let raw = r#"
<outputs>
  <feed id="sk-0001">
    <udp enabled="true"><ip>127.0.0.1</ip><format>raw</format></udp>
  </feed>
</outputs>
"#;
        let parsed: OutputsXml = quick_xml::de::from_str(raw).expect("outputs XML");
        assert_eq!(parsed.feeds.len(), 1);
        assert_eq!(parsed.feeds[0].id, "sk-0001");
        assert!(parsed.feeds[0].udp.is_enabled());
    }
}
