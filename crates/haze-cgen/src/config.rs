use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename = "cgen")]
pub(crate) struct CgenConfig {
    #[serde(rename = "@enabled", default = "default_true")]
    pub(crate) enabled: bool,
    #[serde(rename = "feed", default)]
    pub(crate) feeds: Vec<FeedConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct FeedConfig {
    #[serde(rename = "@id")]
    pub(crate) id: String,
    #[serde(rename = "@name", default)]
    pub(crate) name: String,
    #[serde(rename = "@enabled", default = "default_true")]
    pub(crate) enabled: bool,
    #[serde(rename = "programInput", default)]
    pub(crate) program_input: EndpointConfig,
    #[serde(rename = "priorityInput", default)]
    pub(crate) priority_input: PriorityInputConfig,
    #[serde(rename = "programOutput", default)]
    pub(crate) program_output: EndpointConfig,
    pub(crate) video: VideoConfig,
    #[serde(default)]
    pub(crate) audio: AudioConfig,
    #[serde(default)]
    pub(crate) ladder: LadderConfig,
    #[serde(default)]
    pub(crate) banner: BannerConfig,
    #[serde(default)]
    pub(crate) graphics: GraphicsConfig,
    #[serde(default)]
    pub(crate) clock: ClockConfig,
    #[serde(default)]
    pub(crate) text: TextConfig,
    #[serde(default)]
    pub(crate) state: StateConfig,
    #[serde(default)]
    pub(crate) standby: StandbyConfig,
    #[serde(default)]
    pub(crate) sync: SyncConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct EndpointConfig {
    #[serde(rename = "@url", default)]
    pub(crate) url: String,
    #[serde(rename = "@format", default)]
    pub(crate) format: String,
    #[serde(rename = "@vcodec", default)]
    pub(crate) vcodec: String,
    #[serde(rename = "@acodec", default)]
    pub(crate) acodec: String,
    #[serde(rename = "@video_bitrate_kbps", default)]
    pub(crate) video_bitrate_kbps: Option<u32>,
    #[serde(rename = "@audio_bitrate_kbps", default)]
    pub(crate) audio_bitrate_kbps: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PriorityInputConfig {
    #[serde(rename = "@feed_id", default)]
    pub(crate) feed_id: String,
    #[serde(rename = "@audio_source", default = "priority_audio_source")]
    pub(crate) audio_source: String,
    #[serde(rename = "@format", default)]
    pub(crate) format: String,
}

impl PriorityInputConfig {
    pub(crate) fn priority_audio_enabled(&self) -> bool {
        !matches!(
            self.audio_source.trim().to_ascii_lowercase().as_str(),
            "routine" | "feed" | "program_feed"
        )
    }

    pub(crate) fn routine_audio_enabled(&self) -> bool {
        matches!(
            self.audio_source.trim().to_ascii_lowercase().as_str(),
            "routine" | "feed" | "program_feed" | "both" | "all"
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VideoConfig {
    #[serde(rename = "@width")]
    pub(crate) width: u32,
    #[serde(rename = "@height")]
    pub(crate) height: u32,
    #[serde(rename = "@fps", default)]
    pub(crate) fps: String,
    #[serde(rename = "@interlaced", default)]
    pub(crate) interlaced: bool,
    #[serde(rename = "@field_order", default = "default_field_order")]
    pub(crate) field_order: String,
    #[serde(rename = "@standard", default)]
    pub(crate) standard: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AudioConfig {
    #[serde(rename = "@idle", default = "source_audio")]
    pub(crate) idle: String,
    #[serde(rename = "@alert_mode", default = "replace_audio")]
    pub(crate) alert_mode: String,
    #[serde(rename = "@mute_standby_routine", default = "default_true")]
    pub(crate) mute_standby_routine: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct LadderConfig {
    #[serde(rename = "video", default)]
    pub(crate) videos: Vec<VideoRenditionConfig>,
    #[serde(rename = "audio", default)]
    pub(crate) audios: Vec<AudioRenditionConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VideoRenditionConfig {
    #[serde(rename = "@id")]
    pub(crate) id: String,
    #[serde(rename = "@enabled", default = "default_true_text")]
    pub(crate) enabled: String,
    #[serde(rename = "@width")]
    pub(crate) width: u32,
    #[serde(rename = "@height")]
    pub(crate) height: u32,
    #[serde(rename = "@fps", default)]
    pub(crate) fps: String,
    #[serde(rename = "@interlaced", default)]
    pub(crate) interlaced: bool,
    #[serde(rename = "@field_order", default = "default_field_order")]
    pub(crate) field_order: String,
    #[serde(rename = "@standard", default)]
    pub(crate) standard: String,
    #[serde(rename = "@vcodec", default)]
    pub(crate) vcodec: String,
    #[serde(rename = "@bitrate_kbps", default)]
    pub(crate) bitrate_kbps: Option<u32>,
    #[serde(rename = "@program", default)]
    pub(crate) program: Option<i32>,
    #[serde(rename = "@video_pid", default)]
    pub(crate) video_pid: Option<i32>,
    #[serde(rename = "@pmt_pid", default)]
    pub(crate) pmt_pid: Option<i32>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AudioRenditionConfig {
    #[serde(rename = "@id")]
    pub(crate) id: String,
    #[serde(rename = "@enabled", default = "default_true_text")]
    pub(crate) enabled: String,
    #[serde(rename = "@channels", default = "default_stereo_channels")]
    pub(crate) channels: u16,
    #[serde(rename = "@acodec", default)]
    pub(crate) acodec: String,
    #[serde(rename = "@bitrate_kbps", default)]
    pub(crate) bitrate_kbps: Option<u32>,
    #[serde(rename = "@language", default = "default_audio_language")]
    pub(crate) language: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct BannerConfig {
    #[serde(rename = "@mode", default = "auto_mode")]
    pub(crate) mode: String,
    #[serde(rename = "@ticker_height", default = "default_ticker_height")]
    pub(crate) ticker_height: u32,
    #[serde(rename = "@font", default)]
    pub(crate) font: String,
    #[serde(rename = "@font_size", default = "default_font_size")]
    pub(crate) font_size: u32,
    #[serde(rename = "@scroll_speed", default = "default_scroll_speed")]
    pub(crate) scroll_speed: u32,
    #[serde(rename = "@background_enabled", default = "default_true")]
    pub(crate) background_enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GraphicsConfig {
    #[serde(rename = "@font", default)]
    pub(crate) font: String,
    #[serde(rename = "@font_size", default = "default_font_size")]
    pub(crate) font_size: u32,
    #[serde(rename = "@banner_height", default = "default_ticker_height")]
    pub(crate) banner_height: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ClockConfig {
    #[serde(rename = "@enabled", default)]
    pub(crate) enabled: bool,
    #[serde(rename = "@format", default)]
    pub(crate) format: String,
    #[serde(rename = "@x", default = "default_text_x")]
    pub(crate) x: i32,
    #[serde(rename = "@y", default = "default_clock_y")]
    pub(crate) y: i32,
    #[serde(rename = "@font_size", default = "default_clock_font_size")]
    pub(crate) font_size: u32,
    #[serde(rename = "@color", default)]
    pub(crate) color: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct TextConfig {
    #[serde(rename = "@enabled", default)]
    pub(crate) enabled: bool,
    #[serde(rename = "@x", default = "default_text_x")]
    pub(crate) x: i32,
    #[serde(rename = "@y", default = "default_text_y")]
    pub(crate) y: i32,
    #[serde(rename = "@font_size", default = "default_insert_font_size")]
    pub(crate) font_size: u32,
    #[serde(rename = "@color", default)]
    pub(crate) color: String,
    #[serde(rename = "$text", default)]
    pub(crate) content: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StateConfig {
    #[serde(rename = "@mode", default = "release_mode")]
    pub(crate) mode: String,
    #[serde(rename = "@smpte_bars", default)]
    pub(crate) smpte_bars: bool,
    #[serde(rename = "@updated_at", default)]
    pub(crate) updated_at: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StandbyConfig {
    #[serde(rename = "@mode", default = "standby_banner_mode")]
    pub(crate) mode: String,
    #[serde(rename = "@text", default = "default_standby_text")]
    pub(crate) text: String,
    #[serde(rename = "@font_size", default)]
    pub(crate) font_size: u32,
    #[serde(rename = "@y_percent", default = "default_standby_y_percent")]
    pub(crate) y_percent: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SyncConfig {
    #[serde(rename = "@hard_reset_ms", default = "default_hard_reset_ms")]
    pub(crate) hard_reset_ms: u32,
    #[serde(
        rename = "@max_audio_frames_per_video",
        default = "default_max_audio_frames_per_video"
    )]
    pub(crate) max_audio_frames_per_video: u32,
    #[serde(rename = "@source_buffer_ms", default = "default_source_buffer_ms")]
    pub(crate) source_buffer_ms: u32,
    #[serde(
        rename = "@reconnect_initial_ms",
        default = "default_reconnect_initial_ms"
    )]
    pub(crate) reconnect_initial_ms: u32,
    #[serde(rename = "@reconnect_max_ms", default = "default_reconnect_max_ms")]
    pub(crate) reconnect_max_ms: u32,
    #[serde(rename = "@status_interval_ms", default = "default_status_interval_ms")]
    pub(crate) status_interval_ms: u32,
}

impl CgenConfig {
    pub(crate) fn enabled_feeds(&self) -> Result<Vec<FeedConfig>> {
        if !self.enabled {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for feed in &self.feeds {
            if !feed.enabled {
                continue;
            }
            feed.validate()?;
            out.push(feed.clone());
        }
        Ok(out)
    }
}

impl FeedConfig {
    #[allow(dead_code)]
    pub(crate) fn matches_feed(&self, feed_id: &str) -> bool {
        self.id.trim() == "*" || self.id.trim() == feed_id.trim()
    }

    fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            bail!("cgen feed id is required");
        }
        if self.program_input_url().trim().is_empty() {
            bail!("cgen feed {} input url is required", self.id);
        }
        if self.program_output_url().trim().is_empty() {
            bail!("cgen feed {} output url is required", self.id);
        }
        if !allowed_video_size(self.video.width, self.video.height) {
            bail!(
                "cgen feed {} video size {}x{} is not supported",
                self.id,
                self.video.width,
                self.video.height
            );
        }
        let enabled_videos = self.enabled_video_renditions(self.video.width, self.video.height);
        if enabled_videos.is_empty() {
            bail!("cgen feed {} has no enabled video renditions", self.id);
        }
        for video in enabled_videos {
            if !allowed_video_size(video.width, video.height) {
                bail!(
                    "cgen feed {} video rendition {} size {}x{} is not supported",
                    self.id,
                    video.id,
                    video.width,
                    video.height
                );
            }
        }
        if self.enabled_audio_renditions().is_empty() {
            bail!("cgen feed {} has no enabled audio renditions", self.id);
        }
        if !self.audio.alert_mode.eq_ignore_ascii_case("replace") {
            bail!("cgen feed {} audio alert_mode must be replace", self.id);
        }
        Ok(())
    }

    pub(crate) fn program_input_url(&self) -> &str {
        &self.program_input.url
    }

    pub(crate) fn program_output_url(&self) -> &str {
        &self.program_output.url
    }

    pub(crate) fn output(&self) -> &EndpointConfig {
        &self.program_output
    }

    pub(crate) fn enabled_video_renditions(
        &self,
        source_width: u32,
        source_height: u32,
    ) -> Vec<VideoRenditionConfig> {
        let mut out = Vec::new();
        if self.ladder.videos.is_empty() {
            return out;
        }
        for video in &self.ladder.videos {
            if video.enabled_for_source(source_width, source_height) {
                out.push(video.clone());
            }
        }
        if out.is_empty() {
            return out;
        }
        out
    }

    pub(crate) fn enabled_audio_renditions(&self) -> Vec<AudioRenditionConfig> {
        if self.ladder.audios.is_empty() {
            return Vec::new();
        }
        let out = self
            .ladder
            .audios
            .iter()
            .filter(|audio| audio.enabled_bool())
            .cloned()
            .collect::<Vec<_>>();
        if out.is_empty() {
            return out;
        }
        out
    }
}

impl VideoRenditionConfig {
    pub(crate) fn enabled_for_source(&self, source_width: u32, source_height: u32) -> bool {
        let enabled = self.enabled.trim();
        if enabled.eq_ignore_ascii_case("auto") {
            return self.id.eq_ignore_ascii_case("hd")
                && self.width == source_width
                && self.height == source_height;
        }
        xml_bool_text(enabled, true)
    }

    pub(crate) fn codec_name<'a>(&'a self, fallback: &'a str) -> &'a str {
        non_empty(self.vcodec.as_str()).unwrap_or(fallback)
    }

    pub(crate) fn frame_rate_text<'a>(&'a self, fallback: &'a str) -> &'a str {
        non_empty(self.fps.as_str()).unwrap_or(fallback)
    }

    pub(crate) fn program_number(&self, index: usize) -> i32 {
        self.program
            .unwrap_or_else(|| i32::try_from(index + 1).unwrap_or(1))
            .max(1)
    }

    pub(crate) fn video_pid(&self, index: usize) -> i32 {
        self.video_pid
            .unwrap_or_else(|| 0x100 + (i32::try_from(index).unwrap_or(0) * 0x20))
    }

    pub(crate) fn pmt_pid(&self, index: usize) -> i32 {
        self.pmt_pid
            .unwrap_or_else(|| 0x1000 + i32::try_from(index).unwrap_or(0))
    }
}

impl AudioRenditionConfig {
    pub(crate) fn enabled_bool(&self) -> bool {
        xml_bool_text(&self.enabled, true)
    }

    pub(crate) fn channels(&self) -> u16 {
        self.channels.clamp(1, 6)
    }

    pub(crate) fn codec_name<'a>(&'a self, fallback: &'a str) -> &'a str {
        non_empty(self.acodec.as_str()).unwrap_or(fallback)
    }

    pub(crate) fn bitrate_bps(&self) -> i64 {
        self.bitrate_kbps
            .map(|kbps| i64::from(kbps) * 1000)
            .unwrap_or_else(|| {
                if self.channels() > 2 {
                    384_000
                } else {
                    192_000
                }
            })
    }
}

pub(crate) fn load_config(path: &Path) -> Result<CgenConfig> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    reject_legacy_cgen_xml(path, &raw)?;
    let config: CgenConfig = quick_xml::de::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(config)
}

fn reject_legacy_cgen_xml(path: &Path, raw: &str) -> Result<()> {
    let mut reader = Reader::from_str(raw);
    reader.config_mut().trim_text(true);
    loop {
        match reader.read_event() {
            Ok(Event::Start(event)) | Ok(Event::Empty(event)) => {
                let element_name = event.name();
                let name_bytes = element_name.as_ref();
                let name = std::str::from_utf8(name_bytes).unwrap_or_default();
                if matches!(name, "input" | "output" | "alertOutput") {
                    bail!(
                        "legacy cgen <{}> element is no longer supported in {}; use programInput, priorityInput, and programOutput",
                        name,
                        path.display()
                    );
                }
                for attr in event.attributes() {
                    let attr = attr.with_context(|| {
                        format!("failed to read cgen XML attribute in {}", path.display())
                    })?;
                    let key_bytes = attr.key.as_ref();
                    let key = std::str::from_utf8(key_bytes).unwrap_or_default();
                    if legacy_attribute(name, key) {
                        bail!(
                            "legacy cgen {} @{} attribute is no longer supported in {}",
                            name,
                            key,
                            path.display()
                        );
                    }
                }
            }
            Ok(Event::Eof) => return Ok(()),
            Ok(_) => {}
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("failed to scan cgen XML {}", path.display()))
            }
        }
    }
}

fn legacy_attribute(element: &str, attr: &str) -> bool {
    match element {
        "cgen" => matches!(attr, "graphics_backend" | "ffmpeg"),
        "priorityInput" => attr == "url",
        "audio" => attr == "duck_db",
        "banner" => matches!(attr, "x" | "y"),
        "graphics" => matches!(
            attr,
            "background_color" | "text_x" | "text_y" | "banner_x" | "banner_y" | "banner_width"
        ),
        _ => false,
    }
}

pub(crate) fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

pub(crate) fn allowed_video_size(width: u32, height: u32) -> bool {
    matches!(
        (width, height),
        (720, 480) | (720, 576) | (1280, 720) | (1920, 1080)
    )
}

fn default_true() -> bool {
    true
}

fn default_true_text() -> String {
    "true".to_string()
}

fn default_stereo_channels() -> u16 {
    2
}

fn default_audio_language() -> String {
    "eng".to_string()
}

fn source_audio() -> String {
    "source".to_string()
}

fn priority_audio_source() -> String {
    "priority".to_string()
}

fn replace_audio() -> String {
    "replace".to_string()
}

fn auto_mode() -> String {
    "auto".to_string()
}

fn default_ticker_height() -> u32 {
    128
}

fn default_font_size() -> u32 {
    58
}

fn default_scroll_speed() -> u32 {
    8
}

fn default_text_x() -> i32 {
    48
}

fn default_text_y() -> i32 {
    128
}

fn default_clock_y() -> i32 {
    48
}

fn default_clock_font_size() -> u32 {
    30
}

fn default_insert_font_size() -> u32 {
    58
}

fn default_field_order() -> String {
    "tff".to_string()
}

fn release_mode() -> String {
    "release".to_string()
}

fn standby_banner_mode() -> String {
    "banner".to_string()
}

fn default_standby_text() -> String {
    "EAS Details Channel".to_string()
}

fn default_standby_y_percent() -> u32 {
    10
}

pub(crate) fn default_hard_reset_ms() -> u32 {
    250
}

pub(crate) fn default_max_audio_frames_per_video() -> u32 {
    8
}

pub(crate) fn default_source_buffer_ms() -> u32 {
    240
}

pub(crate) fn default_reconnect_initial_ms() -> u32 {
    500
}

pub(crate) fn default_reconnect_max_ms() -> u32 {
    10_000
}

pub(crate) fn default_status_interval_ms() -> u32 {
    750
}

fn xml_bool_text(value: &str, default: bool) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return default;
    }
    matches!(
        value.to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enabled"
    )
}

fn non_empty(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_disabled_example_feed() {
        let xml = r#"
<cgen enabled="true">
  <feed id="*" enabled="false">
    <programInput url="udp://127.0.0.1:5000" format="mpegts"/>
    <programOutput url="udp://127.0.0.1:5001" format="mpegts" vcodec="libx264" acodec="aac"/>
    <video width="1280" height="720" fps="source" interlaced="false" field_order="tff"/>
    <audio idle="source" alert_mode="replace"/>
    <banner mode="auto" ticker_height="128" font="Arial"/>
  </feed>
</cgen>"#;
        let parsed: CgenConfig = quick_xml::de::from_str(xml).expect("parse");
        assert!(parsed.enabled);
        assert_eq!(parsed.feeds.len(), 1);
        assert_eq!(parsed.enabled_feeds().expect("feeds").len(), 0);
    }

    #[test]
    fn load_config_rejects_legacy_cgen_elements() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cgen.xml");
        std::fs::write(
            &path,
            r#"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <input></input>
    <programInput url="udp://127.0.0.1:5000" format="mpegts"/>
    <programOutput url="udp://127.0.0.1:5001" format="mpegts"/>
    <video width="1280" height="720"/>
  </feed>
</cgen>"#,
        )
        .expect("write config");

        let err = load_config(&path).expect_err("legacy config rejected");
        assert!(err.to_string().contains("legacy cgen <input> element"));
    }

    #[test]
    fn load_config_rejects_legacy_cgen_attributes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cgen.xml");
        std::fs::write(
            &path,
            r#"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <programInput url="udp://127.0.0.1:5000" format="mpegts"/>
    <priorityInput feed_id="*" url="udp://127.0.0.1:5002" format="priority-audio"/>
    <programOutput url="udp://127.0.0.1:5001" format="mpegts"/>
    <video width="1280" height="720"/>
    <audio idle="source" alert_mode="replace" duck_db="-12"/>
  </feed>
</cgen>"#,
        )
        .expect("write config");

        let err = load_config(&path).expect_err("legacy config rejected");
        assert!(err
            .to_string()
            .contains("legacy cgen priorityInput @url attribute"));
    }

    #[test]
    fn load_config_accepts_current_cgen_shape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cgen.xml");
        std::fs::write(
            &path,
            r#"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <programInput url="udp://127.0.0.1:5000" format="mpegts"/>
    <priorityInput feed_id="*" audio_source="both" format="priority-audio"/>
    <programOutput url="udp://127.0.0.1:5001" format="mpegts" vcodec="mpeg2video" acodec="ac3"/>
    <video width="1280" height="720" fps="30000/1001"/>
    <audio idle="source" alert_mode="replace" mute_standby_routine="true"/>
    <ladder>
      <video id="p720" enabled="true" width="1280" height="720" fps="30000/1001"/>
      <audio id="stereo" enabled="true" channels="2"/>
    </ladder>
  </feed>
</cgen>"#,
        )
        .expect("write config");

        let config = load_config(&path).expect("current config");
        assert_eq!(config.enabled_feeds().expect("feeds").len(), 1);
    }

    #[test]
    fn validates_supported_sizes() {
        assert!(allowed_video_size(720, 480));
        assert!(allowed_video_size(720, 576));
        assert!(allowed_video_size(1280, 720));
        assert!(allowed_video_size(1920, 1080));
        assert!(!allowed_video_size(1024, 576));
    }

    #[test]
    fn parses_program_priority_and_output_sections() {
        let xml = r##"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" name="CAP CGEN" enabled="true">
    <programInput url="udp://239.0.0.1:9000?reuse=1" format="mpegts"/>
    <priorityInput feed_id="CAP-IT-ALL" audio_source="both" format="priority-audio"/>
    <programOutput url="udp://239.0.0.2:9001?pkt_size=1316" format="mpegts" vcodec="libx264" acodec="aac"/>
    <video width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc"/>
    <audio idle="source" alert_mode="replace" mute_standby_routine="false"/>
    <ladder>
      <video id="hd" enabled="auto" width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc" vcodec="mpeg2video" bitrate_kbps="12000" program="1" video_pid="256" pmt_pid="4096"/>
      <audio id="stereo" enabled="true" channels="2" acodec="ac3" bitrate_kbps="192" language="eng"/>
    </ladder>
    <banner mode="auto" ticker_height="128" font="Arial" font_size="58" scroll_speed="5" background_enabled="true"/>
    <graphics font="Arial" font_size="58"/>
    <clock enabled="true" x="48" y="48"/>
    <text enabled="true" x="48" y="96">Hello</text>
    <state mode="overlay" smpte_bars="true"/>
    <sync hard_reset_ms="500" max_audio_frames_per_video="10" source_buffer_ms="400" reconnect_initial_ms="250" reconnect_max_ms="5000" status_interval_ms="750"/>
  </feed>
</cgen>"##;
        let parsed: CgenConfig = quick_xml::de::from_str(xml).expect("parse");
        let feeds = parsed.enabled_feeds().expect("feeds");

        assert_eq!(feeds.len(), 1);
        assert_eq!(feeds[0].program_input_url(), "udp://239.0.0.1:9000?reuse=1");
        assert_eq!(
            feeds[0].program_output_url(),
            "udp://239.0.0.2:9001?pkt_size=1316"
        );
        assert_eq!(feeds[0].priority_input.feed_id, "CAP-IT-ALL");
        assert_eq!(feeds[0].priority_input.audio_source, "both");
        assert!(feeds[0].priority_input.priority_audio_enabled());
        assert!(feeds[0].priority_input.routine_audio_enabled());
        assert!(!feeds[0].audio.mute_standby_routine);
        assert_eq!(feeds[0].video.width, 1920);
        assert!(feeds[0].video.interlaced);
        assert_eq!(feeds[0].video.field_order, "tff");
        assert_eq!(feeds[0].video.standard, "atsc");
        assert_eq!(feeds[0].banner.scroll_speed, 5);
        assert!(feeds[0].clock.enabled);
        assert!(feeds[0].state.smpte_bars);
        assert_eq!(feeds[0].sync.hard_reset_ms, 500);
        assert_eq!(feeds[0].sync.max_audio_frames_per_video, 10);
        assert_eq!(feeds[0].sync.source_buffer_ms, 400);
        assert_eq!(feeds[0].sync.reconnect_initial_ms, 250);
        assert_eq!(feeds[0].sync.reconnect_max_ms, 5000);
        assert_eq!(feeds[0].sync.status_interval_ms, 750);
    }

    #[test]
    fn wildcard_feed_matches_any_feed() {
        let feed = FeedConfig {
            id: "*".to_string(),
            name: String::new(),
            enabled: true,
            program_input: EndpointConfig {
                url: "udp://in".to_string(),
                format: "mpegts".to_string(),
                vcodec: String::new(),
                acodec: String::new(),
                video_bitrate_kbps: None,
                audio_bitrate_kbps: None,
            },
            priority_input: PriorityInputConfig::default(),
            program_output: EndpointConfig {
                url: "udp://out".to_string(),
                format: "mpegts".to_string(),
                vcodec: String::new(),
                acodec: String::new(),
                video_bitrate_kbps: None,
                audio_bitrate_kbps: None,
            },
            video: VideoConfig {
                width: 1280,
                height: 720,
                fps: "source".to_string(),
                interlaced: false,
                field_order: "tff".to_string(),
                standard: String::new(),
            },
            audio: AudioConfig {
                idle: "source".to_string(),
                alert_mode: "replace".to_string(),
                mute_standby_routine: true,
            },
            ladder: LadderConfig {
                videos: vec![VideoRenditionConfig {
                    id: "hd".to_string(),
                    enabled: "true".to_string(),
                    width: 1280,
                    height: 720,
                    fps: "source".to_string(),
                    interlaced: false,
                    field_order: "tff".to_string(),
                    standard: String::new(),
                    vcodec: "mpeg2video".to_string(),
                    bitrate_kbps: Some(8_000),
                    program: Some(1),
                    video_pid: Some(0x100),
                    pmt_pid: Some(0x1000),
                }],
                audios: vec![AudioRenditionConfig {
                    id: "stereo".to_string(),
                    enabled: "true".to_string(),
                    channels: 2,
                    acodec: "ac3".to_string(),
                    bitrate_kbps: Some(192),
                    language: "eng".to_string(),
                }],
            },
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
            standby: StandbyConfig::default(),
            sync: SyncConfig::default(),
        };
        assert!(feed.matches_feed("sk-0001"));
        assert!(feed.matches_feed("CAP-IT-ALL"));
    }

    #[test]
    fn parses_mpts_output_ladder() {
        let xml = r#"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <programInput url="udp://239.0.0.1:9000" format="mpegts"/>
    <priorityInput feed_id="*" format="priority-audio"/>
    <programOutput url="udp://239.0.0.2:9001?pkt_size=1316" format="mpegts" vcodec="mpeg2video" acodec="ac3"/>
    <video width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc"/>
    <audio idle="source" alert_mode="replace"/>
    <ladder>
      <video id="hd" enabled="auto" width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc" bitrate_kbps="12000" program="1" video_pid="256" pmt_pid="4096"/>
      <video id="p720" enabled="false" width="1280" height="720" fps="30000/1001" bitrate_kbps="8000" program="2"/>
      <audio id="surround_51" enabled="true" channels="6" acodec="ac3" bitrate_kbps="384" language="eng"/>
      <audio id="stereo" enabled="true" channels="2" acodec="ac3" bitrate_kbps="192" language="eng"/>
    </ladder>
  </feed>
</cgen>"#;
        let parsed: CgenConfig = quick_xml::de::from_str(xml).expect("parse");
        let feeds = parsed.enabled_feeds().expect("feeds");
        let videos = feeds[0].enabled_video_renditions(1920, 1080);
        let audios = feeds[0].enabled_audio_renditions();

        assert_eq!(videos.len(), 1);
        assert_eq!(videos[0].id, "hd");
        assert_eq!(videos[0].program_number(0), 1);
        assert_eq!(videos[0].video_pid(0), 256);
        assert_eq!(audios.len(), 2);
        assert_eq!(audios[0].channels(), 6);
        assert_eq!(audios[1].channels(), 2);
    }

    #[test]
    fn parses_standby_output_settings() {
        let xml = r#"
<cgen enabled="true">
  <feed id="standby" enabled="true">
    <programInput url="udp://in" format="mpegts"/>
    <programOutput url="udp://out" format="mpegts" vcodec="mpeg2video" acodec="ac3"/>
    <video width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc"/>
    <audio idle="source" alert_mode="replace"/>
    <ladder>
      <video id="hd" enabled="auto" width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc" vcodec="mpeg2video" bitrate_kbps="12000" program="1" video_pid="256" pmt_pid="4096"/>
      <audio id="stereo" enabled="true" channels="2" acodec="ac3" bitrate_kbps="192" language="eng"/>
    </ladder>
    <standby mode="smpte" text="EAS Details Channel" font_size="72" y_percent="10"/>
  </feed>
</cgen>"#;
        let parsed: CgenConfig = quick_xml::de::from_str(xml).expect("parse");
        let feeds = parsed.enabled_feeds().expect("feeds");

        assert_eq!(feeds[0].standby.mode, "smpte");
        assert_eq!(feeds[0].standby.text, "EAS Details Channel");
        assert_eq!(feeds[0].standby.font_size, 72);
        assert_eq!(feeds[0].standby.y_percent, 10);
    }

    #[test]
    fn missing_ladder_outputs_are_invalid() {
        let xml = r#"
<cgen enabled="true">
  <feed id="missing-ladder" enabled="true">
    <programInput url="udp://in" format="mpegts"/>
    <programOutput url="udp://out" format="mpegts" vcodec="mpeg2video" acodec="ac3" video_bitrate_kbps="12000" audio_bitrate_kbps="192"/>
    <video width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc"/>
    <audio idle="source" alert_mode="replace"/>
  </feed>
</cgen>"#;
        let parsed: CgenConfig = quick_xml::de::from_str(xml).expect("parse");
        let err = parsed
            .enabled_feeds()
            .expect_err("missing ladder should fail");

        assert!(err.to_string().contains("no enabled video renditions"));
    }
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            hard_reset_ms: default_hard_reset_ms(),
            max_audio_frames_per_video: default_max_audio_frames_per_video(),
            source_buffer_ms: default_source_buffer_ms(),
            reconnect_initial_ms: default_reconnect_initial_ms(),
            reconnect_max_ms: default_reconnect_max_ms(),
            status_interval_ms: default_status_interval_ms(),
        }
    }
}

impl Default for BannerConfig {
    fn default() -> Self {
        Self {
            mode: auto_mode(),
            ticker_height: default_ticker_height(),
            font: String::new(),
            font_size: default_font_size(),
            scroll_speed: default_scroll_speed(),
            background_enabled: true,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            idle: source_audio(),
            alert_mode: replace_audio(),
            mute_standby_routine: true,
        }
    }
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            font: String::new(),
            font_size: default_font_size(),
            banner_height: default_ticker_height(),
        }
    }
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            format: String::new(),
            x: default_text_x(),
            y: default_clock_y(),
            font_size: default_clock_font_size(),
            color: String::new(),
        }
    }
}

impl Default for TextConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            x: default_text_x(),
            y: default_text_y(),
            font_size: default_insert_font_size(),
            color: String::new(),
            content: String::new(),
        }
    }
}

impl Default for StateConfig {
    fn default() -> Self {
        Self {
            mode: release_mode(),
            smpte_bars: false,
            updated_at: String::new(),
        }
    }
}

impl Default for StandbyConfig {
    fn default() -> Self {
        Self {
            mode: standby_banner_mode(),
            text: default_standby_text(),
            font_size: 0,
            y_percent: default_standby_y_percent(),
        }
    }
}
