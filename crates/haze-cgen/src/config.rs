use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
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
    #[serde(default)]
    pub(crate) input: EndpointConfig,
    #[serde(default)]
    pub(crate) output: EndpointConfig,
    #[serde(rename = "programInput", default)]
    pub(crate) program_input: EndpointConfig,
    #[serde(rename = "priorityInput", default)]
    pub(crate) priority_input: PriorityInputConfig,
    #[serde(rename = "programOutput", default)]
    pub(crate) program_output: EndpointConfig,
    #[serde(rename = "alertOutput", default)]
    pub(crate) alert_output: EndpointConfig,
    pub(crate) video: VideoConfig,
    #[serde(default)]
    pub(crate) audio: AudioConfig,
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
    #[serde(rename = "@url", default)]
    pub(crate) url: String,
    #[serde(rename = "@format", default)]
    pub(crate) format: String,
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
    #[serde(rename = "@duck_db", default)]
    pub(crate) duck_db: String,
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
    #[serde(rename = "@x", default)]
    pub(crate) x: i32,
    #[serde(rename = "@y", default)]
    pub(crate) y: i32,
    #[serde(rename = "@background_color", default)]
    pub(crate) background_color: String,
    #[serde(rename = "@background_gradient_color", default)]
    pub(crate) background_gradient_color: String,
    #[serde(rename = "@background_enabled", default = "default_true")]
    pub(crate) background_enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GraphicsConfig {
    #[serde(rename = "@background_color", default)]
    pub(crate) background_color: String,
    #[serde(rename = "@font", default)]
    pub(crate) font: String,
    #[serde(rename = "@font_size", default = "default_font_size")]
    pub(crate) font_size: u32,
    #[serde(rename = "@text_x", default = "default_text_x")]
    pub(crate) text_x: i32,
    #[serde(rename = "@text_y", default = "default_text_y")]
    pub(crate) text_y: i32,
    #[serde(rename = "@banner_x", default)]
    pub(crate) banner_x: i32,
    #[serde(rename = "@banner_y", default)]
    pub(crate) banner_y: i32,
    #[serde(rename = "@banner_width", default)]
    pub(crate) banner_width: u32,
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
        if !self.audio.alert_mode.eq_ignore_ascii_case("replace") {
            bail!("cgen feed {} audio alert_mode must be replace", self.id);
        }
        Ok(())
    }

    pub(crate) fn program_input_url(&self) -> &str {
        if self.program_input.url.trim().is_empty() {
            &self.input.url
        } else {
            &self.program_input.url
        }
    }

    pub(crate) fn program_output_url(&self) -> &str {
        if self.program_output.url.trim().is_empty() {
            &self.output.url
        } else {
            &self.program_output.url
        }
    }

    pub(crate) fn output(&self) -> &EndpointConfig {
        if self.program_output.url.trim().is_empty() {
            &self.output
        } else {
            &self.program_output
        }
    }
}

pub(crate) fn load_config(path: &Path) -> Result<CgenConfig> {
    let raw =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let config: CgenConfig = quick_xml::de::from_str(&raw)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(config)
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

fn source_audio() -> String {
    "source".to_string()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_disabled_example_feed() {
        let xml = r#"
<cgen enabled="true">
  <feed id="*" enabled="false">
    <input url="udp://127.0.0.1:5000" format="mpegts"/>
    <output url="udp://127.0.0.1:5001" format="mpegts" vcodec="libx264" acodec="aac"/>
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
    <priorityInput feed_id="CAP-IT-ALL" format="priority-audio"/>
    <programOutput url="udp://239.0.0.2:9001?pkt_size=1316" format="mpegts" vcodec="libx264" acodec="aac"/>
    <alertOutput url="udp://239.0.0.2:9001?pkt_size=1316" format="mpegts" vcodec="libx264" acodec="aac"/>
    <video width="1920" height="1080" fps="30000/1001" interlaced="true" field_order="tff" standard="atsc"/>
    <audio idle="source" alert_mode="replace" duck_db="-18"/>
    <banner mode="auto" ticker_height="128" font="Arial" font_size="58" scroll_speed="5" background_gradient_color="#7f1d1d" background_enabled="true"/>
    <graphics background_color="#000000" font="Arial" font_size="58"/>
    <clock enabled="true" x="48" y="48"/>
    <text enabled="true" x="48" y="96">Hello</text>
    <state mode="overlay" smpte_bars="true"/>
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
        assert_eq!(feeds[0].video.width, 1920);
        assert!(feeds[0].video.interlaced);
        assert_eq!(feeds[0].video.field_order, "tff");
        assert_eq!(feeds[0].video.standard, "atsc");
        assert_eq!(feeds[0].banner.scroll_speed, 5);
        assert_eq!(feeds[0].banner.background_gradient_color, "#7f1d1d");
        assert!(feeds[0].clock.enabled);
        assert!(feeds[0].state.smpte_bars);
    }

    #[test]
    fn wildcard_feed_matches_any_feed() {
        let feed = FeedConfig {
            id: "*".to_string(),
            name: String::new(),
            enabled: true,
            input: EndpointConfig {
                url: "udp://in".to_string(),
                format: "mpegts".to_string(),
                vcodec: String::new(),
                acodec: String::new(),
                video_bitrate_kbps: None,
                audio_bitrate_kbps: None,
            },
            output: EndpointConfig {
                url: "udp://out".to_string(),
                format: "mpegts".to_string(),
                vcodec: String::new(),
                acodec: String::new(),
                video_bitrate_kbps: None,
                audio_bitrate_kbps: None,
            },
            program_input: EndpointConfig::default(),
            priority_input: PriorityInputConfig::default(),
            program_output: EndpointConfig::default(),
            alert_output: EndpointConfig::default(),
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
                duck_db: String::new(),
            },
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
        };
        assert!(feed.matches_feed("sk-0001"));
        assert!(feed.matches_feed("CAP-IT-ALL"));
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
            x: 0,
            y: 0,
            background_color: String::new(),
            background_gradient_color: String::new(),
            background_enabled: true,
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            idle: source_audio(),
            alert_mode: replace_audio(),
            duck_db: String::new(),
        }
    }
}

impl Default for GraphicsConfig {
    fn default() -> Self {
        Self {
            background_color: String::new(),
            font: String::new(),
            font_size: default_font_size(),
            text_x: default_text_x(),
            text_y: default_text_y(),
            banner_x: 0,
            banner_y: 0,
            banner_width: 0,
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
