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
    #[serde(rename = "@enabled", default = "default_true")]
    pub(crate) enabled: bool,
    pub(crate) input: EndpointConfig,
    pub(crate) output: EndpointConfig,
    pub(crate) video: VideoConfig,
    #[serde(default)]
    pub(crate) audio: AudioConfig,
    #[serde(default)]
    pub(crate) banner: BannerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct EndpointConfig {
    #[serde(rename = "@url")]
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

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VideoConfig {
    #[serde(rename = "@width")]
    pub(crate) width: u32,
    #[serde(rename = "@height")]
    pub(crate) height: u32,
    #[serde(rename = "@fps", default)]
    pub(crate) fps: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct AudioConfig {
    #[serde(rename = "@idle", default = "source_audio")]
    pub(crate) idle: String,
    #[serde(rename = "@alert_mode", default = "replace_audio")]
    pub(crate) alert_mode: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct BannerConfig {
    #[serde(rename = "@mode", default = "auto_mode")]
    pub(crate) mode: String,
    #[serde(rename = "@ticker_height", default = "default_ticker_height")]
    pub(crate) ticker_height: u32,
    #[serde(rename = "@font", default)]
    pub(crate) font: String,
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
        if self.input.url.trim().is_empty() {
            bail!("cgen feed {} input url is required", self.id);
        }
        if self.output.url.trim().is_empty() {
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
    96
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
    <video width="1280" height="720" fps="source"/>
    <audio idle="source" alert_mode="replace"/>
    <banner mode="auto" ticker_height="96" font="Zalando Sans SemiExpanded"/>
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
    fn wildcard_feed_matches_any_feed() {
        let feed = FeedConfig {
            id: "*".to_string(),
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
            video: VideoConfig {
                width: 1280,
                height: 720,
                fps: "source".to_string(),
            },
            audio: AudioConfig {
                idle: "source".to_string(),
                alert_mode: "replace".to_string(),
            },
            banner: BannerConfig::default(),
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
        }
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            idle: source_audio(),
            alert_mode: replace_audio(),
        }
    }
}
