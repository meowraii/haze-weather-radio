use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use serde::Deserialize;
use serde_json::Value;

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
    #[serde(rename = "program", default)]
    pub(crate) program: Vec<ProgramSectionConfig>,
    #[serde(default)]
    pub(crate) priority: PrioritySectionConfig,
    #[serde(default)]
    pub(crate) media: MediaSectionConfig,
    #[serde(default)]
    pub(crate) presentation: PresentationSectionConfig,
    #[serde(default)]
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
    #[serde(skip)]
    pub(crate) encoder: FeedEncoderSettings,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct EndpointConfig {
    #[serde(rename = "@url", default)]
    pub(crate) url: String,
    #[serde(rename = "@type", default)]
    pub(crate) input_type: String,
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
    #[serde(rename = "@browser_auto_size", default)]
    pub(crate) browser_auto_size: String,
    #[serde(rename = "@browser_width", default)]
    pub(crate) browser_width: Option<u32>,
    #[serde(rename = "@browser_height", default)]
    pub(crate) browser_height: Option<u32>,
    #[serde(rename = "@browser_fps", default)]
    pub(crate) browser_fps: Option<u32>,
    #[serde(rename = "@hardware_decoder_enabled", default)]
    pub(crate) hardware_decoder_enabled: String,
    #[serde(rename = "@hardware_decoder", default)]
    pub(crate) hardware_decoder: String,
}

impl EndpointConfig {
    fn has_values(&self) -> bool {
        !self.url.trim().is_empty()
            || !self.input_type.trim().is_empty()
            || !self.format.trim().is_empty()
            || !self.vcodec.trim().is_empty()
            || !self.acodec.trim().is_empty()
            || self.video_bitrate_kbps.is_some()
            || self.audio_bitrate_kbps.is_some()
            || !self.browser_auto_size.trim().is_empty()
            || self.browser_width.is_some()
            || self.browser_height.is_some()
            || self.browser_fps.is_some()
            || !self.hardware_decoder_enabled.trim().is_empty()
            || !self.hardware_decoder.trim().is_empty()
    }

    pub(crate) fn input_type(&self) -> &str {
        match self.input_type.trim().to_ascii_lowercase().as_str() {
            "browser" | "cef" | "browser_source" => "browser",
            _ => "stream",
        }
    }

    pub(crate) fn is_browser_source(&self) -> bool {
        self.input_type() == "browser"
    }

    pub(crate) fn browser_auto_size(&self) -> bool {
        xml_bool_text(&self.browser_auto_size, true)
    }

    pub(crate) fn browser_size(&self, fallback_width: u32, fallback_height: u32) -> (u32, u32) {
        if self.browser_auto_size() {
            return (fallback_width, fallback_height);
        }
        (
            self.browser_width.unwrap_or(fallback_width).clamp(1, 7680),
            self.browser_height
                .unwrap_or(fallback_height)
                .clamp(1, 4320),
        )
    }

    pub(crate) fn browser_fps(&self) -> u32 {
        match self.browser_fps.unwrap_or(60) {
            0 => 0,
            value => value.clamp(5, 120),
        }
    }

    pub(crate) fn hardware_decoder(&self) -> Option<&str> {
        if !xml_bool_text(&self.hardware_decoder_enabled, false) {
            return None;
        }
        let decoder = self.hardware_decoder.trim();
        if decoder.is_empty() || !decoder.chars().all(gst_element_char) {
            return None;
        }
        Some(decoder)
    }
}

fn gst_element_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_' || ch == '-'
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

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct VideoConfig {
    #[serde(rename = "@width", default)]
    pub(crate) width: u32,
    #[serde(rename = "@height", default)]
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

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct FeedEncoderSettings {
    pub(crate) video: EncoderCodecSettings,
    pub(crate) audio: EncoderCodecSettings,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct EncoderCodecSettings {
    pub(crate) codec: String,
    pub(crate) bitrate_kbps: Option<u32>,
    pub(crate) gop: Option<u32>,
    pub(crate) bframes: Option<u32>,
    pub(crate) preset: String,
    pub(crate) tune: String,
    pub(crate) profile: String,
    pub(crate) level: String,
    pub(crate) options: Vec<EncoderOptionConfig>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
pub(crate) struct EncoderOptionConfig {
    #[serde(rename = "@name", default)]
    pub(crate) name: String,
    #[serde(rename = "@value", default)]
    pub(crate) value: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename = "cgenEncoders")]
struct CgenEncodersConfig {
    #[serde(rename = "feed", default)]
    feeds: Vec<FeedEncoderConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FeedEncoderConfig {
    #[serde(rename = "@id", default)]
    id: String,
    #[serde(default)]
    video: EncoderCodecConfig,
    #[serde(default)]
    audio: EncoderCodecConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct EncoderCodecConfig {
    #[serde(rename = "@codec", default)]
    codec: String,
    #[serde(rename = "@bitrate_kbps", default)]
    bitrate_kbps: Option<u32>,
    #[serde(rename = "@gop", default)]
    gop: Option<u32>,
    #[serde(rename = "@bframes", default)]
    bframes: Option<u32>,
    #[serde(rename = "@preset", default)]
    preset: String,
    #[serde(rename = "@tune", default)]
    tune: String,
    #[serde(rename = "@profile", default)]
    profile: String,
    #[serde(rename = "@level", default)]
    level: String,
    #[serde(rename = "option", default)]
    options: Vec<EncoderOptionConfig>,
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
    #[serde(rename = "@font_weight", default = "regular_font_weight")]
    pub(crate) font_weight: String,
    #[serde(rename = "@font_size", default = "default_font_size")]
    pub(crate) font_size: u32,
    #[serde(rename = "@scroll_speed", default = "default_scroll_speed")]
    pub(crate) scroll_speed: u32,
    #[serde(rename = "@scroll_repeat_mode", default = "default_scroll_repeat_mode")]
    pub(crate) scroll_repeat_mode: String,
    #[serde(rename = "@after_eom_repeats", default)]
    pub(crate) after_eom_repeats: u32,
    #[serde(rename = "@fixed_repeats", default = "default_fixed_repeats")]
    pub(crate) fixed_repeats: u32,
    #[serde(rename = "@background_enabled", default = "default_true")]
    pub(crate) background_enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GraphicsConfig {
    #[serde(rename = "@font", default)]
    pub(crate) font: String,
    #[serde(rename = "@font_weight", default = "regular_font_weight")]
    pub(crate) font_weight: String,
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
    #[serde(rename = "@sunny_cat", default)]
    pub(crate) sunny_cat: bool,
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

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ProgramSectionConfig {
    #[serde(default)]
    input: EndpointConfig,
    #[serde(default)]
    output: EndpointConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PrioritySectionConfig {
    #[serde(default)]
    input: PriorityInputConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct MediaSectionConfig {
    #[serde(default)]
    video: Option<VideoConfig>,
    #[serde(default)]
    audio: Option<AudioConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PresentationSectionConfig {
    #[serde(default)]
    banner: Option<PresentationBannerSectionConfig>,
    #[serde(default)]
    graphics: Option<PresentationGraphicsSectionConfig>,
    #[serde(default)]
    clock: Option<ClockConfig>,
    #[serde(default)]
    text: Option<TextConfig>,
    #[serde(default)]
    standby: Option<StandbyConfig>,
    #[serde(default)]
    state: Option<StateConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PresentationBannerSectionConfig {
    #[serde(rename = "@mode", default)]
    mode: String,
    #[serde(rename = "@background_enabled", default)]
    background_enabled: Option<bool>,
    #[serde(default)]
    font: FontSectionConfig,
    #[serde(default)]
    ticker: TickerSectionConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct PresentationGraphicsSectionConfig {
    #[serde(default)]
    font: FontSectionConfig,
    #[serde(rename = "@banner_height", default)]
    banner_height: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FontSectionConfig {
    #[serde(rename = "@family", default)]
    family: String,
    #[serde(rename = "@weight", default)]
    weight: String,
    #[serde(rename = "@size", default)]
    size: u32,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TickerSectionConfig {
    #[serde(rename = "@height", default)]
    height: u32,
    #[serde(rename = "@speed", default)]
    speed: u32,
    #[serde(default)]
    repeat: TickerRepeatSectionConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TickerRepeatSectionConfig {
    #[serde(rename = "@mode", default)]
    mode_name: String,
    #[serde(rename = "@after_eom", default)]
    after_eom: u32,
    #[serde(rename = "@count", default)]
    count: u32,
}

impl CgenConfig {
    fn apply_nested_sections(&mut self) {
        for feed in &mut self.feeds {
            feed.apply_nested_sections();
        }
    }

    fn apply_encoder_settings(&mut self, encoders: CgenEncodersConfig) {
        for encoder_feed in encoders.feeds {
            if encoder_feed.id.trim().is_empty() {
                continue;
            }
            if let Some(feed) = self
                .feeds
                .iter_mut()
                .find(|feed| feed.id.eq_ignore_ascii_case(encoder_feed.id.trim()))
            {
                feed.encoder = FeedEncoderSettings::from_config(encoder_feed);
            }
        }
    }

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

impl FeedEncoderSettings {
    fn from_config(config: FeedEncoderConfig) -> Self {
        Self {
            video: EncoderCodecSettings::from_config(config.video),
            audio: EncoderCodecSettings::from_config(config.audio),
        }
    }
}

impl EncoderCodecSettings {
    fn from_config(config: EncoderCodecConfig) -> Self {
        Self {
            codec: config.codec.trim().to_string(),
            bitrate_kbps: config.bitrate_kbps.filter(|value| *value > 0),
            gop: config.gop.filter(|value| *value > 0),
            bframes: config.bframes,
            preset: config.preset.trim().to_string(),
            tune: config.tune.trim().to_string(),
            profile: config.profile.trim().to_string(),
            level: config.level.trim().to_string(),
            options: config
                .options
                .into_iter()
                .filter(|option| !option.name.trim().is_empty())
                .map(|option| EncoderOptionConfig {
                    name: option.name.trim().to_string(),
                    value: option.value.trim().to_string(),
                })
                .collect(),
        }
    }

    pub(crate) fn applies_to(&self, codec: &str) -> bool {
        self.codec.trim().is_empty() || self.codec.trim().eq_ignore_ascii_case(codec.trim())
    }

    pub(crate) fn bitrate_or(&self, fallback: Option<u32>) -> Option<u32> {
        self.bitrate_kbps.or(fallback)
    }
}

impl FeedConfig {
    fn apply_nested_sections(&mut self) {
        for program in &self.program {
            if program.input.has_values() {
                self.program_input = program.input.clone();
            }
            if program.output.has_values() {
                self.program_output = program.output.clone();
            }
        }
        if !self.priority.input.feed_id.trim().is_empty()
            || !self.priority.input.audio_source.trim().is_empty()
            || !self.priority.input.format.trim().is_empty()
        {
            self.priority_input = self.priority.input.clone();
        }
        if let Some(video) = self.media.video.as_ref() {
            self.video = video.clone();
        }
        if let Some(audio) = self.media.audio.as_ref() {
            self.audio = audio.clone();
        }
        self.apply_presentation_section();
    }

    fn apply_presentation_section(&mut self) {
        if let Some(banner) = self.presentation.banner.as_ref() {
            if !banner.mode.trim().is_empty() {
                self.banner.mode = banner.mode.clone();
            }
            if let Some(background_enabled) = banner.background_enabled {
                self.banner.background_enabled = background_enabled;
            }
            if banner.ticker.height > 0 {
                self.banner.ticker_height = banner.ticker.height;
            }
            if banner.ticker.speed > 0 {
                self.banner.scroll_speed = banner.ticker.speed;
            }
            if !banner.ticker.repeat.mode_name.trim().is_empty() {
                self.banner.scroll_repeat_mode = banner.ticker.repeat.mode_name.clone();
            }
            if banner.ticker.repeat.after_eom > 0 {
                self.banner.after_eom_repeats = banner.ticker.repeat.after_eom;
            }
            if banner.ticker.repeat.count > 0 {
                self.banner.fixed_repeats = banner.ticker.repeat.count;
            }
            if !banner.font.family.trim().is_empty() {
                self.banner.font = banner.font.family.clone();
            }
            if !banner.font.weight.trim().is_empty() {
                self.banner.font_weight = banner.font.weight.clone();
            }
            if banner.font.size > 0 {
                self.banner.font_size = banner.font.size;
            }
        }

        if let Some(graphics) = self.presentation.graphics.as_ref() {
            if !graphics.font.family.trim().is_empty() {
                self.graphics.font = graphics.font.family.clone();
            }
            if !graphics.font.weight.trim().is_empty() {
                self.graphics.font_weight = graphics.font.weight.clone();
            }
            if graphics.font.size > 0 {
                self.graphics.font_size = graphics.font.size;
            }
            if graphics.banner_height > 0 {
                self.graphics.banner_height = graphics.banner_height;
            }
        }
        if let Some(clock) = self.presentation.clock.as_ref() {
            self.clock = clock.clone();
        }
        if let Some(text) = self.presentation.text.as_ref() {
            self.text = text.clone();
        }
        if let Some(standby) = self.presentation.standby.as_ref() {
            self.standby = standby.clone();
        }
        if let Some(state) = self.presentation.state.as_ref() {
            self.state = state.clone();
        }
    }

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

    pub(crate) fn redacted_program_input_url(&self) -> String {
        redact_endpoint_url(self.program_input_url())
    }

    pub(crate) fn redacted_program_output_url(&self) -> String {
        redact_endpoint_url(self.program_output_url())
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

pub(crate) fn redact_feed_endpoint_status(feed: &FeedConfig, value: &mut Value) {
    let replacements = [
        (
            feed.program_input_url().trim(),
            feed.redacted_program_input_url(),
        ),
        (
            feed.program_output_url().trim(),
            feed.redacted_program_output_url(),
        ),
    ];
    redact_status_value(value, &replacements);
}

pub(crate) fn redacted_pipeline_description(feed: &FeedConfig, text: &str) -> String {
    let mut value = Value::String(text.to_string());
    redact_feed_endpoint_status(feed, &mut value);
    value.as_str().unwrap_or(text).to_string()
}

fn redact_status_value(value: &mut Value, replacements: &[(&str, String); 2]) {
    match value {
        Value::String(text) => {
            for (raw, redacted) in replacements {
                if !raw.is_empty() && *raw != redacted {
                    *text = text.replace(raw, redacted);
                    let quoted = format!("\"{}\"", raw.replace('\\', "\\\\").replace('"', "\\\""));
                    let redacted_quoted = format!(
                        "\"{}\"",
                        redacted.replace('\\', "\\\\").replace('"', "\\\"")
                    );
                    *text = text.replace(&quoted, &redacted_quoted);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_status_value(item, replacements);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                redact_status_value(item, replacements);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn redact_endpoint_url(value: &str) -> String {
    let value = value.trim();
    let Some(scheme_end) = value.find("://") else {
        return value.to_string();
    };
    let scheme = &value[..scheme_end];
    let rest = &value[scheme_end + 3..];
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    if authority.is_empty() {
        return format!("{scheme}://...");
    }
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    format!("{scheme}://{host}/...")
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
    let expanded = expand_env_vars(&raw);
    reject_legacy_cgen_xml(path, &expanded)?;
    let mut config: CgenConfig = quick_xml::de::from_str(&expanded)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    config.apply_nested_sections();
    if let Some(encoders) = load_encoder_settings(path)? {
        config.apply_encoder_settings(encoders);
    }
    Ok(config)
}

fn load_encoder_settings(cgen_path: &Path) -> Result<Option<CgenEncodersConfig>> {
    let Some(parent) = cgen_path.parent() else {
        return Ok(None);
    };
    let path = parent.join("cgen-encoders.xml");
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("failed to read {}", path.display())),
    };
    let expanded = expand_env_vars(&raw);
    let encoders: CgenEncodersConfig = quick_xml::de::from_str(&expanded)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(encoders))
}

fn expand_env_vars(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '$' {
            out.push(ch);
            continue;
        }
        if chars.peek() == Some(&'{') {
            chars.next();
            let mut name = String::new();
            for next in chars.by_ref() {
                if next == '}' {
                    break;
                }
                name.push(next);
            }
            out.push_str(&std::env::var(name).unwrap_or_default());
            continue;
        }
        let mut name = String::new();
        while let Some(next) = chars.peek().copied() {
            if next == '_' || next.is_ascii_alphanumeric() {
                name.push(next);
                chars.next();
            } else {
                break;
            }
        }
        if name.is_empty() {
            out.push('$');
        } else {
            out.push_str(&std::env::var(name).unwrap_or_default());
        }
    }
    out
}

fn reject_legacy_cgen_xml(path: &Path, raw: &str) -> Result<()> {
    let mut reader = Reader::from_str(raw);
    reader.config_mut().trim_text(true);
    let mut stack: Vec<String> = Vec::new();
    loop {
        match reader.read_event() {
            Ok(Event::Start(event)) => {
                let element_name = event.name();
                let name_bytes = element_name.as_ref();
                let name = std::str::from_utf8(name_bytes).unwrap_or_default();
                let parent = stack.last().map(String::as_str).unwrap_or_default();
                if legacy_element(parent, name) {
                    bail!(
                        "legacy cgen <{}> element is no longer supported in {}; use programInput, priorityInput, and programOutput",
                        name,
                        path.display()
                    );
                }
                reject_legacy_attrs(path, name, event.attributes())?;
                stack.push(name.to_string());
            }
            Ok(Event::Empty(event)) => {
                let element_name = event.name();
                let name_bytes = element_name.as_ref();
                let name = std::str::from_utf8(name_bytes).unwrap_or_default();
                let parent = stack.last().map(String::as_str).unwrap_or_default();
                if legacy_element(parent, name) {
                    bail!(
                        "legacy cgen <{}> element is no longer supported in {}; use programInput, priorityInput, and programOutput",
                        name,
                        path.display()
                    );
                }
                reject_legacy_attrs(path, name, event.attributes())?;
            }
            Ok(Event::End(_)) => {
                stack.pop();
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

fn reject_legacy_attrs<'a>(
    path: &Path,
    name: &str,
    attributes: quick_xml::events::attributes::Attributes<'a>,
) -> Result<()> {
    for attr in attributes {
        let attr = attr
            .with_context(|| format!("failed to read cgen XML attribute in {}", path.display()))?;
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
    Ok(())
}

fn legacy_element(parent: &str, element: &str) -> bool {
    (element == "output" && parent != "program")
        || element == "alertOutput"
        || (element == "input" && !matches!(parent, "program" | "priority"))
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

fn regular_font_weight() -> String {
    "regular".to_string()
}

fn default_scroll_speed() -> u32 {
    8
}

fn default_scroll_repeat_mode() -> String {
    "until_audio_end".to_string()
}

fn default_fixed_repeats() -> u32 {
    1
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
    fn load_config_merges_cgen_encoder_settings() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cgen.xml");
        std::fs::write(
            &path,
            r#"
<cgen enabled="true">
  <feed id="CAP" enabled="true">
    <program>
      <input url="udp://in" format="mpegts"></input>
      <output url="udp://out" format="mpegts" vcodec="x264enc" acodec="avenc_ac3"></output>
    </program>
    <media>
      <video width="1280" height="720" fps="source"></video>
      <audio idle="source" alert_mode="replace"></audio>
    </media>
    <ladder>
      <video id="hd" enabled="true" width="1280" height="720" fps="source" vcodec="x264enc" bitrate_kbps="6000"/>
      <audio id="stereo" enabled="true" channels="2" acodec="avenc_ac3" bitrate_kbps="192"/>
    </ladder>
  </feed>
</cgen>"#,
        )
        .expect("write cgen");
        std::fs::write(
            dir.path().join("cgen-encoders.xml"),
            r#"
<cgenEncoders>
  <feed id="CAP">
    <video codec="x264enc" bitrate_kbps="7000" gop="30" bframes="2" preset="veryfast" tune="zerolatency"></video>
    <audio codec="avenc_ac3" bitrate_kbps="256" profile="main"></audio>
  </feed>
</cgenEncoders>"#,
        )
        .expect("write encoders");

        let config = load_config(&path).expect("config");
        let feed = &config.feeds[0];

        assert_eq!(feed.encoder.video.codec, "x264enc");
        assert_eq!(feed.encoder.video.bitrate_kbps, Some(7000));
        assert_eq!(feed.encoder.video.gop, Some(30));
        assert_eq!(feed.encoder.video.bframes, Some(2));
        assert_eq!(feed.encoder.video.preset, "veryfast");
        assert_eq!(feed.encoder.audio.bitrate_kbps, Some(256));
    }

    #[test]
    fn load_config_accepts_nested_cgen_shape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cgen.xml");
        std::fs::write(
            &path,
            r#"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <program>
      <input url="udp://127.0.0.1:5000" format="mpegts"/>
      <output url="udp://127.0.0.1:5001" format="mpegts" vcodec="mpeg2video" acodec="ac3"/>
    </program>
    <priority>
      <input feed_id="*" audio_source="both" format="priority-audio"/>
    </priority>
    <media>
      <video width="1280" height="720" fps="30000/1001"/>
      <audio idle="source" alert_mode="replace" mute_standby_routine="true"/>
    </media>
    <ladder>
      <video id="p720" enabled="true" width="1280" height="720" fps="30000/1001"/>
      <audio id="stereo" enabled="true" channels="2"/>
    </ladder>
    <presentation>
      <banner mode="ticker" background_enabled="true">
        <font family="Arial" weight="regular" size="58"/>
        <ticker height="96" speed="12">
          <repeat mode="until_audio_end" after_eom="2" count="1"/>
        </ticker>
      </banner>
    </presentation>
  </feed>
</cgen>"#,
        )
        .expect("write config");

        let config = load_config(&path).expect("nested config");
        let feeds = config.enabled_feeds().expect("feeds");
        assert_eq!(feeds[0].program_input_url(), "udp://127.0.0.1:5000");
        assert_eq!(feeds[0].priority_input.feed_id, "*");
        assert_eq!(feeds[0].video.width, 1280);
        assert_eq!(feeds[0].banner.ticker_height, 96);
        assert_eq!(feeds[0].banner.scroll_speed, 12);
        assert_eq!(feeds[0].banner.scroll_repeat_mode, "until_audio_end");
        assert_eq!(feeds[0].banner.after_eom_repeats, 2);
    }

    #[test]
    fn load_config_expands_environment_variables_at_runtime_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cgen.xml");
        std::env::set_var(
            "HAZE_CGEN_TEST_OUTPUT",
            "rtmp://example.invalid/live/super-secret-key",
        );
        std::fs::write(
            &path,
            r#"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <programInput url="udp://127.0.0.1:5000" format="mpegts"/>
    <priorityInput feed_id="CAP-IT-ALL" audio_source="priority"/>
    <programOutput url="${HAZE_CGEN_TEST_OUTPUT}" format="rtmp" vcodec="h264" acodec="ac3"/>
    <video width="1920" height="1080" fps="30000/1001"/>
    <audio idle="source" alert_mode="replace"/>
    <ladder>
      <video id="hd" enabled="true" width="1920" height="1080"/>
      <audio id="stereo" enabled="true" channels="2"/>
    </ladder>
  </feed>
</cgen>"#,
        )
        .expect("write config");

        let config = load_config(&path).expect("config");
        let feeds = config.enabled_feeds().expect("feeds");
        assert_eq!(
            feeds[0].program_output_url(),
            "rtmp://example.invalid/live/super-secret-key"
        );
    }

    #[test]
    fn redacts_configured_endpoint_values_from_status() {
        let config: CgenConfig = quick_xml::de::from_str(
            r#"
<cgen enabled="true">
  <feed id="CAP-IT-ALL" enabled="true">
    <programInput url="http://user:pass@example.invalid/input.ts" format="mpegts"/>
    <priorityInput feed_id="CAP-IT-ALL" audio_source="priority"/>
    <programOutput url="rtmp://publish.example.invalid/live/secret-key" format="rtmp" vcodec="h264" acodec="ac3"/>
    <video width="1920" height="1080" fps="30000/1001"/>
    <audio idle="source" alert_mode="replace"/>
    <ladder>
      <video id="hd" enabled="true" width="1920" height="1080"/>
      <audio id="stereo" enabled="true" channels="2"/>
    </ladder>
  </feed>
</cgen>"#,
        )
        .expect("parse");
        let feed = config.feeds.into_iter().next().expect("feed");
        let mut status = serde_json::json!({
            "pipeline_description": "uridecodebin uri=\"http://user:pass@example.invalid/input.ts\" ! rtmpsink location=\"rtmp://publish.example.invalid/live/secret-key\"",
            "nested": {
                "output": "rtmp://publish.example.invalid/live/secret-key"
            }
        });

        redact_feed_endpoint_status(&feed, &mut status);
        let rendered = status.to_string();
        assert!(!rendered.contains("user:pass"));
        assert!(!rendered.contains("secret-key"));
        assert!(rendered.contains("http://example.invalid/..."));
        assert!(rendered.contains("rtmp://publish.example.invalid/..."));
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
    <banner mode="auto" ticker_height="128" font="Arial" font_weight="regular" font_size="58" scroll_speed="5" background_enabled="true"/>
    <graphics font="Arial" font_weight="regular" font_size="58"/>
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
        assert_eq!(feeds[0].banner.font_weight, "regular");
        assert_eq!(feeds[0].graphics.font_weight, "regular");
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
                ..Default::default()
            },
            priority_input: PriorityInputConfig::default(),
            program_output: EndpointConfig {
                url: "udp://out".to_string(),
                format: "mpegts".to_string(),
                vcodec: String::new(),
                acodec: String::new(),
                video_bitrate_kbps: None,
                audio_bitrate_kbps: None,
                ..Default::default()
            },
            program: Default::default(),
            priority: Default::default(),
            media: Default::default(),
            presentation: Default::default(),
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
            encoder: Default::default(),
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
            font_weight: regular_font_weight(),
            font_size: default_font_size(),
            scroll_speed: default_scroll_speed(),
            scroll_repeat_mode: default_scroll_repeat_mode(),
            after_eom_repeats: 0,
            fixed_repeats: default_fixed_repeats(),
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
            font_weight: regular_font_weight(),
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
            sunny_cat: false,
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
