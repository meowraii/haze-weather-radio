use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tokio::sync::watch;
use tokio::time::{sleep, Duration};
use tracing::{info, warn};

use crate::config::FeedConfig;
use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

pub(crate) struct PipelineWorker {
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    ffmpeg: Option<String>,
    graphics_backend: String,
    base_dir: PathBuf,
}

impl PipelineWorker {
    pub(crate) fn new(
        feed: FeedConfig,
        state_rx: watch::Receiver<RuntimeState>,
        ffmpeg: Option<String>,
        graphics_backend: String,
        base_dir: PathBuf,
    ) -> Self {
        Self {
            feed,
            state_rx,
            ffmpeg,
            graphics_backend,
            base_dir,
        }
    }

    pub(crate) async fn run(mut self) -> Result<()> {
        if self.feed.audio.idle.trim() != "source" {
            bail!(
                "cgen feed {} only supports idle source audio in this version",
                self.feed.id
            );
        }
        let output = self.feed.output();
        info!(
            feed_id = %self.feed.id,
            name = %self.feed.name,
            input = %self.feed.program_input_url(),
            priority_feed = %self.feed.priority_input.feed_id,
            priority_input = %self.feed.priority_input.url,
            priority_format = %self.feed.priority_input.format,
            output = %self.feed.program_output_url(),
            alert_output = %self.feed.alert_output.url,
            video = format_args!("{}x{}", self.feed.video.width, self.feed.video.height),
            fps = %self.feed.video.fps,
            input_format = %self.feed.program_input.format,
            output_format = %output.format,
            alert_output_format = %self.feed.alert_output.format,
            vcodec = %output.vcodec,
            acodec = %output.acodec,
            video_bitrate_kbps = ?output.video_bitrate_kbps,
            audio_bitrate_kbps = ?output.audio_bitrate_kbps,
            duck_db = %self.feed.audio.duck_db,
            banner_mode = %self.feed.banner.mode,
            ticker_height = self.feed.banner.ticker_height,
            banner_font = %self.feed.banner.font,
            banner_font_size = self.feed.banner.font_size,
            banner_x = self.feed.banner.x,
            banner_y = self.feed.banner.y,
            banner_background_color = %self.feed.banner.background_color,
            banner_background_enabled = self.feed.banner.background_enabled,
            font = %self.feed.graphics.font,
            font_size = self.feed.graphics.font_size,
            background_color = %self.feed.graphics.background_color,
            text_x = self.feed.graphics.text_x,
            text_y = self.feed.graphics.text_y,
            banner_graphics_x = self.feed.graphics.banner_x,
            banner_graphics_y = self.feed.graphics.banner_y,
            banner_graphics_width = self.feed.graphics.banner_width,
            banner_graphics_height = self.feed.graphics.banner_height,
            clock_enabled = self.feed.clock.enabled,
            clock_format = %self.feed.clock.format,
            clock_x = self.feed.clock.x,
            clock_y = self.feed.clock.y,
            clock_font_size = self.feed.clock.font_size,
            clock_color = %self.feed.clock.color,
            text_enabled = self.feed.text.enabled,
            inserted_text_x = self.feed.text.x,
            inserted_text_y = self.feed.text.y,
            inserted_text_font_size = self.feed.text.font_size,
            inserted_text_color = %self.feed.text.color,
            inserted_text_len = self.feed.text.content.len(),
            cgen_mode = %self.feed.state.mode,
            smpte_bars = self.feed.state.smpte_bars,
            state_updated_at = %self.feed.state.updated_at,
            ffmpeg = ?self.ffmpeg,
            graphics_backend = %self.graphics_backend,
            "cgen feed pipeline supervised"
        );

        self.run_ffmpeg_relay().await
    }

    async fn run_ffmpeg_relay(&mut self) -> Result<()> {
        let ffmpeg = self.ffmpeg.clone().unwrap_or_else(|| "ffmpeg".to_string());
        let mut restart_delay = Duration::from_millis(500);
        loop {
            let state = self.state_rx.borrow().clone();
            let mode = desired_mode(&self.feed, &state);
            let args = ffmpeg_args(&self.feed, &mode, &self.base_dir);
            info!(
                feed_id = %self.feed.id,
                input = %self.feed.program_input_url(),
                output = %mode.output_url(&self.feed),
                mode = %mode.name(),
                visual_active = mode.visual_active(),
                priority_audio = mode.priority_audio_path().unwrap_or_default(),
                "starting cgen FFmpeg relay"
            );
            let mut child = tokio::process::Command::new(&ffmpeg)
                .args(&args)
                .kill_on_drop(true)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .with_context(|| format!("failed to start cgen FFmpeg relay using {ffmpeg}"))?;

            if let Some(stderr) = child.stderr.take() {
                let feed_id = self.feed.id.clone();
                tokio::spawn(async move {
                    log_ffmpeg_stderr(feed_id, stderr).await;
                });
            }

            tokio::select! {
                changed = self.state_rx.changed() => {
                    if changed.is_err() {
                        let _ = child.kill().await;
                        return Ok(());
                    }
                    self.log_state();
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    restart_delay = Duration::from_millis(500);
                }
                status = child.wait() => {
                    match status {
                        Ok(status) if status.success() => {
                            info!(feed_id = %self.feed.id, "cgen FFmpeg relay exited cleanly");
                            restart_delay = Duration::from_millis(500);
                        }
                        Ok(status) => {
                            warn!(feed_id = %self.feed.id, status = %status, "cgen FFmpeg relay exited; restarting");
                        }
                        Err(err) => {
                            warn!(feed_id = %self.feed.id, "cgen FFmpeg relay wait failed: {err}");
                        }
                    }
                    sleep(restart_delay).await;
                    restart_delay = (restart_delay * 2).min(Duration::from_secs(10));
                }
            }
        }
    }

    fn log_state(&self) {
        let state = self.state_rx.borrow();
        let visual = if self.feed.id == "*" {
            state.banner_for("*")
        } else {
            state.banner_for(&self.feed.id)
        };
        let audio = if self.feed.id == "*" {
            state.priority_audio_for("*")
        } else {
            state.priority_audio_for(&self.feed.id)
        };
        if visual.is_some() || audio.is_some() {
            info!(
                feed_id = %self.feed.id,
                visual_active = visual.is_some(),
                priority_audio = audio.and_then(|audio| audio.audio_path.as_ref()).map(|path| path.display().to_string()).unwrap_or_default(),
                "cgen alert state active"
            );
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum PipelineMode {
    Release,
    Overlay {
        banner: Option<BannerPayload>,
        audio: Option<PriorityAudio>,
        alert_audio_missing: bool,
    },
}

impl PipelineMode {
    fn name(&self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Overlay {
                audio,
                alert_audio_missing,
                ..
            } if audio.is_some() || *alert_audio_missing => "alert",
            Self::Overlay { .. } => "overlay",
        }
    }

    fn visual_active(&self) -> bool {
        matches!(self, Self::Overlay { .. })
    }

    fn priority_audio_path(&self) -> Option<String> {
        match self {
            Self::Overlay {
                audio: Some(audio), ..
            } => audio
                .audio_path
                .as_ref()
                .map(|path| path.display().to_string()),
            _ => None,
        }
    }

    fn output_url<'a>(&'a self, feed: &'a FeedConfig) -> &'a str {
        match self {
            Self::Overlay { audio: Some(_), .. }
            | Self::Overlay {
                alert_audio_missing: true,
                ..
            } if !feed.alert_output.url.trim().is_empty() => &feed.alert_output.url,
            _ => feed.program_output_url(),
        }
    }
}

fn desired_mode(feed: &FeedConfig, state: &RuntimeState) -> PipelineMode {
    let priority_feed = if feed.priority_input.feed_id.trim().is_empty() {
        feed.id.as_str()
    } else {
        feed.priority_input.feed_id.as_str()
    };
    let banner_feed = if feed.id.trim().is_empty() {
        priority_feed
    } else {
        feed.id.as_str()
    };
    let audio = state.priority_audio_for(priority_feed).cloned();
    let banner = if audio.is_some() {
        state.banner_for(banner_feed).cloned()
    } else {
        None
    };
    let alert_audio_missing = audio
        .as_ref()
        .is_some_and(|audio| audio.audio_path.is_none());
    let static_overlay = feed.state.mode.eq_ignore_ascii_case("overlay")
        || feed.state.smpte_bars
        || feed.text.enabled
        || feed.clock.enabled;
    if banner.is_some() || audio.is_some() || static_overlay {
        PipelineMode::Overlay {
            banner,
            audio,
            alert_audio_missing,
        }
    } else {
        PipelineMode::Release
    }
}

fn ffmpeg_args(feed: &FeedConfig, mode: &PipelineMode, base_dir: &Path) -> Vec<String> {
    match mode {
        PipelineMode::Release => ffmpeg_release_args(feed),
        PipelineMode::Overlay {
            banner,
            audio,
            alert_audio_missing,
        } => ffmpeg_overlay_args(
            feed,
            banner.as_ref(),
            audio.as_ref(),
            *alert_audio_missing,
            base_dir,
        ),
    }
}

fn ffmpeg_release_args(feed: &FeedConfig) -> Vec<String> {
    let output = feed.output();
    let mut args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "warning".to_string(),
        "-fflags".to_string(),
        "nobuffer".to_string(),
        "-flags".to_string(),
        "low_delay".to_string(),
        "-thread_queue_size".to_string(),
        "512".to_string(),
        "-i".to_string(),
        feed.program_input_url().to_string(),
        "-map".to_string(),
        "0".to_string(),
        "-c".to_string(),
        "copy".to_string(),
        "-muxdelay".to_string(),
        "0".to_string(),
        "-muxpreload".to_string(),
        "0".to_string(),
    ];
    if !output.format.trim().is_empty() {
        args.extend(["-f".to_string(), output.format.clone()]);
    }
    args.push(feed.program_output_url().to_string());
    args
}

fn ffmpeg_overlay_args(
    feed: &FeedConfig,
    banner: Option<&BannerPayload>,
    audio: Option<&PriorityAudio>,
    alert_audio_missing: bool,
    base_dir: &Path,
) -> Vec<String> {
    let output = if audio.is_some() || alert_audio_missing {
        if feed.alert_output.url.trim().is_empty() {
            feed.output()
        } else {
            &feed.alert_output
        }
    } else {
        feed.output()
    };
    let alert_audio_path = audio.and_then(|audio| audio.audio_path.as_ref());
    let replace_audio = audio.is_some() || alert_audio_missing;
    let mut args = vec![
        "-hide_banner".to_string(),
        "-loglevel".to_string(),
        "warning".to_string(),
    ];
    args.extend([
        "-fflags".to_string(),
        "nobuffer".to_string(),
        "-flags".to_string(),
        "low_delay".to_string(),
        "-thread_queue_size".to_string(),
        "512".to_string(),
        "-i".to_string(),
        feed.program_input_url().to_string(),
    ]);

    if replace_audio {
        let mut audio_input = vec!["-thread_queue_size".to_string(), "512".to_string()];
        if let Some(path) = alert_audio_path {
            audio_input.extend([
                "-re".to_string(),
                "-f".to_string(),
                "s16le".to_string(),
                "-ar".to_string(),
                audio
                    .map(|audio| audio.sample_rate.max(8_000).to_string())
                    .unwrap_or_else(|| "48000".to_string()),
                "-ac".to_string(),
                audio
                    .map(|audio| audio.channels.max(1).to_string())
                    .unwrap_or_else(|| "1".to_string()),
                "-i".to_string(),
                resolve_audio_path(base_dir, path),
            ]);
        } else {
            audio_input.extend([
                "-f".to_string(),
                "lavfi".to_string(),
                "-i".to_string(),
                "anullsrc=r=48000:cl=mono".to_string(),
            ]);
        }
        args.extend(audio_input);
    }

    let mut filters = vec![video_filter(feed, banner, audio)];
    if replace_audio && alert_audio_path.is_some() {
        filters.push("[1:a]apad[aout]".to_string());
    }
    args.extend(["-filter_complex".to_string(), filters.join(";")]);
    args.extend(["-map".to_string(), "[v]".to_string()]);
    if replace_audio {
        if alert_audio_path.is_some() {
            args.extend(["-map".to_string(), "[aout]".to_string()]);
        } else {
            args.extend(["-map".to_string(), "1:a:0".to_string()]);
        }
    } else {
        args.extend(["-map".to_string(), "0:a?".to_string()]);
    }
    args.extend([
        "-c:v".to_string(),
        codec_or(&output.vcodec, "libx264"),
        "-preset".to_string(),
        "ultrafast".to_string(),
        "-tune".to_string(),
        "zerolatency".to_string(),
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-g".to_string(),
        "30".to_string(),
        "-bf".to_string(),
        "0".to_string(),
        "-c:a".to_string(),
        codec_or(&output.acodec, "aac"),
    ]);
    if let Some(kbps) = output.video_bitrate_kbps {
        args.extend(["-b:v".to_string(), format!("{kbps}k")]);
    }
    if let Some(kbps) = output.audio_bitrate_kbps {
        args.extend(["-b:a".to_string(), format!("{kbps}k")]);
    }
    args.extend([
        "-muxdelay".to_string(),
        "0".to_string(),
        "-muxpreload".to_string(),
        "0".to_string(),
    ]);
    if !output.format.trim().is_empty() {
        args.extend(["-f".to_string(), output.format.clone()]);
    }
    args.push(
        if replace_audio && !feed.alert_output.url.trim().is_empty() {
            feed.alert_output.url.clone()
        } else {
            feed.program_output_url().to_string()
        },
    );
    args
}

fn video_filter(
    feed: &FeedConfig,
    banner: Option<&BannerPayload>,
    audio: Option<&PriorityAudio>,
) -> String {
    if feed.state.smpte_bars {
        return format!(
            "smptebars=size={}x{},format=yuv420p[v]",
            feed.video.width, feed.video.height
        );
    }
    let color = hex_color(
        audio
            .and_then(|audio| audio.background_color.as_deref())
            .or_else(|| banner.and_then(|banner| non_empty_ref(&banner.primary_color)))
            .or_else(|| non_empty_ref(&feed.banner.background_color))
            .unwrap_or("#b45309"),
    );
    let text = overlay_text(feed, banner, audio);
    let font_size = feed.banner.font_size.max(16);
    let font_option = drawtext_font_option(
        non_empty_ref(&feed.banner.font).or_else(|| non_empty_ref(&feed.graphics.font)),
    );
    let font_color = hex_color(non_empty_ref(&feed.text.color).unwrap_or("#ffffff"));
    let fullscreen = feed.banner.mode.eq_ignore_ascii_case("fullscreen");
    let box_x = if fullscreen { 0 } else { feed.banner.x };
    let box_y = if fullscreen { 0 } else { feed.banner.y };
    let box_w = if fullscreen {
        "iw".to_string()
    } else {
        let configured = if feed.graphics.banner_width == 0 {
            feed.video.width
        } else {
            feed.graphics.banner_width
        };
        configured.to_string()
    };
    let box_h = if fullscreen {
        "ih".to_string()
    } else {
        feed.banner
            .ticker_height
            .max(feed.graphics.banner_height)
            .to_string()
    };
    let text_x = if fullscreen {
        "48".to_string()
    } else if feed.banner.mode.eq_ignore_ascii_case("ticker")
        || feed.banner.mode.eq_ignore_ascii_case("auto")
    {
        "w-mod(t*120\\,w+tw)".to_string()
    } else {
        (box_x + 48).to_string()
    };
    let text_y = if fullscreen {
        "(h-text_h)/2".to_string()
    } else {
        format!("{box_y}+({box_h}-text_h)/2")
    };
    let drawbox = if feed.banner.background_enabled {
        format!("drawbox=x={box_x}:y={box_y}:w={box_w}:h={box_h}:color={color}@0.92:t=fill")
    } else {
        "null".to_string()
    };
    format!(
        "[0:v]{drawbox},drawtext={font_option}:text='{text}':x={text_x}:y={text_y}:fontsize={font_size}:fontcolor={font_color},format=yuv420p[v]",
        text = ffmpeg_filter_escape(&text)
    )
}

fn overlay_text(
    feed: &FeedConfig,
    banner: Option<&BannerPayload>,
    audio: Option<&PriorityAudio>,
) -> String {
    let mut parts = Vec::new();
    if let Some(text) = audio.and_then(|audio| audio.banner_text.as_deref()) {
        push_text_part(&mut parts, text);
    }
    if let Some(banner) = banner {
        for alert in &banner.alerts {
            if let Some(text) = alert_text(alert) {
                push_text_part(&mut parts, text);
            }
        }
    }
    if feed.text.enabled {
        push_text_part(&mut parts, &feed.text.content);
    }
    if parts.is_empty() {
        parts.push("Weather alert".to_string());
    }
    let joined = parts.join("     ");
    joined.chars().take(1800).collect()
}

fn alert_text(alert: &SerializedAlert) -> Option<&str> {
    for key in [
        "banner_text",
        "message",
        "scroll_text",
        "headline",
        "title",
        "event_name",
        "event",
        "description",
        "identifier",
    ] {
        if let Some(text) = alert.fields.get(key).and_then(Value::as_str) {
            let text = text.trim();
            if !text.is_empty() {
                return Some(text);
            }
        }
    }
    None
}

fn push_text_part(parts: &mut Vec<String>, text: &str) {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if !collapsed.is_empty() && !parts.iter().any(|existing| existing == &collapsed) {
        parts.push(collapsed);
    }
}

fn resolve_audio_path(base_dir: &Path, path: &Path) -> String {
    if path.is_absolute() {
        path.display().to_string()
    } else {
        base_dir.join(path).display().to_string()
    }
}

fn codec_or(value: &str, default: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        default.to_string()
    } else {
        value.to_string()
    }
}

fn hex_color(value: &str) -> String {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix('#') {
        format!("0x{}", hex.chars().take(6).collect::<String>())
    } else if value.starts_with("0x") {
        value.to_string()
    } else if value.is_empty() {
        "0xffffff".to_string()
    } else {
        value.to_string()
    }
}

fn ffmpeg_filter_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace(':', "\\:")
        .replace('\'', "\\'")
        .replace('%', "\\%")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace(',', "\\,")
}

fn drawtext_font_option(configured: Option<&str>) -> String {
    if let Some(font) = configured {
        if looks_like_font_file(font) {
            return format!("fontfile='{}'", escape_font_path(font));
        }
    }
    if let Some(path) = default_font_file() {
        return format!("fontfile='{}'", escape_font_path(path));
    }
    let family = configured.unwrap_or("Arial");
    format!("font='{}'", ffmpeg_filter_escape(family))
}

fn looks_like_font_file(value: &str) -> bool {
    let lower = value.trim().to_ascii_lowercase();
    lower.ends_with(".ttf")
        || lower.ends_with(".otf")
        || lower.contains('/')
        || lower.contains('\\')
}

fn escape_font_path(value: &str) -> String {
    ffmpeg_filter_escape(&value.replace('\\', "/"))
}

#[cfg(windows)]
fn default_font_file() -> Option<&'static str> {
    let path = "C:/Windows/Fonts/arial.ttf";
    Path::new(path).exists().then_some(path)
}

#[cfg(not(windows))]
fn default_font_file() -> Option<&'static str> {
    None
}

fn non_empty_ref(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

async fn log_ffmpeg_stderr(feed_id: String, stderr: tokio::process::ChildStderr) {
    use tokio::io::{AsyncBufReadExt, BufReader};

    let mut lines = BufReader::new(stderr).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim();
        if !line.is_empty() {
            warn!(feed_id = %feed_id, "cgen FFmpeg: {line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use std::path::Path;

    use crate::config::{
        AudioConfig, BannerConfig, ClockConfig, EndpointConfig, FeedConfig, GraphicsConfig,
        PriorityInputConfig, StateConfig, TextConfig, VideoConfig,
    };
    use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

    use super::{desired_mode, ffmpeg_args, ffmpeg_release_args, PipelineMode};

    #[test]
    fn relay_args_copy_program_input_to_udp_output() {
        let feed = FeedConfig {
            id: "CAP-IT-ALL".to_string(),
            name: "CAP CGEN".to_string(),
            enabled: true,
            input: EndpointConfig::default(),
            output: EndpointConfig::default(),
            program_input: EndpointConfig {
                url: "udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1".to_string(),
                format: "mpegts".to_string(),
                ..Default::default()
            },
            priority_input: PriorityInputConfig {
                feed_id: "CAP-IT-ALL".to_string(),
                ..Default::default()
            },
            program_output: EndpointConfig {
                url: "udp://239.0.0.2:9001?pkt_size=1316".to_string(),
                format: "mpegts".to_string(),
                ..Default::default()
            },
            alert_output: EndpointConfig::default(),
            video: VideoConfig {
                width: 1280,
                height: 720,
                fps: "source".to_string(),
            },
            audio: AudioConfig::default(),
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
        };

        let args = ffmpeg_release_args(&feed);

        assert!(args
            .windows(2)
            .any(|pair| pair == ["-i", "udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1"]));
        assert!(args.windows(2).any(|pair| pair == ["-c", "copy"]));
        assert!(args.windows(2).any(|pair| pair == ["-f", "mpegts"]));
        assert_eq!(
            args.last().map(String::as_str),
            Some("udp://239.0.0.2:9001?pkt_size=1316")
        );
    }

    #[test]
    fn alert_args_replace_source_audio_and_draw_banner() {
        let feed = test_feed();
        let mode = PipelineMode::Overlay {
            banner: Some(BannerPayload {
                active: true,
                primary_color: "#b91c1c".to_string(),
                alerts: vec![SerializedAlert {
                    fields: [(
                        "message".to_string(),
                        json!("The National Weather Service has issued a Tornado Warning."),
                    )]
                    .into_iter()
                    .collect(),
                }],
                ..Default::default()
            }),
            audio: Some(PriorityAudio {
                queue_id: "q1".to_string(),
                audio_path: Some("runtime/audio/alerts/q1.raw".into()),
                duration_ms: Some(12000),
                sample_rate: 48_000,
                channels: 1,
                alert_packet: None,
                banner_text: Some("Alert crawl".to_string()),
                background_color: Some("#7f1d1d".to_string()),
                priority: Some("1".to_string()),
                started_at: chrono::Utc::now(),
            }),
            alert_audio_missing: false,
        };

        let args = ffmpeg_args(&feed, &mode, Path::new("C:/haze"));

        assert!(args
            .windows(2)
            .any(|pair| pair == ["-i", "udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1"]));
        assert!(args.windows(2).any(|pair| pair == ["-f", "s16le"]));
        assert!(args
            .iter()
            .any(|arg| arg.ends_with("runtime\\audio\\alerts\\q1.raw")
                || arg.ends_with("runtime/audio/alerts/q1.raw")));
        assert!(args.windows(2).any(|pair| pair == ["-map", "[aout]"]));
        assert!(args.iter().any(|arg| arg.contains("drawtext")));
        if cfg!(windows) {
            assert!(args.iter().any(|arg| arg.contains("fontfile='C\\:")));
        }
        assert!(args.iter().any(|arg| arg.contains("apad")));
        assert!(!args.iter().any(|arg| arg == "-shortest"));
        assert_eq!(
            args.last().map(String::as_str),
            Some("udp://239.0.0.2:9001?pkt_size=1316")
        );
    }

    #[test]
    fn desired_mode_uses_priority_feed_id() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "cap.alert.audio.ready",
            "feed_ids": ["CAP-IT-ALL"],
            "data": {
                "queue_id": "alert-1",
                "audio_path": "runtime/audio/alerts/alert.raw"
            }
        })));

        assert!(matches!(
            desired_mode(&feed, &state),
            PipelineMode::Overlay { audio: Some(_), .. }
        ));
    }

    #[test]
    fn wildcard_priority_input_uses_alert_from_any_feed() {
        let mut feed = test_feed();
        feed.priority_input.feed_id = "*".to_string();
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["sk-0001"],
            "queue_id": "sk-alert",
            "data": {
                "queue_id": "sk-alert",
                "audio_path": "runtime/audio/alerts/sk-alert.raw"
            }
        })));

        assert!(matches!(
            desired_mode(&feed, &state),
            PipelineMode::Overlay { audio: Some(_), .. }
        ));
    }

    #[test]
    fn banner_state_without_playout_timing_does_not_hold_overlay() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "stale banner"}]
            }
        })));

        assert!(matches!(desired_mode(&feed, &state), PipelineMode::Release));
    }

    fn test_feed() -> FeedConfig {
        FeedConfig {
            id: "CAP-IT-ALL".to_string(),
            name: "CAP CGEN".to_string(),
            enabled: true,
            input: EndpointConfig::default(),
            output: EndpointConfig::default(),
            program_input: EndpointConfig {
                url: "udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1".to_string(),
                format: "mpegts".to_string(),
                ..Default::default()
            },
            priority_input: PriorityInputConfig {
                feed_id: "CAP-IT-ALL".to_string(),
                ..Default::default()
            },
            program_output: EndpointConfig {
                url: "udp://239.0.0.2:9001?pkt_size=1316".to_string(),
                format: "mpegts".to_string(),
                vcodec: "libx264".to_string(),
                acodec: "aac".to_string(),
                video_bitrate_kbps: Some(4500),
                audio_bitrate_kbps: Some(128),
            },
            alert_output: EndpointConfig {
                url: "udp://239.0.0.2:9001?pkt_size=1316".to_string(),
                format: "mpegts".to_string(),
                vcodec: "libx264".to_string(),
                acodec: "aac".to_string(),
                video_bitrate_kbps: Some(4500),
                audio_bitrate_kbps: Some(128),
            },
            video: VideoConfig {
                width: 1280,
                height: 720,
                fps: "source".to_string(),
            },
            audio: AudioConfig::default(),
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
        }
    }
}
