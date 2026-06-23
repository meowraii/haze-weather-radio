use std::fs;
use std::net::{SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::thread;

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use tokio::time::{sleep, Duration, Instant};
use tracing::{info, warn};

use crate::bridge::BridgeClient;
use crate::config::FeedConfig;
use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

pub(crate) struct PipelineWorker {
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    ffmpeg: Option<String>,
    graphics_backend: String,
    base_dir: PathBuf,
    bridge: Option<BridgeClient>,
}

impl PipelineWorker {
    pub(crate) fn new(
        feed: FeedConfig,
        state_rx: watch::Receiver<RuntimeState>,
        ffmpeg: Option<String>,
        graphics_backend: String,
        base_dir: PathBuf,
        bridge: Option<BridgeClient>,
    ) -> Self {
        Self {
            feed,
            state_rx,
            ffmpeg,
            graphics_backend,
            base_dir,
            bridge,
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
            video_standard = %self.feed.video.standard,
            interlaced = self.feed.video.interlaced,
            field_order = %self.feed.video.field_order,
            video_bitrate_kbps = ?output.video_bitrate_kbps,
            audio_bitrate_kbps = ?output.audio_bitrate_kbps,
            duck_db = %self.feed.audio.duck_db,
            banner_mode = %self.feed.banner.mode,
            ticker_height = self.feed.banner.ticker_height,
            banner_font = %self.feed.banner.font,
            banner_font_size = self.feed.banner.font_size,
            banner_scroll_speed = self.feed.banner.scroll_speed,
            banner_x = self.feed.banner.x,
            banner_y = self.feed.banner.y,
            banner_background_color = %self.feed.banner.background_color,
            banner_background_gradient_color = %self.feed.banner.background_gradient_color,
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

        #[cfg(feature = "ffmpeg-rsmpeg")]
        if should_use_native_transport(&self.feed) {
            let status_tx = self.spawn_status_publisher();
            return crate::native::run_remux_supervised(
                self.feed.clone(),
                self.state_rx.clone(),
                self.base_dir.clone(),
                status_tx,
            )
            .await;
        }
        #[cfg(not(feature = "ffmpeg-rsmpeg"))]
        if should_use_native_transport(&self.feed) {
            warn!(
                feed_id = %self.feed.id,
                "native rsmpeg cgen transport requested but ffmpeg-rsmpeg feature is disabled; using subprocess fallback"
            );
        }

        if should_use_constant_compositor(&self.feed) {
            self.run_constant_compositor().await
        } else {
            self.run_ffmpeg_relay().await
        }
    }

    fn spawn_status_publisher(&self) -> Option<mpsc::UnboundedSender<Value>> {
        let bridge = self.bridge.clone()?;
        let feed_id = self.feed.id.clone();
        let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
        tokio::spawn(async move {
            while let Some(data) = rx.recv().await {
                let event = serde_json::json!({
                    "type": "cgen.status.updated",
                    "source": "haze-cgen",
                    "subject": feed_id,
                    "data": data,
                });
                if let Err(err) = bridge.publish(event).await {
                    warn!(feed_id = %feed_id, "failed to publish cgen status: {err:#}");
                    break;
                }
            }
        });
        Some(tx)
    }

    async fn run_constant_compositor(&mut self) -> Result<()> {
        let ffmpeg = self.ffmpeg.clone().unwrap_or_else(|| "ffmpeg".to_string());
        let cgen_dir = self.base_dir.join("runtime").join("cgen");
        fs::create_dir_all(&cgen_dir).context("failed to create cgen runtime directory")?;
        let text_path = cgen_dir.join(format!("{}.txt", safe_file_id(&self.feed.id)));
        fs::write(&text_path, "").context("failed to initialize cgen overlay text file")?;
        let priority_addr = allocate_loopback_udp_addr()?;
        spawn_priority_audio_streamer(
            self.feed.clone(),
            self.state_rx.clone(),
            priority_addr,
            self.base_dir.clone(),
        );
        spawn_overlay_text_writer(self.feed.clone(), self.state_rx.clone(), text_path.clone());

        let mut restart_delay = Duration::from_millis(500);
        loop {
            let args = ffmpeg_constant_compositor_args(
                &self.feed,
                priority_addr,
                &text_path,
                &self.base_dir,
            );
            info!(
                feed_id = %self.feed.id,
                input = %self.feed.program_input_url(),
                output = %self.feed.program_output_url(),
                priority_audio_udp = %priority_addr,
                textfile = %text_path.display(),
                "starting persistent cgen compositor"
            );
            let mut child = tokio::process::Command::new(&ffmpeg)
                .args(&args)
                .kill_on_drop(true)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .with_context(|| {
                    format!("failed to start cgen FFmpeg compositor using {ffmpeg}")
                })?;

            if let Some(stderr) = child.stderr.take() {
                let feed_id = self.feed.id.clone();
                tokio::spawn(async move {
                    log_ffmpeg_stderr(feed_id, stderr).await;
                });
            }

            match child.wait().await {
                Ok(status) if status.success() => {
                    info!(feed_id = %self.feed.id, "cgen persistent compositor exited cleanly");
                    restart_delay = Duration::from_millis(500);
                }
                Ok(status) => {
                    warn!(feed_id = %self.feed.id, status = %status, "cgen persistent compositor exited; restarting");
                }
                Err(err) => {
                    warn!(feed_id = %self.feed.id, "cgen persistent compositor wait failed: {err}");
                }
            }
            sleep(restart_delay).await;
            restart_delay = (restart_delay * 2).min(Duration::from_secs(10));
        }
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

fn should_use_constant_compositor(feed: &FeedConfig) -> bool {
    feed.video.standard.eq_ignore_ascii_case("atsc")
}

fn should_use_native_transport(feed: &FeedConfig) -> bool {
    feed.video.standard.eq_ignore_ascii_case("atsc")
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

fn ffmpeg_constant_compositor_args(
    feed: &FeedConfig,
    priority_addr: SocketAddr,
    text_path: &Path,
    _base_dir: &Path,
) -> Vec<String> {
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
        "-thread_queue_size".to_string(),
        "512".to_string(),
        "-f".to_string(),
        "s16le".to_string(),
        "-ar".to_string(),
        "48000".to_string(),
        "-ac".to_string(),
        "2".to_string(),
        "-i".to_string(),
        format!(
            "udp://{}?fifo_size=1000000&overrun_nonfatal=1",
            priority_addr
        ),
    ];
    let filters = [
        video_textfile_filter(feed, text_path),
        "[0:a]aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo[src]".to_string(),
        "[1:a]aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo[prio]".to_string(),
        "[src][prio]sidechaincompress=threshold=0.001:ratio=20:attack=10:release=250[ducked]"
            .to_string(),
        "[ducked][prio]amix=inputs=2:duration=longest:normalize=0[aout]".to_string(),
    ];
    args.extend(["-filter_complex".to_string(), filters.join(";")]);
    args.extend([
        "-map".to_string(),
        "[v]".to_string(),
        "-map".to_string(),
        "[aout]".to_string(),
    ]);
    append_encode_args(&mut args, feed, output);
    append_mux_args(&mut args, output);
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
                "anullsrc=r=48000:cl=stereo".to_string(),
            ]);
        }
        args.extend(audio_input);
    }

    let mut filters = vec![video_filter(feed, banner, audio)];
    if replace_audio && alert_audio_path.is_some() {
        filters.push("[1:a]aresample=48000,pan=stereo|c0=c0|c1=c0,apad[aout]".to_string());
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
    append_encode_args(&mut args, feed, output);
    append_mux_args(&mut args, output);
    args.push(
        if replace_audio && !feed.alert_output.url.trim().is_empty() {
            feed.alert_output.url.clone()
        } else {
            feed.program_output_url().to_string()
        },
    );
    args
}

fn append_encode_args(
    args: &mut Vec<String>,
    feed: &FeedConfig,
    output: &crate::config::EndpointConfig,
) {
    let video_codec = codec_or(&output.vcodec, "libx264");
    args.extend(["-c:v".to_string(), video_codec.clone()]);
    if is_h264_codec(&video_codec) {
        args.extend([
            "-preset".to_string(),
            "ultrafast".to_string(),
            "-tune".to_string(),
            "zerolatency".to_string(),
        ]);
    }
    args.extend([
        "-pix_fmt".to_string(),
        "yuv420p".to_string(),
        "-color_primaries".to_string(),
        "bt709".to_string(),
        "-color_trc".to_string(),
        "bt709".to_string(),
        "-colorspace".to_string(),
        "bt709".to_string(),
        "-g".to_string(),
        if feed.video.interlaced { "15" } else { "30" }.to_string(),
        "-bf".to_string(),
        "0".to_string(),
    ]);
    if feed.video.interlaced && is_mpeg2_codec(&video_codec) {
        args.extend([
            "-flags".to_string(),
            "+ildct+ilme".to_string(),
            "-top".to_string(),
            top_field_value(feed).to_string(),
            "-alternate_scan".to_string(),
            "1".to_string(),
        ]);
    }
    let audio_codec = codec_or(&output.acodec, "aac");
    args.extend([
        "-c:a".to_string(),
        audio_codec.clone(),
        "-ar".to_string(),
        "48000".to_string(),
    ]);
    if is_ac3_codec(&audio_codec) {
        args.extend(["-ac".to_string(), "2".to_string()]);
    }
    if let Some(kbps) = output.video_bitrate_kbps {
        args.extend(["-b:v".to_string(), format!("{kbps}k")]);
    }
    if let Some(kbps) = output.audio_bitrate_kbps {
        args.extend(["-b:a".to_string(), format!("{kbps}k")]);
    }
}

fn append_mux_args(args: &mut Vec<String>, output: &crate::config::EndpointConfig) {
    args.extend([
        "-muxdelay".to_string(),
        "0".to_string(),
        "-muxpreload".to_string(),
        "0".to_string(),
    ]);
    if !output.format.trim().is_empty() {
        args.extend(["-f".to_string(), output.format.clone()]);
    }
}

fn video_filter(
    feed: &FeedConfig,
    banner: Option<&BannerPayload>,
    audio: Option<&PriorityAudio>,
) -> String {
    if feed.state.smpte_bars {
        return format!(
            "smptebars=size={}x{},{}[v]",
            feed.video.width,
            feed.video.height,
            video_output_filter_suffix(feed)
        );
    }
    let color = hex_color(
        banner
            .and_then(|banner| non_empty_ref(&banner.primary_color))
            .or_else(|| audio.and_then(|audio| audio.background_color.as_deref()))
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
    let box_y = if fullscreen { 0 } else { ticker_y(feed) };
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
        format!("w+1-mod(t*{}\\,w+tw+2)", ticker_pixels_per_second(feed))
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
        "[0:v]scale={}:{}:flags=bicubic,{drawbox},drawtext={font_option}:text='{text}':x={text_x}:y={text_y}:fontsize={font_size}:fontcolor={font_color},{}[v]",
        feed.video.width,
        feed.video.height,
        video_output_filter_suffix(feed),
        text = ffmpeg_filter_escape(&text)
    )
}

fn video_textfile_filter(feed: &FeedConfig, text_path: &Path) -> String {
    let color = hex_color(non_empty_ref(&feed.banner.background_color).unwrap_or("#b45309"));
    let font_size = feed.banner.font_size.max(16);
    let font_option = drawtext_font_option(
        non_empty_ref(&feed.banner.font).or_else(|| non_empty_ref(&feed.graphics.font)),
    );
    let font_color = hex_color(non_empty_ref(&feed.text.color).unwrap_or("#ffffff"));
    let box_y = ticker_y(feed);
    let box_h = feed
        .banner
        .ticker_height
        .max(feed.graphics.banner_height)
        .to_string();
    let text_y = format!("{box_y}+({box_h}-text_h)/2");
    format!(
        "[0:v]scale={}:{}:flags=bicubic,drawtext={font_option}:textfile='{}':reload=1:x=w+1-mod(t*{}\\,w+tw+2):y={text_y}:fontsize={font_size}:fontcolor={font_color}:box=1:boxcolor={color}@0.92:boxborderw=24,{}[v]",
        feed.video.width,
        feed.video.height,
        escape_font_path(&text_path.display().to_string()),
        ticker_pixels_per_second(feed),
        video_output_filter_suffix(feed)
    )
}

fn video_output_filter_suffix(feed: &FeedConfig) -> String {
    let mut filters = vec!["format=yuv420p".to_string()];
    if feed.video.interlaced {
        filters.push(format!(
            "fps={}",
            interlace_input_rate(feed.video.height, &feed.video.fps)
        ));
        filters.push(format!("interlace=scan={}:lowpass=0", field_order(feed)));
        filters.push(format!("setfield={}", field_order(feed)));
    }
    filters.join(",")
}

fn interlace_input_rate(height: u32, fps: &str) -> String {
    let fps = fps.trim();
    if height == 576 || fps == "25" || fps == "25/1" || fps.eq_ignore_ascii_case("pal") {
        return "50".to_string();
    }
    "60000/1001".to_string()
}

fn field_order(feed: &FeedConfig) -> &'static str {
    if feed.video.field_order.eq_ignore_ascii_case("bff") {
        "bff"
    } else {
        "tff"
    }
}

fn top_field_value(feed: &FeedConfig) -> &'static str {
    if field_order(feed) == "bff" {
        "0"
    } else {
        "1"
    }
}

fn is_h264_codec(codec: &str) -> bool {
    matches!(
        codec.trim().to_ascii_lowercase().as_str(),
        "libx264" | "h264" | "h264_nvenc" | "h264_qsv" | "h264_amf"
    )
}

fn is_mpeg2_codec(codec: &str) -> bool {
    codec.trim().eq_ignore_ascii_case("mpeg2video")
}

fn is_ac3_codec(codec: &str) -> bool {
    matches!(codec.trim().to_ascii_lowercase().as_str(), "ac3" | "eac3")
}

fn spawn_overlay_text_writer(
    feed: FeedConfig,
    mut state_rx: watch::Receiver<RuntimeState>,
    text_path: PathBuf,
) {
    tokio::spawn(async move {
        let mut current = String::new();
        let mut linger_until: Option<Instant> = None;
        loop {
            let next = current_overlay_text(&feed, &state_rx.borrow());
            if !next.is_empty() {
                if current != next {
                    if let Err(err) = fs::write(&text_path, &next) {
                        warn!(feed_id = %feed.id, "failed to update cgen crawl text: {err}");
                    }
                    current = next;
                }
                linger_until = None;
            } else if !current.is_empty() && linger_until.is_none() {
                linger_until = Some(Instant::now() + estimate_scroll_linger(&feed, &current));
            }

            if let Some(deadline) = linger_until {
                tokio::select! {
                    changed = state_rx.changed() => {
                        if changed.is_err() {
                            return;
                        }
                    }
                    _ = tokio::time::sleep_until(deadline) => {
                        if let Err(err) = fs::write(&text_path, "") {
                            warn!(feed_id = %feed.id, "failed to clear cgen crawl text: {err}");
                        }
                        current.clear();
                        linger_until = None;
                    }
                }
            } else if state_rx.changed().await.is_err() {
                return;
            }
        }
    });
}

fn current_overlay_text(feed: &FeedConfig, state: &RuntimeState) -> String {
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
    let audio = state.priority_audio_for(priority_feed);
    let banner = state.banner_for(banner_feed);
    if audio.is_some() {
        overlay_text(feed, banner, audio)
    } else {
        String::new()
    }
}

fn estimate_scroll_linger(feed: &FeedConfig, text: &str) -> Duration {
    let width = feed.video.width.max(720) as f64;
    let text_width = text.chars().count() as f64 * feed.banner.font_size.max(16) as f64 * 0.58;
    let seconds =
        ((width + text_width) / ticker_pixels_per_second(feed) as f64 + 2.0).clamp(8.0, 90.0);
    Duration::from_secs_f64(seconds)
}

fn ticker_pixels_per_second(feed: &FeedConfig) -> u32 {
    feed.banner.scroll_speed.max(1).saturating_mul(30)
}

fn ticker_y(feed: &FeedConfig) -> i32 {
    if feed.banner.y != 0 {
        return feed.banner.y;
    }
    ((feed.video.height.max(1) as f32) * 0.08).round() as i32
}

fn spawn_priority_audio_streamer(
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    target: SocketAddr,
    base_dir: PathBuf,
) {
    thread::spawn(move || {
        let Ok(socket) = UdpSocket::bind("127.0.0.1:0") else {
            warn!(feed_id = %feed.id, "failed to bind cgen priority audio UDP injector");
            return;
        };
        let frame_samples = 480usize;
        let mut silence = vec![0u8; frame_samples * 2 * 2];
        let mut loaded_queue = String::new();
        let mut loaded_audio = Vec::<u8>::new();
        let mut cursor = 0usize;
        loop {
            let audio = priority_audio_for_feed(&feed, &state_rx.borrow()).cloned();
            if let Some(audio) = audio {
                if loaded_queue != audio.queue_id {
                    loaded_queue.clone_from(&audio.queue_id);
                    cursor = 0;
                    loaded_audio = audio
                        .audio_path
                        .as_ref()
                        .and_then(|path| fs::read(resolve_audio_path_buf(&base_dir, path)).ok())
                        .unwrap_or_default();
                }
                if !loaded_audio.is_empty() && audio.sample_rate == 48_000 {
                    let frame = next_stereo_frame(
                        &loaded_audio,
                        &mut cursor,
                        audio.channels,
                        frame_samples,
                    );
                    let _ = socket.send_to(&frame, target);
                } else {
                    let _ = socket.send_to(&silence, target);
                }
            } else {
                loaded_queue.clear();
                loaded_audio.clear();
                cursor = 0;
                silence.fill(0);
                let _ = socket.send_to(&silence, target);
            }
            thread::sleep(Duration::from_millis(10));
        }
    });
}

fn priority_audio_for_feed<'a>(
    feed: &FeedConfig,
    state: &'a RuntimeState,
) -> Option<&'a PriorityAudio> {
    let priority_feed = if feed.priority_input.feed_id.trim().is_empty() {
        feed.id.as_str()
    } else {
        feed.priority_input.feed_id.as_str()
    };
    state.priority_audio_for(priority_feed)
}

fn next_stereo_frame(
    raw: &[u8],
    cursor: &mut usize,
    channels: u16,
    frame_samples: usize,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame_samples * 4);
    let channels = channels.max(1) as usize;
    for _ in 0..frame_samples {
        if *cursor + 2 > raw.len() {
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        let left = [raw[*cursor], raw[*cursor + 1]];
        let right = if channels >= 2 && *cursor + 4 <= raw.len() {
            [raw[*cursor + 2], raw[*cursor + 3]]
        } else {
            left
        };
        out.extend_from_slice(&left);
        out.extend_from_slice(&right);
        *cursor = (*cursor).saturating_add(channels * 2);
    }
    out
}

fn resolve_audio_path_buf(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn allocate_loopback_udp_addr() -> Result<SocketAddr> {
    let socket = UdpSocket::bind("127.0.0.1:0").context("failed to allocate cgen UDP port")?;
    socket
        .local_addr()
        .context("failed to inspect cgen UDP port")
}

fn safe_file_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
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
        PriorityInputConfig, StateConfig, SyncConfig, TextConfig, VideoConfig,
    };
    use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

    use super::{
        desired_mode, ffmpeg_args, ffmpeg_constant_compositor_args, ffmpeg_release_args,
        PipelineMode,
    };

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
                interlaced: false,
                field_order: "tff".to_string(),
                standard: String::new(),
            },
            audio: AudioConfig::default(),
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
            sync: SyncConfig::default(),
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
    fn alert_args_support_atsc_interlaced_mpegts() {
        let mut feed = test_feed();
        feed.program_output.vcodec = "mpeg2video".to_string();
        feed.program_output.acodec = "ac3".to_string();
        feed.alert_output.vcodec = "mpeg2video".to_string();
        feed.alert_output.acodec = "ac3".to_string();
        feed.video.width = 1920;
        feed.video.height = 1080;
        feed.video.fps = "30000/1001".to_string();
        feed.video.interlaced = true;
        feed.video.standard = "atsc".to_string();

        let mode = PipelineMode::Overlay {
            banner: None,
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

        assert!(args.windows(2).any(|pair| pair == ["-c:v", "mpeg2video"]));
        assert!(args.windows(2).any(|pair| pair == ["-c:a", "ac3"]));
        assert!(args.windows(2).any(|pair| pair == ["-ar", "48000"]));
        assert!(args.windows(2).any(|pair| pair == ["-ac", "2"]));
        assert!(args
            .windows(2)
            .any(|pair| pair == ["-flags", "+ildct+ilme"]));
        assert!(args.windows(2).any(|pair| pair == ["-top", "1"]));
        assert!(args.iter().any(|arg| arg.contains("scale=1920:1080")));
        assert!(args.iter().any(|arg| arg.contains("fps=60000/1001")));
        assert!(args.iter().any(|arg| arg.contains("interlace=scan=tff")));
    }

    #[test]
    fn persistent_compositor_args_keep_program_stream_and_priority_sidechain() {
        let mut feed = test_feed();
        feed.program_output.vcodec = "mpeg2video".to_string();
        feed.program_output.acodec = "ac3".to_string();
        feed.video.width = 1920;
        feed.video.height = 1080;
        feed.video.interlaced = true;
        feed.video.standard = "atsc".to_string();

        let args = ffmpeg_constant_compositor_args(
            &feed,
            "127.0.0.1:39000".parse().expect("socket addr"),
            Path::new("C:/haze/runtime/cgen/CAP-IT-ALL.txt"),
            Path::new("C:/haze"),
        );

        assert!(args.iter().any(|arg| arg.contains("textfile='C\\:")));
        assert!(args
            .iter()
            .any(|arg| arg.contains("sidechaincompress=threshold=0.001:ratio=20")));
        assert!(args.iter().any(|arg| arg.contains("amix=inputs=2")));
        assert!(args
            .iter()
            .any(|arg| arg == "udp://127.0.0.1:39000?fifo_size=1000000&overrun_nonfatal=1"));
        assert!(args.windows(2).any(|pair| pair == ["-c:v", "mpeg2video"]));
        assert!(args.windows(2).any(|pair| pair == ["-c:a", "ac3"]));
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
                interlaced: false,
                field_order: "tff".to_string(),
                standard: String::new(),
            },
            audio: AudioConfig::default(),
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}
