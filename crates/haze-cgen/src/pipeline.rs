use anyhow::{bail, Context, Result};
use tokio::sync::watch;
use tokio::time::{sleep, Duration};
use tracing::{info, warn};

use crate::config::FeedConfig;
use crate::state::RuntimeState;

pub(crate) struct PipelineWorker {
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    ffmpeg: Option<String>,
    graphics_backend: String,
}

impl PipelineWorker {
    pub(crate) fn new(
        feed: FeedConfig,
        state_rx: watch::Receiver<RuntimeState>,
        ffmpeg: Option<String>,
        graphics_backend: String,
    ) -> Self {
        Self {
            feed,
            state_rx,
            ffmpeg,
            graphics_backend,
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
        let args = ffmpeg_relay_args(&self.feed);
        let mut restart_delay = Duration::from_millis(500);
        loop {
            info!(
                feed_id = %self.feed.id,
                input = %self.feed.program_input_url(),
                output = %self.feed.program_output_url(),
                "starting cgen FFmpeg release relay"
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

fn ffmpeg_relay_args(feed: &FeedConfig) -> Vec<String> {
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
    use crate::config::{
        AudioConfig, BannerConfig, ClockConfig, EndpointConfig, FeedConfig, GraphicsConfig,
        PriorityInputConfig, StateConfig, TextConfig, VideoConfig,
    };

    use super::ffmpeg_relay_args;

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

        let args = ffmpeg_relay_args(&feed);

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
}
