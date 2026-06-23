use anyhow::{bail, Result};
use tokio::sync::watch;
use tokio::time::{interval, Duration, MissedTickBehavior};
use tracing::info;

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

        let mut ticker = interval(Duration::from_secs(5));
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                changed = self.state_rx.changed() => {
                    if changed.is_err() {
                        return Ok(());
                    }
                    self.log_state();
                }
                _ = ticker.tick() => {
                    self.log_state();
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
