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
        info!(
            feed_id = %self.feed.id,
            input = %self.feed.input.url,
            output = %self.feed.output.url,
            video = format_args!("{}x{}", self.feed.video.width, self.feed.video.height),
            fps = %self.feed.video.fps,
            input_format = %self.feed.input.format,
            output_format = %self.feed.output.format,
            vcodec = %self.feed.output.vcodec,
            acodec = %self.feed.output.acodec,
            video_bitrate_kbps = ?self.feed.output.video_bitrate_kbps,
            audio_bitrate_kbps = ?self.feed.output.audio_bitrate_kbps,
            banner_mode = %self.feed.banner.mode,
            ticker_height = self.feed.banner.ticker_height,
            font = %self.feed.banner.font,
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
