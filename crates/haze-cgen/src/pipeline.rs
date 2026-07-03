use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::Value;
use tokio::sync::{mpsc, watch};
use tokio::time::MissedTickBehavior;
use tracing::{info, warn};

use crate::bridge::BridgeClient;
use crate::config::FeedConfig;
use crate::state::RuntimeState;

const STATUS_MIN_PUBLISH_INTERVAL: Duration = Duration::from_millis(250);
const STATUS_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) struct PipelineWorker {
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    shutdown_rx: watch::Receiver<bool>,
    base_dir: PathBuf,
    bridge: Option<BridgeClient>,
}

impl PipelineWorker {
    pub(crate) fn new(
        feed: FeedConfig,
        state_rx: watch::Receiver<RuntimeState>,
        shutdown_rx: watch::Receiver<bool>,
        base_dir: PathBuf,
        bridge: Option<BridgeClient>,
    ) -> Self {
        Self {
            feed,
            state_rx,
            shutdown_rx,
            base_dir,
            bridge,
        }
    }

    pub(crate) async fn run(self) -> Result<()> {
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
            input = %self.feed.redacted_program_input_url(),
            priority_feed = %self.feed.priority_input.feed_id,
            audio_source = %self.feed.priority_input.audio_source,
            output = %self.feed.redacted_program_output_url(),
            video = format_args!("{}x{}", self.feed.video.width, self.feed.video.height),
            fps = %self.feed.video.fps,
            input_format = %self.feed.program_input.format,
            output_format = %output.format,
            vcodec = %output.vcodec,
            acodec = %output.acodec,
            video_standard = %self.feed.video.standard,
            interlaced = self.feed.video.interlaced,
            field_order = %self.feed.video.field_order,
            video_bitrate_kbps = ?output.video_bitrate_kbps,
            audio_bitrate_kbps = ?output.audio_bitrate_kbps,
            banner_mode = %self.feed.banner.mode,
            ticker_height = self.feed.banner.ticker_height,
            banner_font = %self.feed.banner.font,
            banner_font_size = self.feed.banner.font_size,
            banner_scroll_speed = self.feed.banner.scroll_speed,
            banner_background_enabled = self.feed.banner.background_enabled,
            clock_enabled = self.feed.clock.enabled,
            text_enabled = self.feed.text.enabled,
            cgen_mode = %self.feed.state.mode,
            smpte_bars = self.feed.state.smpte_bars,
            "cgen media pipeline supervised"
        );

        let status_tx = self.spawn_status_publisher();
        crate::gst_backend::run_supervised(
            self.feed,
            self.state_rx,
            self.shutdown_rx,
            self.base_dir,
            status_tx,
        )
        .await
    }

    fn spawn_status_publisher(&self) -> Option<mpsc::UnboundedSender<Value>> {
        let bridge = self.bridge.clone()?;
        let feed_id = self.feed.id.clone();
        let status_path = self
            .base_dir
            .join("runtime")
            .join("cgen")
            .join(format!("{}.status.json", safe_file_id(&feed_id)));
        let (tx, mut rx) = mpsc::unbounded_channel::<Value>();
        tokio::spawn(async move {
            let mut pending: Option<Value> = None;
            let mut last_sent: Option<Value> = None;
            let mut last_sent_at: Option<Instant> = None;
            let mut ticker = tokio::time::interval(STATUS_MIN_PUBLISH_INTERVAL);
            ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

            loop {
                tokio::select! {
                    data = rx.recv() => {
                        let Some(data) = data else {
                            if pending.is_some() {
                                let _ = flush_cgen_status(
                                    &feed_id,
                                    &status_path,
                                    &bridge,
                                    &mut pending,
                                    &mut last_sent,
                                    &mut last_sent_at,
                                ).await;
                            }
                            break;
                        };
                        pending = Some(data);
                        if status_flush_due(pending.as_ref(), last_sent.as_ref(), last_sent_at, Instant::now())
                            && !flush_cgen_status(
                                &feed_id,
                                &status_path,
                                &bridge,
                                &mut pending,
                                &mut last_sent,
                                &mut last_sent_at,
                            ).await
                        {
                            break;
                        }
                    }
                    _ = ticker.tick() => {
                        if status_flush_due(pending.as_ref(), last_sent.as_ref(), last_sent_at, Instant::now())
                            && !flush_cgen_status(
                                &feed_id,
                                &status_path,
                                &bridge,
                                &mut pending,
                                &mut last_sent,
                                &mut last_sent_at,
                            ).await
                        {
                            break;
                        }
                    }
                }
            }
        });
        Some(tx)
    }
}

fn status_flush_due(
    pending: Option<&Value>,
    last_sent: Option<&Value>,
    last_sent_at: Option<Instant>,
    now: Instant,
) -> bool {
    let Some(pending) = pending else {
        return false;
    };
    let Some(last_sent_at) = last_sent_at else {
        return true;
    };
    let elapsed = now.saturating_duration_since(last_sent_at);
    if elapsed >= STATUS_HEARTBEAT_INTERVAL {
        return true;
    }
    elapsed >= STATUS_MIN_PUBLISH_INTERVAL && last_sent != Some(pending)
}

async fn flush_cgen_status(
    feed_id: &str,
    status_path: &Path,
    bridge: &BridgeClient,
    pending: &mut Option<Value>,
    last_sent: &mut Option<Value>,
    last_sent_at: &mut Option<Instant>,
) -> bool {
    let Some(data) = pending.take() else {
        return true;
    };
    let write_path = status_path.to_path_buf();
    let write_data = data.clone();
    match tokio::task::spawn_blocking(move || write_cgen_status_file(&write_path, &write_data))
        .await
    {
        Ok(Ok(())) => {}
        Ok(Err(err)) => {
            warn!(feed_id = %feed_id, "failed to write cgen status file: {err:#}");
        }
        Err(err) => {
            warn!(feed_id = %feed_id, "cgen status file writer panicked: {err}");
        }
    }

    let event = serde_json::json!({
        "type": "cgen.status.updated",
        "source": "haze-cgen",
        "subject": feed_id,
        "data": data.clone(),
    });
    if let Err(err) = bridge.publish(event).await {
        warn!(feed_id = %feed_id, "failed to publish cgen status: {err:#}");
        return false;
    }

    *last_sent = Some(data);
    *last_sent_at = Some(Instant::now());
    true
}

fn write_cgen_status_file(path: &Path, data: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("failed to create cgen status directory")?;
    }
    let mut data = data.clone();
    if let Some(object) = data.as_object_mut() {
        object.insert(
            "updated_at".to_string(),
            Value::String(chrono::Utc::now().to_rfc3339()),
        );
    }
    let raw = serde_json::to_vec_pretty(&data).context("failed to encode cgen status")?;
    let tmp = path.with_extension("status.json.tmp");
    fs::write(&tmp, raw).context("failed to write cgen status temp file")?;
    fs::rename(&tmp, path).context("failed to replace cgen status file")?;
    Ok(())
}

fn safe_file_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        safe_file_id, status_flush_due, STATUS_HEARTBEAT_INTERVAL, STATUS_MIN_PUBLISH_INTERVAL,
    };
    use serde_json::json;
    use std::time::{Duration, Instant};

    #[test]
    fn safe_file_id_replaces_pathish_chars() {
        assert_eq!(safe_file_id("CAP/IT:ALL"), "CAP_IT_ALL");
    }

    #[test]
    fn status_flush_due_sends_first_update() {
        assert!(status_flush_due(
            Some(&json!({"input_connected": true})),
            None,
            None,
            Instant::now(),
        ));
    }

    #[test]
    fn status_flush_due_coalesces_fast_duplicates() {
        let now = Instant::now();
        let value = json!({"input_connected": true});
        assert!(!status_flush_due(
            Some(&value),
            Some(&value),
            Some(now - Duration::from_millis(100)),
            now,
        ));
    }

    #[test]
    fn status_flush_due_sends_changed_update_after_min_interval() {
        let now = Instant::now();
        assert!(status_flush_due(
            Some(&json!({"input_connected": false})),
            Some(&json!({"input_connected": true})),
            Some(now - STATUS_MIN_PUBLISH_INTERVAL),
            now,
        ));
    }

    #[test]
    fn status_flush_due_heartbeats_unchanged_updates() {
        let now = Instant::now();
        let value = json!({"input_connected": true});
        assert!(status_flush_due(
            Some(&value),
            Some(&value),
            Some(now - STATUS_HEARTBEAT_INTERVAL),
            now,
        ));
    }
}
