use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use base64::Engine;
use chrono::{DateTime, Duration as ChronoDuration, Timelike, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio::time::{interval, Instant, MissedTickBehavior};

use crate::bridge::{self, BridgeClient, ProductRenderRequest, RenderedProduct, SynthJob};
use crate::config::{display_text, resolve_path, FeedConfig, LoadedConfig};
use crate::sinks::{sinks_for_feed, Sink};
use haze_media::{decode_wav, normalize_pcm, silence_chunk, Pcm};

const SOURCE_ID: &str = "haze-playout";
const ALERT_QUEUE_DIR: &str = "runtime/queues/alerts";
const PCM_CHUNK_MS: u32 = 20;
const MEDIA_PUBLISH_CHUNK_MS: u32 = 200;
const LIVE_BREAKIN_MAX_BUFFER_MS: u32 = 750;
const MAX_CATCH_UP_CHUNKS: usize = 1;
const PCM_PUBLISH_QUEUE_CAPACITY: usize = 4;
const REALTIME_LAG_WARN_BACKLOG_MS: u64 = 60;

#[derive(Debug, Clone)]
pub(crate) struct Options {
    pub(crate) config_path: PathBuf,
    pub(crate) bridge_addr: String,
    pub(crate) alert_poll: Duration,
}

#[derive(Debug, Clone)]
struct PackageRequest {
    package_id: String,
    force: bool,
}

#[derive(Debug, Clone)]
struct PlayoutControl {
    action: String,
}

#[derive(Debug, Clone)]
struct AudioItem {
    id: String,
    package_id: String,
    title: String,
    pcm: Vec<u8>,
    gap_after: Duration,
    not_before: Option<DateTime<Utc>>,
    queued_at: String,
    target_start: String,
    predicted_start: String,
    predicted_finish: String,
    source: ItemSource,
}

#[derive(Debug, Clone)]
struct PcmPublish {
    data: Vec<u8>,
    duration_ms: u32,
}

#[derive(Debug, Clone)]
enum BreakInCommand {
    Start {
        id: String,
        title: String,
    },
    Chunk {
        id: String,
        pcm: Vec<u8>,
        sample_rate: u32,
        channels: u16,
    },
    Finish {
        id: String,
    },
    Cancel {
        id: String,
    },
}

#[derive(Debug)]
struct LiveBreakIn {
    id: String,
    title: String,
    buffer: VecDeque<u8>,
    finishing: bool,
}

#[derive(Debug, Clone)]
enum ItemSource {
    Playlist,
    Generated,
    Alert {
        manifest_path: PathBuf,
        header: String,
        event: String,
    },
}

#[derive(Debug, Clone)]
struct FeedHandle {
    feed: FeedConfig,
    audio_tx: mpsc::Sender<AudioItem>,
    breakin_tx: mpsc::Sender<BreakInCommand>,
    request_tx: mpsc::Sender<PackageRequest>,
    control_tx: mpsc::Sender<PlayoutControl>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct AlertQueueItem {
    #[serde(default)]
    id: String,
    #[serde(default)]
    alert_id: String,
    #[serde(default)]
    r#type: String,
    #[serde(default)]
    priority: String,
    #[serde(default)]
    source: String,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    feed_ids: Vec<String>,
    #[serde(default)]
    header: String,
    #[serde(default)]
    event: String,
    #[serde(default)]
    alert_sent_at: String,
    #[serde(default)]
    alert_expires_at: String,
    #[serde(default)]
    message_type: String,
    #[serde(default)]
    audio_path: String,
    #[serde(default)]
    sample_rate: u32,
    #[serde(default)]
    channels: u16,
    #[serde(default)]
    claimed_at: Option<String>,
    #[serde(default)]
    played_at: Option<String>,
    #[serde(default)]
    failed_at: Option<String>,
    #[serde(default)]
    last_error: Option<String>,
}

#[derive(Debug)]
struct AlertCandidate {
    manifest: PathBuf,
    item: AlertQueueItem,
    id: String,
    sort_key: AlertSortKey,
}

#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
struct AlertSortKey {
    group: String,
    rank: u8,
    created_at: String,
    id: String,
}

enum ConnectedOutcome {
    Reconnect,
    Shutdown,
}

pub(crate) async fn run(options: Options) -> Result<()> {
    if options.bridge_addr.trim().is_empty() {
        anyhow::bail!("missing host event bridge address");
    }

    loop {
        let cfg = Arc::new(crate::config::load_config(&options.config_path)?);
        let connection = bridge::connect_retry(&options.bridge_addr).await?;
        match run_connected(
            Arc::clone(&cfg),
            connection.client,
            connection.events,
            &options,
        )
        .await
        {
            ConnectedOutcome::Shutdown => return Ok(()),
            ConnectedOutcome::Reconnect => {
                tracing::warn!("host event bridge disconnected; reconnecting playout service");
            }
        }
    }
}

async fn run_connected(
    cfg: Arc<LoadedConfig>,
    client: BridgeClient,
    mut events: mpsc::Receiver<Value>,
    options: &Options,
) -> ConnectedOutcome {
    let mut handles = HashMap::<String, FeedHandle>::new();
    for feed in cfg.enabled_feeds().cloned() {
        let handle = FeedHandle::spawn(Arc::clone(&cfg), client.clone(), feed, options.alert_poll);
        handles.insert(handle.feed.id.clone(), handle);
    }
    client.service_ready(handles.len()).await;

    while let Some(event) = events.recv().await {
        if bridge::string_at(&event, "type") == "system.shutdown" {
            return ConnectedOutcome::Shutdown;
        }
        dispatch_event(&cfg, &handles, event).await;
    }
    ConnectedOutcome::Reconnect
}

async fn dispatch_event(
    cfg: &Arc<LoadedConfig>,
    handles: &HashMap<String, FeedHandle>,
    event: Value,
) {
    match bridge::string_at(&event, "type") {
        "scheduled_package" => {
            if cfg.root.services.go.playlist.enabled
                && bridge::string_at(&event, "source") == "daemon-scheduler"
            {
                return;
            }
            let data = bridge::data(&event);
            let package_id = bridge::first_text(&event, data, &["pkg_id", "package_id"]);
            let feed_id = bridge::first_text(&event, data, &["feed_id"]);
            if !package_id.is_empty() {
                dispatch_package(handles, feed_id, package_id, true).await;
            }
        }
        "playlist_refill" => {
            if cfg.root.services.go.playlist.enabled {
                return;
            }
            for handle in handles.values() {
                if let Some(package_id) = cfg.routine_playlist_order().first() {
                    let _ = handle
                        .request_tx
                        .send(PackageRequest {
                            package_id: package_id.clone(),
                            force: false,
                        })
                        .await;
                }
            }
        }
        "playlist.item.ready" => {
            let data = bridge::data(&event);
            let feed_id = bridge::first_text(&event, data, &["feed_id"]);
            let Some(handle) = handles.get(feed_id).cloned() else {
                return;
            };
            let cfg = Arc::clone(cfg);
            let data = data.clone();
            let feed_id = feed_id.to_string();
            tokio::spawn(async move {
                match audio_item_from_ready(&cfg, &handle.feed, data).await {
                    Ok(item) => {
                        if handle.audio_tx.send(item).await.is_err() {
                            tracing::warn!(
                                feed_id,
                                "playout queue closed before playlist item could be queued"
                            );
                        }
                    }
                    Err(err) => tracing::warn!(feed_id, "playlist item rejected: {err}"),
                }
            });
        }
        "playlist.control" => {
            let data = bridge::data(&event);
            let feed_id = bridge::first_text(&event, data, &["feed_id"]);
            let action = bridge::first_text(&event, data, &["action"]);
            dispatch_control(handles, feed_id, action).await;
        }
        "operator.breakin.start"
        | "operator.breakin.chunk"
        | "operator.breakin.finish"
        | "operator.breakin.cancel" => {
            dispatch_breakin(handles, event).await;
        }
        _ => {}
    }
}

async fn dispatch_package(
    handles: &HashMap<String, FeedHandle>,
    feed_id: &str,
    package_id: &str,
    force: bool,
) {
    let request = PackageRequest {
        package_id: package_id.to_string(),
        force,
    };
    if feed_id.is_empty() || feed_id == "*" {
        for handle in handles.values() {
            let _ = handle.request_tx.send(request.clone()).await;
        }
    } else if let Some(handle) = handles.get(feed_id) {
        let _ = handle.request_tx.send(request).await;
    }
}

async fn dispatch_control(handles: &HashMap<String, FeedHandle>, feed_id: &str, action: &str) {
    if action.trim().is_empty() {
        return;
    }
    let control = PlayoutControl {
        action: action.trim().to_ascii_lowercase(),
    };
    if feed_id.is_empty() || feed_id == "*" {
        for handle in handles.values() {
            let _ = handle.control_tx.send(control.clone()).await;
        }
    } else if let Some(handle) = handles.get(feed_id) {
        let _ = handle.control_tx.send(control).await;
    }
}

async fn dispatch_breakin(handles: &HashMap<String, FeedHandle>, event: Value) {
    let data = bridge::data(&event);
    let session_id = fallback_text(
        bridge::first_text(&event, data, &["session_id", "subject"]),
        "operator-breakin",
    );
    let command = match bridge::string_at(&event, "type") {
        "operator.breakin.start" => BreakInCommand::Start {
            id: session_id,
            title: fallback_text(
                bridge::first_text(&Value::Null, data, &["title", "header"]),
                "Operator Break-in",
            ),
        },
        "operator.breakin.chunk" => {
            let encoded = bridge::first_text(&Value::Null, data, &["data", "pcm"]);
            let Ok(pcm) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
                return;
            };
            if pcm.is_empty() {
                return;
            }
            BreakInCommand::Chunk {
                id: session_id,
                pcm,
                sample_rate: u32_at(data, "sample_rate", 48_000),
                channels: u16_at(data, "channels", 1),
            }
        }
        "operator.breakin.finish" => BreakInCommand::Finish { id: session_id },
        "operator.breakin.cancel" => BreakInCommand::Cancel { id: session_id },
        _ => return,
    };
    let targets = feed_targets_from_event(&event);
    for handle in matching_feed_handles(handles, &targets) {
        let _ = handle.breakin_tx.send(command.clone()).await;
    }
}

fn feed_targets_from_event(event: &Value) -> Vec<String> {
    let data = bridge::data(event);
    let mut out = Vec::new();
    for source in [data, event] {
        if let Some(feed_id) = source.get("feed_id").and_then(Value::as_str) {
            if !feed_id.trim().is_empty() {
                out.push(feed_id.trim().to_string());
            }
        }
        if let Some(values) = source.get("feed_ids").and_then(Value::as_array) {
            for value in values {
                if let Some(feed_id) = value.as_str() {
                    if !feed_id.trim().is_empty() {
                        out.push(feed_id.trim().to_string());
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

fn matching_feed_handles<'a>(
    handles: &'a HashMap<String, FeedHandle>,
    targets: &[String],
) -> Vec<&'a FeedHandle> {
    if targets.is_empty() || targets.iter().any(|target| target == "*") {
        return handles.values().collect();
    }
    targets
        .iter()
        .filter_map(|target| handles.get(target))
        .collect()
}

fn u32_at(value: &Value, key: &str, fallback: u32) -> u32 {
    value
        .get(key)
        .and_then(|value| {
            value
                .as_u64()
                .and_then(|raw| u32::try_from(raw).ok())
                .or_else(|| value.as_str().and_then(|raw| raw.trim().parse().ok()))
        })
        .unwrap_or(fallback)
}

fn u16_at(value: &Value, key: &str, fallback: u16) -> u16 {
    value
        .get(key)
        .and_then(|value| {
            value
                .as_u64()
                .and_then(|raw| u16::try_from(raw).ok())
                .or_else(|| value.as_str().and_then(|raw| raw.trim().parse().ok()))
        })
        .unwrap_or(fallback)
}

impl FeedHandle {
    fn spawn(
        cfg: Arc<LoadedConfig>,
        client: BridgeClient,
        feed: FeedConfig,
        alert_poll: Duration,
    ) -> Self {
        let (audio_tx, audio_rx) = mpsc::channel(32);
        let (priority_tx, priority_rx) = mpsc::channel(32);
        let (breakin_tx, breakin_rx) = mpsc::channel(128);
        let (request_tx, request_rx) = mpsc::channel(32);
        let (control_tx, control_rx) = mpsc::channel(16);

        tokio::spawn(package_builder(
            Arc::clone(&cfg),
            client.clone(),
            feed.clone(),
            request_rx,
            audio_tx.clone(),
        ));

        if feed.same_enabled() {
            tokio::spawn(alert_scanner(
                Arc::clone(&cfg),
                client.clone(),
                feed.clone(),
                priority_tx.clone(),
                alert_poll,
            ));
        }

        if feed.routine_enabled() && !cfg.root.services.go.playlist.enabled {
            if let Some(package_id) = cfg.routine_playlist_order().first() {
                let _ = request_tx.try_send(PackageRequest {
                    package_id: package_id.clone(),
                    force: true,
                });
            }
        }

        tokio::spawn(
            FeedRunner {
                cfg,
                client,
                feed: feed.clone(),
                audio_rx,
                priority_rx,
                breakin_rx,
                control_rx,
                sinks: Vec::new(),
                after_current_action: String::new(),
                paused: false,
            }
            .run(),
        );

        Self {
            feed,
            audio_tx,
            breakin_tx,
            request_tx,
            control_tx,
        }
    }
}

struct FeedRunner {
    cfg: Arc<LoadedConfig>,
    client: BridgeClient,
    feed: FeedConfig,
    audio_rx: mpsc::Receiver<AudioItem>,
    priority_rx: mpsc::Receiver<AudioItem>,
    breakin_rx: mpsc::Receiver<BreakInCommand>,
    control_rx: mpsc::Receiver<PlayoutControl>,
    sinks: Vec<Box<dyn Sink>>,
    after_current_action: String,
    paused: bool,
}

impl FeedRunner {
    async fn run(mut self) {
        self.sinks = sinks_for_feed(&self.cfg, &self.feed);
        let pcm_tx = spawn_pcm_publisher(
            self.client.clone(),
            self.feed.id.clone(),
            self.cfg.root.playout.sample_rate,
            self.cfg.root.playout.channels,
        );
        update_runtime(&self.cfg, &self.feed.id, "Idle").await;
        self.pump(pcm_tx).await;
        for sink in &mut self.sinks {
            if let Err(err) = sink.close() {
                tracing::warn!(
                    feed_id = self.feed.id,
                    sink = sink.name(),
                    "failed to close playout sink: {err}"
                );
            }
        }
    }

    async fn pump(&mut self, pcm_tx: mpsc::Sender<PcmPublish>) {
        let sample_rate = self.cfg.root.playout.sample_rate;
        let channels = self.cfg.root.playout.channels;
        let chunk = silence_chunk(sample_rate, channels, PCM_CHUNK_MS);
        let publish_chunk_count = (MEDIA_PUBLISH_CHUNK_MS / PCM_CHUNK_MS).max(1) as usize;
        let mut publish_buffer = Vec::with_capacity(chunk.len() * publish_chunk_count);
        let mut publish_duration_ms = 0u32;
        let chunk_interval = Duration::from_millis(u64::from(PCM_CHUNK_MS));
        let mut ticker = interval(chunk_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut last_media_tick = Instant::now() - chunk_interval;
        let mut media_remainder = Duration::from_millis(0);
        let mut last_lag_log = Instant::now() - Duration::from_secs(60);
        let mut dropped_pcm_publishes = 0u64;
        let mut current: Option<AudioItem> = None;
        let mut pending: Option<AudioItem> = None;
        let mut priority_pending = VecDeque::<AudioItem>::new();
        let mut deferred_routine = VecDeque::<AudioItem>::new();
        let mut live_breakin: Option<LiveBreakIn> = None;
        let mut out = vec![0u8; chunk.len()];
        let mut position = 0usize;
        let mut gap_until = Instant::now();

        loop {
            tokio::select! {
                Some(control) = self.control_rx.recv() => {
                    self.apply_control(&control.action, current.is_some());
                    if clears_deferred_routine(&control.action) {
                        deferred_routine.clear();
                    }
                }
                Some(command) = self.breakin_rx.recv() => {
                    match command {
                        BreakInCommand::Start { id, title } => {
                            if let Some(done) = live_breakin.take() {
                                complete_live_breakin(&self.client, &self.cfg, &self.feed.id, &done, true).await;
                            }
                            if let Some(item) = current.take() {
                                interrupt_item(&self.client, &self.feed.id, &item).await;
                                if item.is_alert() {
                                    finish_item(&self.client, &self.feed.id, &item).await;
                                } else {
                                    deferred_routine.push_front(item);
                                }
                            }
                            if let Some(item) = pending.take() {
                                if item.is_alert() {
                                    priority_pending.push_front(item);
                                } else {
                                    deferred_routine.push_back(item);
                                }
                            }
                            position = 0;
                            gap_until = Instant::now();
                            let live = LiveBreakIn {
                                id,
                                title,
                                buffer: VecDeque::new(),
                                finishing: false,
                            };
                            start_live_breakin(&self.client, &self.cfg, &self.feed.id, &live).await;
                            live_breakin = Some(live);
                        }
                        BreakInCommand::Chunk { id, pcm, sample_rate, channels } => {
                            if let Some(live) = live_breakin.as_mut().filter(|live| live.id == id) {
                                let pcm = normalize_pcm(
                                    Pcm {
                                        sample_rate,
                                        channels: channels.max(1),
                                        data: pcm,
                                    },
                                    self.cfg.root.playout.sample_rate,
                                    self.cfg.root.playout.channels,
                                );
                                live.buffer.extend(pcm.data);
                                trim_live_breakin_buffer(
                                    live,
                                    self.cfg.root.playout.sample_rate,
                                    self.cfg.root.playout.channels,
                                );
                            }
                        }
                        BreakInCommand::Finish { id } => {
                            if let Some(live) = live_breakin.as_mut().filter(|live| live.id == id) {
                                live.finishing = true;
                            }
                        }
                        BreakInCommand::Cancel { id } => {
                            if live_breakin.as_ref().is_some_and(|live| live.id == id) {
                                if let Some(done) = live_breakin.take() {
                                    complete_live_breakin(&self.client, &self.cfg, &self.feed.id, &done, true).await;
                                }
                            }
                        }
                    }
                }
                _ = ticker.tick() => {
                    let tick_now = Instant::now();
                    let elapsed = tick_now.saturating_duration_since(last_media_tick);
                    media_remainder += elapsed;
                    last_media_tick = tick_now;
                    let mut chunks_due = 0usize;
                    while media_remainder >= chunk_interval && chunks_due < MAX_CATCH_UP_CHUNKS {
                        media_remainder -= chunk_interval;
                        chunks_due += 1;
                    }
                    if media_remainder >= chunk_interval {
                        let dropped_backlog = media_remainder;
                        if dropped_backlog >= realtime_lag_warn_backlog()
                            && last_lag_log.elapsed() >= Duration::from_secs(10)
                        {
                            tracing::warn!(
                                feed_id = self.feed.id,
                                elapsed_ms = elapsed.as_millis(),
                                backlog_ms = dropped_backlog.as_millis(),
                                "playout tick lagged; dropping missed realtime audio instead of bursting stale chunks"
                            );
                            last_lag_log = Instant::now();
                        }
                        media_remainder = Duration::from_millis(0);
                    }
                    if chunks_due == 0 {
                        chunks_due = 1;
                    }

                    for _ in 0..chunks_due {
                        let now = Utc::now();
                        while let Ok(item) = self.priority_rx.try_recv() {
                            accept_item(&self.client, &self.feed.id, &item).await;
                            priority_pending.push_back(item);
                        }
                        if live_breakin.is_none() && !priority_pending.is_empty()
                            && current.as_ref().is_some_and(|item| !item.is_alert())
                        {
                            if let Some(item) = current.take() {
                                interrupt_item(&self.client, &self.feed.id, &item).await;
                                deferred_routine.push_front(item);
                            }
                            if let Some(item) = pending.take() {
                                if !item.is_alert() {
                                    deferred_routine.push_back(item);
                                } else {
                                    priority_pending.push_front(item);
                                }
                            }
                            position = 0;
                            gap_until = Instant::now();
                        }
                        if live_breakin.is_none() && !priority_pending.is_empty()
                            && pending.as_ref().is_some_and(|item| !item.is_alert())
                        {
                            if let Some(item) = pending.take() {
                                deferred_routine.push_back(item);
                            }
                            gap_until = Instant::now();
                        }
                        if live_breakin.is_none() && current.is_none() && pending.is_none() && !self.paused {
                            if let Some(item) = priority_pending.pop_front() {
                                pending = Some(item);
                            } else if let Some(item) = deferred_routine.pop_front() {
                                pending = Some(item);
                            } else if let Ok(item) = self.audio_rx.try_recv() {
                                accept_item(&self.client, &self.feed.id, &item).await;
                                pending = Some(item);
                            }
                        }
                        if live_breakin.is_none() && current.is_none() && !self.paused && Instant::now() >= gap_until {
                            if let Some(item) = pending.as_ref() {
                                if item.not_before.is_none_or(|not_before| now >= not_before) {
                                    current = pending.take();
                                    position = 0;
                                    if let Some(item) = current.as_ref() {
                                        start_item(&self.client, &self.cfg, &self.feed.id, item).await;
                                    }
                                }
                            }
                        }

                        out.fill(0);
                        let mut breakin_drained = false;
                        if let Some(live) = live_breakin.as_mut() {
                            drain_live_breakin_buffer(live, &mut out);
                            if live.finishing && live.buffer.is_empty() {
                                breakin_drained = true;
                            }
                        } else if let Some(item) = current.as_ref() {
                            let end = position.saturating_add(chunk.len());
                            if end <= item.pcm.len() {
                                out.copy_from_slice(&item.pcm[position..end]);
                            } else if position < item.pcm.len() {
                                let remaining = item.pcm.len() - position;
                                out[..remaining].copy_from_slice(&item.pcm[position..]);
                            }
                            position = end;
                        }
                        let completed_breakin = if breakin_drained {
                            live_breakin.take()
                        } else {
                            None
                        };

                        for sink in &mut self.sinks {
                            if let Err(err) = sink.write(&out) {
                                tracing::warn!(feed_id = self.feed.id, sink = sink.name(), "sink write failed: {err}");
                            }
                        }
                        publish_buffer.extend_from_slice(&out);
                        publish_duration_ms = publish_duration_ms.saturating_add(PCM_CHUNK_MS);
                        if publish_duration_ms >= MEDIA_PUBLISH_CHUNK_MS {
                            let data = std::mem::take(&mut publish_buffer);
                            let duration_ms = publish_duration_ms;
                            publish_duration_ms = 0;
                            publish_buffer = Vec::with_capacity(chunk.len() * publish_chunk_count);
                            match pcm_tx.try_send(PcmPublish { data, duration_ms }) {
                                Ok(()) => {}
                                Err(TrySendError::Full(_)) => {
                                    dropped_pcm_publishes = dropped_pcm_publishes.saturating_add(1);
                                    if dropped_pcm_publishes == 1 || dropped_pcm_publishes.is_multiple_of(50) {
                                        tracing::warn!(
                                            feed_id = self.feed.id,
                                            dropped_chunks = dropped_pcm_publishes,
                                            "media publisher is behind; dropping stale bridge PCM"
                                        );
                                    }
                                }
                                Err(TrySendError::Closed(_)) => {
                                    tracing::warn!(
                                        feed_id = self.feed.id,
                                        "media publisher stopped before PCM could be forwarded"
                                    );
                                }
                            }
                        }
                        if let Some(done) = completed_breakin {
                            complete_live_breakin(&self.client, &self.cfg, &self.feed.id, &done, false).await;
                        }

                        if live_breakin.is_none() && current.as_ref().is_some_and(|item| position >= item.pcm.len()) {
                            if let Some(item) = current.take() {
                                finish_item(&self.client, &self.feed.id, &item).await;
                                gap_until = Instant::now() + item.gap_after;
                                update_runtime(&self.cfg, &self.feed.id, "Idle").await;
                                if !self.after_current_action.is_empty() {
                                    let action = std::mem::take(&mut self.after_current_action);
                                    self.apply_control(&action, false);
                                }
                            }
                        }
                    }
                }
                else => break,
            }
        }
    }

    fn apply_control(&mut self, action: &str, has_current: bool) {
        match action {
            "pause" => self.paused = true,
            "resume" => self.paused = false,
            "restart" => {
                self.paused = false;
                self.drain_audio_queue();
            }
            "flush" | "flush_pending" => self.drain_audio_queue(),
            "flush_restart" => {
                self.paused = false;
                self.drain_audio_queue();
            }
            "flush_stop" => {
                self.paused = true;
                self.drain_audio_queue();
            }
            "pause_after_current" | "flush_restart_after_current" | "flush_stop_after_current" => {
                if has_current {
                    self.after_current_action = action.to_string();
                } else {
                    self.apply_control(action.trim_end_matches("_after_current"), false);
                }
            }
            _ => {}
        }
    }

    fn drain_audio_queue(&mut self) {
        while self.audio_rx.try_recv().is_ok() {}
    }
}

impl AudioItem {
    fn is_alert(&self) -> bool {
        matches!(self.source, ItemSource::Alert { .. })
    }
}

async fn accept_item(client: &BridgeClient, feed_id: &str, item: &AudioItem) {
    let _ = client
        .publish(json!({
            "type": "playout.accepted",
            "source": SOURCE_ID,
            "feed_id": feed_id,
            "queue_id": item.id,
            "pkg_id": item.package_id,
            "title": item.title,
            "data": item_event_data(feed_id, item),
        }))
        .await;
}

async fn start_item(client: &BridgeClient, cfg: &LoadedConfig, feed_id: &str, item: &AudioItem) {
    tracing::info!("[{}] Now playing: {}", feed_id, item.title);
    update_runtime(cfg, feed_id, &item.title).await;
    match &item.source {
        ItemSource::Alert {
            manifest_path,
            header,
            event,
        } => {
            if let Err(err) = mark_alert_started(manifest_path) {
                tracing::warn!(
                    feed_id,
                    queue_id = item.id,
                    "failed to mark alert started: {err}"
                );
            }
            let _ = client
                .publish(json!({
                    "type": "alert.playout.started",
                    "source": SOURCE_ID,
                    "feed_ids": [feed_id],
                    "queue_id": item.id,
                    "header": header,
                    "event": event,
                    "data": {
                        "feed_id": feed_id,
                        "queue_id": item.id,
                        "header": header,
                        "event": event,
                    }
                }))
                .await;
        }
        ItemSource::Playlist | ItemSource::Generated => {
            let _ = client
                .publish(json!({
                    "type": "playout.started",
                    "source": SOURCE_ID,
                    "feed_id": feed_id,
                    "queue_id": item.id,
                    "pkg_id": item.package_id,
                    "title": item.title,
                    "data": item_event_data(feed_id, item),
                }))
                .await;
        }
    }
}

async fn finish_item(client: &BridgeClient, feed_id: &str, item: &AudioItem) {
    match &item.source {
        ItemSource::Alert {
            manifest_path,
            header,
            event,
        } => {
            if let Err(err) = mark_alert_played(manifest_path) {
                tracing::warn!(
                    feed_id,
                    queue_id = item.id,
                    "failed to mark alert played: {err}"
                );
            }
            let _ = client
                .publish(json!({
                    "type": "alert.playout.completed",
                    "source": SOURCE_ID,
                    "feed_ids": [feed_id],
                    "queue_id": item.id,
                    "header": header,
                    "event": event,
                    "data": {
                        "feed_id": feed_id,
                        "queue_id": item.id,
                        "header": header,
                        "event": event,
                    }
                }))
                .await;
        }
        ItemSource::Playlist | ItemSource::Generated => {
            let _ = client
                .publish(json!({
                    "type": "playout.completed",
                    "source": SOURCE_ID,
                    "feed_id": feed_id,
                    "queue_id": item.id,
                    "pkg_id": item.package_id,
                    "title": item.title,
                    "data": item_event_data(feed_id, item),
                }))
                .await;
        }
    }
}

async fn interrupt_item(client: &BridgeClient, feed_id: &str, item: &AudioItem) {
    let _ = client
        .publish(json!({
            "type": "playout.interrupted",
            "source": SOURCE_ID,
            "feed_id": feed_id,
            "queue_id": item.id,
            "pkg_id": item.package_id,
            "title": item.title,
            "data": {
                "feed_id": feed_id,
                "queue_id": item.id,
                "pkg_id": item.package_id,
                "package_id": item.package_id,
                "title": item.title,
                "interrupted": true,
            }
        }))
        .await;
}

async fn start_live_breakin(
    client: &BridgeClient,
    cfg: &LoadedConfig,
    feed_id: &str,
    live: &LiveBreakIn,
) {
    tracing::info!("[{}] Operator break-in started: {}", feed_id, live.title);
    update_runtime(cfg, feed_id, &live.title).await;
    let _ = client
        .publish(json!({
            "type": "alert.playout.started",
            "source": SOURCE_ID,
            "feed_ids": [feed_id],
            "queue_id": live.id,
            "header": live.title,
            "event": "OPR",
            "data": {
                "feed_id": feed_id,
                "queue_id": live.id,
                "header": live.title,
                "event": "OPR",
                "source": "operator-breakin",
            }
        }))
        .await;
    let _ = client
        .publish(json!({
            "type": "operator.breakin.playout.started",
            "source": SOURCE_ID,
            "feed_id": feed_id,
            "queue_id": live.id,
            "title": live.title,
            "data": {
                "feed_id": feed_id,
                "session_id": live.id,
                "title": live.title,
            }
        }))
        .await;
}

async fn complete_live_breakin(
    client: &BridgeClient,
    cfg: &LoadedConfig,
    feed_id: &str,
    live: &LiveBreakIn,
    cancelled: bool,
) {
    tracing::info!("[{}] Operator break-in ended: {}", feed_id, live.title);
    update_runtime(cfg, feed_id, "Idle").await;
    let _ = client
        .publish(json!({
            "type": "alert.playout.completed",
            "source": SOURCE_ID,
            "feed_ids": [feed_id],
            "queue_id": live.id,
            "header": live.title,
            "event": "OPR",
            "data": {
                "feed_id": feed_id,
                "queue_id": live.id,
                "header": live.title,
                "event": "OPR",
                "source": "operator-breakin",
                "cancelled": cancelled,
            }
        }))
        .await;
    let _ = client
        .publish(json!({
            "type": "operator.breakin.playout.completed",
            "source": SOURCE_ID,
            "feed_id": feed_id,
            "queue_id": live.id,
            "title": live.title,
            "data": {
                "feed_id": feed_id,
                "session_id": live.id,
                "title": live.title,
                "cancelled": cancelled,
            }
        }))
        .await;
}

fn drain_live_breakin_buffer(live: &mut LiveBreakIn, out: &mut [u8]) {
    let take = out.len().min(live.buffer.len());
    if take == 0 {
        return;
    }
    let pending = live.buffer.make_contiguous();
    out[..take].copy_from_slice(&pending[..take]);
    live.buffer.drain(..take);
}

fn trim_live_breakin_buffer(live: &mut LiveBreakIn, sample_rate: u32, channels: u16) {
    let drop = live_breakin_drop_bytes(live.buffer.len(), sample_rate, channels);
    if drop > 0 {
        live.buffer.drain(..drop);
    }
}

fn align_down_to_frame(bytes: usize, frame_bytes: usize) -> usize {
    let frame_bytes = frame_bytes.max(1);
    bytes - (bytes % frame_bytes)
}

fn align_up_to_frame(bytes: usize, frame_bytes: usize) -> usize {
    let frame_bytes = frame_bytes.max(1);
    if bytes == 0 {
        return 0;
    }
    bytes + ((frame_bytes - (bytes % frame_bytes)) % frame_bytes)
}

fn live_breakin_max_buffer_bytes(sample_rate: u32, channels: u16) -> usize {
    let frame_bytes = usize::from(channels.max(1)) * 2;
    align_down_to_frame(
        usize::try_from(
            sample_rate
                .saturating_mul(u32::from(channels.max(1)))
                .saturating_mul(2)
                .saturating_mul(LIVE_BREAKIN_MAX_BUFFER_MS)
                / 1000,
        )
        .unwrap_or(usize::MAX),
        frame_bytes,
    )
}

#[cfg(test)]
fn max_live_breakin_buffer_for(sample_rate: u32, channels: u16) -> usize {
    live_breakin_max_buffer_bytes(sample_rate, channels)
}

fn stale_pcm_publish_log_due(dropped: u64) -> bool {
    dropped == 1 || dropped.is_multiple_of(25)
}

fn realtime_lag_warn_backlog() -> Duration {
    Duration::from_millis(REALTIME_LAG_WARN_BACKLOG_MS)
}

fn pcm_publish_queue_capacity() -> usize {
    PCM_PUBLISH_QUEUE_CAPACITY
}

fn live_breakin_frame_bytes(channels: u16) -> usize {
    usize::from(channels.max(1)) * 2
}

fn live_breakin_drop_bytes(buffer_len: usize, sample_rate: u32, channels: u16) -> usize {
    let frame_bytes = live_breakin_frame_bytes(channels);
    let max_bytes = live_breakin_max_buffer_bytes(sample_rate, channels);
    let overflow = buffer_len.saturating_sub(max_bytes);
    align_up_to_frame(overflow, frame_bytes).min(buffer_len)
}

fn clears_deferred_routine(action: &str) -> bool {
    matches!(
        action.trim().to_ascii_lowercase().as_str(),
        "restart" | "flush" | "flush_pending" | "flush_restart" | "flush_stop"
    )
}

async fn update_runtime(cfg: &LoadedConfig, feed_id: &str, now_playing: &str) {
    let now = Utc::now().to_rfc3339();
    let payload = json!({
        "feed_id": feed_id,
        "now_playing": now_playing,
        "on_air_now_playing": now_playing,
        "on_air_last_played_at": now,
        "public_stream_now_playing": now_playing,
        "public_stream_started_at": now,
        "updated_at": now,
    });
    let path = cfg
        .base_dir
        .join("runtime/feeds")
        .join(format!("{}.json", safe_id(feed_id)));
    if let Some(parent) = path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    if let Ok(raw) = serde_json::to_vec_pretty(&payload) {
        let tmp = path.with_extension("json.tmp");
        if tokio::fs::write(&tmp, raw).await.is_ok() {
            let _ = tokio::fs::rename(&tmp, &path).await;
        }
    }
}

fn spawn_pcm_publisher(
    client: BridgeClient,
    feed_id: String,
    sample_rate: u32,
    channels: u16,
) -> mpsc::Sender<PcmPublish> {
    let (tx, mut rx) = mpsc::channel::<PcmPublish>(pcm_publish_queue_capacity());
    tokio::spawn(async move {
        let mut dropped_stale = 0u64;
        while let Some(mut chunk) = rx.recv().await {
            let mut skipped = 0u64;
            while let Ok(newer) = rx.try_recv() {
                chunk = newer;
                skipped = skipped.saturating_add(1);
            }
            if skipped > 0 {
                dropped_stale = dropped_stale.saturating_add(skipped);
                if stale_pcm_publish_log_due(dropped_stale) {
                    tracing::warn!(
                        feed_id = feed_id.as_str(),
                        dropped_chunks = dropped_stale,
                        "media publisher dropped stale bridge PCM to keep realtime audio current"
                    );
                }
            }
            let pcm = base64::engine::general_purpose::STANDARD.encode(&chunk.data);
            let _ = client
                .publish(json!({
                    "type": "playout.pcm",
                    "source": SOURCE_ID,
                    "feed_id": feed_id,
                    "data": {
                        "feed_id": feed_id,
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "duration_ms": chunk.duration_ms,
                        "pcm": pcm,
                    }
                }))
                .await;
        }
    });
    tx
}

async fn package_builder(
    cfg: Arc<LoadedConfig>,
    client: BridgeClient,
    feed: FeedConfig,
    mut requests: mpsc::Receiver<PackageRequest>,
    audio_tx: mpsc::Sender<AudioItem>,
) {
    let mut recent = HashMap::<String, Instant>::new();
    while let Some(request) = requests.recv().await {
        if !request.force
            && recent
                .get(&request.package_id)
                .is_some_and(|last| last.elapsed() < Duration::from_secs(45))
        {
            continue;
        }
        match build_package(&cfg, &client, &feed, &request.package_id).await {
            Ok(item) => {
                recent.insert(request.package_id.clone(), Instant::now());
                if audio_tx.send(item).await.is_err() {
                    break;
                }
            }
            Err(err) => tracing::warn!(
                feed_id = feed.id,
                package_id = request.package_id,
                "package failed: {err}"
            ),
        }
    }
}

async fn build_package(
    cfg: &LoadedConfig,
    client: &BridgeClient,
    feed: &FeedConfig,
    package_id: &str,
) -> Result<AudioItem> {
    if !cfg.package_enabled(package_id) || !feed.routine_enabled() {
        anyhow::bail!("package {package_id} is disabled for feed {}", feed.id);
    }
    let product = render_product_with_fallback(cfg, client, feed, package_id).await?;
    if let Some(item) = audio_item_from_rendered_product(cfg, feed, package_id, &product).await? {
        return Ok(item);
    }
    if product.text.trim().is_empty() {
        anyhow::bail!("package {package_id} rendered empty text");
    }
    let title = fallback_text(&product.title, &title_for_package(package_id));
    let reader_id = fallback_text(&product.reader_id, &cfg.reader_id(package_id));
    let language = fallback_text(&product.language, &feed.language());
    let job_id = queue_id(&format!("{}-{package_id}", feed.id));
    let output_path = cfg
        .base_dir
        .join("runtime/audio/playout")
        .join(&feed.id)
        .join(format!("{job_id}.wav"));
    let wav_path = client
        .synthesize(SynthJob {
            id: job_id.clone(),
            text: product.text,
            reader_id,
            language,
            output_path,
        })
        .await?;
    let pcm = read_wav_pcm(&PathBuf::from(wav_path), cfg).await?;
    Ok(AudioItem {
        id: job_id,
        package_id: package_id.to_string(),
        title,
        pcm: pcm.data,
        gap_after: package_gap(cfg),
        not_before: None,
        queued_at: Utc::now().to_rfc3339(),
        target_start: String::new(),
        predicted_start: String::new(),
        predicted_finish: String::new(),
        source: ItemSource::Generated,
    })
}

async fn audio_item_from_rendered_product(
    cfg: &LoadedConfig,
    feed: &FeedConfig,
    package_id: &str,
    product: &RenderedProduct,
) -> Result<Option<AudioItem>> {
    let content_type = metadata_text(&product.metadata, "content_type");
    if !content_type.eq_ignore_ascii_case("audio") {
        return Ok(None);
    }
    let audio_path = metadata_text(&product.metadata, "audio_path");
    if audio_path.is_empty() {
        let audio_url = metadata_text(&product.metadata, "audio_url");
        if audio_url.is_empty() {
            anyhow::bail!("audio product {package_id} is missing audio_path or audio_url");
        }
        anyhow::bail!("audio URL products require the Go playlist downloader path");
    }
    let path = resolve_path(&cfg.base_dir, audio_path);
    let pcm = read_wav_pcm(&path, cfg).await?;
    Ok(Some(AudioItem {
        id: queue_id(&format!("{}-{package_id}", feed.id)),
        package_id: package_id.to_string(),
        title: fallback_text(&product.title, &title_for_package(package_id)),
        pcm: pcm.data,
        gap_after: package_gap(cfg),
        not_before: None,
        queued_at: Utc::now().to_rfc3339(),
        target_start: String::new(),
        predicted_start: String::new(),
        predicted_finish: String::new(),
        source: ItemSource::Generated,
    }))
}

fn metadata_text<'a>(metadata: &'a HashMap<String, String>, key: &str) -> &'a str {
    metadata
        .iter()
        .find_map(|(raw_key, value)| {
            if raw_key.trim().eq_ignore_ascii_case(key) {
                Some(value.trim())
            } else {
                None
            }
        })
        .unwrap_or("")
}

async fn render_product_with_fallback(
    cfg: &LoadedConfig,
    client: &BridgeClient,
    feed: &FeedConfig,
    package_id: &str,
) -> Result<RenderedProduct> {
    if cfg.root.services.go.product_render.enabled {
        let request_id = queue_id(&format!("{}-{package_id}", feed.id));
        let request = ProductRenderRequest {
            request_id,
            feed_id: feed.id.clone(),
            package_id: package_id.to_string(),
            force: true,
        };
        if let Ok(product) = client.render_product(request).await {
            return Ok(product);
        }
    }
    let (text, title) = package_text(cfg, feed, package_id);
    if text.trim().is_empty() {
        anyhow::bail!("package {package_id} requires the product render service");
    }
    Ok(RenderedProduct {
        id: queue_id(package_id),
        feed_id: feed.id.clone(),
        package_id: package_id.to_string(),
        title,
        text,
        reader_id: cfg.reader_id(package_id),
        language: feed.language(),
        metadata: HashMap::new(),
    })
}

fn package_text(cfg: &LoadedConfig, feed: &FeedConfig, package_id: &str) -> (String, String) {
    match package_id {
        "date_time" => (
            date_time_announcement(chrono::Local::now()),
            "Date and Time".to_string(),
        ),
        "station_id" => {
            let on_air = fallback_text(
                &display_text(&cfg.root.operator.on_air_name),
                "Haze Weather Radio",
            );
            let site = feed.site_name();
            let callsign = feed.station_callsign();
            let frequency = feed.station_frequency_mhz();
            let mut parts = vec![format!("You are listening to {on_air}.")];
            let callsign_spoken = spoken_callsign(&callsign);
            if !callsign_spoken.is_empty() {
                parts.push(format!("Callsign {callsign_spoken}."));
            }
            if !frequency.trim().is_empty() {
                let location = fallback_text(&site, &feed.id);
                parts.push(format!(
                    "Broadcasting from {location} on a frequency of {frequency} megahertz."
                ));
            } else if !site.is_empty() {
                parts.push(format!("Serving {site}."));
            }
            if let Some(replacement) = replacement_station_statement(feed) {
                parts.push(replacement);
            }
            (parts.join(" "), "Station Identification".to_string())
        }
        _ => (String::new(), String::new()),
    }
}

fn replacement_station_statement(feed: &FeedConfig) -> Option<String> {
    let transmitter = feed.replacement_transmitter()?;
    let callsign = transmitter.callsign.trim();
    let site = transmitter.site_name.trim();
    if callsign.is_empty() && site.is_empty() {
        return None;
    }
    let mut parts = vec![
        "This station replaces former".to_string(),
        "Weatheradio Canada".to_string(),
        "station".to_string(),
    ];
    if !callsign.is_empty() {
        parts.push(callsign.to_string());
    }
    if !site.is_empty() {
        parts.push("in".to_string());
        parts.push(site.to_string());
    }
    Some(format!("{}.", parts.join(" ")))
}

fn date_time_announcement(now: DateTime<chrono::Local>) -> String {
    format!(
        "{} The current time is {}.",
        time_greeting(now.hour()),
        spoken_clock_time(now)
    )
}

fn time_greeting(hour: u32) -> &'static str {
    match hour {
        5..=11 => "Good morning.",
        12..=16 => "Good afternoon.",
        17..=21 => "Good evening.",
        _ => "Good night.",
    }
}

fn spoken_clock_time(now: DateTime<chrono::Local>) -> String {
    let mut hour = now.hour() % 12;
    if hour == 0 {
        hour = 12;
    }
    let minute = now.minute();
    let ampm = if now.hour() >= 12 { "P.M." } else { "A.M." };
    let tz = timezone_name(&now.format("%Z").to_string());
    if minute == 0 {
        return format!("{} {ampm}, {tz}", number_to_words(hour));
    }
    let minute_text = if minute < 10 {
        format!("oh {}", number_to_words(minute))
    } else {
        number_to_words(minute)
    };
    format!("{} {minute_text} {ampm}, {tz}", number_to_words(hour))
}

fn number_to_words(value: u32) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 6] = ["", "", "twenty", "thirty", "forty", "fifty"];
    if value < ONES.len() as u32 {
        return ONES[value as usize].to_string();
    }
    if value < 60 {
        let ten = value / 10;
        let one = value % 10;
        if one == 0 {
            return TENS[ten as usize].to_string();
        }
        return format!("{} {}", TENS[ten as usize], ONES[one as usize]);
    }
    value.to_string()
}

fn timezone_name(abbrev: &str) -> String {
    match abbrev {
        "CST" => "Central Standard Time",
        "CDT" => "Central Daylight Time",
        "MST" => "Mountain Standard Time",
        "MDT" => "Mountain Daylight Time",
        "EST" => "Eastern Standard Time",
        "EDT" => "Eastern Daylight Time",
        "PST" => "Pacific Standard Time",
        "PDT" => "Pacific Daylight Time",
        "UTC" => "Coordinated Universal Time",
        _ => abbrev,
    }
    .to_string()
}

fn spoken_callsign(callsign: &str) -> String {
    callsign
        .trim()
        .chars()
        .filter(|ch| !matches!(ch, '-' | '_' | ' '))
        .map(|ch| ch.to_ascii_uppercase().to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

async fn audio_item_from_ready(
    cfg: &LoadedConfig,
    _feed: &FeedConfig,
    data: Value,
) -> Result<AudioItem> {
    let audio_path = bridge::first_text(&Value::Null, &data, &["audio_path"]);
    if audio_path.is_empty() {
        anyhow::bail!("audio_path is required");
    }
    let path = resolve_path(&cfg.base_dir, audio_path);
    let pcm = read_wav_pcm(&path, cfg).await?;
    let package_id = bridge::first_text(&Value::Null, &data, &["package_id", "pkg_id"]).to_string();
    let title = fallback_text(
        bridge::first_text(&Value::Null, &data, &["title"]),
        &title_for_package(&package_id),
    );
    Ok(AudioItem {
        id: fallback_text(
            bridge::first_text(&Value::Null, &data, &["queue_id", "id"]),
            &queue_id(&package_id),
        ),
        package_id,
        title,
        pcm: pcm.data,
        gap_after: package_gap(cfg),
        not_before: parse_time(bridge::first_text(
            &Value::Null,
            &data,
            &["not_before", "predicted_start_at", "target_start_at"],
        )),
        queued_at: bridge::first_text(&Value::Null, &data, &["queued_at"]).to_string(),
        target_start: bridge::first_text(&Value::Null, &data, &["target_start_at"]).to_string(),
        predicted_start: bridge::first_text(&Value::Null, &data, &["predicted_start_at"])
            .to_string(),
        predicted_finish: bridge::first_text(&Value::Null, &data, &["predicted_finish_at"])
            .to_string(),
        source: ItemSource::Playlist,
    })
}

async fn read_wav_pcm(path: &Path, cfg: &LoadedConfig) -> Result<Pcm> {
    let path = path.to_path_buf();
    let raw = tokio::fs::read(&path)
        .await
        .with_context(|| format!("failed to read audio {}", path.display()))?;
    let pcm = tokio::task::spawn_blocking(move || decode_wav(&raw))
        .await
        .context("WAV decoder task failed")??;
    Ok(normalize_pcm(
        pcm,
        cfg.root.playout.sample_rate,
        cfg.root.playout.channels,
    ))
}

async fn alert_scanner(
    cfg: Arc<LoadedConfig>,
    _client: BridgeClient,
    feed: FeedConfig,
    audio_tx: mpsc::Sender<AudioItem>,
    poll: Duration,
) {
    let mut seen = HashSet::<String>::new();
    let mut ticker = interval(poll.max(Duration::from_millis(100)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        if let Err(err) = scan_alerts_once(&cfg, &feed, &audio_tx, &mut seen).await {
            tracing::warn!(feed_id = feed.id, "alert queue scan failed: {err}");
        }
    }
}

async fn scan_alerts_once(
    cfg: &LoadedConfig,
    feed: &FeedConfig,
    audio_tx: &mpsc::Sender<AudioItem>,
    seen: &mut HashSet<String>,
) -> Result<()> {
    let queue_dir = cfg.base_dir.join(ALERT_QUEUE_DIR);
    let Ok(entries) = fs::read_dir(&queue_dir) else {
        return Ok(());
    };
    let mut manifests: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
        })
        .collect();
    manifests.sort();
    let mut candidates = Vec::<AlertCandidate>::new();
    for manifest in manifests {
        let Ok(item) = read_alert_item(&manifest) else {
            continue;
        };
        if split_same_part(&item) || legacy_split_alert_item(&item) {
            let mut item = item;
            item.status = "superseded".to_string();
            item.last_error = Some(
                "split legacy alert item is superseded by combined SAME alert audio".to_string(),
            );
            let _ = write_alert_item(&manifest, &item);
            continue;
        }
        if !alert_targets_feed(&item, &feed.id) || !alert_pending(&item.status) {
            continue;
        }
        if alert_item_stale_for_priority(&item, Utc::now()) {
            let mut item = item;
            item.status = "superseded".to_string();
            item.last_error = Some("alert queue item is stale or cancelled".to_string());
            let _ = write_alert_item(&manifest, &item);
            continue;
        }
        let id = fallback_text(
            &item.id,
            &safe_id(
                manifest
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("alert"),
            ),
        );
        if seen.contains(&id) {
            continue;
        }
        let sort_key = alert_sort_key(&item, &id, &feed.id);
        candidates.push(AlertCandidate {
            manifest,
            item,
            id,
            sort_key,
        });
    }
    candidates.sort_by(|left, right| left.sort_key.cmp(&right.sort_key));

    for candidate in candidates {
        let AlertCandidate {
            manifest,
            mut item,
            id,
            ..
        } = candidate;
        match audio_item_from_alert(cfg, feed, &manifest, &mut item, &id) {
            Ok(audio) => {
                item.status = "queued".to_string();
                item.claimed_at = Some(Utc::now().to_rfc3339());
                item.failed_at = None;
                item.last_error = None;
                if let Err(err) = write_alert_item(&manifest, &item) {
                    tracing::warn!(
                        feed_id = feed.id,
                        alert_id = id,
                        "failed to claim alert queue item: {err}"
                    );
                    continue;
                }
                seen.insert(id);
                if audio_tx.send(audio).await.is_err() {
                    break;
                }
            }
            Err(err) => {
                item.status = "failed".to_string();
                item.failed_at = Some(Utc::now().to_rfc3339());
                item.last_error = Some(err.to_string());
                let _ = write_alert_item(&manifest, &item);
            }
        }
    }
    Ok(())
}

fn alert_sort_key(item: &AlertQueueItem, id: &str, feed_id: &str) -> AlertSortKey {
    AlertSortKey {
        group: alert_group_key(item, id, feed_id),
        rank: alert_sequence_rank(item, id),
        created_at: item.created_at.trim().to_string(),
        id: id.to_string(),
    }
}

fn alert_group_key(item: &AlertQueueItem, id: &str, feed_id: &str) -> String {
    let explicit = item.alert_id.trim();
    if !explicit.is_empty() {
        return explicit.to_string();
    }
    let mut value = id.trim().to_string();
    for prefix in [
        format!("000_{feed_id}_"),
        format!("001_{feed_id}_"),
        format!("002_{feed_id}_"),
        format!("cap_alert_{feed_id}_"),
    ] {
        if let Some(stripped) = value.strip_prefix(&prefix) {
            value = stripped.to_string();
            break;
        }
    }
    for suffix in ["_same_header", "_same_eom", "_same_full", "_same", "_cap"] {
        if let Some(stripped) = value.strip_suffix(suffix) {
            value = stripped.to_string();
            break;
        }
    }
    value
}

fn alert_sequence_rank(item: &AlertQueueItem, id: &str) -> u8 {
    let marker = format!(
        "{} {} {}",
        item.r#type.to_ascii_lowercase(),
        item.priority.to_ascii_lowercase(),
        id.to_ascii_lowercase()
    );
    if marker.contains("header") || marker.starts_with("  000_") || id.starts_with("000_") {
        0
    } else if marker.contains("eom") || id.starts_with("002_") {
        2
    } else if marker.contains("cap") || id.starts_with("001_") || id.starts_with("cap_alert_") {
        1
    } else {
        3
    }
}

fn audio_item_from_alert(
    cfg: &LoadedConfig,
    _feed: &FeedConfig,
    manifest: &Path,
    item: &mut AlertQueueItem,
    id: &str,
) -> Result<AudioItem> {
    let audio_path = resolve_path(&cfg.base_dir, &item.audio_path);
    let raw = fs::read(&audio_path)
        .with_context(|| format!("failed to read alert audio {}", audio_path.display()))?;
    let pcm = normalize_pcm(
        Pcm {
            sample_rate: first_positive(item.sample_rate, 48_000),
            channels: item.channels.max(1),
            data: raw,
        },
        cfg.root.playout.sample_rate,
        cfg.root.playout.channels,
    );
    let title = fallback_text(&item.header, "SAME Alert");
    Ok(AudioItem {
        id: id.to_string(),
        package_id: "same_alert".to_string(),
        title,
        pcm: pcm.data,
        gap_after: Duration::from_millis(500),
        not_before: None,
        queued_at: Utc::now().to_rfc3339(),
        target_start: String::new(),
        predicted_start: String::new(),
        predicted_finish: String::new(),
        source: ItemSource::Alert {
            manifest_path: manifest.to_path_buf(),
            header: item.header.clone(),
            event: item.event.clone(),
        },
    })
}

fn read_alert_item(path: &Path) -> Result<AlertQueueItem> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read alert manifest {}", path.display()))?;
    let mut item: AlertQueueItem = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse alert manifest {}", path.display()))?;
    if item.id.trim().is_empty() {
        item.id = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("alert")
            .to_string();
    }
    Ok(item)
}

fn write_alert_item(path: &Path, item: &AlertQueueItem) -> Result<()> {
    let raw = serde_json::to_vec_pretty(item)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, raw).with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("failed to replace {}", path.display()))
}

fn mark_alert_played(path: &Path) -> Result<()> {
    let mut item = read_alert_item(path)?;
    item.status = "played".to_string();
    item.played_at = Some(Utc::now().to_rfc3339());
    item.last_error = None;
    write_alert_item(path, &item)
}

fn mark_alert_started(path: &Path) -> Result<()> {
    let mut item = read_alert_item(path)?;
    item.status = "playing".to_string();
    item.claimed_at = Some(Utc::now().to_rfc3339());
    item.last_error = None;
    write_alert_item(path, &item)
}

fn alert_targets_feed(item: &AlertQueueItem, feed_id: &str) -> bool {
    item.feed_ids.iter().any(|id| id.trim() == feed_id)
}

fn alert_pending(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "" | "pending" | "queued" | "claimed" | "playing"
    )
}

fn alert_item_stale_for_priority(item: &AlertQueueItem, now: DateTime<Utc>) -> bool {
    if item.message_type.eq_ignore_ascii_case("cancel") {
        return true;
    }
    let header = item.header.to_ascii_lowercase();
    if header.contains("ended") || header.contains("cancelled") || header.contains("canceled") {
        return true;
    }
    if let Some(expires) = parse_time(&item.alert_expires_at) {
        if now > expires {
            return true;
        }
    }
    let Some(sent) = parse_time(&item.alert_sent_at) else {
        return false;
    };
    if now < sent {
        return false;
    }
    let event = item.event.trim().to_ascii_uppercase();
    let limit = if event == "SVR" || event == "TOR" {
        ChronoDuration::minutes(30)
    } else {
        ChronoDuration::minutes(60)
    };
    now.signed_duration_since(sent) > limit
}

fn split_same_part(item: &AlertQueueItem) -> bool {
    let marker = format!(
        "{} {} {}",
        item.r#type.to_ascii_lowercase(),
        item.priority.to_ascii_lowercase(),
        item.id.to_ascii_lowercase()
    );
    marker.contains("same_header") || marker.contains("same_eom")
}

fn legacy_split_alert_item(item: &AlertQueueItem) -> bool {
    if !item.source.trim().is_empty() {
        return false;
    }
    let id = item.id.to_ascii_lowercase();
    id.starts_with("000_") || id.starts_with("cap_alert_")
}

fn item_event_data(feed_id: &str, item: &AudioItem) -> Value {
    json!({
        "feed_id": feed_id,
        "queue_id": item.id,
        "pkg_id": item.package_id,
        "package_id": item.package_id,
        "title": item.title,
        "queued_at": item.queued_at,
        "target_start_at": item.target_start,
        "predicted_start_at": item.predicted_start,
        "predicted_finish_at": item.predicted_finish,
    })
}

fn package_gap(cfg: &LoadedConfig) -> Duration {
    Duration::from_secs_f64(cfg.root.playout.pacing.package_gap_s.max(0.0))
}

fn parse_time(raw: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(raw.trim())
        .map(|value| value.with_timezone(&Utc))
        .ok()
}

fn first_positive(value: u32, fallback: u32) -> u32 {
    if value == 0 {
        fallback
    } else {
        value
    }
}

fn fallback_text(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.trim().to_string()
    } else {
        value.to_string()
    }
}

fn title_for_package(package_id: &str) -> String {
    match package_id.trim().to_ascii_lowercase().as_str() {
        "date_time" => "Date and Time".to_string(),
        "station_id" => "Station Identification".to_string(),
        "current_conditions" => "Current Conditions".to_string(),
        "forecast" => "Forecast".to_string(),
        "air_quality" => "Air Quality".to_string(),
        "climate_summary" => "Climate Summary".to_string(),
        "geophysical_alert" => "Geophysical Alert".to_string(),
        "user_bulletin" => "User Bulletin".to_string(),
        "alerts" => "Weather Alerts".to_string(),
        "" => "Queued Audio".to_string(),
        other => other.to_string(),
    }
}

fn queue_id(prefix: &str) -> String {
    safe_id(&format!(
        "{}-{}",
        fallback_text(prefix, "item"),
        Utc::now().timestamp_nanos_opt().unwrap_or(0)
    ))
}

fn safe_id(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            out.push(ch);
        }
    }
    if out.is_empty() {
        "item".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alert_pending_accepts_empty_and_pending_only() {
        assert!(alert_pending(""));
        assert!(alert_pending("pending"));
        assert!(alert_pending("queued"));
        assert!(alert_pending("claimed"));
        assert!(alert_pending("playing"));
        assert!(!alert_pending("played"));
    }

    #[test]
    fn safe_id_removes_path_characters() {
        assert_eq!(safe_id("../sk 0001:forecast"), "sk0001forecast");
    }

    #[test]
    fn alert_sort_key_keeps_same_speech_eom_sequence_together() {
        let feed_id = "sk-0001";
        let mut parts = [
            AlertQueueItem {
                id: "002_sk-0001_ALERT_same_eom".to_string(),
                alert_id: "ALERT".to_string(),
                r#type: "same_eom".to_string(),
                ..AlertQueueItem::default()
            },
            AlertQueueItem {
                id: "001_sk-0001_ALERT_cap".to_string(),
                alert_id: "ALERT".to_string(),
                r#type: "cap_alert".to_string(),
                ..AlertQueueItem::default()
            },
            AlertQueueItem {
                id: "000_sk-0001_ALERT_same_header".to_string(),
                alert_id: "ALERT".to_string(),
                r#type: "same_header".to_string(),
                ..AlertQueueItem::default()
            },
        ];

        parts.sort_by_key(|item| alert_sort_key(item, &item.id, feed_id));

        let ordered: Vec<&str> = parts.iter().map(|item| item.r#type.as_str()).collect();
        assert_eq!(ordered, ["same_header", "cap_alert", "same_eom"]);
    }

    #[test]
    fn split_same_parts_are_not_standalone_alert_items() {
        assert!(split_same_part(&AlertQueueItem {
            r#type: "same_header".to_string(),
            ..AlertQueueItem::default()
        }));
        assert!(split_same_part(&AlertQueueItem {
            id: "002_sk-0001_ALERT_same_eom".to_string(),
            ..AlertQueueItem::default()
        }));
        assert!(!split_same_part(&AlertQueueItem {
            r#type: "same_alert".to_string(),
            ..AlertQueueItem::default()
        }));
    }

    #[test]
    fn legacy_split_alert_items_are_superseded() {
        assert!(legacy_split_alert_item(&AlertQueueItem {
            id: "000_sk-0001_ALERT_same".to_string(),
            ..AlertQueueItem::default()
        }));
        assert!(legacy_split_alert_item(&AlertQueueItem {
            id: "cap_alert_sk-0001_ALERT".to_string(),
            ..AlertQueueItem::default()
        }));
        assert!(legacy_split_alert_item(&AlertQueueItem {
            id: "000_sk-0001_ALERT_same".to_string(),
            r#type: "same_alert".to_string(),
            priority: "same".to_string(),
            ..AlertQueueItem::default()
        }));
        assert!(!legacy_split_alert_item(&AlertQueueItem {
            id: "000_sk-0001_ALERT_same".to_string(),
            r#type: "same_alert".to_string(),
            priority: "same".to_string(),
            source: "cap-same-tts".to_string(),
            ..AlertQueueItem::default()
        }));
    }

    #[test]
    fn stale_priority_alerts_are_superseded_by_alert_sent_time() {
        let now = DateTime::parse_from_rfc3339("2026-06-16T23:40:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(!alert_item_stale_for_priority(
            &AlertQueueItem {
                event: "SVR".to_string(),
                alert_sent_at: "2026-06-16T23:11:00Z".to_string(),
                ..AlertQueueItem::default()
            },
            now
        ));
        assert!(alert_item_stale_for_priority(
            &AlertQueueItem {
                event: "SVR".to_string(),
                alert_sent_at: "2026-06-16T23:09:00Z".to_string(),
                ..AlertQueueItem::default()
            },
            now
        ));
        assert!(!alert_item_stale_for_priority(
            &AlertQueueItem {
                event: "SVA".to_string(),
                alert_sent_at: "2026-06-16T22:41:00Z".to_string(),
                ..AlertQueueItem::default()
            },
            now
        ));
        assert!(alert_item_stale_for_priority(
            &AlertQueueItem {
                event: "SVA".to_string(),
                alert_sent_at: "2026-06-16T22:39:00Z".to_string(),
                ..AlertQueueItem::default()
            },
            now
        ));
    }

    #[test]
    fn cancelled_priority_alerts_are_superseded() {
        let now = DateTime::parse_from_rfc3339("2026-06-16T23:40:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(alert_item_stale_for_priority(
            &AlertQueueItem {
                message_type: "Cancel".to_string(),
                ..AlertQueueItem::default()
            },
            now
        ));
        assert!(alert_item_stale_for_priority(
            &AlertQueueItem {
                header: "yellow warning - severe thunderstorm - ended".to_string(),
                ..AlertQueueItem::default()
            },
            now
        ));
    }

    #[test]
    fn live_breakin_trim_drops_oldest_audio_on_frame_boundaries() {
        let channels = 1;
        let max = max_live_breakin_buffer_for(48_000, channels);
        let mut live = LiveBreakIn {
            id: "breakin".to_string(),
            title: "Break-in".to_string(),
            buffer: (0..(max + 8)).map(|value| (value % 251) as u8).collect(),
            finishing: false,
        };

        trim_live_breakin_buffer(&mut live, 48_000, channels);

        assert!(live.buffer.len() <= max);
        assert_eq!(live.buffer.len() % live_breakin_frame_bytes(channels), 0);
        assert_eq!(live.buffer.front().copied(), Some(8 % 251));
    }

    #[test]
    fn live_breakin_drain_copies_available_audio_without_touching_tail() {
        let mut live = LiveBreakIn {
            id: "breakin".to_string(),
            title: "Break-in".to_string(),
            buffer: VecDeque::from(vec![1, 2, 3]),
            finishing: false,
        };
        let mut out = [0u8; 6];

        drain_live_breakin_buffer(&mut live, &mut out);

        assert_eq!(out, [1, 2, 3, 0, 0, 0]);
        assert!(live.buffer.is_empty());
    }

    #[test]
    fn realtime_pcm_publish_queue_stays_short_for_low_latency() {
        assert!(pcm_publish_queue_capacity() <= 4);
        assert_eq!(MAX_CATCH_UP_CHUNKS, 1);
    }

    #[test]
    fn realtime_lag_warning_ignores_single_frame_jitter() {
        assert!(realtime_lag_warn_backlog() > Duration::from_millis(u64::from(PCM_CHUNK_MS)));
    }
}
