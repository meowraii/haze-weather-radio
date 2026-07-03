use std::collections::{BTreeMap, HashMap, VecDeque};
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "gstreamer-backend")]
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use base64::Engine as _;
use clap::Parser;
use haze_media::{normalize_pcm, pcm16_samples, push_i16, AudioFormat, Pcm};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};
use tokio::time::{interval, sleep, MissedTickBehavior};
use tracing::{debug, info, warn};

#[cfg(feature = "gstreamer-backend")]
use std::io::ErrorKind;

#[cfg(windows)]
use windows_sys::Win32::Media::{timeBeginPeriod, timeEndPeriod};

const SOURCE_ID: &str = "haze-media";
const DEFAULT_CONFIG: &str = "config.yaml";
const DEFAULT_OUTPUTS_FILE: &str = "managed/configs/output.xml";
const DEFAULT_LISTEN: &str = "127.0.0.1:8097";
const SAMPLE_RATE: u32 = 48_000;
const CHANNELS: u16 = 1;
const FRAME_DURATION: Duration = Duration::from_millis(20);
const FRAME_SAMPLES: usize = SAMPLE_RATE as usize / 50;
const FRAME_BYTES: usize = FRAME_SAMPLES * CHANNELS as usize * 2;
const INPUT_QUEUE_CAPACITY: usize = 32;
const PACED_FRAME_CAPACITY: usize = 64;
const TARGET_SOURCE_QUEUE_MS: u64 = 280;
const SOFT_SOURCE_QUEUE_MS: u64 = 440;
const MAX_SOURCE_QUEUE_MS: u64 = 900;
const SOURCE_DRIFT_TRIM_MS: u64 = 2;
const MIN_SOURCE_DRIFT_TRIM_SAMPLES: usize = 4;
const CONCEALMENT_FRAMES: u8 = 3;
const STATUS_INTERVAL: Duration = Duration::from_secs(5);
const HTTP_HEADER_LIMIT: usize = 16 * 1024;
const HTTP_BODY_LIMIT: usize = 512 * 1024;
#[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
const ENCODED_HTTP_QUEUE_CAPACITY: usize = 64;
const RAW_HTTP_LAG_RESYNC_SILENCE_FRAMES: u64 = 1;
#[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
const WEBRTC_OFFER_TIMEOUT: Duration = Duration::from_secs(8);
#[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
const WEBRTC_ICE_GATHER_WINDOW: Duration = Duration::from_millis(900);
const NEAR_SILENT_PEAK: u16 = 20;
const CLIP_SAMPLE_PEAK: u16 = 32_760;
const AUDIO_HEALTH_INPUT_STALE_MS: u128 = 1_000;
const AUDIO_HEALTH_TICK_GAP_WARN_MS: u128 = 30;
const DEFAULT_LOUDNESS_ENABLED: bool = false;
const DEFAULT_LOUDNESS_GAIN_DB: f32 = 5.0;
const DEFAULT_COMPRESSOR_THRESHOLD_DBFS: f32 = -18.0;
const DEFAULT_COMPRESSOR_RATIO: f32 = 3.0;
const DEFAULT_COMPRESSOR_MAKEUP_DB: f32 = 1.5;
const DEFAULT_COMPRESSOR_ATTACK_MS: f32 = 5.0;
const DEFAULT_COMPRESSOR_RELEASE_MS: f32 = 120.0;
const DEFAULT_LIMITER_CEILING: f32 = 0.98;

#[derive(Debug, Parser)]
#[command(about = "Haze GStreamer-backed live media service")]
struct Args {
    #[arg(long, default_value = DEFAULT_CONFIG)]
    config: PathBuf,

    #[arg(long, env = "HAZE_HOST_BRIDGE_ADDR")]
    bridge: String,

    #[arg(long, env = "HAZE_MEDIA_BRIDGE_ADDR")]
    media_bridge: Option<String>,

    #[arg(long)]
    listen: Option<String>,

    #[arg(long)]
    outputs: Option<PathBuf>,

    #[arg(long)]
    backend: Option<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum BackendMode {
    Auto,
    GStreamer,
    Legacy,
}

impl BackendMode {
    fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "gstreamer" | "gst" => Self::GStreamer,
            "legacy" | "builtin" => Self::Legacy,
            _ => Self::Auto,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::GStreamer => "gstreamer",
            Self::Legacy => "legacy",
        }
    }
}

#[derive(Debug, Deserialize)]
struct RootConfig {
    #[serde(default = "default_outputs_file")]
    outputs_file: String,
    #[serde(default)]
    services: ServicesConfig,
}

#[derive(Debug, Default, Deserialize)]
struct ServicesConfig {
    #[serde(default)]
    rust: RustServicesConfig,
}

#[derive(Debug, Default, Deserialize)]
struct RustServicesConfig {
    #[serde(default)]
    media: MediaServiceConfig,
}

#[derive(Debug, Default, Deserialize)]
struct MediaServiceConfig {
    #[serde(default, rename = "enabled")]
    _enabled: bool,
    #[serde(default)]
    addr: String,
    #[serde(default)]
    listen: String,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    audio_processing: MediaAudioProcessingConfig,
}

#[derive(Debug, Default, Deserialize)]
struct MediaAudioProcessingConfig {
    #[serde(default)]
    loudness_enabled: Option<bool>,
    #[serde(default)]
    gain_db: Option<f32>,
    #[serde(default)]
    compressor_threshold_dbfs: Option<f32>,
    #[serde(default)]
    compressor_ratio: Option<f32>,
    #[serde(default)]
    compressor_makeup_db: Option<f32>,
    #[serde(default)]
    limiter_ceiling: Option<f32>,
    #[serde(default)]
    compressor_attack_ms: Option<f32>,
    #[serde(default)]
    compressor_release_ms: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct OutputsXml {
    #[serde(rename = "feed", default)]
    feeds: Vec<OutputFeedXml>,
}

#[derive(Debug, Default, Deserialize)]
struct OutputFeedXml {
    #[serde(rename = "@id", default)]
    id: String,
    #[serde(default)]
    webrtc: OutputNodeXml,
    #[serde(default)]
    stream: OutputNodeXml,
    #[serde(default)]
    udp: OutputNodeXml,
    #[serde(default)]
    rtp: OutputNodeXml,
    #[serde(default)]
    rtmp: OutputNodeXml,
    #[serde(default)]
    srt: OutputNodeXml,
    #[serde(default)]
    rtsp: OutputNodeXml,
    #[serde(default)]
    audio_device: OutputNodeXml,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct OutputNodeXml {
    #[serde(rename = "@enabled", default)]
    enabled: Option<String>,
}

impl OutputNodeXml {
    fn is_enabled(&self) -> bool {
        xml_bool(self.enabled.as_deref(), false)
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct OutputFeed {
    id: String,
    webrtc: bool,
    http: bool,
    external: bool,
    webrtc_ready: bool,
}

#[derive(Debug, Clone)]
struct PcmChunk {
    feed_id: String,
    sample_rate: u32,
    channels: u16,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct PacedFrame {
    _sequence: u64,
    data: Arc<[u8]>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum WebRTCAudioCodec {
    Opus,
    G722,
    Pcmu,
    Pcma,
}

impl WebRTCAudioCodec {
    fn id(self) -> &'static str {
        match self {
            Self::Opus => "opus",
            Self::G722 => "g722",
            Self::Pcmu => "pcmu",
            Self::Pcma => "pcma",
        }
    }

    fn rtpmap_match(self) -> &'static str {
        match self {
            Self::Opus => "OPUS/48000",
            Self::G722 => "G722/8000",
            Self::Pcmu => "PCMU/8000",
            Self::Pcma => "PCMA/8000",
        }
    }

    fn is_static_payload(self, payload_type: u8) -> bool {
        matches!(
            (self, payload_type),
            (Self::G722, 9) | (Self::Pcmu, 0) | (Self::Pcma, 8)
        )
    }

    #[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
    fn webrtc_caps(self, payload_type: u8) -> String {
        match self {
            Self::Opus => format!(
                "application/x-rtp,media=audio,encoding-name=OPUS,payload={payload_type},clock-rate=48000"
            ),
            Self::G722 => format!(
                "application/x-rtp,media=audio,encoding-name=G722,payload={payload_type},clock-rate=8000"
            ),
            Self::Pcmu => format!(
                "application/x-rtp,media=audio,encoding-name=PCMU,payload={payload_type},clock-rate=8000"
            ),
            Self::Pcma => format!(
                "application/x-rtp,media=audio,encoding-name=PCMA,payload={payload_type},clock-rate=8000"
            ),
        }
    }
}

#[derive(Debug, Clone)]
struct AudioLoudness {
    enabled: bool,
    gain: f32,
    threshold_db: f32,
    ratio: f32,
    makeup: f32,
    ceiling: f32,
    attack_coeff: f32,
    release_coeff: f32,
    envelope: f32,
}

impl AudioLoudness {
    fn from_env() -> Self {
        let enabled = env_bool("HAZE_MEDIA_LOUDNESS_ENABLED", DEFAULT_LOUDNESS_ENABLED);
        Self {
            enabled,
            gain: db_to_gain(env_f32("HAZE_MEDIA_GAIN_DB", DEFAULT_LOUDNESS_GAIN_DB)),
            threshold_db: env_f32(
                "HAZE_MEDIA_COMPRESSOR_THRESHOLD_DBFS",
                DEFAULT_COMPRESSOR_THRESHOLD_DBFS,
            ),
            ratio: env_f32("HAZE_MEDIA_COMPRESSOR_RATIO", DEFAULT_COMPRESSOR_RATIO).max(1.0),
            makeup: db_to_gain(env_f32(
                "HAZE_MEDIA_COMPRESSOR_MAKEUP_DB",
                DEFAULT_COMPRESSOR_MAKEUP_DB,
            )),
            ceiling: env_f32("HAZE_MEDIA_LIMITER_CEILING", DEFAULT_LIMITER_CEILING).clamp(0.1, 1.0),
            attack_coeff: compressor_coeff_ms(env_f32(
                "HAZE_MEDIA_COMPRESSOR_ATTACK_MS",
                DEFAULT_COMPRESSOR_ATTACK_MS,
            )),
            release_coeff: compressor_coeff_ms(env_f32(
                "HAZE_MEDIA_COMPRESSOR_RELEASE_MS",
                DEFAULT_COMPRESSOR_RELEASE_MS,
            )),
            envelope: 0.0,
        }
    }

    fn process(&mut self, frame: &mut [u8]) {
        if !self.enabled || frame.is_empty() {
            return;
        }
        for sample in frame.chunks_exact_mut(2) {
            let raw = i16::from_le_bytes([sample[0], sample[1]]);
            let processed = self.process_sample(raw);
            sample.copy_from_slice(&processed.to_le_bytes());
        }
    }

    fn process_sample(&mut self, sample: i16) -> i16 {
        let mut value = f32::from(sample) / 32768.0;
        value *= self.gain;
        let level = value.abs();
        let coeff = if level > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = (coeff * self.envelope) + ((1.0 - coeff) * level);
        if self.ratio > 1.0 && self.envelope > 0.0 {
            let envelope_db = gain_to_db(self.envelope.max(1.0e-9));
            if envelope_db > self.threshold_db {
                let compressed_db =
                    self.threshold_db + ((envelope_db - self.threshold_db) / self.ratio);
                value *= db_to_gain(compressed_db - envelope_db);
            }
        }
        value *= self.makeup;
        value = value.clamp(-self.ceiling, self.ceiling);
        (value * f32::from(i16::MAX)).round() as i16
    }
}

#[derive(Debug, Clone, Default, Serialize)]
struct FeedStats {
    feed_id: String,
    audio_ok: bool,
    audio_warnings: Vec<String>,
    frames: u64,
    real_frames: u64,
    silence_frames: u64,
    concealed_frames: u64,
    partial_frames: u64,
    late_ticks: u64,
    catchup_ticks: u64,
    dropped_input_chunks: u64,
    stale_samples_dropped: u64,
    source_drift_samples_trimmed: u64,
    near_silent_frames: u64,
    consecutive_near_silent_frames: u64,
    repeated_frames: u64,
    repeated_non_silent_frames: u64,
    consecutive_repeated_frames: u64,
    clipped_samples: u64,
    last_peak: u16,
    last_rms_dbfs: f64,
    last_max_sample_jump: u16,
    subscribers: usize,
    queued_samples: usize,
    last_tick_gap_ms: Option<u128>,
    max_tick_gap_ms: u128,
    last_input_age_ms: Option<u128>,
    last_frame_age_ms: Option<u128>,
}

struct FeedRuntime {
    feed_id: String,
    input_tx: mpsc::Sender<PcmChunk>,
    frame_tx: broadcast::Sender<PacedFrame>,
    stats: Arc<Mutex<FeedStats>>,
}

impl FeedRuntime {
    fn new(feed_id: String) -> Arc<Self> {
        let (input_tx, input_rx) = mpsc::channel(INPUT_QUEUE_CAPACITY);
        let (frame_tx, _) = broadcast::channel(PACED_FRAME_CAPACITY);
        let stats = Arc::new(Mutex::new(FeedStats {
            feed_id: feed_id.clone(),
            ..FeedStats::default()
        }));
        let runtime = Arc::new(Self {
            feed_id,
            input_tx,
            frame_tx,
            stats,
        });
        start_feed_clock_thread(Arc::clone(&runtime), input_rx);
        runtime
    }

    fn push(&self, chunk: PcmChunk) {
        if let Err(mpsc::error::TrySendError::Full(_)) = self.input_tx.try_send(chunk) {
            if let Ok(mut stats) = self.stats.lock() {
                stats.dropped_input_chunks = stats.dropped_input_chunks.saturating_add(1);
            }
        }
    }

    fn subscribe(&self) -> broadcast::Receiver<PacedFrame> {
        self.frame_tx.subscribe()
    }

    fn snapshot(&self) -> FeedStats {
        let mut snapshot = self
            .stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_else(|_| FeedStats {
                feed_id: self.feed_id.clone(),
                ..FeedStats::default()
            });
        snapshot.subscribers = self.frame_tx.receiver_count();
        let (audio_ok, audio_warnings) = classify_audio_health(&snapshot);
        snapshot.audio_ok = audio_ok;
        snapshot.audio_warnings = audio_warnings;
        snapshot
    }
}

#[derive(Clone)]
struct MediaState {
    feeds: Arc<Mutex<HashMap<String, Arc<FeedRuntime>>>>,
    configured_outputs: Arc<BTreeMap<String, OutputFeed>>,
    http_clients: Arc<Mutex<HashMap<u64, HttpClientRuntime>>>,
    webrtc_peers: Arc<Mutex<HashMap<u64, WebRTCPeerRuntime>>>,
    next_http_client_id: Arc<AtomicU64>,
    next_webrtc_peer_id: Arc<AtomicU64>,
    backend: BackendMode,
    gstreamer_available: bool,
    webrtc_available: bool,
    webrtc_codecs: Arc<Vec<WebRTCAudioCodec>>,
}

#[derive(Debug, Clone)]
struct HttpClientRuntime {
    id: u64,
    feed_id: String,
    codec: &'static str,
    connected_at: Instant,
    last_write_at: Option<Instant>,
    bytes_written: u64,
    chunks_written: u64,
    encoded_chunks_dropped: u64,
    lagged_frames: u64,
}

#[derive(Debug, Serialize)]
struct HttpClientSnapshot {
    id: u64,
    feed_id: String,
    codec: &'static str,
    connected_ms: u128,
    last_write_age_ms: Option<u128>,
    bytes_written: u64,
    chunks_written: u64,
    encoded_chunks_dropped: u64,
    lagged_frames: u64,
}

#[derive(Debug, Clone)]
struct WebRTCPeerRuntime {
    id: u64,
    feed_id: String,
    codec: &'static str,
    connected_at: Instant,
    last_push_at: Option<Instant>,
    bytes_pushed: u64,
    frames_pushed: u64,
    dropped_frames: u64,
}

#[derive(Debug, Serialize)]
struct WebRTCPeerSnapshot {
    id: u64,
    feed_id: String,
    codec: &'static str,
    connected_ms: u128,
    last_push_age_ms: Option<u128>,
    bytes_pushed: u64,
    frames_pushed: u64,
    dropped_frames: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct WebRTCCodecSelection {
    codec: WebRTCAudioCodec,
    payload_type: u8,
}

struct HttpClientGuard {
    id: u64,
    clients: Arc<Mutex<HashMap<u64, HttpClientRuntime>>>,
}

impl HttpClientGuard {
    fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for HttpClientGuard {
    fn drop(&mut self) {
        if let Ok(mut clients) = self.clients.lock() {
            clients.remove(&self.id);
        }
    }
}

impl MediaState {
    fn new(
        mut outputs: BTreeMap<String, OutputFeed>,
        backend: BackendMode,
        gstreamer_available: bool,
        webrtc_codecs: Vec<WebRTCAudioCodec>,
    ) -> Self {
        let webrtc_available = gstreamer_available && !webrtc_codecs.is_empty();
        let webrtc_ready = webrtc_available;
        for output in outputs.values_mut() {
            output.webrtc_ready = output.webrtc && webrtc_ready;
        }
        Self {
            feeds: Arc::new(Mutex::new(HashMap::new())),
            configured_outputs: Arc::new(outputs),
            http_clients: Arc::new(Mutex::new(HashMap::new())),
            webrtc_peers: Arc::new(Mutex::new(HashMap::new())),
            next_http_client_id: Arc::new(AtomicU64::new(1)),
            next_webrtc_peer_id: Arc::new(AtomicU64::new(1)),
            backend,
            gstreamer_available,
            webrtc_available,
            webrtc_codecs: Arc::new(webrtc_codecs),
        }
    }

    fn feed(&self, feed_id: &str) -> Arc<FeedRuntime> {
        let feed_id = feed_id.trim().to_string();
        let mut feeds = self.feeds.lock().expect("feed registry poisoned");
        if let Some(feed) = feeds.get(&feed_id) {
            return Arc::clone(feed);
        }
        let runtime = FeedRuntime::new(feed_id.clone());
        feeds.insert(feed_id, Arc::clone(&runtime));
        runtime
    }

    fn publish_pcm(&self, chunk: PcmChunk) {
        if chunk.feed_id.trim().is_empty() || chunk.data.is_empty() {
            return;
        }
        self.feed(&chunk.feed_id).push(chunk);
    }

    fn snapshots(&self) -> Vec<FeedStats> {
        let feeds = self.feeds.lock().expect("feed registry poisoned");
        let mut out = feeds
            .values()
            .map(|feed| feed.snapshot())
            .collect::<Vec<_>>();
        out.sort_by(|a, b| a.feed_id.cmp(&b.feed_id));
        out
    }

    fn register_http_client(&self, feed_id: &str, codec: &'static str) -> HttpClientGuard {
        let id = self.next_http_client_id.fetch_add(1, Ordering::Relaxed);
        let client = HttpClientRuntime {
            id,
            feed_id: feed_id.to_string(),
            codec,
            connected_at: Instant::now(),
            last_write_at: None,
            bytes_written: 0,
            chunks_written: 0,
            encoded_chunks_dropped: 0,
            lagged_frames: 0,
        };
        if let Ok(mut clients) = self.http_clients.lock() {
            clients.insert(id, client);
        }
        HttpClientGuard {
            id,
            clients: Arc::clone(&self.http_clients),
        }
    }

    fn record_http_client_write(&self, client_id: u64, bytes: usize) {
        if let Ok(mut clients) = self.http_clients.lock() {
            if let Some(client) = clients.get_mut(&client_id) {
                client.last_write_at = Some(Instant::now());
                client.bytes_written = client.bytes_written.saturating_add(bytes as u64);
                client.chunks_written = client.chunks_written.saturating_add(1);
            }
        }
    }

    #[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
    fn record_http_client_encoded_drop(&self, client_id: u64) {
        if let Ok(mut clients) = self.http_clients.lock() {
            if let Some(client) = clients.get_mut(&client_id) {
                client.encoded_chunks_dropped = client.encoded_chunks_dropped.saturating_add(1);
            }
        }
    }

    fn record_http_client_lag(&self, client_id: u64, frames: u64) {
        if let Ok(mut clients) = self.http_clients.lock() {
            if let Some(client) = clients.get_mut(&client_id) {
                client.lagged_frames = client.lagged_frames.saturating_add(frames);
            }
        }
    }

    fn http_client_snapshots(&self) -> Vec<HttpClientSnapshot> {
        let now = Instant::now();
        let clients = self
            .http_clients
            .lock()
            .expect("HTTP client registry poisoned");
        let mut out = clients
            .values()
            .map(|client| HttpClientSnapshot {
                id: client.id,
                feed_id: client.feed_id.clone(),
                codec: client.codec,
                connected_ms: now.duration_since(client.connected_at).as_millis(),
                last_write_age_ms: client
                    .last_write_at
                    .map(|instant| now.duration_since(instant).as_millis()),
                bytes_written: client.bytes_written,
                chunks_written: client.chunks_written,
                encoded_chunks_dropped: client.encoded_chunks_dropped,
                lagged_frames: client.lagged_frames,
            })
            .collect::<Vec<_>>();
        out.sort_by_key(|client| client.id);
        out
    }

    fn register_webrtc_peer(&self, feed_id: &str, codec: &'static str) -> u64 {
        let id = self.next_webrtc_peer_id.fetch_add(1, Ordering::Relaxed);
        let peer = WebRTCPeerRuntime {
            id,
            feed_id: feed_id.to_string(),
            codec,
            connected_at: Instant::now(),
            last_push_at: None,
            bytes_pushed: 0,
            frames_pushed: 0,
            dropped_frames: 0,
        };
        if let Ok(mut peers) = self.webrtc_peers.lock() {
            peers.insert(id, peer);
        }
        id
    }

    #[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
    fn unregister_webrtc_peer(&self, peer_id: u64) {
        if let Ok(mut peers) = self.webrtc_peers.lock() {
            peers.remove(&peer_id);
        }
    }

    #[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
    fn record_webrtc_peer_push(&self, peer_id: u64, bytes: usize) {
        if let Ok(mut peers) = self.webrtc_peers.lock() {
            if let Some(peer) = peers.get_mut(&peer_id) {
                peer.last_push_at = Some(Instant::now());
                peer.bytes_pushed = peer.bytes_pushed.saturating_add(bytes as u64);
                peer.frames_pushed = peer.frames_pushed.saturating_add(1);
            }
        }
    }

    #[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
    fn record_webrtc_peer_drop(&self, peer_id: u64) {
        if let Ok(mut peers) = self.webrtc_peers.lock() {
            if let Some(peer) = peers.get_mut(&peer_id) {
                peer.dropped_frames = peer.dropped_frames.saturating_add(1);
            }
        }
    }

    fn webrtc_peer_snapshots(&self) -> Vec<WebRTCPeerSnapshot> {
        let now = Instant::now();
        let peers = self
            .webrtc_peers
            .lock()
            .expect("WebRTC peer registry poisoned");
        let mut out = peers
            .values()
            .map(|peer| WebRTCPeerSnapshot {
                id: peer.id,
                feed_id: peer.feed_id.clone(),
                codec: peer.codec,
                connected_ms: now.duration_since(peer.connected_at).as_millis(),
                last_push_age_ms: peer
                    .last_push_at
                    .map(|instant| now.duration_since(instant).as_millis()),
                bytes_pushed: peer.bytes_pushed,
                frames_pushed: peer.frames_pushed,
                dropped_frames: peer.dropped_frames,
            })
            .collect::<Vec<_>>();
        out.sort_by_key(|peer| peer.id);
        out
    }

    fn health(&self) -> Value {
        let http_clients = self.http_client_snapshots();
        let webrtc_peers = self.webrtc_peer_snapshots();
        let webrtc_ready = self.gstreamer_available && self.webrtc_available;
        let webrtc_codecs = self
            .webrtc_codecs
            .iter()
            .map(|codec| codec.id())
            .collect::<Vec<_>>();
        json!({
            "ok": true,
            "service": "haze-media",
            "backend": self.backend.as_str(),
            "gstreamer_available": self.gstreamer_available,
            "capabilities": {
                "http_audio": true,
                "encoded_http_audio": self.gstreamer_available,
                "webrtc": webrtc_ready,
                "webrtc_reason": if webrtc_ready { Value::Null } else if !self.gstreamer_available {
                    json!("gstreamer backend is unavailable")
                } else {
                    json!("gstreamer webrtc elements are unavailable")
                },
                "webrtc_codecs": webrtc_codecs,
            },
            "audio_processing": audio_processing_health(),
            "clock": media_clock_health(),
            "configured_outputs": self.configured_outputs.values().collect::<Vec<_>>(),
            "http_client_count": http_clients.len(),
            "http_clients": http_clients,
            "webrtc_peer_count": webrtc_peers.len(),
            "webrtc_peers": webrtc_peers,
            "feeds": self.snapshots(),
        })
    }
}

fn audio_processing_health() -> Value {
    let enabled = env_bool("HAZE_MEDIA_LOUDNESS_ENABLED", DEFAULT_LOUDNESS_ENABLED);
    let gain_db = env_f32("HAZE_MEDIA_GAIN_DB", DEFAULT_LOUDNESS_GAIN_DB);
    let threshold_dbfs = env_f32(
        "HAZE_MEDIA_COMPRESSOR_THRESHOLD_DBFS",
        DEFAULT_COMPRESSOR_THRESHOLD_DBFS,
    );
    let ratio = env_f32("HAZE_MEDIA_COMPRESSOR_RATIO", DEFAULT_COMPRESSOR_RATIO).max(1.0);
    let makeup_db = env_f32(
        "HAZE_MEDIA_COMPRESSOR_MAKEUP_DB",
        DEFAULT_COMPRESSOR_MAKEUP_DB,
    );
    let limiter_ceiling =
        env_f32("HAZE_MEDIA_LIMITER_CEILING", DEFAULT_LIMITER_CEILING).clamp(0.1, 1.0);
    json!({
        "loudness_enabled": enabled,
        "gain_db": gain_db,
        "compressor_threshold_dbfs": threshold_dbfs,
        "compressor_ratio": ratio,
        "compressor_makeup_db": makeup_db,
        "limiter_ceiling": limiter_ceiling,
        "source": env::var("HAZE_MEDIA_AUDIO_PROCESSING_SOURCE")
            .unwrap_or_else(|_| "environment".to_string()),
    })
}

fn apply_media_audio_processing_config(config: &MediaAudioProcessingConfig) {
    let has_config = config.loudness_enabled.is_some()
        || config.gain_db.is_some()
        || config.compressor_threshold_dbfs.is_some()
        || config.compressor_ratio.is_some()
        || config.compressor_makeup_db.is_some()
        || config.limiter_ceiling.is_some()
        || config.compressor_attack_ms.is_some()
        || config.compressor_release_ms.is_some();
    if has_config && env::var_os("HAZE_MEDIA_AUDIO_PROCESSING_SOURCE").is_none() {
        env::set_var("HAZE_MEDIA_AUDIO_PROCESSING_SOURCE", "config.yaml");
    }
    set_env_bool_if_absent("HAZE_MEDIA_LOUDNESS_ENABLED", config.loudness_enabled);
    set_env_f32_if_absent("HAZE_MEDIA_GAIN_DB", config.gain_db);
    set_env_f32_if_absent(
        "HAZE_MEDIA_COMPRESSOR_THRESHOLD_DBFS",
        config.compressor_threshold_dbfs,
    );
    set_env_f32_if_absent("HAZE_MEDIA_COMPRESSOR_RATIO", config.compressor_ratio);
    set_env_f32_if_absent(
        "HAZE_MEDIA_COMPRESSOR_MAKEUP_DB",
        config.compressor_makeup_db,
    );
    set_env_f32_if_absent("HAZE_MEDIA_LIMITER_CEILING", config.limiter_ceiling);
    set_env_f32_if_absent(
        "HAZE_MEDIA_COMPRESSOR_ATTACK_MS",
        config.compressor_attack_ms,
    );
    set_env_f32_if_absent(
        "HAZE_MEDIA_COMPRESSOR_RELEASE_MS",
        config.compressor_release_ms,
    );
}

fn set_env_bool_if_absent(name: &str, value: Option<bool>) {
    if env::var_os(name).is_some() {
        return;
    }
    if let Some(value) = value {
        env::set_var(name, if value { "true" } else { "false" });
    }
}

fn set_env_f32_if_absent(name: &str, value: Option<f32>) {
    if env::var_os(name).is_some() {
        return;
    }
    if let Some(value) = value {
        env::set_var(name, value.to_string());
    }
}

fn media_clock_health() -> Value {
    json!({
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "frame_ms": FRAME_DURATION.as_millis(),
        "frame_samples": FRAME_SAMPLES,
        "target_source_queue_ms": TARGET_SOURCE_QUEUE_MS,
        "soft_source_queue_ms": SOFT_SOURCE_QUEUE_MS,
        "max_source_queue_ms": MAX_SOURCE_QUEUE_MS,
        "source_drift_trim_ms": SOURCE_DRIFT_TRIM_MS,
        "input_queue_capacity": INPUT_QUEUE_CAPACITY,
        "paced_frame_capacity": PACED_FRAME_CAPACITY,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "haze_media=info,info".into()),
        )
        .init();
    let _timer_resolution = TimerResolutionGuard::request();

    let args = Args::parse();
    let root = load_root_config(&args.config)?;
    apply_media_audio_processing_config(&root.services.rust.media.audio_processing);
    let listen = first_non_blank(&[
        args.listen.as_deref(),
        Some(root.services.rust.media.listen.as_str()),
        Some(root.services.rust.media.addr.as_str()),
        Some(DEFAULT_LISTEN),
    ]);
    let backend = BackendMode::parse(&first_non_blank(&[
        args.backend.as_deref(),
        Some(root.services.rust.media.backend.as_str()),
        Some("auto"),
    ]));
    let base_dir = args
        .config
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    let outputs_path = args.outputs.unwrap_or_else(|| {
        resolve_path(
            &base_dir,
            Path::new(first_non_blank(&[Some(root.outputs_file.as_str())]).as_str()),
        )
    });
    let outputs = load_outputs(&outputs_path)?;
    let gstreamer_available = initialize_gstreamer(backend)?;
    let webrtc_codecs = detect_gstreamer_webrtc_codecs(gstreamer_available);
    let state = MediaState::new(outputs, backend, gstreamer_available, webrtc_codecs);

    info!(
        listen,
        bridge = %args.bridge,
        media_bridge = %first_non_blank(&[args.media_bridge.as_deref(), Some(args.bridge.as_str())]),
        backend = state.backend.as_str(),
        gstreamer_available,
        webrtc_available = state.webrtc_available,
        outputs = %outputs_path.display(),
        "starting haze-media"
    );

    let status_state = state.clone();
    let status_bridge_addr = args.bridge.clone();
    tokio::spawn(async move {
        run_status_bridge_loop(status_bridge_addr, status_state).await;
    });

    let pcm_state = state.clone();
    let pcm_bridge_addr =
        first_non_blank(&[args.media_bridge.as_deref(), Some(args.bridge.as_str())]);
    tokio::spawn(async move {
        run_pcm_bridge_loop(pcm_bridge_addr, pcm_state).await;
    });

    let server_state = state.clone();
    let server = tokio::spawn(async move { run_http_server(&listen, server_state).await });

    tokio::select! {
        result = server => {
            result.context("haze-media HTTP task panicked")??;
        }
        signal = tokio::signal::ctrl_c() => {
            signal.context("failed waiting for shutdown signal")?;
            info!("haze-media shutdown requested");
        }
    }
    Ok(())
}

struct TimerResolutionGuard;

impl TimerResolutionGuard {
    fn request() -> Self {
        request_timer_resolution();
        Self
    }
}

impl Drop for TimerResolutionGuard {
    fn drop(&mut self) {
        release_timer_resolution();
    }
}

#[cfg(windows)]
fn request_timer_resolution() {
    // A 20 ms media clock is audibly sensitive to the default Windows timer quantum.
    let result = unsafe { timeBeginPeriod(1) };
    if result == 0 {
        info!("haze-media requested 1 ms Windows timer resolution");
    } else {
        warn!("haze-media failed to request 1 ms Windows timer resolution: {result}");
    }
}

#[cfg(not(windows))]
fn request_timer_resolution() {}

#[cfg(windows)]
fn release_timer_resolution() {
    unsafe {
        timeEndPeriod(1);
    }
}

#[cfg(not(windows))]
fn release_timer_resolution() {}

fn start_feed_clock_thread(runtime: Arc<FeedRuntime>, input_rx: mpsc::Receiver<PcmChunk>) {
    let feed_id = runtime.feed_id.clone();
    if let Err(err) = thread::Builder::new()
        .name(feed_clock_thread_name(&feed_id))
        .spawn(move || feed_clock_thread(runtime, input_rx))
    {
        warn!("failed to start haze-media feed clock thread for {feed_id}: {err}");
    }
}

fn feed_clock_thread_name(feed_id: &str) -> String {
    format!("haze-media-clock-{}", feed_id.trim())
}

fn feed_clock_thread(runtime: Arc<FeedRuntime>, mut input_rx: mpsc::Receiver<PcmChunk>) {
    let mut next_tick = Instant::now();
    let mut samples = VecDeque::<i16>::with_capacity(FRAME_SAMPLES * 12);
    let mut frame = Vec::with_capacity(FRAME_BYTES);
    let mut last_real_frame = vec![0i16; FRAME_SAMPLES];
    let mut previous_output_frame = Vec::with_capacity(FRAME_BYTES);
    let mut concealment_remaining = 0u8;
    let mut smooth_trim_budget_samples = 0usize;
    let mut primed = false;
    let mut sequence = 0u64;
    let mut last_input_at: Option<Instant> = None;
    let mut last_tick_at: Option<Instant> = None;
    let mut loudness = AudioLoudness::from_env();
    loop {
        let scheduled_tick = next_tick;
        let before_sleep = Instant::now();
        if next_tick > before_sleep {
            thread::sleep(next_tick.duration_since(before_sleep));
        }
        let ticked_at = Instant::now();
        let schedule_lag = ticked_at.saturating_duration_since(scheduled_tick);
        next_tick = if schedule_lag > Duration::from_millis(80) {
            ticked_at
                .checked_add(FRAME_DURATION)
                .unwrap_or_else(Instant::now)
        } else {
            scheduled_tick
                .checked_add(FRAME_DURATION)
                .unwrap_or_else(Instant::now)
        };
        let tick_gap = last_tick_at.map(|previous| ticked_at.duration_since(previous));
        last_tick_at = Some(ticked_at);
        let late_tick = tick_gap
            .map(|gap| gap > FRAME_DURATION + Duration::from_millis(10))
            .unwrap_or(false);
        let catchup_tick = tick_gap
            .map(|gap| gap < FRAME_DURATION.saturating_sub(Duration::from_millis(10)))
            .unwrap_or(false);
        let mut drained = 0usize;
        while drained < INPUT_QUEUE_CAPACITY {
            match input_rx.try_recv() {
                Ok(chunk) => {
                    append_normalized_chunk(&mut samples, chunk);
                    last_input_at = Some(Instant::now());
                    drained += 1;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }
        let (stale_dropped, smooth_trim_budget_hint) = trim_source_queue(&mut samples, primed);
        smooth_trim_budget_samples = smooth_trim_budget_samples.max(smooth_trim_budget_hint);

        frame.clear();
        let mut real = 0usize;
        let mut soft_trimmed = 0u64;
        let mut concealed = false;
        let target_samples = target_source_queue_samples();
        if !primed && samples.len() >= target_samples {
            primed = true;
            concealment_remaining = CONCEALMENT_FRAMES;
        }
        if primed && samples.len() >= FRAME_SAMPLES {
            let correction_excess = samples
                .len()
                .saturating_sub(target_source_queue_samples() + FRAME_SAMPLES);
            let trim_now = smooth_trim_budget_samples
                .min(source_drift_trim_samples(correction_excess))
                .min(samples.len().saturating_sub(FRAME_SAMPLES));
            if trim_now > 0 {
                render_exact_frame_with_trim(
                    &mut samples,
                    &mut frame,
                    &mut last_real_frame,
                    trim_now,
                );
                smooth_trim_budget_samples = smooth_trim_budget_samples.saturating_sub(trim_now);
                soft_trimmed = trim_now as u64;
            } else {
                render_exact_frame(&mut samples, &mut frame, &mut last_real_frame);
            }
            real = FRAME_SAMPLES;
            concealment_remaining = CONCEALMENT_FRAMES;
        } else if primed && concealment_remaining > 0 {
            render_concealment_frame(&last_real_frame, concealment_remaining, &mut frame);
            concealment_remaining = concealment_remaining.saturating_sub(1);
            concealed = true;
        } else {
            for _ in 0..FRAME_SAMPLES {
                push_i16(&mut frame, 0);
            }
            primed = false;
        }
        loudness.process(&mut frame);
        sequence = sequence.saturating_add(1);
        let signal = analyze_pcm_frame(&frame, &previous_output_frame);
        previous_output_frame.clear();
        previous_output_frame.extend_from_slice(&frame);
        let data: Arc<[u8]> = Arc::from(frame.as_slice());
        let _ = runtime.frame_tx.send(PacedFrame {
            _sequence: sequence,
            data,
        });
        let now = Instant::now();
        if let Ok(mut stats) = runtime.stats.lock() {
            stats.frames = stats.frames.saturating_add(1);
            if late_tick {
                stats.late_ticks = stats.late_ticks.saturating_add(1);
            }
            if catchup_tick {
                stats.catchup_ticks = stats.catchup_ticks.saturating_add(1);
            }
            if real == FRAME_SAMPLES {
                stats.real_frames = stats.real_frames.saturating_add(1);
            } else if concealed {
                stats.concealed_frames = stats.concealed_frames.saturating_add(1);
            } else if real == 0 {
                stats.silence_frames = stats.silence_frames.saturating_add(1);
            } else {
                stats.partial_frames = stats.partial_frames.saturating_add(1);
            }
            stats.stale_samples_dropped = stats.stale_samples_dropped.saturating_add(stale_dropped);
            stats.source_drift_samples_trimmed = stats
                .source_drift_samples_trimmed
                .saturating_add(soft_trimmed);
            if signal.near_silent {
                stats.near_silent_frames = stats.near_silent_frames.saturating_add(1);
                stats.consecutive_near_silent_frames =
                    stats.consecutive_near_silent_frames.saturating_add(1);
            } else {
                stats.consecutive_near_silent_frames = 0;
            }
            if signal.repeated {
                stats.repeated_frames = stats.repeated_frames.saturating_add(1);
                stats.consecutive_repeated_frames =
                    stats.consecutive_repeated_frames.saturating_add(1);
                if !signal.near_silent {
                    stats.repeated_non_silent_frames =
                        stats.repeated_non_silent_frames.saturating_add(1);
                }
            } else {
                stats.consecutive_repeated_frames = 0;
            }
            stats.clipped_samples = stats.clipped_samples.saturating_add(signal.clipped_samples);
            stats.last_peak = signal.peak;
            stats.last_rms_dbfs = signal.rms_dbfs;
            stats.last_max_sample_jump = signal.max_sample_jump;
            stats.queued_samples = samples.len();
            stats.last_tick_gap_ms = tick_gap.map(|gap| gap.as_millis());
            if let Some(gap) = tick_gap {
                stats.max_tick_gap_ms = stats.max_tick_gap_ms.max(gap.as_millis());
            }
            stats.last_input_age_ms =
                last_input_at.map(|instant| now.duration_since(instant).as_millis());
            stats.last_frame_age_ms = Some(0);
        }
    }
}

fn target_source_queue_samples() -> usize {
    (SAMPLE_RATE as u64 * TARGET_SOURCE_QUEUE_MS / 1_000) as usize
}

fn max_source_queue_samples() -> usize {
    (SAMPLE_RATE as u64 * MAX_SOURCE_QUEUE_MS / 1_000) as usize
}

fn soft_source_queue_samples() -> usize {
    let target = target_source_queue_samples();
    let max = max_source_queue_samples();
    let soft = (SAMPLE_RATE as u64 * SOFT_SOURCE_QUEUE_MS / 1_000) as usize;
    soft.clamp(target + FRAME_SAMPLES, max)
}

fn max_source_drift_trim_samples() -> usize {
    ((SAMPLE_RATE as u64 * SOURCE_DRIFT_TRIM_MS / 1_000) as usize).max(1)
}

fn source_drift_trim_samples(excess_samples: usize) -> usize {
    if excess_samples == 0 {
        return 0;
    }
    (excess_samples / 20)
        .clamp(
            MIN_SOURCE_DRIFT_TRIM_SAMPLES,
            max_source_drift_trim_samples(),
        )
        .min(excess_samples)
}

fn trim_source_queue(samples: &mut VecDeque<i16>, primed: bool) -> (u64, usize) {
    let max_samples = max_source_queue_samples();
    let mut stale_dropped = 0u64;
    while samples.len() > max_samples {
        samples.pop_front();
        stale_dropped = stale_dropped.saturating_add(1);
    }

    let mut smooth_trim_budget_hint = 0usize;
    if primed && samples.len() > soft_source_queue_samples() {
        let target_samples = target_source_queue_samples();
        let trim_samples = samples
            .len()
            .saturating_sub(target_samples)
            .min(max_source_queue_samples());
        smooth_trim_budget_hint = trim_samples;
    }
    (stale_dropped, smooth_trim_budget_hint)
}

fn render_exact_frame(
    samples: &mut VecDeque<i16>,
    frame: &mut Vec<u8>,
    last_real_frame: &mut [i16],
) {
    for output_index in 0..FRAME_SAMPLES {
        let sample = samples.pop_front().unwrap_or_default();
        if let Some(slot) = last_real_frame.get_mut(output_index) {
            *slot = sample;
        }
        push_i16(frame, sample);
    }
}

fn render_exact_frame_with_trim(
    samples: &mut VecDeque<i16>,
    frame: &mut Vec<u8>,
    last_real_frame: &mut [i16],
    trim_samples: usize,
) {
    let source_samples = FRAME_SAMPLES.saturating_add(trim_samples);
    let mut dropped = 0usize;
    let mut drop_accumulator = 0usize;
    let mut output_index = 0usize;
    for _ in 0..source_samples {
        let sample = samples.pop_front().unwrap_or_default();
        if dropped < trim_samples {
            drop_accumulator = drop_accumulator.saturating_add(trim_samples);
            if drop_accumulator >= source_samples {
                drop_accumulator -= source_samples;
                dropped += 1;
                continue;
            }
        }
        if output_index < FRAME_SAMPLES {
            if let Some(slot) = last_real_frame.get_mut(output_index) {
                *slot = sample;
            }
            push_i16(frame, sample);
            output_index += 1;
        }
    }
    while output_index < FRAME_SAMPLES {
        if let Some(slot) = last_real_frame.get_mut(output_index) {
            *slot = 0;
        }
        push_i16(frame, 0);
        output_index += 1;
    }
}

fn render_concealment_frame(last_real_frame: &[i16], remaining: u8, frame: &mut Vec<u8>) {
    let denominator = f32::from(CONCEALMENT_FRAMES) + 1.0;
    let gain = (f32::from(remaining) / denominator).clamp(0.0, 1.0);
    for sample in last_real_frame.iter().take(FRAME_SAMPLES) {
        push_i16(frame, (f32::from(*sample) * gain).round() as i16);
    }
    let missing = FRAME_SAMPLES.saturating_sub(last_real_frame.len().min(FRAME_SAMPLES));
    for _ in 0..missing {
        push_i16(frame, 0);
    }
}

#[derive(Debug, Clone, Copy)]
struct FrameSignal {
    peak: u16,
    rms_dbfs: f64,
    max_sample_jump: u16,
    clipped_samples: u64,
    near_silent: bool,
    repeated: bool,
}

fn analyze_pcm_frame(frame: &[u8], previous: &[u8]) -> FrameSignal {
    let mut peak = 0u16;
    let mut max_sample_jump = 0u16;
    let mut clipped_samples = 0u64;
    let mut sum_squares = 0f64;
    let mut count = 0usize;
    let mut previous_sample: Option<i16> = None;

    for sample_bytes in frame.chunks_exact(2) {
        let sample = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
        let abs = sample.unsigned_abs();
        peak = peak.max(abs);
        if abs >= CLIP_SAMPLE_PEAK {
            clipped_samples = clipped_samples.saturating_add(1);
        }
        if let Some(previous_sample) = previous_sample {
            let jump = i32::from(sample)
                .saturating_sub(i32::from(previous_sample))
                .unsigned_abs()
                .min(u32::from(u16::MAX)) as u16;
            max_sample_jump = max_sample_jump.max(jump);
        }
        previous_sample = Some(sample);
        let sample_f64 = f64::from(sample);
        sum_squares += sample_f64 * sample_f64;
        count += 1;
    }

    let rms = if count == 0 {
        0.0
    } else {
        (sum_squares / count as f64).sqrt()
    };
    let rms_dbfs = if rms <= f64::EPSILON {
        -120.0
    } else {
        20.0 * (rms / 32768.0).log10()
    };
    FrameSignal {
        peak,
        rms_dbfs,
        max_sample_jump,
        clipped_samples,
        near_silent: peak <= NEAR_SILENT_PEAK,
        repeated: !previous.is_empty() && previous == frame,
    }
}

fn classify_audio_health(stats: &FeedStats) -> (bool, Vec<String>) {
    let mut warnings = Vec::new();
    if stats.dropped_input_chunks > 0 {
        warnings.push(format!(
            "dropped_input_chunks={}",
            stats.dropped_input_chunks
        ));
    }
    if stats.stale_samples_dropped > 0 {
        warnings.push(format!(
            "stale_samples_dropped={}",
            stats.stale_samples_dropped
        ));
    }
    if stats.concealed_frames > 0 {
        warnings.push(format!("concealed_frames={}", stats.concealed_frames));
    }
    if stats.partial_frames > 0 {
        warnings.push(format!("partial_frames={}", stats.partial_frames));
    }
    if stats.repeated_non_silent_frames > 0 {
        warnings.push(format!(
            "repeated_non_silent_frames={}",
            stats.repeated_non_silent_frames
        ));
    }
    if stats.clipped_samples > 0 {
        warnings.push(format!("clipped_samples={}", stats.clipped_samples));
    }
    if stats.late_ticks > 0 {
        warnings.push(format!("late_ticks={}", stats.late_ticks));
    }
    if stats.catchup_ticks > 0 {
        warnings.push(format!("catchup_ticks={}", stats.catchup_ticks));
    }
    if stats.max_tick_gap_ms > AUDIO_HEALTH_TICK_GAP_WARN_MS {
        warnings.push(format!("max_tick_gap_ms={}", stats.max_tick_gap_ms));
    }
    if stats.frames > 50 && stats.queued_samples < FRAME_SAMPLES {
        warnings.push(format!("source_queue_low_samples={}", stats.queued_samples));
    }
    if stats.frames > 50 {
        match stats.last_input_age_ms {
            Some(age) if age > AUDIO_HEALTH_INPUT_STALE_MS => {
                warnings.push(format!("last_input_age_ms={age}"));
            }
            None => warnings.push("no_input_seen".to_string()),
            _ => {}
        }
    }
    (warnings.is_empty(), warnings)
}

fn append_normalized_chunk(samples: &mut VecDeque<i16>, chunk: PcmChunk) {
    let pcm = normalize_pcm(
        Pcm {
            sample_rate: chunk.sample_rate,
            channels: chunk.channels,
            data: chunk.data,
        },
        SAMPLE_RATE,
        CHANNELS,
    );
    samples.extend(pcm16_samples(&pcm.data));
}

async fn run_status_bridge_loop(addr: String, state: MediaState) {
    if addr.trim().is_empty() {
        warn!(
            "haze-media event bridge address is empty; media status events will not be published"
        );
        return;
    }
    loop {
        match TcpStream::connect(addr.trim()).await {
            Ok(stream) => {
                info!(
                    "haze-media connected to host event bridge at {}",
                    addr.trim()
                );
                if let Err(err) = run_status_bridge_connection(stream, &state).await {
                    warn!("haze-media host event bridge disconnected: {err:#}");
                }
            }
            Err(err) => {
                warn!(
                    "haze-media waiting for host event bridge at {}: {err}",
                    addr.trim()
                );
            }
        }
        sleep(Duration::from_secs(1)).await;
    }
}

async fn run_status_bridge_connection(stream: TcpStream, state: &MediaState) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        while matches!(lines.next_line().await, Ok(Some(_))) {}
    });
    write_bridge_event(
        &mut writer,
        json!({
            "type": "bridge.client",
            "source": SOURCE_ID,
            "data": { "receive_events": false },
        }),
    )
    .await?;
    write_bridge_event(
        &mut writer,
        json!({
            "type": "service.ready",
            "source": SOURCE_ID,
            "data": {
                "service": SOURCE_ID,
                "backend": state.backend.as_str(),
                "gstreamer_available": state.gstreamer_available,
            },
        }),
    )
    .await?;

    let mut status = interval(STATUS_INTERVAL);
    status.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        status.tick().await;
        publish_status(&mut writer, state).await?;
    }
}

async fn run_pcm_bridge_loop(addr: String, state: MediaState) {
    if addr.trim().is_empty() {
        warn!("haze-media media bridge address is empty; media service will only serve silence");
        return;
    }
    loop {
        match TcpStream::connect(addr.trim()).await {
            Ok(stream) => {
                info!(
                    "haze-media connected to host media bridge at {}",
                    addr.trim()
                );
                if let Err(err) = run_pcm_bridge_connection(stream, &state).await {
                    warn!("haze-media host media bridge disconnected: {err:#}");
                }
            }
            Err(err) => {
                warn!(
                    "haze-media waiting for host media bridge at {}: {err}",
                    addr.trim()
                );
            }
        }
        sleep(Duration::from_secs(1)).await;
    }
}

async fn run_pcm_bridge_connection(stream: TcpStream, state: &MediaState) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    write_bridge_event(
        &mut writer,
        json!({
            "type": "bridge.client",
            "source": SOURCE_ID,
            "data": { "receive_events": true },
        }),
    )
    .await?;

    let mut lines = BufReader::new(reader).lines();
    loop {
        let Some(line) = lines
            .next_line()
            .await
            .context("failed to read host media bridge line")?
        else {
            bail!("host media bridge closed");
        };
        if let Some(chunk) = decode_pcm_event(line.as_bytes()) {
            state.publish_pcm(chunk);
        }
    }
}

async fn publish_status<W>(writer: &mut W, state: &MediaState) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    write_bridge_event(
        writer,
        json!({
            "type": "media.status",
            "source": SOURCE_ID,
            "data": state.health(),
        }),
    )
    .await?;
    for feed in state.snapshots() {
        write_bridge_event(
            writer,
            json!({
                "type": "media.feed.status",
                "source": SOURCE_ID,
                "feed_id": feed.feed_id,
                "data": feed,
            }),
        )
        .await?;
    }
    Ok(())
}

async fn write_bridge_event<W>(writer: &mut W, mut value: Value) -> Result<()>
where
    W: AsyncWrite + Unpin,
{
    if value.get("timestamp").is_none() {
        value["timestamp"] = json!(chrono::Utc::now().to_rfc3339());
    }
    let mut raw = serde_json::to_vec(&value)?;
    raw.push(b'\n');
    writer
        .write_all(&raw)
        .await
        .context("failed to write host bridge event")
}

fn decode_pcm_event(raw: &[u8]) -> Option<PcmChunk> {
    #[derive(Deserialize)]
    struct Event {
        #[serde(default)]
        r#type: String,
        #[serde(default)]
        feed_id: String,
        #[serde(default)]
        data: EventData,
    }
    #[derive(Default, Deserialize)]
    struct EventData {
        #[serde(default)]
        feed_id: String,
        #[serde(default)]
        sample_rate: u32,
        #[serde(default)]
        channels: u16,
        #[serde(default)]
        pcm: String,
    }
    let event = serde_json::from_slice::<Event>(raw).ok()?;
    if event.r#type != "playout.pcm" {
        return None;
    }
    let feed_id = first_non_blank(&[
        Some(event.data.feed_id.as_str()),
        Some(event.feed_id.as_str()),
    ]);
    if feed_id.is_empty() || event.data.pcm.trim().is_empty() {
        return None;
    }
    let data = base64::engine::general_purpose::STANDARD
        .decode(event.data.pcm.trim())
        .ok()?;
    let sample_rate = event.data.sample_rate.max(8_000);
    let channels = event.data.channels.max(1);
    let frame_bytes = AudioFormat::new(sample_rate, channels).frame_bytes();
    if frame_bytes == 0 || data.len() < frame_bytes {
        return None;
    }
    let aligned = data.len() - data.len() % frame_bytes;
    Some(PcmChunk {
        feed_id,
        sample_rate,
        channels,
        data: data[..aligned].to_vec(),
    })
}

async fn run_http_server(addr: &str, state: MediaState) -> Result<()> {
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind haze-media HTTP listener at {addr}"))?;
    loop {
        let (stream, peer) = listener.accept().await?;
        if let Err(err) = stream.set_nodelay(true) {
            debug!(%peer, "failed to set TCP_NODELAY for haze-media HTTP connection: {err}");
        }
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(err) = handle_http_connection(stream, state).await {
                debug!(%peer, "haze-media HTTP connection ended: {err:#}");
            }
        });
    }
}

async fn handle_http_connection(mut stream: TcpStream, state: MediaState) -> Result<()> {
    let request = read_http_request(&mut stream).await?;
    match request.path.as_str() {
        "/health" | "/api/v1/health" => {
            write_json_response(&mut stream, 200, state.health()).await?;
        }
        "/api/v1/feed/audio" => {
            let feed_id = query_value(&request.query, "feed").unwrap_or_default();
            let codec = query_value(&request.query, "codec").unwrap_or_else(|| "pcm16".to_string());
            if !valid_feed_id(&feed_id) {
                write_text_response(&mut stream, 400, "feed is required\n").await?;
                return Ok(());
            }
            stream_feed_audio(&mut stream, &state, &feed_id, &codec).await?;
        }
        "/api/v1/webrtc/offer" => {
            handle_webrtc_offer(&mut stream, state, request).await?;
        }
        _ => {
            write_text_response(&mut stream, 404, "not found\n").await?;
        }
    }
    Ok(())
}

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    query: String,
    body: Vec<u8>,
}

async fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest> {
    let mut buffer = Vec::with_capacity(1024);
    let mut chunk = [0u8; 1024];
    let header_end = loop {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            bail!("connection closed before request");
        }
        buffer.extend_from_slice(&chunk[..read]);
        if let Some(pos) = buffer.windows(4).position(|window| window == b"\r\n\r\n") {
            break pos + 4;
        }
        if buffer.len() > HTTP_HEADER_LIMIT {
            bail!("HTTP header too large");
        }
    };
    let headers = String::from_utf8_lossy(&buffer[..header_end]);
    let mut header_lines = headers.lines();
    let first_line = header_lines.next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("/");
    if method != "GET" && method != "HEAD" && method != "POST" {
        bail!("unsupported HTTP method {method}");
    }
    let mut content_length = 0usize;
    for line in header_lines {
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        if name.trim().eq_ignore_ascii_case("content-length") {
            content_length = value
                .trim()
                .parse::<usize>()
                .context("invalid HTTP Content-Length")?;
            if content_length > HTTP_BODY_LIMIT {
                bail!("HTTP body too large");
            }
        }
    }
    let mut body = buffer[header_end..].to_vec();
    while body.len() < content_length {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            bail!("connection closed before HTTP body");
        }
        body.extend_from_slice(&chunk[..read]);
        if body.len() > HTTP_BODY_LIMIT {
            bail!("HTTP body too large");
        }
    }
    body.truncate(content_length);
    let (path, query) = target
        .split_once('?')
        .map(|(path, query)| (path.to_string(), query.to_string()))
        .unwrap_or_else(|| (target.to_string(), String::new()));
    Ok(HttpRequest {
        method: method.to_string(),
        path,
        query,
        body,
    })
}

async fn stream_feed_audio(
    stream: &mut TcpStream,
    state: &MediaState,
    feed_id: &str,
    codec: &str,
) -> Result<()> {
    let format = HttpAudioFormat::parse(codec);
    let feed = state.feed(feed_id);
    let client = state.register_http_client(feed_id, format.id);
    let client_id = client.id();
    match format.kind {
        HttpAudioKind::Wav => stream_wav(stream, state, client_id, feed).await,
        HttpAudioKind::Raw => stream_raw_pcm(stream, state, client_id, feed).await,
        HttpAudioKind::GStreamer => {
            if !state.gstreamer_available {
                write_text_response(stream, 503, "gstreamer backend is not available\n").await?;
                return Ok(());
            }
            stream_gstreamer_encoded(stream, state, client_id, feed, format).await
        }
        HttpAudioKind::Unsupported => {
            write_text_response(stream, 415, "unsupported media codec\n").await?;
            Ok(())
        }
    }
}

#[derive(Debug, Deserialize)]
struct WebRTCOfferRequest {
    #[serde(default)]
    feed_id: String,
    #[serde(default)]
    sdp: String,
    #[serde(default)]
    _sdp_type: String,
    #[serde(default)]
    preferred_codec: String,
    #[serde(default)]
    codec: String,
    #[serde(default)]
    require_opus: bool,
    #[serde(default)]
    disable_g722: bool,
}

async fn handle_webrtc_offer(
    stream: &mut TcpStream,
    state: MediaState,
    request: HttpRequest,
) -> Result<()> {
    if request.method != "POST" {
        write_text_response(stream, 405, "method not allowed\n").await?;
        return Ok(());
    }
    if !state.gstreamer_available || !state.webrtc_available {
        write_json_response(
            stream,
            503,
            json!({
                "error": "gstreamer webrtc backend is unavailable",
            }),
        )
        .await?;
        return Ok(());
    }
    let payload = match serde_json::from_slice::<WebRTCOfferRequest>(&request.body) {
        Ok(payload) => payload,
        Err(err) => {
            write_json_response(
                stream,
                400,
                json!({"error": format!("invalid WebRTC offer JSON: {err}")}),
            )
            .await?;
            return Ok(());
        }
    };
    let feed_id = payload.feed_id.trim().to_string();
    if !valid_feed_id(&feed_id) {
        write_json_response(stream, 400, json!({"error": "feed_id is required"})).await?;
        return Ok(());
    }
    let offer_sdp = payload.sdp.trim().to_string();
    if offer_sdp.is_empty() || offer_sdp.len() > HTTP_BODY_LIMIT {
        write_json_response(stream, 400, json!({"error": "sdp is required"})).await?;
        return Ok(());
    }
    let preferred_codec = first_non_blank(&[
        Some(payload.preferred_codec.as_str()),
        Some(payload.codec.as_str()),
    ]);
    let Some(selection) = select_webrtc_audio_codec(
        &offer_sdp,
        &state.webrtc_codecs,
        &preferred_codec,
        payload.require_opus,
        payload.disable_g722,
    ) else {
        write_json_response(
            stream,
            415,
            json!({
                "error": "the WebRTC offer did not include a codec supported by haze-media",
                "supported_codecs": state.webrtc_codecs.iter().map(|codec| codec.id()).collect::<Vec<_>>(),
            }),
        )
        .await?;
        return Ok(());
    };

    let feed = state.feed(&feed_id);
    let media_recent = feed
        .snapshot()
        .last_input_age_ms
        .map(|age| age <= 5_000)
        .unwrap_or(false);
    let setup = build_gstreamer_webrtc_peer(&offer_sdp, selection).await;
    let peer = match setup {
        Ok(peer) => peer,
        Err(err) => {
            warn!(feed_id, "failed to create GStreamer WebRTC peer: {err:#}");
            write_json_response(
                stream,
                503,
                json!({"error": format!("failed to create GStreamer WebRTC peer: {err:#}")}),
            )
            .await?;
            return Ok(());
        }
    };
    let answer_sdp = peer.answer_sdp.clone();
    write_json_response(
        stream,
        200,
        json!({
            "feed_id": feed_id,
            "sdp": answer_sdp,
            "sdp_type": "answer",
            "codec": selection.codec.id(),
            "payload_type": selection.payload_type,
            "media_recent": media_recent,
        }),
    )
    .await?;
    let peer_id = state.register_webrtc_peer(&feed_id, selection.codec.id());
    start_webrtc_peer_feeder(state.clone(), peer_id, feed, peer);
    Ok(())
}

#[cfg(feature = "gstreamer-backend")]
struct GStreamerWebRTCPeer {
    pipeline: gstreamer::Pipeline,
    appsrc: gstreamer_app::AppSrc,
    answer_sdp: String,
    closed_rx: std_mpsc::Receiver<()>,
}

#[cfg(not(feature = "gstreamer-backend"))]
struct GStreamerWebRTCPeer {
    answer_sdp: String,
}

#[cfg(feature = "gstreamer-backend")]
async fn build_gstreamer_webrtc_peer(
    offer_sdp: &str,
    selection: WebRTCCodecSelection,
) -> Result<GStreamerWebRTCPeer> {
    let offer_sdp = offer_sdp.to_string();
    tokio::task::spawn_blocking(move || build_gstreamer_webrtc_peer_sync(&offer_sdp, selection))
        .await
        .context("GStreamer WebRTC setup task panicked")?
}

#[cfg(not(feature = "gstreamer-backend"))]
async fn build_gstreamer_webrtc_peer(
    _offer_sdp: &str,
    _selection: WebRTCCodecSelection,
) -> Result<GStreamerWebRTCPeer> {
    bail!("haze-media was built without GStreamer support")
}

#[cfg(feature = "gstreamer-backend")]
fn build_gstreamer_webrtc_peer_sync(
    offer_sdp: &str,
    selection: WebRTCCodecSelection,
) -> Result<GStreamerWebRTCPeer> {
    use gst::prelude::*;
    use gstreamer as gst;
    use gstreamer_app as gst_app;
    use gstreamer_sdp as gst_sdp;
    use gstreamer_webrtc as gst_webrtc;

    let pipeline_description = gstreamer_webrtc_pipeline(selection);
    let element = gst::parse::launch(&pipeline_description)
        .context("failed to build GStreamer WebRTC pipeline")?;
    let pipeline = element
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow::anyhow!("GStreamer WebRTC description did not produce a pipeline"))?;
    let appsrc = pipeline
        .by_name("src")
        .context("GStreamer WebRTC pipeline is missing appsrc")?
        .downcast::<gst_app::AppSrc>()
        .map_err(|_| anyhow::anyhow!("GStreamer WebRTC src is not appsrc"))?;
    let webrtc = pipeline
        .by_name("webrtc")
        .context("GStreamer WebRTC pipeline is missing webrtcbin")?;
    let (closed_tx, closed_rx) = std_mpsc::channel();
    let closed_tx_notify = closed_tx.clone();
    webrtc.connect_notify(Some("connection-state"), move |element, _| {
        let state = element.property::<gst_webrtc::WebRTCPeerConnectionState>("connection-state");
        if matches!(
            state,
            gst_webrtc::WebRTCPeerConnectionState::Failed
                | gst_webrtc::WebRTCPeerConnectionState::Closed
                | gst_webrtc::WebRTCPeerConnectionState::Disconnected
        ) {
            let _ = closed_tx_notify.send(());
        }
    });
    let closed_tx_ice = closed_tx.clone();
    webrtc.connect_notify(Some("ice-connection-state"), move |element, _| {
        let state =
            element.property::<gst_webrtc::WebRTCICEConnectionState>("ice-connection-state");
        if matches!(
            state,
            gst_webrtc::WebRTCICEConnectionState::Failed
                | gst_webrtc::WebRTCICEConnectionState::Closed
                | gst_webrtc::WebRTCICEConnectionState::Disconnected
        ) {
            let _ = closed_tx_ice.send(());
        }
    });

    pipeline
        .set_state(gst::State::Playing)
        .map_err(|err| anyhow::anyhow!("failed to start GStreamer WebRTC pipeline: {err:?}"))?;

    let sdp = gst_sdp::SDPMessage::parse_buffer(offer_sdp.as_bytes())
        .context("failed to parse WebRTC offer SDP")?;
    let offer = gst_webrtc::WebRTCSessionDescription::new(gst_webrtc::WebRTCSDPType::Offer, sdp);
    webrtc.emit_by_name::<()>("set-remote-description", &[&offer, &None::<gst::Promise>]);

    let (answer_tx, answer_rx) = std::sync::mpsc::channel();
    let promise = gst::Promise::with_change_func(move |reply| {
        let result = reply
            .map_err(|err| format!("create-answer promise failed: {err:?}"))
            .and_then(|reply| reply.ok_or_else(|| "create-answer returned no reply".to_string()))
            .and_then(|reply| {
                reply
                    .get::<gst_webrtc::WebRTCSessionDescription>("answer")
                    .map_err(|err| format!("create-answer did not include an answer: {err}"))
            });
        let _ = answer_tx.send(result);
    });
    webrtc.emit_by_name::<()>("create-answer", &[&None::<gst::Structure>, &promise]);
    let answer = answer_rx
        .recv_timeout(WEBRTC_OFFER_TIMEOUT)
        .context("timed out waiting for GStreamer WebRTC answer")?
        .map_err(|err| anyhow::anyhow!(err))?;

    webrtc.emit_by_name::<()>("set-local-description", &[&answer, &None::<gst::Promise>]);
    std::thread::sleep(WEBRTC_ICE_GATHER_WINDOW);
    let final_answer = webrtc
        .property::<Option<gst_webrtc::WebRTCSessionDescription>>("local-description")
        .unwrap_or(answer);
    let answer_sdp = final_answer
        .sdp()
        .as_text()
        .context("failed to serialize WebRTC answer SDP")?;
    Ok(GStreamerWebRTCPeer {
        pipeline,
        appsrc,
        answer_sdp,
        closed_rx,
    })
}

#[cfg(feature = "gstreamer-backend")]
fn gstreamer_webrtc_pipeline(selection: WebRTCCodecSelection) -> String {
    let payload_type = selection.payload_type;
    let source = "appsrc name=src is-live=true block=true do-timestamp=false max-bytes=96000 format=time stream-type=stream \
         caps=audio/x-raw,format=S16LE,layout=interleaved,rate=48000,channels=1 \
         ! queue max-size-time=800000000 max-size-buffers=50 max-size-bytes=0 \
         ! audioconvert ! audioresample quality=4";
    let caps = selection.codec.webrtc_caps(payload_type);
    match selection.codec {
        WebRTCAudioCodec::Opus => format!(
            "{source} \
             ! opusenc bitrate=96000 bitrate-type=cbr frame-size=20 audio-type=generic dtx=false perfect-timestamp=true inband-fec=true \
             ! rtpopuspay pt={payload_type} \
             ! {caps} \
             ! webrtcbin name=webrtc bundle-policy=max-bundle"
        ),
        WebRTCAudioCodec::G722 => format!(
            "{source} \
             ! audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1 \
             ! avenc_g722 \
             ! rtpg722pay pt={payload_type} \
             ! {caps} \
             ! webrtcbin name=webrtc bundle-policy=max-bundle"
        ),
        WebRTCAudioCodec::Pcmu => format!(
            "{source} \
             ! audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1 \
             ! mulawenc \
             ! rtppcmupay pt={payload_type} \
             ! {caps} \
             ! webrtcbin name=webrtc bundle-policy=max-bundle"
        ),
        WebRTCAudioCodec::Pcma => format!(
            "{source} \
             ! audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1 \
             ! alawenc \
             ! rtppcmapay pt={payload_type} \
             ! {caps} \
             ! webrtcbin name=webrtc bundle-policy=max-bundle"
        ),
    }
}

#[cfg(feature = "gstreamer-backend")]
fn next_gst_audio_pts(
    next_pts_ns: &mut Option<u64>,
    running_time_ns: Option<u64>,
    frame_duration_ns: u64,
) -> u64 {
    let pts_ns = next_pts_ns.unwrap_or_else(|| {
        running_time_ns
            .map(|time| time.saturating_add(frame_duration_ns))
            .unwrap_or(frame_duration_ns)
    });
    *next_pts_ns = Some(pts_ns.saturating_add(frame_duration_ns));
    pts_ns
}

#[cfg(feature = "gstreamer-backend")]
fn build_gst_audio_buffer(
    data: &[u8],
    pts_ns: u64,
    frame_duration_ns: u64,
) -> Result<gstreamer::Buffer> {
    use gstreamer as gst;

    let mut buffer =
        gst::Buffer::with_size(data.len()).context("failed to allocate GStreamer audio buffer")?;
    {
        let buffer_mut = buffer
            .get_mut()
            .context("new GStreamer audio buffer is shared")?;
        {
            let mut map = buffer_mut
                .map_writable()
                .context("failed to map GStreamer audio buffer")?;
            map.as_mut_slice().copy_from_slice(data);
        }
        buffer_mut.set_duration(gst::ClockTime::from_nseconds(frame_duration_ns));
        buffer_mut.set_pts(gst::ClockTime::from_nseconds(pts_ns));
    }
    Ok(buffer)
}

#[cfg(feature = "gstreamer-backend")]
fn start_webrtc_peer_feeder(
    state: MediaState,
    peer_id: u64,
    feed: Arc<FeedRuntime>,
    peer: GStreamerWebRTCPeer,
) {
    use gstreamer::prelude::*;

    tokio::spawn(async move {
        let frame_duration_ns = FRAME_DURATION.as_nanos() as u64;
        let mut next_pts_ns: Option<u64> = None;
        let silence = vec![0u8; FRAME_BYTES];
        sleep(Duration::from_millis(800)).await;
        let mut rx = feed.subscribe();
        'feed: loop {
            if peer.closed_rx.try_recv().is_ok() {
                break;
            }
            match rx.recv().await {
                Ok(frame) => {
                    let pts_ns = next_gst_audio_pts(
                        &mut next_pts_ns,
                        peer.pipeline
                            .current_running_time()
                            .map(|time| time.nseconds()),
                        frame_duration_ns,
                    );
                    let buffer =
                        match build_gst_audio_buffer(&frame.data, pts_ns, frame_duration_ns) {
                            Ok(buffer) => buffer,
                            Err(err) => {
                                warn!("failed to build WebRTC audio buffer: {err:#}");
                                break;
                            }
                        };
                    if peer.appsrc.push_buffer(buffer).is_err() {
                        break;
                    }
                    state.record_webrtc_peer_push(peer_id, frame.data.len());
                }
                Err(broadcast::error::RecvError::Lagged(skipped)) => {
                    if let Some(pts_ns) = &mut next_pts_ns {
                        *pts_ns =
                            pts_ns.saturating_add(frame_duration_ns.saturating_mul(skipped.max(1)));
                    }
                    for _ in 0..skipped {
                        state.record_webrtc_peer_drop(peer_id);
                    }
                    let fill_frames = skipped.min(RAW_HTTP_LAG_RESYNC_SILENCE_FRAMES);
                    for _ in 0..fill_frames {
                        let pts_ns = next_gst_audio_pts(
                            &mut next_pts_ns,
                            peer.pipeline
                                .current_running_time()
                                .map(|time| time.nseconds()),
                            frame_duration_ns,
                        );
                        let buffer =
                            match build_gst_audio_buffer(&silence, pts_ns, frame_duration_ns) {
                                Ok(buffer) => buffer,
                                Err(err) => {
                                    warn!("failed to build WebRTC silence buffer: {err:#}");
                                    break 'feed;
                                }
                            };
                        if peer.appsrc.push_buffer(buffer).is_err() {
                            break 'feed;
                        }
                        state.record_webrtc_peer_push(peer_id, silence.len());
                    }
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
        let _ = peer.appsrc.end_of_stream();
        let _ = peer.pipeline.set_state(gstreamer::State::Null);
        state.unregister_webrtc_peer(peer_id);
    });
}

#[cfg(not(feature = "gstreamer-backend"))]
fn start_webrtc_peer_feeder(
    _state: MediaState,
    _peer_id: u64,
    _feed: Arc<FeedRuntime>,
    _peer: GStreamerWebRTCPeer,
) {
}

fn select_webrtc_audio_codec(
    sdp: &str,
    available_codecs: &[WebRTCAudioCodec],
    preferred_codec: &str,
    require_opus: bool,
    disable_g722: bool,
) -> Option<WebRTCCodecSelection> {
    let mut candidates = Vec::new();
    if let Some(codec) = parse_webrtc_audio_codec(preferred_codec) {
        candidates.push(codec);
    } else if require_opus {
        candidates.push(WebRTCAudioCodec::Opus);
    } else {
        if let Some(codec) =
            parse_webrtc_audio_codec(&env::var("HAZE_WEBRTC_DEFAULT_CODEC").unwrap_or_default())
        {
            candidates.push(codec);
        } else {
            candidates.push(WebRTCAudioCodec::Opus);
        }
        if !disable_g722 {
            candidates.push(WebRTCAudioCodec::G722);
        }
        candidates.push(WebRTCAudioCodec::Pcmu);
        candidates.push(WebRTCAudioCodec::Pcma);
        candidates.push(WebRTCAudioCodec::Opus);
    }
    for codec in candidates {
        if disable_g722 && codec == WebRTCAudioCodec::G722 {
            continue;
        }
        if !available_codecs.contains(&codec) {
            continue;
        }
        if let Some(payload_type) = offered_audio_payload_type(sdp, codec) {
            return Some(WebRTCCodecSelection {
                codec,
                payload_type,
            });
        }
    }
    None
}

fn parse_webrtc_audio_codec(value: &str) -> Option<WebRTCAudioCodec> {
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['.', '-', '_'], "")
        .as_str()
    {
        "opus" => Some(WebRTCAudioCodec::Opus),
        "g722" => Some(WebRTCAudioCodec::G722),
        "pcmu" | "mulaw" | "ulaw" => Some(WebRTCAudioCodec::Pcmu),
        "pcma" | "alaw" => Some(WebRTCAudioCodec::Pcma),
        _ => None,
    }
}

fn offered_audio_payload_type(sdp: &str, target_codec: WebRTCAudioCodec) -> Option<u8> {
    for line in sdp.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("a=rtpmap:") else {
            continue;
        };
        let Some((payload, codec)) = rest.split_once(' ') else {
            continue;
        };
        if codec
            .to_ascii_uppercase()
            .starts_with(target_codec.rtpmap_match())
        {
            if let Ok(payload_type) = payload.parse::<u8>() {
                return Some(payload_type);
            }
        }
    }
    for payload_type in offered_media_payload_types(sdp) {
        if target_codec.is_static_payload(payload_type) {
            return Some(payload_type);
        }
    }
    None
}

fn offered_media_payload_types(sdp: &str) -> Vec<u8> {
    let mut payload_types = Vec::new();
    for line in sdp.lines() {
        let line = line.trim();
        if !line.to_ascii_lowercase().starts_with("m=audio ") {
            continue;
        }
        for part in line.split_whitespace().skip(3) {
            if let Ok(payload_type) = part.parse::<u8>() {
                payload_types.push(payload_type);
            }
        }
    }
    payload_types
}

#[allow(dead_code)]
fn offered_opus_payload_type(sdp: &str) -> Option<u8> {
    offered_audio_payload_type(sdp, WebRTCAudioCodec::Opus)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum HttpAudioKind {
    Wav,
    Raw,
    GStreamer,
    Unsupported,
}

#[derive(Debug, Clone)]
struct HttpAudioFormat {
    kind: HttpAudioKind,
    #[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
    id: &'static str,
    #[cfg_attr(not(feature = "gstreamer-backend"), allow(dead_code))]
    content_type: &'static str,
}

impl HttpAudioFormat {
    fn parse(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "" | "pcm16" | "wav" | "wav_pcm16" | "pcm_s16le" => Self {
                kind: HttpAudioKind::Wav,
                id: "pcm16",
                content_type: "audio/wav",
            },
            "raw" | "raw_pcm16" | "s16le" => Self {
                kind: HttpAudioKind::Raw,
                id: "raw",
                content_type: "audio/L16; rate=48000; channels=1",
            },
            "opus" | "ogg_opus" | "opus_ogg" => Self {
                kind: HttpAudioKind::GStreamer,
                id: "opus",
                content_type: "audio/ogg; codecs=opus",
            },
            "aac" | "adts" => Self {
                kind: HttpAudioKind::GStreamer,
                id: "aac",
                content_type: "audio/aac",
            },
            _ => Self {
                kind: HttpAudioKind::Unsupported,
                id: "unsupported",
                content_type: "application/octet-stream",
            },
        }
    }
}

async fn stream_wav(
    stream: &mut TcpStream,
    state: &MediaState,
    client_id: u64,
    feed: Arc<FeedRuntime>,
) -> Result<()> {
    write_stream_headers(stream, "audio/wav").await?;
    stream.write_all(&wav_stream_header()).await?;
    state.record_http_client_write(client_id, 44);
    let mut rx = feed.subscribe();
    let silence = vec![0u8; FRAME_BYTES];
    loop {
        match rx.recv().await {
            Ok(frame) => {
                stream.write_all(&frame.data).await?;
                state.record_http_client_write(client_id, frame.data.len());
            }
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                state.record_http_client_lag(client_id, skipped);
                let fill_frames = skipped.min(RAW_HTTP_LAG_RESYNC_SILENCE_FRAMES);
                for _ in 0..fill_frames {
                    stream.write_all(&silence).await?;
                    state.record_http_client_write(client_id, silence.len());
                }
                continue;
            }
            Err(broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

async fn stream_raw_pcm(
    stream: &mut TcpStream,
    state: &MediaState,
    client_id: u64,
    feed: Arc<FeedRuntime>,
) -> Result<()> {
    write_stream_headers(stream, "audio/L16; rate=48000; channels=1").await?;
    let mut rx = feed.subscribe();
    let silence = vec![0u8; FRAME_BYTES];
    loop {
        match rx.recv().await {
            Ok(frame) => {
                stream.write_all(&frame.data).await?;
                state.record_http_client_write(client_id, frame.data.len());
            }
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                state.record_http_client_lag(client_id, skipped);
                let fill_frames = skipped.min(RAW_HTTP_LAG_RESYNC_SILENCE_FRAMES);
                for _ in 0..fill_frames {
                    stream.write_all(&silence).await?;
                    state.record_http_client_write(client_id, silence.len());
                }
                continue;
            }
            Err(broadcast::error::RecvError::Closed) => return Ok(()),
        }
    }
}

#[cfg(feature = "gstreamer-backend")]
async fn stream_gstreamer_encoded(
    stream: &mut TcpStream,
    state: &MediaState,
    client_id: u64,
    feed: Arc<FeedRuntime>,
    format: HttpAudioFormat,
) -> Result<()> {
    use gst::prelude::*;
    use gstreamer as gst;
    use gstreamer_app as gst_app;

    let setup = (|| -> Result<(gst::Pipeline, gst_app::AppSrc, gst_app::AppSink)> {
        let pipeline_description = gstreamer_audio_pipeline(format.id)?;
        let element = gst::parse::launch(&pipeline_description)
            .with_context(|| format!("failed to build {} audio pipeline", format.id))?;
        let pipeline = element.downcast::<gst::Pipeline>().map_err(|_| {
            anyhow::anyhow!("GStreamer audio description did not produce a pipeline")
        })?;
        let appsrc = pipeline
            .by_name("src")
            .context("GStreamer audio pipeline is missing appsrc")?
            .downcast::<gst_app::AppSrc>()
            .map_err(|_| anyhow::anyhow!("GStreamer src is not appsrc"))?;
        let appsink = pipeline
            .by_name("sink")
            .context("GStreamer audio pipeline is missing appsink")?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| anyhow::anyhow!("GStreamer sink is not appsink"))?;
        Ok((pipeline, appsrc, appsink))
    })();
    let (pipeline, appsrc, appsink) = match setup {
        Ok(parts) => parts,
        Err(err) => {
            warn!(
                "failed to prepare GStreamer {} audio stream: {err:#}",
                format.id
            );
            write_text_response(
                stream,
                503,
                &format!("gstreamer audio pipeline is unavailable: {err:#}\n"),
            )
            .await?;
            return Ok(());
        }
    };
    let (encoded_tx, mut encoded_rx) = mpsc::channel::<Vec<u8>>(ENCODED_HTTP_QUEUE_CAPACITY);
    let callback_state = state.clone();
    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let Some(buffer) = sample.buffer() else {
                    return Err(gst::FlowError::Error);
                };
                let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                let data = map.as_slice().to_vec();
                match encoded_tx.try_send(data) {
                    Ok(()) => {}
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        callback_state.record_http_client_encoded_drop(client_id);
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        return Err(gst::FlowError::Eos);
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    if let Err(err) = pipeline.set_state(gst::State::Playing) {
        warn!(
            "failed to start GStreamer {} audio stream: {err:#}",
            format.id
        );
        write_text_response(
            stream,
            503,
            &format!("gstreamer audio pipeline failed to start: {err:#}\n"),
        )
        .await?;
        let _ = pipeline.set_state(gst::State::Null);
        return Ok(());
    }
    write_stream_headers(stream, format.content_type).await?;

    let mut rx = feed.subscribe();
    let frame_duration_ns = FRAME_DURATION.as_nanos() as u64;
    let mut next_pts_ns: Option<u64> = None;
    let silence = vec![0u8; FRAME_BYTES];
    'stream: loop {
        tokio::select! {
            frame = rx.recv() => {
                match frame {
                    Ok(frame) => {
                        let pts_ns = next_gst_audio_pts(
                            &mut next_pts_ns,
                            pipeline.current_running_time().map(|time| time.nseconds()),
                            frame_duration_ns,
                        );
                        let buffer = build_gst_audio_buffer(&frame.data, pts_ns, frame_duration_ns)?;
                        if appsrc.push_buffer(buffer).is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        state.record_http_client_lag(client_id, skipped);
                        let fill_frames = skipped.min(RAW_HTTP_LAG_RESYNC_SILENCE_FRAMES);
                        for _ in 0..fill_frames {
                            let pts_ns = next_gst_audio_pts(
                                &mut next_pts_ns,
                                pipeline.current_running_time().map(|time| time.nseconds()),
                                frame_duration_ns,
                            );
                            let buffer = build_gst_audio_buffer(&silence, pts_ns, frame_duration_ns)?;
                            if appsrc.push_buffer(buffer).is_err() {
                                break 'stream;
                            }
                        }
                        continue;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
            encoded = encoded_rx.recv() => {
                let Some(encoded) = encoded else {
                    break;
                };
                if let Err(err) = stream.write_all(&encoded).await {
                    if err.kind() != ErrorKind::BrokenPipe && err.kind() != ErrorKind::ConnectionReset {
                        warn!("haze-media encoded HTTP write failed: {err}");
                    }
                    break;
                }
                state.record_http_client_write(client_id, encoded.len());
            }
        }
    }
    let _ = appsrc.end_of_stream();
    let _ = pipeline.set_state(gst::State::Null);
    Ok(())
}

#[cfg(not(feature = "gstreamer-backend"))]
async fn stream_gstreamer_encoded(
    stream: &mut TcpStream,
    _state: &MediaState,
    _client_id: u64,
    _feed: Arc<FeedRuntime>,
    _format: HttpAudioFormat,
) -> Result<()> {
    write_text_response(
        stream,
        503,
        "haze-media was built without GStreamer support\n",
    )
    .await?;
    Ok(())
}

#[cfg(feature = "gstreamer-backend")]
fn gstreamer_audio_pipeline(codec: &str) -> Result<String> {
    let source = "appsrc name=src is-live=true block=true do-timestamp=false max-bytes=38400 format=time stream-type=stream caps=audio/x-raw,format=S16LE,layout=interleaved,rate=48000,channels=1";
    let common = "queue max-size-time=1000000000 max-size-buffers=100 ! audioconvert ! audioresample quality=4";
    match codec {
        "opus" => Ok(format!(
            "{source} ! {common} ! opusenc bitrate=96000 frame-size=20 inband-fec=true ! oggmux ! appsink name=sink emit-signals=true sync=false"
        )),
        "aac" => Ok(format!(
            "{source} ! {common} ! avenc_aac bitrate=96000 ! aacparse ! appsink name=sink emit-signals=true sync=false"
        )),
        _ => bail!("unsupported GStreamer audio codec {codec}"),
    }
}

async fn write_stream_headers(stream: &mut TcpStream, content_type: &str) -> Result<()> {
    let headers = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-store\r\nConnection: close\r\nX-Accel-Buffering: no\r\n\r\n"
    );
    stream.write_all(headers.as_bytes()).await?;
    Ok(())
}

async fn write_json_response(stream: &mut TcpStream, status: u16, value: Value) -> Result<()> {
    let body = serde_json::to_vec(&value)?;
    let reason = status_reason(status);
    let headers = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(&body).await?;
    Ok(())
}

async fn write_text_response(stream: &mut TcpStream, status: u16, body: &str) -> Result<()> {
    let reason = status_reason(status);
    let headers = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(headers.as_bytes()).await?;
    stream.write_all(body.as_bytes()).await?;
    Ok(())
}

fn wav_stream_header() -> [u8; 44] {
    let byte_rate = SAMPLE_RATE * u32::from(CHANNELS) * 16 / 8;
    let block_align = CHANNELS * 16 / 8;
    let mut header = [0u8; 44];
    header[0..4].copy_from_slice(b"RIFF");
    header[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    header[8..12].copy_from_slice(b"WAVE");
    header[12..16].copy_from_slice(b"fmt ");
    header[16..20].copy_from_slice(&16u32.to_le_bytes());
    header[20..22].copy_from_slice(&1u16.to_le_bytes());
    header[22..24].copy_from_slice(&CHANNELS.to_le_bytes());
    header[24..28].copy_from_slice(&SAMPLE_RATE.to_le_bytes());
    header[28..32].copy_from_slice(&byte_rate.to_le_bytes());
    header[32..34].copy_from_slice(&block_align.to_le_bytes());
    header[34..36].copy_from_slice(&16u16.to_le_bytes());
    header[36..40].copy_from_slice(b"data");
    header[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
    header
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        415 => "Unsupported Media Type",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        _ => "OK",
    }
}

fn load_root_config(path: &Path) -> Result<RootConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    serde_yaml::from_str(&expand_env_vars(&raw))
        .with_context(|| format!("failed to parse config {}", path.display()))
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
            if name.is_empty() {
                continue;
            }
            out.push_str(&std::env::var(name).unwrap_or_default());
            continue;
        }
        out.push(ch);
    }
    out
}

fn load_outputs(path: &Path) -> Result<BTreeMap<String, OutputFeed>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read outputs XML {}", path.display()))?;
    let parsed: OutputsXml = quick_xml::de::from_str(&expand_env_vars(&raw))
        .with_context(|| format!("failed to parse outputs XML {}", path.display()))?;
    let mut outputs = BTreeMap::new();
    for feed in parsed.feeds {
        let id = feed.id.trim();
        if id.is_empty() {
            continue;
        }
        outputs.insert(
            id.to_string(),
            OutputFeed {
                id: id.to_string(),
                webrtc: feed.webrtc.is_enabled(),
                http: true,
                webrtc_ready: false,
                external: feed.stream.is_enabled()
                    || feed.udp.is_enabled()
                    || feed.rtp.is_enabled()
                    || feed.rtmp.is_enabled()
                    || feed.srt.is_enabled()
                    || feed.rtsp.is_enabled()
                    || feed.audio_device.is_enabled(),
            },
        );
    }
    Ok(outputs)
}

fn default_outputs_file() -> String {
    DEFAULT_OUTPUTS_FILE.to_string()
}

fn resolve_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn first_non_blank(values: &[Option<&str>]) -> String {
    for value in values.iter().flatten() {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    String::new()
}

fn env_bool(name: &str, default: bool) -> bool {
    match env::var(name)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
    {
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on" | "enabled") => true,
        Some(value) if matches!(value.as_str(), "0" | "false" | "no" | "off" | "disabled") => false,
        _ => default,
    }
}

fn env_f32(name: &str, default: f32) -> f32 {
    env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn db_to_gain(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

fn gain_to_db(gain: f32) -> f32 {
    20.0 * gain.max(1.0e-9).log10()
}

fn compressor_coeff_ms(ms: f32) -> f32 {
    let ms = ms.max(0.1);
    (-1.0 / ((SAMPLE_RATE as f32) * ms / 1_000.0)).exp()
}

fn query_value(query: &str, key: &str) -> Option<String> {
    for pair in query.split('&') {
        let (raw_key, raw_value) = pair.split_once('=').unwrap_or((pair, ""));
        if url_decode(raw_key) == key {
            return Some(url_decode(raw_value));
        }
    }
    None
}

fn url_decode(value: &str) -> String {
    let mut out = Vec::with_capacity(value.len());
    let mut bytes = value.as_bytes().iter().copied();
    while let Some(byte) = bytes.next() {
        match byte {
            b'+' => out.push(b' '),
            b'%' => {
                let hi = bytes.next();
                let lo = bytes.next();
                match (hi.and_then(hex_value), lo.and_then(hex_value)) {
                    (Some(hi), Some(lo)) => out.push((hi << 4) | lo),
                    _ => out.push(byte),
                }
            }
            _ => out.push(byte),
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn valid_feed_id(feed_id: &str) -> bool {
    let feed_id = feed_id.trim();
    !feed_id.is_empty()
        && feed_id.len() <= 128
        && feed_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.' | ':'))
}

fn xml_bool(value: Option<&str>, default: bool) -> bool {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("true" | "yes" | "1" | "on" | "enabled") => true,
        Some("false" | "no" | "0" | "off" | "disabled") => false,
        _ => default,
    }
}

fn initialize_gstreamer(mode: BackendMode) -> Result<bool> {
    if mode == BackendMode::Legacy {
        return Ok(false);
    }
    initialize_gstreamer_inner().or_else(|err| {
        if mode == BackendMode::GStreamer {
            Err(err)
        } else {
            warn!("GStreamer backend unavailable; HTTP audio will use legacy direct PCM only: {err:#}");
            Ok(false)
        }
    })
}

fn detect_gstreamer_webrtc_codecs(gstreamer_available: bool) -> Vec<WebRTCAudioCodec> {
    if !gstreamer_available {
        return Vec::new();
    }
    detect_gstreamer_webrtc_codecs_inner()
}

#[cfg(feature = "gstreamer-backend")]
fn detect_gstreamer_webrtc_codecs_inner() -> Vec<WebRTCAudioCodec> {
    if let Err(err) = validate_gstreamer_webrtc_base_elements() {
        warn!("GStreamer WebRTC backend unavailable: {err:#}");
        return Vec::new();
    }
    let mut codecs = Vec::new();
    for codec in [
        WebRTCAudioCodec::Opus,
        WebRTCAudioCodec::G722,
        WebRTCAudioCodec::Pcmu,
        WebRTCAudioCodec::Pcma,
    ] {
        let missing = webrtc_codec_elements(codec)
            .iter()
            .copied()
            .filter(|name| gstreamer::ElementFactory::find(name).is_none())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            codecs.push(codec);
        } else {
            warn!(
                codec = codec.id(),
                missing = missing.join(", "),
                "GStreamer WebRTC codec is unavailable"
            );
        }
    }
    if codecs.is_empty() {
        warn!("GStreamer WebRTC backend unavailable: no RTP audio codec branches are available");
    }
    codecs
}

#[cfg(not(feature = "gstreamer-backend"))]
fn detect_gstreamer_webrtc_codecs_inner() -> Vec<WebRTCAudioCodec> {
    Vec::new()
}

#[cfg(feature = "gstreamer-backend")]
fn initialize_gstreamer_inner() -> Result<bool> {
    configure_portable_gstreamer_paths();
    gstreamer::init().context("failed to initialize GStreamer")?;
    validate_gstreamer_audio_elements()?;
    Ok(true)
}

#[cfg(feature = "gstreamer-backend")]
fn configure_portable_gstreamer_paths() {
    let Ok(exe_path) = env::current_exe() else {
        return;
    };
    let Some(exe_dir) = exe_path.parent() else {
        return;
    };
    let plugin_dir = exe_dir.join("gstreamer-1.0");
    if !plugin_dir.is_dir() {
        return;
    }
    prepend_env_path("GST_PLUGIN_PATH", &plugin_dir);
    prepend_env_path("GST_PLUGIN_SYSTEM_PATH_1_0", &plugin_dir);

    let scanner = plugin_dir.join(if cfg!(windows) {
        "gst-plugin-scanner.exe"
    } else {
        "gst-plugin-scanner"
    });
    if scanner.is_file() {
        env::set_var("GST_PLUGIN_SCANNER", scanner);
    }
}

#[cfg(feature = "gstreamer-backend")]
fn prepend_env_path(name: &str, path: &Path) {
    let mut paths = env::var_os(name)
        .map(|raw| env::split_paths(&raw).collect::<Vec<_>>())
        .unwrap_or_default();
    if paths.iter().any(|existing| existing == path) {
        return;
    }
    paths.insert(0, path.to_path_buf());
    match env::join_paths(paths) {
        Ok(joined) => env::set_var(name, joined),
        Err(err) => warn!("failed to set {name} for GStreamer runtime: {err}"),
    }
}

#[cfg(feature = "gstreamer-backend")]
fn validate_gstreamer_audio_elements() -> Result<()> {
    let required = [
        "appsrc",
        "appsink",
        "queue",
        "audioconvert",
        "audioresample",
        "opusenc",
        "oggmux",
    ];
    let missing = required
        .iter()
        .copied()
        .filter(|name| gstreamer::ElementFactory::find(name).is_none())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "GStreamer runtime is missing required audio element(s): {}",
            missing.join(", ")
        );
    }
    Ok(())
}

#[cfg(feature = "gstreamer-backend")]
fn validate_gstreamer_webrtc_base_elements() -> Result<()> {
    let required = [
        "appsrc",
        "queue",
        "audioconvert",
        "audioresample",
        "webrtcbin",
    ];
    let missing = required
        .iter()
        .copied()
        .filter(|name| gstreamer::ElementFactory::find(name).is_none())
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        bail!(
            "GStreamer runtime is missing required WebRTC element(s): {}",
            missing.join(", ")
        );
    }
    Ok(())
}

#[cfg(feature = "gstreamer-backend")]
fn webrtc_codec_elements(codec: WebRTCAudioCodec) -> &'static [&'static str] {
    match codec {
        WebRTCAudioCodec::Opus => &["opusenc", "rtpopuspay"],
        WebRTCAudioCodec::G722 => &["avenc_g722", "rtpg722pay"],
        WebRTCAudioCodec::Pcmu => &["mulawenc", "rtppcmupay"],
        WebRTCAudioCodec::Pcma => &["alawenc", "rtppcmapay"],
    }
}

#[cfg(not(feature = "gstreamer-backend"))]
fn initialize_gstreamer_inner() -> Result<bool> {
    bail!("haze-media was built without the gstreamer-backend feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_playout_pcm_event() {
        let raw_pcm = vec![1u8, 0, 2, 0];
        let event = json!({
            "type": "playout.pcm",
            "feed_id": "sk-0001",
            "data": {
                "sample_rate": 48000,
                "channels": 1,
                "pcm": base64::engine::general_purpose::STANDARD.encode(&raw_pcm),
            }
        });
        let chunk = decode_pcm_event(serde_json::to_string(&event).unwrap().as_bytes()).unwrap();
        assert_eq!(chunk.feed_id, "sk-0001");
        assert_eq!(chunk.sample_rate, 48_000);
        assert_eq!(chunk.channels, 1);
        assert_eq!(chunk.data, raw_pcm);
    }

    #[test]
    fn parses_output_xml() {
        let parsed: OutputsXml = quick_xml::de::from_str(
            r#"<outputs><feed id="sk-0001"><webrtc enabled="true"/><udp enabled="false"/></feed></outputs>"#,
        )
        .unwrap();
        assert_eq!(parsed.feeds.len(), 1);
        assert!(parsed.feeds[0].webrtc.is_enabled());
        assert!(!parsed.feeds[0].udp.is_enabled());
    }

    #[test]
    fn wav_header_uses_streaming_lengths() {
        let header = wav_stream_header();
        assert_eq!(&header[0..4], b"RIFF");
        assert_eq!(&header[8..12], b"WAVE");
        assert_eq!(
            u32::from_le_bytes(header[40..44].try_into().unwrap()),
            u32::MAX
        );
    }

    #[test]
    fn health_reports_active_http_clients() {
        let state = MediaState::new(BTreeMap::new(), BackendMode::Legacy, false, Vec::new());
        let client = state.register_http_client("sk-0001", "raw");
        state.record_http_client_write(client.id(), FRAME_BYTES);
        state.record_http_client_encoded_drop(client.id());
        state.record_http_client_lag(client.id(), 3);

        let health = state.health();
        assert_eq!(health["http_client_count"], json!(1));
        assert_eq!(health["capabilities"]["http_audio"], json!(true));
        assert_eq!(health["capabilities"]["webrtc"], json!(false));
        assert_eq!(health["http_clients"][0]["feed_id"], json!("sk-0001"));
        assert_eq!(health["http_clients"][0]["codec"], json!("raw"));
        assert_eq!(
            health["http_clients"][0]["encoded_chunks_dropped"],
            json!(1)
        );
        assert_eq!(health["http_clients"][0]["lagged_frames"], json!(3));
        assert_eq!(
            health["http_clients"][0]["bytes_written"],
            json!(FRAME_BYTES)
        );
        assert_eq!(health["http_clients"][0]["chunks_written"], json!(1));
        assert_eq!(
            health["audio_processing"]["loudness_enabled"],
            json!(DEFAULT_LOUDNESS_ENABLED)
        );
        assert_eq!(health["clock"]["frame_ms"], json!(20));
        assert_eq!(
            health["clock"]["target_source_queue_ms"],
            json!(TARGET_SOURCE_QUEUE_MS)
        );

        drop(client);
        let health = state.health();
        assert_eq!(health["http_client_count"], json!(0));
    }

    #[test]
    fn feed_clock_uses_named_dedicated_thread() {
        assert_eq!(
            feed_clock_thread_name("sk-0001"),
            "haze-media-clock-sk-0001"
        );
    }

    #[test]
    fn exact_frame_renderer_outputs_one_unstretched_pcm_frame() {
        let mut samples = (0..(FRAME_SAMPLES - 4))
            .map(|value| value as i16)
            .collect::<VecDeque<_>>();
        samples.extend((0..4).map(|value| (FRAME_SAMPLES + value) as i16));
        let mut frame = Vec::with_capacity(FRAME_BYTES);
        let mut last = vec![0i16; FRAME_SAMPLES];

        render_exact_frame(&mut samples, &mut frame, &mut last);

        assert_eq!(frame.len(), FRAME_BYTES);
        assert!(samples.is_empty());
        assert_eq!(last[0], 0);
        assert_eq!(last[FRAME_SAMPLES - 1], (FRAME_SAMPLES + 3) as i16);
        assert_ne!(last[0], last[FRAME_SAMPLES - 1]);
    }

    #[test]
    fn source_queue_hard_drop_keeps_queue_bounded() {
        let max_samples = max_source_queue_samples();
        let mut samples = (0..(max_samples + 10))
            .map(|value| value as i16)
            .collect::<VecDeque<_>>();

        let (stale_dropped, soft_trimmed) = trim_source_queue(&mut samples, false);

        assert_eq!(stale_dropped, 10);
        assert_eq!(soft_trimmed, 0);
        assert_eq!(samples.len(), max_samples);
    }

    #[test]
    fn source_queue_soft_trim_corrects_primed_drift_gradually() {
        let soft_samples = soft_source_queue_samples();
        let target_samples = target_source_queue_samples();
        let mut samples = (0..(soft_samples + 500))
            .map(|value| value as i16)
            .collect::<VecDeque<_>>();

        let (stale_dropped, soft_trim_budget) = trim_source_queue(&mut samples, true);

        assert_eq!(stale_dropped, 0);
        assert_eq!(soft_trim_budget, soft_samples + 500 - target_samples);
        assert_eq!(samples.len(), soft_samples + 500);
        assert!(samples.len() > target_samples);
    }

    #[test]
    fn exact_frame_renderer_smoothly_spreads_trimmed_samples() {
        let trim_samples = 10usize;
        let mut samples = (0..(FRAME_SAMPLES + trim_samples))
            .map(|value| value as i16)
            .collect::<VecDeque<_>>();
        let mut frame = Vec::with_capacity(FRAME_BYTES);
        let mut last = vec![0i16; FRAME_SAMPLES];

        render_exact_frame_with_trim(&mut samples, &mut frame, &mut last, trim_samples);

        let rendered = pcm16_samples(&frame).collect::<Vec<_>>();
        assert_eq!(rendered.len(), FRAME_SAMPLES);
        assert!(samples.is_empty());
        assert_eq!(last.len(), FRAME_SAMPLES);
        assert!(rendered[0] < rendered[FRAME_SAMPLES - 1]);
        assert!(rendered
            .windows(2)
            .all(|pair| pair[1].saturating_sub(pair[0]) <= 2));
    }

    #[test]
    fn concealment_frame_decays_last_real_audio() {
        let mut frame = Vec::new();
        let last_real = vec![12_000i16; FRAME_SAMPLES];

        render_concealment_frame(&last_real, CONCEALMENT_FRAMES, &mut frame);

        let samples = pcm16_samples(&frame).collect::<Vec<_>>();
        assert_eq!(samples.len(), FRAME_SAMPLES);
        assert!(samples[0] > 0);
        assert!(samples[0] < last_real[0]);
        assert_eq!(samples[0], samples[FRAME_SAMPLES - 1]);
    }

    #[test]
    fn frame_signal_detects_silence_repetition_and_clipping() {
        let silence = vec![0u8; FRAME_BYTES];
        let silent_signal = analyze_pcm_frame(&silence, &[]);
        assert!(silent_signal.near_silent);
        assert!(!silent_signal.repeated);
        assert_eq!(silent_signal.clipped_samples, 0);

        let repeated_signal = analyze_pcm_frame(&silence, &silence);
        assert!(repeated_signal.repeated);

        let mut clipped = Vec::with_capacity(FRAME_BYTES);
        for _ in 0..FRAME_SAMPLES {
            push_i16(&mut clipped, i16::MAX);
        }
        let clipped_signal = analyze_pcm_frame(&clipped, &silence);
        assert!(!clipped_signal.near_silent);
        assert_eq!(clipped_signal.peak, i16::MAX as u16);
        assert_eq!(clipped_signal.clipped_samples, FRAME_SAMPLES as u64);
    }

    #[test]
    fn webrtc_selection_honors_receiver_g722_preference() {
        let offer = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111 9 0 8\r\na=rtpmap:111 opus/48000/2\r\na=rtpmap:9 G722/8000\r\na=rtpmap:0 PCMU/8000\r\na=rtpmap:8 PCMA/8000\r\n";
        let selection = select_webrtc_audio_codec(
            offer,
            &[
                WebRTCAudioCodec::Opus,
                WebRTCAudioCodec::G722,
                WebRTCAudioCodec::Pcmu,
                WebRTCAudioCodec::Pcma,
            ],
            "g722",
            false,
            false,
        )
        .unwrap();

        assert_eq!(selection.codec, WebRTCAudioCodec::G722);
        assert_eq!(selection.payload_type, 9);
    }

    #[test]
    fn webrtc_selection_prefers_opus_for_browser_auto() {
        let offer = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111 0 8\r\na=rtpmap:111 opus/48000/2\r\na=rtpmap:0 PCMU/8000\r\na=rtpmap:8 PCMA/8000\r\n";
        let selection = select_webrtc_audio_codec(
            offer,
            &[WebRTCAudioCodec::Opus, WebRTCAudioCodec::Pcmu],
            "",
            false,
            false,
        )
        .unwrap();

        assert_eq!(selection.codec, WebRTCAudioCodec::Opus);
        assert_eq!(selection.payload_type, 111);
    }

    #[test]
    fn webrtc_selection_falls_back_to_static_pcmu_payload() {
        let offer = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 0\r\n";
        let selection =
            select_webrtc_audio_codec(offer, &[WebRTCAudioCodec::Pcmu], "", false, false).unwrap();

        assert_eq!(selection.codec, WebRTCAudioCodec::Pcmu);
        assert_eq!(selection.payload_type, 0);
    }

    #[test]
    fn loudness_processor_raises_quiet_audio_and_limits_hot_audio() {
        let mut loudness = AudioLoudness {
            enabled: true,
            gain: db_to_gain(6.0),
            threshold_db: -18.0,
            ratio: 3.0,
            makeup: db_to_gain(1.5),
            ceiling: 0.98,
            attack_coeff: compressor_coeff_ms(5.0),
            release_coeff: compressor_coeff_ms(120.0),
            envelope: 0.0,
        };

        let quiet = loudness.process_sample(1_000);
        let hot = loudness.process_sample(32_000);

        assert!(quiet > 1_000);
        assert!(hot <= (f32::from(i16::MAX) * 0.98).round() as i16);
    }

    #[test]
    fn disabled_loudness_processor_is_bypass() {
        let mut frame = Vec::new();
        push_i16(&mut frame, -1234);
        push_i16(&mut frame, 1234);
        let original = frame.clone();
        AudioLoudness {
            enabled: false,
            gain: db_to_gain(12.0),
            threshold_db: -30.0,
            ratio: 10.0,
            makeup: db_to_gain(12.0),
            ceiling: 0.5,
            attack_coeff: compressor_coeff_ms(5.0),
            release_coeff: compressor_coeff_ms(120.0),
            envelope: 0.0,
        }
        .process(&mut frame);

        assert_eq!(frame, original);
    }

    #[test]
    fn media_loudness_is_opt_in_by_default() {
        std::env::remove_var("HAZE_MEDIA_LOUDNESS_ENABLED");

        let loudness = AudioLoudness::from_env();

        assert!(!loudness.enabled);
    }

    #[test]
    fn audio_health_allows_standby_silence() {
        let stats = FeedStats {
            feed_id: "CAP-IT-ALL".to_string(),
            frames: 10_000,
            real_frames: 10_000,
            near_silent_frames: 10_000,
            repeated_frames: 9_999,
            consecutive_near_silent_frames: 9_999,
            consecutive_repeated_frames: 9_998,
            queued_samples: FRAME_SAMPLES * 12,
            last_input_age_ms: Some(20),
            max_tick_gap_ms: 20,
            ..FeedStats::default()
        };

        let (ok, warnings) = classify_audio_health(&stats);
        assert!(ok, "{warnings:?}");
        assert!(warnings.is_empty());
    }

    #[test]
    fn audio_health_flags_robotic_failure_modes() {
        let stats = FeedStats {
            feed_id: "sk-0001".to_string(),
            frames: 10_000,
            real_frames: 9_990,
            concealed_frames: 5,
            repeated_non_silent_frames: 4,
            clipped_samples: 2,
            late_ticks: 1,
            queued_samples: FRAME_SAMPLES * 12,
            last_input_age_ms: Some(20),
            max_tick_gap_ms: 55,
            ..FeedStats::default()
        };

        let (ok, warnings) = classify_audio_health(&stats);
        assert!(!ok);
        assert!(warnings
            .iter()
            .any(|warning| warning.starts_with("concealed_frames=")));
        assert!(warnings
            .iter()
            .any(|warning| warning.starts_with("repeated_non_silent_frames=")));
        assert!(warnings
            .iter()
            .any(|warning| warning.starts_with("clipped_samples=")));
        assert!(warnings
            .iter()
            .any(|warning| warning.starts_with("late_ticks=")));
    }

    #[cfg(feature = "gstreamer-backend")]
    #[test]
    fn gstreamer_opus_pipeline_uses_clean_broadcast_bitrate() {
        let pipeline = gstreamer_audio_pipeline("opus").unwrap();
        assert!(pipeline.contains("bitrate=96000"));
        assert!(pipeline.contains("frame-size=20"));
    }
}
