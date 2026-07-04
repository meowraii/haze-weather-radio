use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File};
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::Context;
use anyhow::{bail, Result};
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch};
use tokio::time::sleep;
use tokio::time::Duration;
use tracing::{info, warn};

use crate::config::{EncoderCodecSettings, FeedConfig};
use crate::state::RuntimeState;
use crate::wgpu_renderer::{OverlayRenderState, WgpuFrameRenderer};

use gst::prelude::*;
use gstreamer as gst;
use gstreamer_app as gst_app;

const DEFAULT_UDP_BUFFER_BYTES: u32 = 4 * 1024 * 1024;
const GST_BUS_POLL_INTERVAL_MS: u64 = 10;
const PRIORITY_AUDIO_RESTART_SUPPRESS: Duration = Duration::from_secs(300);

pub(crate) async fn run_supervised(
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    mut shutdown_rx: watch::Receiver<bool>,
    base_dir: PathBuf,
    status_tx: Option<mpsc::UnboundedSender<Value>>,
) -> Result<()> {
    configure_portable_gstreamer_paths();
    gst::init().context("failed to initialize GStreamer")?;
    let restart = restart_backoff_bounds(&feed);
    let mut restart_delay = restart.initial;
    loop {
        if *shutdown_rx.borrow() {
            info!(feed_id = %feed.id, "gstreamer cgen supervisor shutting down");
            return Ok(());
        }
        let feed_once = feed.clone();
        let state_once = state_rx.clone();
        let shutdown_once = shutdown_rx.clone();
        let base_once = base_dir.clone();
        let status_once = status_tx.clone();
        let result = tokio::task::spawn_blocking(move || {
            run_pipeline_once(feed_once, state_once, shutdown_once, base_once, status_once)
        })
        .await
        .context("gstreamer cgen worker panicked")?;
        match result {
            Ok(()) => {
                info!(feed_id = %feed.id, "gstreamer cgen pipeline exited cleanly");
                restart_delay = restart.initial;
            }
            Err(err) => {
                warn!(feed_id = %feed.id, "gstreamer cgen pipeline failed: {err:#}");
            }
        }
        tokio::select! {
            _ = sleep(restart_delay) => {}
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    info!(feed_id = %feed.id, "gstreamer cgen supervisor shutdown requested");
                    return Ok(());
                }
            }
        }
        restart_delay = (restart_delay * 2).min(restart.max);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RestartBackoff {
    initial: Duration,
    max: Duration,
}

fn restart_backoff_bounds(feed: &FeedConfig) -> RestartBackoff {
    let initial_ms = feed.sync.reconnect_initial_ms.clamp(100, 60_000);
    let max_ms = feed
        .sync
        .reconnect_max_ms
        .max(initial_ms)
        .clamp(initial_ms, 300_000);
    RestartBackoff {
        initial: Duration::from_millis(u64::from(initial_ms)),
        max: Duration::from_millis(u64::from(max_ms)),
    }
}

fn sync_status_value(feed: &FeedConfig) -> Value {
    json!({
        "hard_reset_ms": feed.sync.hard_reset_ms,
        "max_audio_frames_per_video": feed.sync.max_audio_frames_per_video,
        "source_buffer_ms": feed.sync.source_buffer_ms,
        "reconnect_initial_ms": feed.sync.reconnect_initial_ms,
        "reconnect_max_ms": feed.sync.reconnect_max_ms,
        "status_interval_ms": feed.sync.status_interval_ms,
    })
}

pub(crate) fn gstreamer_catalog_json() -> Result<Value> {
    configure_portable_gstreamer_paths();
    gst::init().context("failed to initialize GStreamer")?;
    Ok(json!({
        "formats": gstreamer_muxer_catalog(),
        "video_codecs": gstreamer_encoder_catalog(gst::ElementFactoryType::VIDEO_ENCODER, "video"),
        "audio_codecs": gstreamer_encoder_catalog(gst::ElementFactoryType::AUDIO_ENCODER, "audio"),
        "gstreamer": {
            "source": "haze-cgen-registry",
            "runtime": "gstreamer-rs",
        },
    }))
}

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
    prepend_env_path("GST_PLUGIN_PATH_1_0", &plugin_dir);

    let scanner = plugin_dir.join(if cfg!(windows) {
        "gst-plugin-scanner.exe"
    } else {
        "gst-plugin-scanner"
    });
    if scanner.is_file() {
        env::set_var("GST_PLUGIN_SCANNER", scanner);
    }
}

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

fn gstreamer_muxer_catalog() -> Vec<Value> {
    let mut entries = Vec::new();
    for factory in
        gst::ElementFactory::factories_with_type(gst::ElementFactoryType::MUXER, gst::Rank::NONE)
    {
        let element = factory.name().to_string();
        if let Some((id, label)) = format_catalog_entry(
            element.as_str(),
            factory.longname(),
            factory.klass(),
            factory.description(),
        ) {
            let source = factory
                .plugin_name()
                .map(|name| name.to_string())
                .unwrap_or_else(|| "gstreamer".to_string());
            entries.push(json!({
                "id": id,
                "label": label,
                "kind": "container",
                "element": element,
                "source": source,
            }));
        }
    }
    sort_catalog_values(entries)
}

fn gstreamer_encoder_catalog(
    factory_type: gst::ElementFactoryType,
    media_kind: &str,
) -> Vec<Value> {
    let mut entries = Vec::new();
    for factory in gst::ElementFactory::factories_with_type(factory_type, gst::Rank::NONE) {
        let element = factory.name().to_string();
        let label = encoder_catalog_label(
            element.as_str(),
            factory.longname(),
            factory.klass(),
            factory.description(),
        );
        let source = factory
            .plugin_name()
            .map(|name| name.to_string())
            .unwrap_or_else(|| "gstreamer".to_string());
        entries.push(json!({
            "id": element,
            "label": label,
            "kind": media_kind,
            "element": element,
            "source": source,
        }));
    }
    sort_catalog_values(entries)
}

fn sort_catalog_values(mut entries: Vec<Value>) -> Vec<Value> {
    entries.sort_by(|left, right| {
        let left_label = left
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        let right_label = right
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        left_label.cmp(&right_label)
    });
    entries
}

fn format_catalog_entry(
    element: &str,
    longname: &str,
    klass: &str,
    description: &str,
) -> Option<(&'static str, &'static str)> {
    let lower = catalog_lower(element, longname, klass, description);
    match () {
        _ if lower.contains("mpeg-ts")
            || lower.contains("mpeg transport")
            || element == "mpegtsmux" =>
        {
            Some(("mpegts", "MPEG-TS"))
        }
        _ if lower.contains("quicktime") || lower.contains("mp4") => Some(("mp4", "MPEG-4 / MP4")),
        _ if lower.contains("matroska") => Some(("matroska", "Matroska")),
        _ if lower.contains("webm") => Some(("webm", "WebM")),
        _ if lower.contains("flv") => Some(("flv", "FLV")),
        _ if lower.contains("wav") => Some(("wav", "WAV")),
        _ if lower.contains("ogg") => Some(("ogg", "Ogg")),
        _ if lower.contains("avi") => Some(("avi", "AVI")),
        _ if lower.contains("mxf") => Some(("mxf", "MXF")),
        _ => None,
    }
}

fn encoder_catalog_label(element: &str, longname: &str, klass: &str, description: &str) -> String {
    let lower = catalog_lower(element, longname, klass, description);
    let codec = catalog_codec_name(lower.as_str()).unwrap_or(longname.trim());
    let implementation = catalog_implementation_name(element, lower.as_str());
    if implementation.is_empty() {
        format!("{codec} ({element})")
    } else {
        format!("{codec} - {implementation} ({element})")
    }
}

fn catalog_lower(element: &str, longname: &str, klass: &str, description: &str) -> String {
    format!("{element} {longname} {klass} {description}").to_ascii_lowercase()
}

fn catalog_codec_name(lower: &str) -> Option<&'static str> {
    match () {
        _ if lower.contains("e-ac-3")
            || lower.contains("eac3")
            || lower.contains("enhanced ac-3") =>
        {
            Some("E-AC-3")
        }
        _ if lower.contains("ac-3") || lower.contains("ac3") => Some("AC-3"),
        _ if lower.contains("h.265") || lower.contains("h265") || lower.contains("hevc") => {
            Some("H.265 / HEVC")
        }
        _ if lower.contains("h.264") || lower.contains("h264") || lower.contains("avc") => {
            Some("H.264 / AVC")
        }
        _ if lower.contains("aac") => Some("AAC"),
        _ if lower.contains("opus") => Some("Opus"),
        _ if lower.contains("vorbis") => Some("Vorbis"),
        _ if lower.contains("flac") => Some("FLAC"),
        _ if lower.contains("mp3") || lower.contains("mpeg layer 3") => Some("MP3"),
        _ if lower.contains("mp2") || lower.contains("mpeg layer ii") => {
            Some("MPEG Layer II Audio")
        }
        _ if lower.contains("alaw") || lower.contains("a-law") => Some("A-law"),
        _ if lower.contains("mulaw") || lower.contains("mu-law") => Some("mu-law"),
        _ if lower.contains("speex") => Some("Speex"),
        _ if lower.contains("mpeg-2") || lower.contains("mpeg2") => Some("MPEG-2 Video"),
        _ if lower.contains("mpeg-4") || lower.contains("mpeg4") => Some("MPEG-4 Part 2"),
        _ if lower.contains("av1") => Some("AV1"),
        _ if lower.contains("vp9") => Some("VP9"),
        _ if lower.contains("vp8") => Some("VP8"),
        _ if lower.contains("theora") => Some("Theora"),
        _ if lower.contains("jpeg") => Some("Motion JPEG"),
        _ if lower.contains("png") => Some("PNG"),
        _ if lower.contains("webp") => Some("WebP"),
        _ => None,
    }
}

fn catalog_implementation_name(element: &str, lower: &str) -> &'static str {
    match element {
        "x264enc" => "x264 software",
        "x265enc" => "x265 software",
        "openh264enc" => "OpenH264",
        "amfh264enc" | "amfh265enc" | "amfav1enc" => "AMD AMF",
        "nvh264enc" | "nvh265enc" | "nvav1enc" => "NVIDIA NVENC",
        "qsvh264enc" | "qsvh265enc" | "qsvmpeg2enc" | "qsvav1enc" => "Intel Quick Sync",
        "svtav1enc" => "SVT-AV1",
        "aomav1enc" | "av1enc" => "AOM",
        "vp8enc" | "vp9enc" => "libvpx",
        "opusenc" => "Opus",
        "vorbisenc" => "Vorbis",
        "flacenc" => "FLAC",
        "faac" => "FAAC",
        "fdkaacenc" => "fdk-aac",
        "mfaacenc" => "Media Foundation",
        "lamemp3enc" => "LAME",
        "twolame" => "TwoLAME",
        _ if element.starts_with("avenc_") => "libav",
        _ if lower.contains("hardware") => "hardware",
        _ => "",
    }
}

fn run_pipeline_once(
    feed: FeedConfig,
    mut state_rx: watch::Receiver<RuntimeState>,
    mut shutdown_rx: watch::Receiver<bool>,
    base_dir: PathBuf,
    status_tx: Option<mpsc::UnboundedSender<Value>>,
) -> Result<()> {
    let plan = GstPipelinePlan::from_feed(&feed)?;
    info!(
        feed_id = %feed.id,
        input = %feed.redacted_program_input_url(),
        output = %feed.redacted_program_output_url(),
        routine_audio = feed.priority_input.routine_audio_enabled(),
        priority_audio = feed.priority_input.priority_audio_enabled(),
        mute_standby_routine = feed.audio.mute_standby_routine,
        pipeline = %crate::config::redacted_pipeline_description(&feed, &plan.description),
        "starting gstreamer cgen pipeline"
    );
    publish_status(
        &status_tx,
        &feed,
        &state_rx,
        false,
        json!({
            "media_backend": "gstreamer-rs",
            "input_connected": false,
            "input_video_connected": false,
            "input_audio_connected": false,
            "output_active": false,
            "pipeline_description": crate::config::redacted_pipeline_description(&feed, &plan.description),
            "output_ladder": plan.status_value(),
            "sync": sync_status_value(&feed),
        }),
    );
    if let Err(err) = validate_gstreamer_elements(&plan.required_elements) {
        publish_status(
            &status_tx,
            &feed,
            &state_rx,
            false,
            json!({
                "media_backend": "gstreamer-rs",
                "input_connected": false,
                "output_active": false,
                "last_error": err.to_string(),
                "required_elements": plan.required_elements,
                "sync": sync_status_value(&feed),
            }),
        );
        return Err(err);
    }

    let element =
        gst::parse::launch(&plan.description).context("failed to parse GStreamer CGEN pipeline")?;
    let pipeline = element
        .downcast::<gst::Pipeline>()
        .map_err(|_| anyhow::anyhow!("GStreamer CGEN description did not produce a pipeline"))?;
    apply_mpegts_program_map(&pipeline, &plan)?;
    let priority_appsrc = priority_audio_appsrc(&pipeline)?;
    let input_health = Arc::new(Mutex::new(InputHealth::default()));
    install_program_video_probe(&pipeline, Arc::clone(&input_health))?;
    install_program_audio_probe(&pipeline, Arc::clone(&input_health))?;
    let priority_drain = Duration::from_millis(
        u64::from(feed.sync.source_buffer_ms.clamp(40, 5_000)).saturating_add(1_000),
    );
    let mut priority_feeder = PriorityAudioFeeder::new(
        priority_appsrc,
        base_dir.clone(),
        feed.id.clone(),
        priority_drain,
    );
    let mut text_overlay = TextOverlayController::default();
    let wgpu_compositors = match install_wgpu_overlay_probes(
        &pipeline,
        &feed,
        &plan,
        &base_dir,
        text_overlay.shared_state(),
    ) {
        Ok(compositors) => compositors,
        Err(err) => {
            publish_status(
                &status_tx,
                &feed,
                &state_rx,
                false,
                json!({
                    "media_backend": "gstreamer-rs",
                    "graphics_backend": "wgpu",
                    "input_connected": false,
                    "output_active": false,
                    "fatal": true,
                    "last_error": err.to_string(),
                    "sync": sync_status_value(&feed),
                }),
            );
            return Err(err);
        }
    };
    let (initial_audio_mode, initial_video_mode) = {
        let state = state_rx.borrow();
        let video_connected = input_video_connected(&input_health, &feed);
        let audio_connected = input_audio_connected(&input_health, &feed);
        let feeder_status = priority_feeder.update(priority_audio_for_feed(&feed, &state));
        let video_mode = desired_video_selector_mode(&feed, &state, video_connected);
        (
            desired_audio_selector_mode_with_status(&feed, audio_connected, &feeder_status),
            video_mode,
        )
    };
    let mut current_audio_mode = initial_audio_mode;
    let mut current_video_mode = initial_video_mode;
    set_audio_selector_mode(&pipeline, initial_audio_mode)?;
    set_video_selector_mode(&pipeline, initial_video_mode)?;
    let mut diagnostics = PipelineDiagnostics::default();
    pipeline
        .set_state(gst::State::Playing)
        .context("failed to set GStreamer CGEN pipeline to Playing")?;
    {
        let state = state_rx.borrow();
        let feeder_status = priority_feeder.update(priority_audio_for_feed(&feed, &state));
        text_overlay.sync(
            &pipeline,
            &feed,
            &state,
            current_video_mode.no_signal(),
            feeder_status.is_live_priority(),
        )?;
    }
    publish_status(
        &status_tx,
        &feed,
        &state_rx,
        current_video_mode.no_signal(),
        json!({
            "media_backend": "gstreamer-rs",
            "input_connected": false,
            "input_video_connected": false,
            "input_audio_connected": false,
            "output_active": true,
            "routine_audio_enabled": feed.priority_input.routine_audio_enabled(),
            "priority_audio_enabled": feed.priority_input.priority_audio_enabled(),
            "priority_input_format": feed.priority_input.format,
            "mute_standby_routine": feed.audio.mute_standby_routine,
            "audio_selector": initial_audio_mode.as_str(),
            "video_selector": initial_video_mode.as_str(),
            "text_overlay": text_overlay.status_value(),
            "graphics_renderer": wgpu_compositors.status_value(),
            "input_health": input_health_status(&input_health, &feed),
            "pipeline_diagnostics": diagnostics.status_value(),
            "output_ladder": plan.status_value(),
            "sync": sync_status_value(&feed),
        }),
    );

    let bus = pipeline
        .bus()
        .context("GStreamer CGEN pipeline has no bus")?;
    let mut next_status = Instant::now();
    loop {
        if *shutdown_rx.borrow() {
            pipeline
                .set_state(gst::State::Null)
                .context("failed to stop GStreamer CGEN pipeline during shutdown")?;
            return Ok(());
        }
        while let Some(message) =
            bus.timed_pop(gst::ClockTime::from_mseconds(GST_BUS_POLL_INTERVAL_MS))
        {
            use gst::MessageView;
            match message.view() {
                MessageView::Eos(..) => {
                    pipeline
                        .set_state(gst::State::Null)
                        .context("failed to stop GStreamer CGEN pipeline")?;
                    return Ok(());
                }
                MessageView::Error(err) => {
                    let src = err
                        .src()
                        .map(|src| src.path_string())
                        .unwrap_or_else(|| "unknown".into());
                    let debug = err
                        .debug()
                        .map(|debug| debug.to_string())
                        .unwrap_or_default();
                    let _ = pipeline.set_state(gst::State::Null);
                    bail!("GStreamer CGEN error from {src}: {} {debug}", err.error());
                }
                MessageView::Warning(warning) => {
                    diagnostics.record_warning(message_src_path(&message), warning);
                }
                MessageView::StateChanged(state) => {
                    if message
                        .src()
                        .is_some_and(|src| src == pipeline.upcast_ref::<gst::Object>())
                    {
                        publish_status(
                            &status_tx,
                            &feed,
                            &state_rx,
                            current_video_mode.no_signal(),
                            json!({
                                "media_backend": "gstreamer-rs",
                                "gst_state": format!("{:?}", state.current()),
                            }),
                        );
                    }
                }
                MessageView::Element(element) => {
                    if element
                        .structure()
                        .is_some_and(|structure| structure.name() == "GstUDPSrcTimeout")
                    {
                        mark_input_timeout(&input_health);
                    }
                }
                MessageView::Latency(..) => match pipeline.recalculate_latency() {
                    Ok(()) => diagnostics.record_latency_recalculation(None),
                    Err(err) => diagnostics.record_latency_recalculation(Some(err.to_string())),
                },
                MessageView::Qos(qos) => {
                    diagnostics.record_qos(message_src_path(&message), qos);
                }
                _ => {}
            }
        }
        {
            let state = state_rx.borrow();
            let video_connected = input_video_connected(&input_health, &feed);
            let video_mode = desired_video_selector_mode(&feed, &state, video_connected);
            if current_video_mode != video_mode {
                set_video_selector_mode(&pipeline, video_mode)?;
                current_video_mode = video_mode;
            }
            let priority_audio = priority_audio_for_feed(&feed, &state);
            let feeder_status = priority_feeder.update(priority_audio);
            let audio_connected = input_audio_connected(&input_health, &feed);
            let audio_mode =
                desired_audio_selector_mode_with_status(&feed, audio_connected, &feeder_status);
            if current_audio_mode != audio_mode {
                set_audio_selector_mode(&pipeline, audio_mode)?;
                current_audio_mode = audio_mode;
            }
            text_overlay.sync(
                &pipeline,
                &feed,
                &state,
                video_mode.no_signal(),
                feeder_status.is_live_priority(),
            )?;
        }
        if shutdown_rx.has_changed().unwrap_or(false) && *shutdown_rx.borrow_and_update() {
            pipeline
                .set_state(gst::State::Null)
                .context("failed to stop GStreamer CGEN pipeline during shutdown")?;
            return Ok(());
        }
        if state_rx.has_changed().unwrap_or(false) || Instant::now() >= next_status {
            let state = state_rx.borrow_and_update();
            let video_connected = input_video_connected(&input_health, &feed);
            let audio_connected = input_audio_connected(&input_health, &feed);
            let priority_audio = priority_audio_for_feed(&feed, &state);
            let feeder_status = priority_feeder.update(priority_audio);
            let audio_mode =
                desired_audio_selector_mode_with_status(&feed, audio_connected, &feeder_status);
            let video_mode = desired_video_selector_mode(&feed, &state, video_connected);
            let priority_active = priority_audio_active(&feed, &state);
            let visual_lifecycle = if state.banner_for(feed.id.as_str()).is_some() {
                "banner"
            } else if video_mode.no_signal() {
                "standby"
            } else {
                "release"
            };
            drop(state);
            if current_audio_mode != audio_mode {
                set_audio_selector_mode(&pipeline, audio_mode)?;
                current_audio_mode = audio_mode;
            }
            if current_video_mode != video_mode {
                set_video_selector_mode(&pipeline, video_mode)?;
                current_video_mode = video_mode;
            }
            publish_status(
                &status_tx,
                &feed,
                &state_rx,
                video_mode.no_signal(),
                json!({
                    "media_backend": "gstreamer-rs",
                    "input_connected": true,
                    "output_active": true,
                    "routine_audio_enabled": feed.priority_input.routine_audio_enabled(),
                    "priority_audio_enabled": feed.priority_input.priority_audio_enabled(),
                    "mute_standby_routine": feed.audio.mute_standby_routine,
                    "audio_selector": audio_mode.as_str(),
                    "video_selector": video_mode.as_str(),
                    "audio_lifecycle": audio_mode.status_text(priority_active),
                    "priority_audio_active": priority_active,
                    "priority_audio_pipeline": feeder_status.status,
                    "priority_audio_queue_id": feeder_status.queue_id,
                    "priority_audio_error": feeder_status.error,
                    "text_overlay": text_overlay.status_value(),
                    "graphics_renderer": wgpu_compositors.status_value(),
                    "visual_lifecycle": visual_lifecycle,
                    "input_connected": video_connected && audio_connected,
                    "input_video_connected": video_connected,
                    "input_audio_connected": audio_connected,
                    "input_health": input_health_status(&input_health, &feed),
                    "pipeline_diagnostics": diagnostics.status_value(),
                    "output_ladder": plan.status_value(),
                    "sync": sync_status_value(&feed),
                }),
            );
            next_status = Instant::now()
                + Duration::from_millis(u64::from(feed.sync.status_interval_ms.max(100)));
        }
    }
}

type TextOverlayRenderState = OverlayRenderState;

#[derive(Debug, Default)]
struct TextOverlayController {
    key: Option<String>,
    started_at: Option<Instant>,
    pass_index: u32,
    completed_passes: u32,
    audio_was_live: bool,
    post_audio_repeats_started: u32,
    retaining_cleared_ticker: bool,
    last: Option<TextOverlayRenderState>,
    shared: Arc<Mutex<Option<TextOverlayRenderState>>>,
}

impl TextOverlayController {
    fn sync(
        &mut self,
        _pipeline: &gst::Pipeline,
        feed: &FeedConfig,
        state: &RuntimeState,
        no_signal: bool,
        audio_live: bool,
    ) -> Result<()> {
        let next = self.next_state(feed, state, no_signal, audio_live);
        if self.last.as_ref() == Some(&next) {
            return Ok(());
        }
        if let Ok(mut guard) = self.shared.lock() {
            *guard = Some(next.clone());
        }
        self.last = Some(next);
        Ok(())
    }

    fn next_state(
        &mut self,
        feed: &FeedConfig,
        state: &RuntimeState,
        no_signal: bool,
        audio_live: bool,
    ) -> TextOverlayRenderState {
        self.update_audio_lifecycle(audio_live);
        let snapshot = crate::graphics::presentation_snapshot(feed, state, no_signal);
        let text = snapshot.overlay_text.trim().to_string();
        let font_desc = format!("{} {}", snapshot.font, snapshot.font_size.max(16));
        let ypos = overlay_ypos(feed, snapshot.ticker_y);
        if text.is_empty() {
            if snapshot.visual_mode == "release" {
                if let Some(last) = self.last.clone() {
                    if last.visual_mode == "ticker_alert" {
                        if !ticker_state_done(&last) {
                            self.retaining_cleared_ticker = true;
                            return last;
                        }
                        if self.should_repeat_ticker(feed, audio_live) {
                            let next = self.restart_ticker_pass(last);
                            self.retaining_cleared_ticker = true;
                            return next;
                        }
                    }
                }
            }
            self.reset_ticker_lifecycle();
            return TextOverlayRenderState {
                source_text: text.clone(),
                visual_id: String::new(),
                visual_mode: snapshot.visual_mode,
                font_family: snapshot.font,
                font_weight: snapshot.font_weight,
                font_size: snapshot.font_size.max(16),
                clock_text: snapshot.clock_text,
                clock_x: snapshot.clock_x,
                clock_y: snapshot.clock_y,
                clock_font_size: snapshot.clock_font_size,
                clock_color: snapshot.clock_color,
                banner_height: snapshot.ticker_height,
                speed_px_per_frame: snapshot.ticker_speed_px_per_frame.max(1),
                frame_width: feed.video.width.max(1),
                draw_fps: render_draw_fps(feed),
                text_width_px: 0.0,
                started_at: Instant::now(),
                text,
                font_desc,
                ypos,
                x_absolute: 1.0,
                silent: true,
                gradient: snapshot.ticker_gradient,
                sunny_cat: snapshot.sunny_cat,
            };
        }

        let visual_id = if snapshot.visual_id.trim().is_empty() {
            text.clone()
        } else {
            snapshot.visual_id.clone()
        };
        let source_text = snapshot.overlay_text.trim().to_string();
        let key = format!(
            "{}|{}|{}|{}|{}|{}|{}|{}|{}",
            snapshot.visual_mode,
            visual_id,
            snapshot.font,
            snapshot.font_weight,
            snapshot.font_size,
            snapshot.ticker_height,
            snapshot.ticker_speed_px_per_frame,
            feed.banner.scroll_repeat_mode,
            source_text
        );
        if self.retaining_cleared_ticker && snapshot.visual_mode == "ticker_alert" {
            self.start_ticker_lifecycle(key);
        } else if self.key.as_deref() != Some(key.as_str()) {
            self.start_ticker_lifecycle(key);
        } else if let Some(last) = self.last.clone() {
            if snapshot.visual_mode == "ticker_alert" && ticker_state_done(&last) {
                if self.should_repeat_ticker(feed, audio_live) {
                    return self.restart_ticker_pass(last);
                }
                return self.silent_state(feed, snapshot, text, font_desc, ypos);
            }
        }
        let font_size = snapshot.font_size.max(16);
        let started_at = self.started_at.unwrap_or_else(Instant::now);
        let (text, x_absolute, silent) = if snapshot.visual_mode == "ticker_alert" {
            let (x_absolute, _) = ticker_x_absolute(feed, &snapshot, &source_text, started_at);
            (source_text.clone(), x_absolute, false)
        } else {
            ticker_overlay_window(feed, &snapshot, &source_text, started_at)
        };
        TextOverlayRenderState {
            text_width_px: estimated_text_width_px(&source_text, font_size),
            visual_id: self.render_visual_id(&visual_id),
            source_text,
            visual_mode: snapshot.visual_mode,
            font_family: snapshot.font,
            font_weight: snapshot.font_weight,
            font_size,
            clock_text: snapshot.clock_text,
            clock_x: snapshot.clock_x,
            clock_y: snapshot.clock_y,
            clock_font_size: snapshot.clock_font_size,
            clock_color: snapshot.clock_color,
            banner_height: snapshot.ticker_height,
            speed_px_per_frame: snapshot.ticker_speed_px_per_frame.max(1),
            frame_width: feed.video.width.max(1),
            draw_fps: render_draw_fps(feed),
            started_at,
            text,
            font_desc,
            ypos,
            x_absolute,
            silent,
            gradient: snapshot.ticker_gradient,
            sunny_cat: snapshot.sunny_cat,
        }
    }

    fn update_audio_lifecycle(&mut self, audio_live: bool) {
        if audio_live {
            self.audio_was_live = true;
            self.post_audio_repeats_started = 0;
        } else if self.audio_was_live {
            self.audio_was_live = false;
            self.post_audio_repeats_started = 0;
        }
    }

    fn start_ticker_lifecycle(&mut self, key: String) {
        self.key = Some(key);
        self.started_at = Some(Instant::now());
        self.pass_index = 0;
        self.completed_passes = 0;
        self.post_audio_repeats_started = 0;
        self.retaining_cleared_ticker = false;
    }

    fn reset_ticker_lifecycle(&mut self) {
        self.key = None;
        self.started_at = None;
        self.pass_index = 0;
        self.completed_passes = 0;
        self.post_audio_repeats_started = 0;
        self.retaining_cleared_ticker = false;
    }

    fn restart_ticker_pass(&mut self, mut state: TextOverlayRenderState) -> TextOverlayRenderState {
        self.completed_passes = self.completed_passes.saturating_add(1);
        self.pass_index = self.pass_index.saturating_add(1);
        let now = Instant::now();
        self.started_at = Some(now);
        state.started_at = now;
        state.x_absolute = 1.0;
        state.silent = false;
        state.text = state.source_text.clone();
        state.visual_id = self.render_visual_id(&base_visual_id(&state.visual_id));
        state
    }

    fn should_repeat_ticker(&mut self, feed: &FeedConfig, audio_live: bool) -> bool {
        if fixed_scroll_repeats_enabled(feed) {
            let total = feed.banner.fixed_repeats.max(1);
            return self.completed_passes.saturating_add(1) < total;
        }
        if audio_live {
            return true;
        }
        if self.post_audio_repeats_started < feed.banner.after_eom_repeats {
            self.post_audio_repeats_started = self.post_audio_repeats_started.saturating_add(1);
            return true;
        }
        false
    }

    fn render_visual_id(&self, visual_id: &str) -> String {
        if self.pass_index == 0 {
            visual_id.to_string()
        } else {
            format!("{}#pass{}", visual_id, self.pass_index.saturating_add(1))
        }
    }

    fn silent_state(
        &mut self,
        feed: &FeedConfig,
        snapshot: crate::graphics::PresentationSnapshot,
        text: String,
        font_desc: String,
        ypos: f64,
    ) -> TextOverlayRenderState {
        self.reset_ticker_lifecycle();
        TextOverlayRenderState {
            source_text: text.clone(),
            visual_id: String::new(),
            visual_mode: snapshot.visual_mode,
            font_family: snapshot.font,
            font_weight: snapshot.font_weight,
            font_size: snapshot.font_size.max(16),
            clock_text: snapshot.clock_text,
            clock_x: snapshot.clock_x,
            clock_y: snapshot.clock_y,
            clock_font_size: snapshot.clock_font_size,
            clock_color: snapshot.clock_color,
            banner_height: snapshot.ticker_height,
            speed_px_per_frame: snapshot.ticker_speed_px_per_frame.max(1),
            frame_width: feed.video.width.max(1),
            draw_fps: render_draw_fps(feed),
            text_width_px: 0.0,
            started_at: Instant::now(),
            text,
            font_desc,
            ypos,
            x_absolute: 1.0,
            silent: true,
            gradient: snapshot.ticker_gradient,
            sunny_cat: snapshot.sunny_cat,
        }
    }

    fn status_value(&self) -> Value {
        if let Some(state) = &self.last {
            json!({
                "render_text_len": state.text.chars().count(),
                "render_text_preview": state.text.chars().take(160).collect::<String>(),
                "x_absolute": state.x_absolute,
                "ypos": state.ypos,
                "silent": state.silent,
                "sunny_cat": state.sunny_cat,
                "font_desc": state.font_desc,
                "font_weight": state.font_weight,
                "clock_text": state.clock_text,
                "clock_x": state.clock_x,
                "clock_y": state.clock_y,
                "clock_font_size": state.clock_font_size,
                "pass_index": self.pass_index,
                "completed_passes": self.completed_passes,
                "audio_was_live": self.audio_was_live,
                "post_audio_repeats_started": self.post_audio_repeats_started,
            })
        } else {
            json!({
                "render_text_len": 0,
                "render_text_preview": "",
                "x_absolute": 1.0,
                "ypos": 0.08,
                "silent": true,
                "sunny_cat": false,
                "font_desc": "",
                "clock_text": "",
                "clock_x": 48,
                "clock_y": 48,
                "clock_font_size": 30,
                "pass_index": self.pass_index,
                "completed_passes": self.completed_passes,
            })
        }
    }

    fn shared_state(&self) -> Arc<Mutex<Option<TextOverlayRenderState>>> {
        Arc::clone(&self.shared)
    }
}

struct WgpuCompositorSet {
    renderers: Vec<Arc<Mutex<WgpuFrameRenderer>>>,
    _probe_ids: Vec<gst::PadProbeId>,
}

impl WgpuCompositorSet {
    fn status_value(&self) -> Value {
        json!({
            "graphics_backend": "wgpu",
            "renditions": self.renderers.iter().filter_map(|renderer| {
                renderer.lock().ok().map(|renderer| renderer.status_value())
            }).collect::<Vec<_>>(),
        })
    }
}

fn install_wgpu_overlay_probes(
    pipeline: &gst::Pipeline,
    feed: &FeedConfig,
    plan: &GstPipelinePlan,
    base_dir: &Path,
    state: Arc<Mutex<Option<TextOverlayRenderState>>>,
) -> Result<WgpuCompositorSet> {
    let mut renderers = Vec::new();
    let mut probe_ids = Vec::new();
    for (index, video) in plan.videos.iter().enumerate() {
        let overlay_name = gst_element_name("cgen_overlay", &video.id, index);
        let overlay = pipeline
            .by_name(&overlay_name)
            .with_context(|| format!("GStreamer CGEN pipeline is missing {overlay_name}"))?;
        let src_pad = overlay
            .static_pad("src")
            .with_context(|| format!("GStreamer CGEN overlay {overlay_name} has no src pad"))?;
        let renderer = Arc::new(Mutex::new(WgpuFrameRenderer::new(
            video.id.clone(),
            video.width,
            video.height,
            video.interlaced,
        )?));
        let renderer_for_probe = Arc::clone(&renderer);
        let state_for_probe = Arc::clone(&state);
        let preview_writer = if index == 0 {
            Some(Arc::new(Mutex::new(FramePreviewWriter::new(
                base_dir
                    .join("runtime")
                    .join("cgen")
                    .join(format!("{}.preview.jpg", safe_preview_id(&feed.id))),
                video.width,
                video.height,
            ))))
        } else {
            None
        };
        let preview_for_probe = preview_writer.clone();
        let feed_id = feed.id.clone();
        let overlay_for_probe = overlay_name.clone();
        let probe_id = src_pad
            .add_probe(gst::PadProbeType::BUFFER, move |_, info| {
                let state = state_for_probe.lock().ok().and_then(|guard| guard.clone());
                let overlay_active = overlay_render_state_needs_frame_mutation(state.as_ref());
                let preview_due = preview_for_probe
                    .as_ref()
                    .and_then(|preview| preview.lock().ok().map(|preview| preview.write_due()))
                    .unwrap_or(false);
                if !overlay_active && !preview_due {
                    return gst::PadProbeReturn::Ok;
                }
                if let Some(buffer) = info.buffer_mut() {
                    let frame_pts_ns = buffer.pts().map(|pts| pts.nseconds());
                    let buffer = buffer.make_mut();
                    match buffer.map_writable() {
                        Ok(mut map) => {
                            if overlay_active {
                                if let Ok(mut renderer) = renderer_for_probe.lock() {
                                if let Err(err) = renderer.composite_bgrx(map.as_mut_slice(), frame_pts_ns, state.as_ref()) {
                                    warn!(feed_id = %feed_id, overlay = %overlay_for_probe, "wgpu compositor failed: {err:#}");
                                }
                                }
                            }
                            if let Some(preview) = &preview_for_probe {
                                if let Ok(mut preview) = preview.lock() {
                                    if let Err(err) = preview.maybe_write(map.as_slice()) {
                                        warn!(feed_id = %feed_id, overlay = %overlay_for_probe, "failed to write cgen preview frame: {err:#}");
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            warn!(feed_id = %feed_id, overlay = %overlay_for_probe, "failed to map video buffer for wgpu compositor: {err}");
                        }
                    }
                }
                gst::PadProbeReturn::Ok
            })
            .with_context(|| format!("failed to add WGPU probe to {overlay_name}"))?;
        renderers.push(renderer);
        probe_ids.push(probe_id);
    }
    Ok(WgpuCompositorSet {
        renderers,
        _probe_ids: probe_ids,
    })
}

fn overlay_render_state_needs_frame_mutation(state: Option<&TextOverlayRenderState>) -> bool {
    state.is_some_and(|state| {
        state.sunny_cat
            || (!state.silent && !state.text.trim().is_empty())
            || !state.clock_text.trim().is_empty()
    })
}

const FRAME_PREVIEW_INTERVAL: Duration = Duration::from_millis(66);

struct FramePreviewWriter {
    path: PathBuf,
    source_width: u32,
    source_height: u32,
    last_write: Option<Instant>,
}

impl FramePreviewWriter {
    fn new(path: PathBuf, source_width: u32, source_height: u32) -> Self {
        Self {
            path,
            source_width,
            source_height,
            last_write: None,
        }
    }

    fn maybe_write(&mut self, frame: &[u8]) -> Result<()> {
        let now = Instant::now();
        if !self.write_due_at(now) {
            return Ok(());
        }
        let width = usize::try_from(self.source_width).unwrap_or(0);
        let height = usize::try_from(self.source_height).unwrap_or(0);
        let expected = width.saturating_mul(height).saturating_mul(4);
        if width == 0 || height == 0 || frame.len() < expected {
            return Ok(());
        }
        self.last_write = Some(now);
        write_bgrx_preview_jpeg(&self.path, frame, width, height)?;
        Ok(())
    }

    fn write_due(&self) -> bool {
        self.write_due_at(Instant::now())
    }

    fn write_due_at(&self, now: Instant) -> bool {
        self.last_write
            .map(|last| now.saturating_duration_since(last) >= FRAME_PREVIEW_INTERVAL)
            .unwrap_or(true)
    }
}

fn write_bgrx_preview_jpeg(path: &Path, frame: &[u8], width: usize, height: usize) -> Result<()> {
    const MAX_PREVIEW_WIDTH: usize = 320;
    let out_width = width.min(MAX_PREVIEW_WIDTH).max(1);
    let out_height = ((height as f64 * out_width as f64 / width.max(1) as f64).round() as usize)
        .clamp(1, height.max(1));
    let mut rgb = vec![0u8; out_width.saturating_mul(out_height).saturating_mul(3)];
    for out_y in 0..out_height {
        let src_y = out_y.saturating_mul(height) / out_height;
        for out_x in 0..out_width {
            let src_x = out_x.saturating_mul(width) / out_width;
            let src = (src_y * width + src_x) * 4;
            let dst = (out_y * out_width + out_x) * 3;
            rgb[dst] = frame[src + 2];
            rgb[dst + 1] = frame[src + 1];
            rgb[dst + 2] = frame[src];
        }
    }
    let mut jpeg = Vec::with_capacity(rgb.len().saturating_div(3));
    let encoder = jpeg_encoder::Encoder::new(&mut jpeg, 68);
    encoder
        .encode(
            &rgb,
            out_width as u16,
            out_height as u16,
            jpeg_encoder::ColorType::Rgb,
        )
        .context("failed to encode cgen preview jpeg")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).context("failed to create cgen preview directory")?;
    }
    let tmp = path.with_extension("preview.jpg.tmp");
    fs::write(&tmp, jpeg).context("failed to write cgen preview temp file")?;
    fs::rename(&tmp, path).context("failed to replace cgen preview file")?;
    Ok(())
}

fn safe_preview_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn ticker_state_done(state: &TextOverlayRenderState) -> bool {
    let frame_width = f64::from(state.frame_width.max(1));
    let pixels_per_second = crate::graphics::ticker_speed_px_per_second(state.speed_px_per_frame);
    let elapsed = state.started_at.elapsed().as_secs_f64();
    let x_px = frame_width + 1.0 - (elapsed * pixels_per_second);
    let text_width_px =
        ticker_required_text_width_px(&state.source_text, state.font_size, state.text_width_px);
    x_px < -ticker_exit_width_px(text_width_px, frame_width)
}

fn fixed_scroll_repeats_enabled(feed: &FeedConfig) -> bool {
    matches!(
        feed.banner
            .scroll_repeat_mode
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "fixed" | "fixed_repeats" | "count" | "count_only"
    )
}

fn base_visual_id(value: &str) -> String {
    value
        .split_once("#pass")
        .map(|(base, _)| base)
        .unwrap_or(value)
        .to_string()
}

#[cfg(test)]
fn text_overlay_state(feed: &FeedConfig, state: &RuntimeState) -> TextOverlayRenderState {
    TextOverlayController::default().next_state(feed, state, false, false)
}

fn ticker_x_absolute(
    feed: &FeedConfig,
    snapshot: &crate::graphics::PresentationSnapshot,
    text: &str,
    started_at: Instant,
) -> (f64, bool) {
    if snapshot.visual_mode == "fullscreen_alert" || snapshot.visual_mode == "overlay" {
        return (0.5, false);
    }
    let frame_width = f64::from(feed.video.width.max(1));
    let text_width = ticker_required_text_width_px(
        text,
        snapshot.font_size,
        estimated_text_width_px(text, snapshot.font_size),
    );
    let pixels_per_second =
        crate::graphics::ticker_speed_px_per_second(snapshot.ticker_speed_px_per_frame);
    let elapsed = ticker_elapsed_seconds(feed, started_at);
    let x_px = frame_width + 1.0 - (elapsed * pixels_per_second);
    let done = x_px < -ticker_exit_width_px(text_width, frame_width);
    if done {
        return (-1.0, true);
    }
    (x_px / frame_width, done)
}

fn ticker_exit_width_px(text_width_px: f64, frame_width_px: f64) -> f64 {
    let tail_padding = (frame_width_px.max(1.0) * 0.15).clamp(96.0, 480.0);
    text_width_px.max(1.0) + tail_padding
}

fn ticker_required_text_width_px(text: &str, font_size: u32, configured_width_px: f64) -> f64 {
    let chars = text.chars().count().max(1) as f64;
    configured_width_px
        .max(estimated_text_width_px(text, font_size))
        .max(chars * f64::from(font_size.max(16)) * 0.92)
        .max(1.0)
}

fn ticker_elapsed_seconds(feed: &FeedConfig, started_at: Instant) -> f64 {
    let _ = feed;
    started_at.elapsed().as_secs_f64()
}

fn ticker_overlay_window(
    feed: &FeedConfig,
    snapshot: &crate::graphics::PresentationSnapshot,
    text: &str,
    started_at: Instant,
) -> (String, f64, bool) {
    let (x_absolute, done) = ticker_x_absolute(feed, snapshot, text, started_at);
    if done {
        return (String::new(), -1.0, true);
    }
    if snapshot.visual_mode == "fullscreen_alert" || snapshot.visual_mode == "overlay" {
        return (text.to_string(), x_absolute, false);
    }

    let frame_width = f64::from(feed.video.width.max(1));
    let char_count = text.chars().count();
    if char_count == 0 {
        return (String::new(), 1.0, true);
    }
    let text_width = estimated_text_width_px(text, snapshot.font_size);
    let char_width = (text_width / char_count as f64).max(8.0);
    let x_px = x_absolute * frame_width;
    let first_visible_px = (-x_px).max(0.0);
    let first_char = ((first_visible_px / char_width).floor() as usize).saturating_sub(4);
    let char_offset_px = first_char as f64 * char_width;
    let window_left_px = x_px + char_offset_px;
    let visible_width_px = (frame_width - window_left_px).max(frame_width);
    let max_chars = ((visible_width_px / char_width).ceil() as usize)
        .saturating_add(24)
        .min(char_count.saturating_sub(first_char));
    let window_text = text.chars().skip(first_char).take(max_chars).collect();
    (window_text, window_left_px / frame_width, false)
}

fn estimated_text_width_px(text: &str, font_size: u32) -> f64 {
    let font_size = f64::from(font_size.max(16));
    let mut units: f64 = 0.0;
    for ch in text.chars() {
        units += if ch.is_whitespace() {
            0.38
        } else if ch.is_ascii_punctuation() {
            0.42
        } else if ch.is_ascii_lowercase() {
            0.58
        } else if ch.is_ascii_digit() {
            0.60
        } else {
            0.74
        };
    }
    units.max(1.0) * font_size * 1.12
}

fn feed_fps(feed: &FeedConfig) -> f64 {
    let fps = feed.video.fps.trim();
    if fps.is_empty() || fps.eq_ignore_ascii_case("source") {
        return 30.0;
    }
    if let Some((num, den)) = fps.split_once('/') {
        let num = num.trim().parse::<f64>().unwrap_or(30.0);
        let den = den.trim().parse::<f64>().unwrap_or(1.0);
        return if den <= 0.0 { 30.0 } else { num / den };
    }
    fps.parse::<f64>().unwrap_or(30.0).max(1.0)
}

fn render_draw_fps(feed: &FeedConfig) -> f64 {
    feed_fps(feed)
}

fn overlay_ypos(feed: &FeedConfig, ticker_y: i32) -> f64 {
    if feed.video.height == 0 {
        return 0.08;
    }
    (f64::from(ticker_y.max(0)) / f64::from(feed.video.height)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AudioSelectorMode {
    Silence,
    Program,
    Priority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VideoSelectorMode {
    Program,
    Standby,
    Black,
    Smpte,
}

impl VideoSelectorMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Program => "program",
            Self::Standby => "standby",
            Self::Black => "black",
            Self::Smpte => "smpte",
        }
    }

    fn pad_name(self) -> &'static str {
        match self {
            Self::Program => "sink_0",
            Self::Standby | Self::Black => "sink_1",
            Self::Smpte => "sink_2",
        }
    }

    fn no_signal(self) -> bool {
        matches!(self, Self::Standby)
    }
}

impl AudioSelectorMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Silence => "silence",
            Self::Program => "program",
            Self::Priority => "priority",
        }
    }

    fn pad_name(self) -> &'static str {
        match self {
            Self::Silence => "sink_0",
            Self::Program => "sink_1",
            Self::Priority => "sink_2",
        }
    }

    fn status_text(self, priority_active: bool) -> &'static str {
        match (self, priority_active) {
            (Self::Priority, true) => "priority",
            (Self::Silence, true) => "priority_silence",
            (Self::Silence, false) => "standby_silence",
            (Self::Program, _) => "program",
            (Self::Priority, false) => "priority",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct InputHealth {
    last_video_frame: Option<Instant>,
    last_audio_frame: Option<Instant>,
    video_timed_out: bool,
    audio_timed_out: bool,
}

impl Default for InputHealth {
    fn default() -> Self {
        Self {
            last_video_frame: None,
            last_audio_frame: None,
            video_timed_out: true,
            audio_timed_out: true,
        }
    }
}

impl InputHealth {
    fn mark_video_frame(&mut self) {
        self.last_video_frame = Some(Instant::now());
        self.video_timed_out = false;
    }

    fn mark_audio_frame(&mut self) {
        self.last_audio_frame = Some(Instant::now());
        self.audio_timed_out = false;
    }

    fn mark_timeout(&mut self) {
        self.video_timed_out = true;
        self.audio_timed_out = true;
    }

    fn video_connected(self, feed: &FeedConfig) -> bool {
        self.stream_connected(self.last_video_frame, self.video_timed_out, feed)
    }

    fn audio_connected(self, feed: &FeedConfig) -> bool {
        self.stream_connected(self.last_audio_frame, self.audio_timed_out, feed)
    }

    fn stream_connected(
        self,
        last_frame: Option<Instant>,
        timed_out: bool,
        feed: &FeedConfig,
    ) -> bool {
        if timed_out {
            return false;
        }
        let Some(last_frame) = last_frame else {
            return false;
        };
        let timeout = Duration::from_millis(u64::from(
            feed.sync
                .hard_reset_ms
                .max(feed.sync.source_buffer_ms)
                .clamp(100, 10_000),
        ));
        last_frame.elapsed() <= timeout
    }
}

fn input_video_connected(input_health: &Arc<Mutex<InputHealth>>, feed: &FeedConfig) -> bool {
    input_health
        .lock()
        .map(|health| health.video_connected(feed))
        .unwrap_or(false)
}

fn input_audio_connected(input_health: &Arc<Mutex<InputHealth>>, feed: &FeedConfig) -> bool {
    input_health
        .lock()
        .map(|health| health.audio_connected(feed))
        .unwrap_or(false)
}

fn input_health_status(input_health: &Arc<Mutex<InputHealth>>, feed: &FeedConfig) -> Value {
    match input_health.lock() {
        Ok(health) => {
            let video_age_ms = health
                .last_video_frame
                .map(|instant| u64::try_from(instant.elapsed().as_millis()).unwrap_or(u64::MAX));
            let audio_age_ms = health
                .last_audio_frame
                .map(|instant| u64::try_from(instant.elapsed().as_millis()).unwrap_or(u64::MAX));
            json!({
                "connected": health.video_connected(feed) && health.audio_connected(feed),
                "video_connected": health.video_connected(feed),
                "audio_connected": health.audio_connected(feed),
                "video_timed_out": health.video_timed_out,
                "audio_timed_out": health.audio_timed_out,
                "last_program_frame_age_ms": video_age_ms,
                "last_video_frame_age_ms": video_age_ms,
                "last_audio_frame_age_ms": audio_age_ms,
            })
        }
        Err(_) => json!({
            "connected": false,
            "video_connected": false,
            "audio_connected": false,
            "video_timed_out": true,
            "audio_timed_out": true,
            "last_error": "input health lock poisoned",
        }),
    }
}

fn mark_input_timeout(input_health: &Arc<Mutex<InputHealth>>) {
    if let Ok(mut health) = input_health.lock() {
        health.mark_timeout();
    }
}

fn install_program_video_probe(
    pipeline: &gst::Pipeline,
    input_health: Arc<Mutex<InputHealth>>,
) -> Result<()> {
    let selector = pipeline
        .by_name("video_selector")
        .context("GStreamer CGEN pipeline is missing video_selector")?;
    let pad = selector
        .static_pad(VideoSelectorMode::Program.pad_name())
        .context("GStreamer CGEN video selector missing program sink")?;
    pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, _info| {
        if let Ok(mut health) = input_health.lock() {
            health.mark_video_frame();
        }
        gst::PadProbeReturn::Ok
    });
    Ok(())
}

fn install_program_audio_probe(
    pipeline: &gst::Pipeline,
    input_health: Arc<Mutex<InputHealth>>,
) -> Result<()> {
    let selector = pipeline
        .by_name("audio_selector")
        .context("GStreamer CGEN pipeline is missing audio_selector")?;
    let pad = selector
        .static_pad(AudioSelectorMode::Program.pad_name())
        .context("GStreamer CGEN audio selector missing program sink")?;
    pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, _info| {
        if let Ok(mut health) = input_health.lock() {
            health.mark_audio_frame();
        }
        gst::PadProbeReturn::Ok
    });
    Ok(())
}

fn desired_video_selector_mode(
    feed: &FeedConfig,
    state: &RuntimeState,
    input_connected: bool,
) -> VideoSelectorMode {
    let control = state.control_for(feed.id.as_str());
    let mode = control
        .and_then(|control| control.string_field("mode"))
        .unwrap_or(feed.state.mode.as_str());
    let smpte_bars = control
        .and_then(|control| control.bool_field("smpte_bars"))
        .unwrap_or(feed.state.smpte_bars);
    if smpte_bars || mode.eq_ignore_ascii_case("smpte") {
        VideoSelectorMode::Smpte
    } else if mode.eq_ignore_ascii_case("black") {
        VideoSelectorMode::Black
    } else if mode.eq_ignore_ascii_case("standby") || !input_connected {
        VideoSelectorMode::Standby
    } else {
        VideoSelectorMode::Program
    }
}

#[cfg(test)]
fn desired_audio_selector_mode(
    feed: &FeedConfig,
    state: &RuntimeState,
    input_connected: bool,
) -> AudioSelectorMode {
    let status = if let Some(audio) = priority_audio_for_feed(feed, state) {
        if audio.audio_path.is_some() {
            PriorityAudioSourceStatus {
                status: "active",
                queue_id: Some(audio.queue_id.clone()),
                error: None,
            }
        } else {
            PriorityAudioSourceStatus {
                status: "missing-path",
                queue_id: Some(audio.queue_id.clone()),
                error: Some("priority audio event has no audio_path".to_string()),
            }
        }
    } else {
        PriorityAudioSourceStatus {
            status: "idle",
            queue_id: None,
            error: None,
        }
    };
    desired_audio_selector_mode_with_status(feed, input_connected, &status)
}

fn desired_audio_selector_mode_with_status(
    feed: &FeedConfig,
    input_connected: bool,
    feeder_status: &PriorityAudioSourceStatus,
) -> AudioSelectorMode {
    if feeder_status.is_live_priority() {
        AudioSelectorMode::Priority
    } else if feeder_status.forces_silence() {
        AudioSelectorMode::Silence
    } else if !feed.priority_input.routine_audio_enabled() {
        AudioSelectorMode::Silence
    } else if !input_connected || mute_standby_routine_applies(feed) {
        AudioSelectorMode::Silence
    } else {
        AudioSelectorMode::Program
    }
}

fn mute_standby_routine_applies(feed: &FeedConfig) -> bool {
    feed.audio.mute_standby_routine && !feed.audio.idle.eq_ignore_ascii_case("source")
}

fn priority_audio_active(feed: &FeedConfig, state: &RuntimeState) -> bool {
    if !feed.priority_input.priority_audio_enabled() {
        return false;
    }
    let feed_id = if feed.priority_input.feed_id.trim().is_empty() {
        feed.id.as_str()
    } else {
        feed.priority_input.feed_id.as_str()
    };
    state.priority_audio_for(feed_id).is_some()
}

fn priority_audio_for_feed<'a>(
    feed: &FeedConfig,
    state: &'a RuntimeState,
) -> Option<&'a crate::state::PriorityAudio> {
    if !feed.priority_input.priority_audio_enabled() {
        return None;
    }
    let feed_id = if feed.priority_input.feed_id.trim().is_empty() {
        feed.id.as_str()
    } else {
        feed.priority_input.feed_id.as_str()
    };
    state.priority_audio_for(feed_id)
}

fn set_video_selector_mode(pipeline: &gst::Pipeline, mode: VideoSelectorMode) -> Result<()> {
    let selector = pipeline
        .by_name("video_selector")
        .context("GStreamer CGEN pipeline is missing video_selector")?;
    let pad = selector
        .static_pad(mode.pad_name())
        .with_context(|| format!("GStreamer CGEN video selector missing {}", mode.pad_name()))?;
    selector.set_property("active-pad", &pad);
    Ok(())
}

fn set_audio_selector_mode(pipeline: &gst::Pipeline, mode: AudioSelectorMode) -> Result<()> {
    let selector = pipeline
        .by_name("audio_selector")
        .context("GStreamer CGEN pipeline is missing audio_selector")?;
    let pad = selector
        .static_pad(mode.pad_name())
        .with_context(|| format!("GStreamer CGEN audio selector missing {}", mode.pad_name()))?;
    selector.set_property("active-pad", &pad);
    Ok(())
}

fn priority_audio_appsrc(pipeline: &gst::Pipeline) -> Result<gst_app::AppSrc> {
    pipeline
        .by_name("priority_audio_src")
        .context("GStreamer CGEN pipeline is missing priority_audio_src")?
        .downcast::<gst_app::AppSrc>()
        .map_err(|_| anyhow::anyhow!("priority_audio_src is not an appsrc"))
}

fn apply_mpegts_program_map(pipeline: &gst::Pipeline, plan: &GstPipelinePlan) -> Result<()> {
    if plan.mux_kind != MuxKind::MpegTs {
        return Ok(());
    }
    let mux = pipeline
        .by_name("mux")
        .context("GStreamer CGEN pipeline is missing mpegts mux")?;
    let structure = plan
        .program_map
        .parse::<gst::Structure>()
        .with_context(|| format!("failed to parse mpegts program map {}", plan.program_map))?;
    mux.set_property("prog-map", &structure);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PriorityAudioSourceStatus {
    status: &'static str,
    queue_id: Option<String>,
    error: Option<String>,
}

impl PriorityAudioSourceStatus {
    fn is_live_priority(&self) -> bool {
        matches!(self.status, "active" | "padding")
    }

    fn forces_silence(&self) -> bool {
        matches!(
            self.status,
            "silence" | "missing-path" | "finished" | "error"
        )
    }
}

struct PriorityAudioFeeder {
    appsrc: gst_app::AppSrc,
    base_dir: PathBuf,
    feed_id: String,
    eof_drain: Duration,
    active: Option<ActivePriorityAudioFeeder>,
    completed: BTreeMap<String, Instant>,
}

struct ActivePriorityAudioFeeder {
    queue_id: String,
    source_status: &'static str,
    started_at: Instant,
    release_after: Duration,
    stop: Arc<AtomicBool>,
    reached_eof: Arc<AtomicBool>,
    reached_eof_at: Arc<Mutex<Option<Instant>>>,
    finished: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    worker: Option<thread::JoinHandle<()>>,
}

impl PriorityAudioFeeder {
    fn new(
        appsrc: gst_app::AppSrc,
        base_dir: PathBuf,
        feed_id: String,
        eof_drain: Duration,
    ) -> Self {
        Self {
            appsrc,
            base_dir,
            feed_id,
            eof_drain,
            active: None,
            completed: BTreeMap::new(),
        }
    }

    fn update(&mut self, audio: Option<&crate::state::PriorityAudio>) -> PriorityAudioSourceStatus {
        self.prune_completed();
        let Some(audio) = audio else {
            if self.active.as_ref().is_some_and(|active| {
                active.source_status != "silence" && !self.active_ready_to_release(active)
            }) {
                return self.active_status();
            }
            self.stop();
            return PriorityAudioSourceStatus {
                status: "idle",
                queue_id: None,
                error: None,
            };
        };
        if self
            .active
            .as_ref()
            .is_some_and(|active| active.queue_id == audio.queue_id)
        {
            if self
                .active
                .as_ref()
                .is_some_and(|active| self.active_ready_to_release(active))
            {
                self.completed
                    .insert(audio.queue_id.clone(), Instant::now());
                self.stop();
                return PriorityAudioSourceStatus {
                    status: "idle",
                    queue_id: None,
                    error: None,
                };
            }
            return self.active_status();
        }
        if self.completed.contains_key(&audio.queue_id) {
            return PriorityAudioSourceStatus {
                status: "idle",
                queue_id: None,
                error: None,
            };
        }
        self.stop();
        let Some(path) = audio.audio_path.as_ref() else {
            self.active = Some(self.silence_active(
                audio.queue_id.clone(),
                "priority audio event has no audio_path",
            ));
            return self.active_status();
        };

        let sample_rate = audio.sample_rate.max(8_000);
        let channels = audio.channels.clamp(1, 6);
        let path = resolve_media_path(&self.base_dir, path);
        if !path.is_file() {
            warn!(
                feed_id = %self.feed_id,
                queue_id = %audio.queue_id,
                path = %path.display(),
                "priority audio file is missing; forcing alert silence"
            );
            self.active =
                Some(self.silence_active(audio.queue_id.clone(), "priority audio file is missing"));
            return self.active_status();
        }
        let caps = gst::Caps::builder("audio/x-raw")
            .field("format", "S16LE")
            .field("rate", i32::try_from(sample_rate).unwrap_or(48_000))
            .field("channels", i32::from(channels))
            .field("layout", "interleaved")
            .build();
        self.appsrc.set_caps(Some(&caps));
        self.appsrc.set_format(gst::Format::Time);
        self.appsrc.set_is_live(true);

        let stop = Arc::new(AtomicBool::new(false));
        let reached_eof = Arc::new(AtomicBool::new(false));
        let reached_eof_at = Arc::new(Mutex::new(None));
        let finished = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let release_after = Duration::from_millis(
            audio
                .duration_ms
                .unwrap_or(0)
                .saturating_add(self.eof_drain.as_millis().try_into().unwrap_or(u64::MAX))
                .saturating_add(1_000),
        )
        .max(self.eof_drain.saturating_add(Duration::from_secs(2)));
        let worker = spawn_priority_audio_feeder(
            self.appsrc.clone(),
            self.feed_id.clone(),
            audio.queue_id.clone(),
            path,
            sample_rate,
            channels,
            Arc::clone(&stop),
            Arc::clone(&reached_eof),
            Arc::clone(&reached_eof_at),
            Arc::clone(&finished),
            Arc::clone(&error),
        );
        self.active = Some(ActivePriorityAudioFeeder {
            queue_id: audio.queue_id.clone(),
            source_status: "active",
            started_at: Instant::now(),
            release_after,
            stop,
            reached_eof,
            reached_eof_at,
            finished,
            error,
            worker: Some(worker),
        });
        self.active_status()
    }

    fn silence_active(&self, queue_id: String, reason: &str) -> ActivePriorityAudioFeeder {
        ActivePriorityAudioFeeder {
            queue_id,
            source_status: "silence",
            started_at: Instant::now(),
            release_after: Duration::ZERO,
            stop: Arc::new(AtomicBool::new(false)),
            reached_eof: Arc::new(AtomicBool::new(false)),
            reached_eof_at: Arc::new(Mutex::new(None)),
            finished: Arc::new(AtomicBool::new(false)),
            error: Arc::new(Mutex::new(Some(reason.to_string()))),
            worker: None,
        }
    }

    fn active_ready_to_release(&self, active: &ActivePriorityAudioFeeder) -> bool {
        if active.source_status == "silence" {
            return false;
        }
        if active.finished.load(Ordering::Relaxed) {
            return true;
        }
        if active.started_at.elapsed() >= active.release_after {
            return true;
        }
        active
            .reached_eof_at
            .lock()
            .ok()
            .and_then(|guard| *guard)
            .is_some_and(|at| at.elapsed() >= self.eof_drain)
    }

    fn stop(&mut self) {
        if let Some(mut active) = self.active.take() {
            active.stop.store(true, Ordering::Relaxed);
            if let Some(worker) = active.worker.take() {
                if worker.join().is_err() {
                    warn!(
                        feed_id = %self.feed_id,
                        queue_id = %active.queue_id,
                        "priority audio feeder thread panicked during stop"
                    );
                }
            }
        }
    }

    fn prune_completed(&mut self) {
        let now = Instant::now();
        self.completed.retain(|_, completed_at| {
            now.duration_since(*completed_at) < PRIORITY_AUDIO_RESTART_SUPPRESS
        });
    }

    fn active_status(&self) -> PriorityAudioSourceStatus {
        let Some(active) = self.active.as_ref() else {
            return PriorityAudioSourceStatus {
                status: "idle",
                queue_id: None,
                error: None,
            };
        };
        let error = active.error.lock().ok().and_then(|guard| guard.clone());
        let status = if error.is_some() {
            if active.source_status == "silence" {
                "silence"
            } else {
                "error"
            }
        } else if active.reached_eof.load(Ordering::Relaxed) {
            "padding"
        } else if active.finished.load(Ordering::Relaxed) {
            "finished"
        } else {
            active.source_status
        };
        PriorityAudioSourceStatus {
            status,
            queue_id: Some(active.queue_id.clone()),
            error,
        }
    }
}

impl Drop for PriorityAudioFeeder {
    fn drop(&mut self) {
        self.stop();
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_priority_audio_feeder(
    appsrc: gst_app::AppSrc,
    feed_id: String,
    queue_id: String,
    path: PathBuf,
    sample_rate: u32,
    channels: u16,
    stop: Arc<AtomicBool>,
    reached_eof: Arc<AtomicBool>,
    reached_eof_at: Arc<Mutex<Option<Instant>>>,
    finished: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        if let Err(err) = run_priority_audio_feeder(
            &appsrc,
            &path,
            sample_rate,
            channels,
            Arc::clone(&stop),
            Arc::clone(&reached_eof),
            Arc::clone(&reached_eof_at),
        ) {
            warn!(
                feed_id = %feed_id,
                queue_id = %queue_id,
                path = %path.display(),
                "priority audio feeder failed: {err:#}"
            );
            if let Ok(mut guard) = error.lock() {
                *guard = Some(err.to_string());
            }
        }
        finished.store(true, Ordering::Relaxed);
    })
}

fn run_priority_audio_feeder(
    appsrc: &gst_app::AppSrc,
    path: &Path,
    sample_rate: u32,
    channels: u16,
    stop: Arc<AtomicBool>,
    reached_eof: Arc<AtomicBool>,
    reached_eof_at: Arc<Mutex<Option<Instant>>>,
) -> Result<()> {
    let mut file = File::open(path)
        .with_context(|| format!("failed to open priority audio {}", path.display()))?;
    let chunk_bytes = pcm_chunk_bytes(sample_rate, channels, 20);
    let mut chunk = vec![0u8; chunk_bytes];
    let mut eof = false;
    let chunk_period = Duration::from_millis(20);
    let mut next_deadline = Instant::now() + chunk_period;
    while !stop.load(Ordering::Relaxed) {
        let count = if eof {
            0
        } else {
            match file.read(&mut chunk) {
                Ok(0) => {
                    eof = true;
                    reached_eof.store(true, Ordering::Relaxed);
                    if let Ok(mut guard) = reached_eof_at.lock() {
                        if guard.is_none() {
                            *guard = Some(Instant::now());
                        }
                    }
                    0
                }
                Ok(count) => count,
                Err(err) => {
                    return Err(err).with_context(|| {
                        format!("failed to read priority audio {}", path.display())
                    })
                }
            }
        };
        let payload = if count == 0 {
            vec![0u8; chunk_bytes]
        } else {
            chunk[..count].to_vec()
        };
        let duration = pcm_duration(sample_rate, channels, payload.len());
        let mut buffer = gst::Buffer::from_mut_slice(payload);
        buffer.make_mut().set_duration(duration);
        match appsrc.push_buffer(buffer) {
            Ok(_) => {}
            Err(gst::FlowError::Flushing | gst::FlowError::Eos) => return Ok(()),
            Err(err) => bail!("failed to push priority audio to appsrc: {err:?}"),
        }
        let now = Instant::now();
        if now < next_deadline {
            thread::sleep(next_deadline - now);
        }
        while next_deadline <= Instant::now() {
            next_deadline += chunk_period;
        }
    }
    Ok(())
}

fn resolve_media_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn pcm_chunk_bytes(sample_rate: u32, channels: u16, millis: u32) -> usize {
    let frames = sample_rate.saturating_mul(millis).div_ceil(1000).max(1);
    usize::try_from(frames)
        .unwrap_or(960)
        .saturating_mul(usize::from(channels.max(1)))
        .saturating_mul(2)
}

fn pcm_duration(sample_rate: u32, channels: u16, bytes: usize) -> gst::ClockTime {
    let frame_bytes = usize::from(channels.max(1)).saturating_mul(2).max(1);
    let frames = bytes / frame_bytes;
    let nanos =
        (u128::try_from(frames).unwrap_or(0) * 1_000_000_000u128) / u128::from(sample_rate.max(1));
    gst::ClockTime::from_nseconds(u64::try_from(nanos).unwrap_or(u64::MAX))
}

fn publish_status(
    status_tx: &Option<mpsc::UnboundedSender<Value>>,
    feed: &FeedConfig,
    state_rx: &watch::Receiver<RuntimeState>,
    no_signal: bool,
    mut data: Value,
) {
    let compositor = crate::graphics::compositor_status(feed, &state_rx.borrow(), no_signal);
    if let (Some(target), Some(source)) = (data.as_object_mut(), compositor.as_object()) {
        for (key, value) in source {
            target.insert(key.clone(), value.clone());
        }
    }
    if let Some(target) = data.as_object_mut() {
        target.insert("feed_id".to_string(), Value::String(feed.id.clone()));
    }
    crate::config::redact_feed_endpoint_status(feed, &mut data);
    if let Some(tx) = status_tx {
        let _ = tx.send(data);
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
struct PipelineDiagnostics {
    warning_count: u64,
    qos_count: u64,
    latency_recalculation_count: u64,
    last_warning: Option<String>,
    last_warning_source: Option<String>,
    last_qos: Option<QosSnapshot>,
    last_latency_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
struct QosSnapshot {
    source: String,
    live: bool,
    jitter_ns: i64,
    proportion: f64,
    quality: i32,
    processed: String,
    dropped: String,
}

impl PipelineDiagnostics {
    fn record_warning(&mut self, source: String, warning: &gst::message::Warning) {
        self.warning_count = self.warning_count.saturating_add(1);
        let debug = warning
            .debug()
            .map(|debug| debug.to_string())
            .unwrap_or_default();
        self.last_warning = Some(if debug.is_empty() {
            warning.error().to_string()
        } else {
            format!("{} {debug}", warning.error())
        });
        self.last_warning_source = Some(source);
    }

    fn record_latency_recalculation(&mut self, error: Option<String>) {
        self.latency_recalculation_count = self.latency_recalculation_count.saturating_add(1);
        self.last_latency_error = error;
    }

    fn record_qos(&mut self, source: String, qos: &gst::message::Qos) {
        self.qos_count = self.qos_count.saturating_add(1);
        let (jitter_ns, proportion, quality) = qos.values();
        self.last_qos = Some(QosSnapshot {
            source,
            live: qos.live(),
            jitter_ns,
            proportion,
            quality,
            processed: qos.processed().to_string(),
            dropped: qos.dropped().to_string(),
        });
    }

    fn status_value(&self) -> Value {
        json!({
            "warning_count": self.warning_count,
            "qos_count": self.qos_count,
            "latency_recalculation_count": self.latency_recalculation_count,
            "last_warning": self.last_warning,
            "last_warning_source": self.last_warning_source,
            "last_latency_error": self.last_latency_error,
            "last_qos": self.last_qos.as_ref().map(QosSnapshot::status_value),
        })
    }
}

impl QosSnapshot {
    fn status_value(&self) -> Value {
        json!({
            "source": self.source,
            "live": self.live,
            "jitter_ns": self.jitter_ns,
            "proportion": self.proportion,
            "quality": self.quality,
            "processed": self.processed,
            "dropped": self.dropped,
        })
    }
}

fn message_src_path(message: &gst::Message) -> String {
    message
        .src()
        .map(|src| src.path_string())
        .unwrap_or_else(|| "unknown".into())
        .to_string()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GstPipelinePlan {
    description: String,
    videos: Vec<PlannedVideoRendition>,
    audios: Vec<PlannedAudioRendition>,
    program_map: String,
    mux_kind: MuxKind,
    required_elements: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedVideoRendition {
    id: String,
    program: i32,
    video_pid: i32,
    pmt_pid: i32,
    width: u32,
    height: u32,
    fps: String,
    interlaced: bool,
    field_order: String,
    standard: String,
    codec: String,
    bitrate_kbps: Option<u32>,
    encoder: EncoderCodecSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedAudioRendition {
    id: String,
    program: i32,
    audio_pid: i32,
    channels: u16,
    codec: String,
    bitrate_bps: i64,
    language: String,
    encoder: EncoderCodecSettings,
}

impl GstPipelinePlan {
    fn from_feed(feed: &FeedConfig) -> Result<Self> {
        let source = InputSourceFragment::from_url(feed.program_input_url(), feed)?;
        let sink = SinkFragment::from_url(feed.program_output_url(), feed)?;
        let mut videos = feed.enabled_video_renditions(feed.video.width, feed.video.height);
        let mut audios = feed.enabled_audio_renditions();
        if sink.mux_kind == MuxKind::Flv {
            videos.truncate(1);
            audios.truncate(1);
        }
        let planned_videos = videos
            .iter()
            .enumerate()
            .map(|(index, video)| PlannedVideoRendition::from_config(feed, video, index))
            .collect::<Vec<_>>();
        let planned_audios = planned_videos
            .iter()
            .flat_map(|video| {
                audios.iter().enumerate().map(|(audio_index, audio)| {
                    PlannedAudioRendition::from_config(
                        feed,
                        audio,
                        video.program,
                        audio_pid_for(video.video_pid, audio_index),
                    )
                })
            })
            .collect::<Vec<_>>();
        let video_input = source.video_branch();
        let audio_input = source.audio_branch();
        let mux = mux_fragment(feed, sink.mux_kind);
        let queue = queue_fragment(feed, QueueLeak::None);
        let video_live_queue = queue_fragment(feed, QueueLeak::Downstream);
        let audio_live_queue = queue_fragment(feed, QueueLeak::Downstream);
        let priority_max_bytes = priority_appsrc_max_bytes(feed);
        let priority_max_latency = queue_time_ns(feed);
        let standby_caps = source_video_caps_fragment(feed);
        let video_branches = videos
            .iter()
            .enumerate()
            .map(|(index, video)| {
                video_rendition_branch(feed, video, &planned_videos[index], index, sink.mux_kind)
            })
            .collect::<Vec<_>>()
            .join(" ");
        let audio_branches = planned_audios
            .iter()
            .enumerate()
            .map(|(index, audio)| audio_rendition_branch(feed, audio, index, sink.mux_kind))
            .collect::<Vec<_>>()
            .join(" ");
        let description = format!(
            "{} \
             {video_input} ! {video_live_queue} ! videoconvert ! videoscale ! videorate ! {standby_caps} ! video_selector.sink_0 \
             videotestsrc name=standby_video_src pattern=black is-live=true do-timestamp=true ! {standby_caps} ! {queue} ! video_selector.sink_1 \
             videotestsrc name=smpte_video_src pattern=smpte is-live=true do-timestamp=true ! {standby_caps} ! {queue} ! video_selector.sink_2 \
             input-selector name=video_selector sync-streams=true sync-mode=clock cache-buffers=false drop-backwards=true ! {queue} ! tee name=video_tee \
             input-selector name=audio_selector sync-streams=true sync-mode=clock cache-buffers=false drop-backwards=true ! {queue} ! audioconvert ! audioresample ! audiorate ! audio/x-raw,rate=48000,layout=interleaved ! tee name=audio_tee \
             audiotestsrc wave=silence is-live=true do-timestamp=true ! audio/x-raw,format=S16LE,rate=48000,channels=2,layout=interleaved ! {queue} ! audio_selector.sink_0 \
             {audio_input} ! {audio_live_queue} ! audioconvert ! audioresample ! audio/x-raw,rate=48000,layout=interleaved ! {queue} ! audio_selector.sink_1 \
             appsrc name=priority_audio_src is-live=true block=false leaky-type=downstream stream-type=stream format=time do-timestamp=true min-latency=0 max-latency={priority_max_latency} max-bytes={priority_max_bytes} ! {queue} ! audioconvert ! audioresample ! audio/x-raw,rate=48000,layout=interleaved ! {queue} ! audio_selector.sink_2 \
             {video_branches} \
             {audio_branches} \
             {mux} ! {queue} ! {}"
            , source.description
            , sink.description
        );
        let program_map = program_map_fragment(&planned_videos, &planned_audios);
        let required_elements = required_elements(&source, &sink, &planned_videos, &planned_audios);
        Ok(Self {
            description,
            videos: planned_videos,
            audios: planned_audios,
            program_map,
            mux_kind: sink.mux_kind,
            required_elements,
        })
    }

    fn status_value(&self) -> Value {
        json!({
            "videos": self.videos.iter().map(PlannedVideoRendition::status_value).collect::<Vec<_>>(),
            "audios": self.audios.iter().map(PlannedAudioRendition::status_value).collect::<Vec<_>>(),
            "video_count": self.videos.len(),
            "audio_count": self.audios.len(),
            "program_map": self.program_map,
            "required_elements": self.required_elements,
        })
    }
}

fn validate_gstreamer_elements(elements: &[String]) -> Result<()> {
    let missing = elements
        .iter()
        .filter(|name| gst::ElementFactory::find(name.as_str()).is_none())
        .cloned()
        .collect::<Vec<String>>();
    if missing.is_empty() {
        return Ok(());
    }
    bail!(
        "missing required GStreamer element(s): {}",
        missing.join(", ")
    )
}

fn required_elements(
    source: &InputSourceFragment,
    sink: &SinkFragment,
    videos: &[PlannedVideoRendition],
    audios: &[PlannedAudioRendition],
) -> Vec<String> {
    let mut elements = BTreeSet::from([
        "appsrc".to_string(),
        "audioconvert".to_string(),
        "audiorate".to_string(),
        "audioresample".to_string(),
        "audiotestsrc".to_string(),
        "input-selector".to_string(),
        "identity".to_string(),
        "queue".to_string(),
        "tee".to_string(),
        "videoconvert".to_string(),
        "videorate".to_string(),
        "videoscale".to_string(),
        "videotestsrc".to_string(),
    ]);
    elements.insert(sink.mux_kind.required_element().to_string());
    elements.extend(source.required_elements.iter().cloned());
    elements.extend(sink.required_elements.iter().cloned());
    elements.extend(
        videos
            .iter()
            .map(|video| video_encoder_element(video.codec.as_str()).to_string()),
    );
    if videos.iter().any(|video| video.interlaced) {
        elements.insert("interlace".to_string());
    }
    elements.extend(
        audios
            .iter()
            .map(|audio| audio_encoder_element(audio.codec.as_str()).to_string()),
    );
    elements.into_iter().collect()
}

impl PlannedVideoRendition {
    fn from_config(
        feed: &FeedConfig,
        video: &crate::config::VideoRenditionConfig,
        index: usize,
    ) -> Self {
        let codec = video.codec_name(feed.output().vcodec.as_str()).to_string();
        let encoder = if feed.encoder.video.applies_to(codec.as_str()) {
            feed.encoder.video.clone()
        } else {
            EncoderCodecSettings::default()
        };
        Self {
            id: video.id.clone(),
            program: video.program_number(index),
            video_pid: video.video_pid(index),
            pmt_pid: video.pmt_pid(index),
            width: video.width,
            height: video.height,
            fps: video.frame_rate_text(feed.video.fps.as_str()).to_string(),
            interlaced: video.interlaced,
            field_order: video.field_order.clone(),
            standard: video.standard.clone(),
            codec,
            bitrate_kbps: encoder
                .bitrate_or(video.bitrate_kbps.or(feed.output().video_bitrate_kbps)),
            encoder,
        }
    }

    fn status_value(&self) -> Value {
        json!({
            "id": self.id,
            "program": self.program,
            "video_pid": self.video_pid,
            "pmt_pid": self.pmt_pid,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "interlaced": self.interlaced,
            "field_order": self.field_order,
            "standard": self.standard,
            "codec": self.codec,
            "bitrate_kbps": self.bitrate_kbps,
        })
    }
}

impl PlannedAudioRendition {
    fn from_config(
        feed: &FeedConfig,
        audio: &crate::config::AudioRenditionConfig,
        program: i32,
        audio_pid: i32,
    ) -> Self {
        let codec = audio.codec_name(feed.output().acodec.as_str()).to_string();
        let encoder = if feed.encoder.audio.applies_to(codec.as_str()) {
            feed.encoder.audio.clone()
        } else {
            EncoderCodecSettings::default()
        };
        let fallback_bitrate_kbps =
            u32::try_from(audio.bitrate_bps().saturating_div(1_000)).unwrap_or(192);
        Self {
            id: audio.id.clone(),
            program,
            audio_pid,
            channels: audio.channels(),
            codec,
            bitrate_bps: i64::from(
                encoder
                    .bitrate_or(Some(fallback_bitrate_kbps))
                    .unwrap_or(fallback_bitrate_kbps),
            ) * 1_000,
            language: audio.language.clone(),
            encoder,
        }
    }

    fn status_value(&self) -> Value {
        json!({
            "id": self.id,
            "program": self.program,
            "audio_pid": self.audio_pid,
            "channels": self.channels,
            "codec": self.codec,
            "bitrate_bps": self.bitrate_bps,
            "language": self.language,
        })
    }
}

fn video_rendition_branch(
    feed: &FeedConfig,
    video: &crate::config::VideoRenditionConfig,
    planned: &PlannedVideoRendition,
    index: usize,
    mux_kind: MuxKind,
) -> String {
    let caps = video_caps_fragment(feed, video);
    let overlay_name = gst_element_name("cgen_overlay", &video.id, index);
    let encoder = video_encoder_fragment(
        planned.codec.as_str(),
        planned.bitrate_kbps,
        video.interlaced,
        video.field_order.as_str(),
        &planned.encoder,
    );
    let interlace = if video.interlaced {
        if top_field_first(video.field_order.as_str()) {
            " ! interlace field-pattern=1:1 top-field-first=true"
        } else {
            " ! interlace field-pattern=1:1 top-field-first=false"
        }
    } else {
        ""
    };
    let queue = queue_fragment(feed, QueueLeak::None);
    let leaky_queue = queue_fragment(feed, QueueLeak::Downstream);
    let mux_pad = mux_kind.video_sink_pad(planned);
    format!(
        "video_tee. ! {leaky_queue} ! videoconvert ! videoscale ! videorate ! {caps},format=BGRx ! identity name={overlay_name} silent=true ! videoconvert ! video/x-raw,format=I420{interlace} ! {queue} ! {encoder} ! {queue} ! {mux_pad}"
    )
}

fn audio_rendition_branch(
    feed: &FeedConfig,
    audio: &PlannedAudioRendition,
    index: usize,
    mux_kind: MuxKind,
) -> String {
    let name = gst_element_name(
        "audio",
        &format!("{}_p{}_pid{}", audio.id, audio.program, audio.audio_pid),
        index,
    );
    let encoder =
        audio_encoder_fragment_bps(audio.codec.as_str(), audio.bitrate_bps, &audio.encoder);
    let queue = queue_fragment(feed, QueueLeak::None);
    let leaky_queue = queue_fragment(feed, QueueLeak::Downstream);
    let mux_pad = mux_kind.audio_sink_pad(audio);
    format!(
        "audio_tee. ! {leaky_queue} ! audioconvert ! audioresample ! audio/x-raw,rate=48000,channels={},layout=interleaved ! {queue} ! {encoder} ! {queue} name={name}_encoded ! {mux_pad}",
        audio.channels
    )
}

fn audio_pid_for(video_pid: i32, audio_index: usize) -> i32 {
    video_pid + 1 + i32::try_from(audio_index).unwrap_or(0)
}

fn program_map_fragment(
    videos: &[PlannedVideoRendition],
    audios: &[PlannedAudioRendition],
) -> String {
    let mut fields = Vec::with_capacity(videos.len() + audios.len() + 1);
    fields.push("program_map".to_string());
    fields.extend(
        videos
            .iter()
            .map(|video| format!("sink_{}=(int){}", video.video_pid, video.program)),
    );
    fields.extend(
        audios
            .iter()
            .map(|audio| format!("sink_{}=(int){}", audio.audio_pid, audio.program)),
    );
    fields.join(",")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QueueLeak {
    None,
    Downstream,
}

fn queue_time_ns(feed: &FeedConfig) -> u64 {
    u64::from(feed.sync.source_buffer_ms.clamp(40, 5_000)) * 1_000_000
}

fn queue_fragment(feed: &FeedConfig, leak: QueueLeak) -> String {
    let leaky = match leak {
        QueueLeak::None => "",
        QueueLeak::Downstream => " leaky=downstream",
    };
    format!(
        "queue max-size-time={} max-size-buffers=0 max-size-bytes=0 flush-on-eos=true{leaky}",
        queue_time_ns(feed)
    )
}

fn priority_appsrc_max_bytes(feed: &FeedConfig) -> u64 {
    let channels = feed
        .enabled_audio_renditions()
        .iter()
        .map(|audio| u64::from(audio.channels()))
        .max()
        .unwrap_or(2)
        .max(2);
    let bytes_per_ms = 48_000_u64.saturating_mul(channels).saturating_mul(2) / 1_000;
    bytes_per_ms
        .saturating_mul(u64::from(feed.sync.source_buffer_ms.clamp(40, 5_000)))
        .max(48_000)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MuxKind {
    MpegTs,
    Flv,
}

impl MuxKind {
    fn required_element(self) -> &'static str {
        match self {
            Self::MpegTs => "mpegtsmux",
            Self::Flv => "flvmux",
        }
    }

    fn video_sink_pad(self, video: &PlannedVideoRendition) -> String {
        match self {
            Self::MpegTs => format!("mux.sink_{}", video.video_pid),
            Self::Flv => "mux.video".to_string(),
        }
    }

    fn audio_sink_pad(self, audio: &PlannedAudioRendition) -> String {
        match self {
            Self::MpegTs => format!("mux.sink_{}", audio.audio_pid),
            Self::Flv => "mux.audio".to_string(),
        }
    }
}

fn mux_fragment(feed: &FeedConfig, kind: MuxKind) -> String {
    let latency_ns = queue_time_ns(feed) / 2;
    match kind {
        MuxKind::MpegTs => format!(
            "mpegtsmux name=mux alignment=7 latency={latency_ns} pat-interval=4500 pmt-interval=4500 si-interval=4500 pcr-interval=1800"
        ),
        MuxKind::Flv => "flvmux name=mux streamable=true latency=0".to_string(),
    }
}

fn gst_element_name(prefix: &str, id: &str, index: usize) -> String {
    let mut name = String::with_capacity(prefix.len() + id.len() + 8);
    name.push_str(prefix);
    name.push('_');
    if id.trim().is_empty() {
        name.push_str(&index.to_string());
        return name;
    }
    for ch in id.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            name.push(ch);
        } else {
            name.push('_');
        }
    }
    name
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InputSourceFragment {
    description: String,
    pad_name: &'static str,
    decode_chain: &'static str,
    required_elements: Vec<String>,
}

impl InputSourceFragment {
    fn from_url(url: &str, _feed: &FeedConfig) -> Result<Self> {
        let url = url.trim();
        if url.is_empty() {
            bail!("gstreamer cgen input url is empty");
        }
        if let Some(endpoint) = UdpEndpoint::parse(url) {
            return Ok(Self {
                description: format!(
                    "udpsrc address={} port={} auto-multicast=true reuse={} buffer-size={} retrieve-sender-address=false ! tsdemux name=src",
                    gst_quote(&endpoint.host),
                    endpoint.port,
                    endpoint.reuse,
                    endpoint.buffer_size.unwrap_or(DEFAULT_UDP_BUFFER_BYTES)
                ),
                pad_name: "src",
                decode_chain: " ! parsebin ! decodebin",
                required_elements: vec![
                    "decodebin".to_string(),
                    "parsebin".to_string(),
                    "tsdemux".to_string(),
                    "udpsrc".to_string(),
                ],
            });
        }
        Ok(Self {
            description: format!("uridecodebin uri={} name=src", gst_quote(url)),
            pad_name: "src",
            decode_chain: "",
            required_elements: vec!["uridecodebin".to_string()],
        })
    }

    fn video_branch(&self) -> String {
        self.branch("video")
    }

    fn audio_branch(&self) -> String {
        self.branch("audio")
    }

    fn branch(&self, kind: &str) -> String {
        let caps = if kind == "video" {
            " ! video/x-raw"
        } else {
            " ! audio/x-raw"
        };
        format!("{}.{}{}", self.pad_name, self.decode_chain, caps)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SinkFragment {
    description: String,
    required_elements: Vec<String>,
    mux_kind: MuxKind,
}

impl SinkFragment {
    fn from_url(url: &str, feed: &FeedConfig) -> Result<Self> {
        let url = url.trim();
        if url.is_empty() {
            bail!("gstreamer cgen output url is empty");
        }
        let max_lateness_ns = queue_time_ns(feed);
        if let Some(endpoint) = UdpEndpoint::parse(url) {
            return Ok(Self {
                description: format!(
                    "udpsink host={} port={} auto-multicast=true buffer-size={} sync=true async=false qos=true max-lateness={max_lateness_ns}",
                    gst_quote(&endpoint.host),
                    endpoint.port,
                    endpoint.buffer_size.unwrap_or(DEFAULT_UDP_BUFFER_BYTES)
                ),
                required_elements: vec!["udpsink".to_string()],
                mux_kind: MuxKind::MpegTs,
            });
        }
        if url.to_ascii_lowercase().starts_with("rtmp://") {
            return Ok(Self {
                description: format!(
                    "rtmpsink location={} sync=true async=false qos=true max-lateness={max_lateness_ns}",
                    gst_quote(url)
                ),
                required_elements: vec!["rtmpsink".to_string()],
                mux_kind: MuxKind::Flv,
            });
        }
        return Ok(Self {
            description: format!("filesink location={} sync=true async=false", gst_quote(url)),
            required_elements: vec!["filesink".to_string()],
            mux_kind: MuxKind::MpegTs,
        });
    }
}

fn video_caps_fragment(feed: &FeedConfig, video: &crate::config::VideoRenditionConfig) -> String {
    let caps = format!("video/x-raw,width={},height={}", video.width, video.height);
    let Some(output_framerate) = gst_fraction(video.frame_rate_text(feed.video.fps.as_str()))
    else {
        return caps;
    };
    let framerate = rendition_framerate_for_caps(output_framerate.as_str(), video.interlaced);
    format!("{caps},framerate={framerate}")
}

fn source_video_caps_fragment(feed: &FeedConfig) -> String {
    let caps = format!(
        "video/x-raw,width={},height={}",
        feed.video.width, feed.video.height
    );
    let Some(output_framerate) = gst_fraction(feed.video.fps.as_str()) else {
        return caps;
    };
    let framerate = rendition_framerate_for_caps(output_framerate.as_str(), feed.video.interlaced);
    format!("{caps},framerate={framerate}")
}

fn rendition_framerate_for_caps(framerate: &str, interlaced: bool) -> String {
    if interlaced {
        doubled_gst_fraction(framerate).unwrap_or_else(|| framerate.to_string())
    } else {
        framerate.to_string()
    }
}

fn video_encoder_fragment(
    codec: &str,
    bitrate_kbps: Option<u32>,
    interlaced: bool,
    field_order: &str,
    settings: &EncoderCodecSettings,
) -> String {
    let bitrate = settings
        .bitrate_or(bitrate_kbps)
        .unwrap_or(12_000)
        .saturating_mul(1_000);
    match codec.trim().to_ascii_lowercase().as_str() {
        "mpeg2video" | "mpeg2" | "avenc_mpeg2video" => {
            let gop = settings.gop.unwrap_or(15).clamp(1, 600);
            let interlace_flags = if interlaced {
                let field_order = if top_field_first(field_order) {
                    "tt"
                } else {
                    "bb"
                };
                format!(
                    " flags=+ildct+ilme alternate-scan=true field-order={field_order} seq-disp-ext=always"
                )
            } else {
                " field-order=progressive".to_string()
            };
            format!("avenc_mpeg2video bitrate={bitrate} gop-size={gop} qos=true{interlace_flags}")
        }
        "h264" | "avc" | "x264" | "libx264" | "x264enc" => {
            let preset = video_encoder_preset(settings.preset.as_str(), "ultrafast");
            let tune = video_encoder_tune(settings.tune.as_str(), "zerolatency");
            let keyint = settings
                .gop
                .map(|value| format!(" key-int-max={}", value.clamp(1, 600)))
                .unwrap_or_default();
            let bframes = settings
                .bframes
                .map(|value| format!(" bframes={}", value.min(16)))
                .unwrap_or_default();
            format!(
                "x264enc tune={tune} speed-preset={preset} bitrate={} qos=true{keyint}{bframes}",
                bitrate / 1_000
            )
        }
        "avenc_h264" | "h264_libav" => format!("avenc_h264 bitrate={bitrate}"),
        "h264_amf" | "amf_h264" | "amfh264enc" => {
            format!("amfh264enc bitrate={} qos=true", bitrate / 1_000)
        }
        "h264_nvenc" | "nvenc_h264" | "nvh264enc" => {
            format!("nvh264enc bitrate={}", bitrate / 1_000)
        }
        "h264_qsv" | "qsv_h264" | "qsvh264enc" => {
            format!("qsvh264enc bitrate={}", bitrate / 1_000)
        }
        "mpeg2_qsv" | "qsv_mpeg2" | "qsvmpeg2enc" => {
            format!("qsvmpeg2enc bitrate={}", bitrate / 1_000)
        }
        "openh264" | "h264_openh264" | "openh264enc" => format!("openh264enc bitrate={bitrate}"),
        "h265" | "hevc" | "x265" | "libx265" | "x265enc" => {
            let preset = video_encoder_preset(settings.preset.as_str(), "ultrafast");
            let tune = video_encoder_tune(settings.tune.as_str(), "zerolatency");
            let mut options = vec![
                format!("bframes={}", settings.bframes.unwrap_or(0).min(16)),
                "rc-lookahead=0".to_string(),
                "frame-threads=1".to_string(),
                "pools=none".to_string(),
            ];
            if let Some(gop) = settings.gop {
                options.push(format!("keyint={}", gop.clamp(1, 600)));
            }
            format!(
                "x265enc speed-preset={preset} tune={tune} bitrate={} option-string=\"{}\"",
                bitrate / 1_000,
                options.join(":")
            )
        }
        "h265_amf" | "hevc_amf" | "amf_h265" | "amf_hevc" | "amfh265enc" => {
            format!("amfh265enc bitrate={}", bitrate / 1_000)
        }
        "h265_nvenc" | "hevc_nvenc" | "nvenc_h265" | "nvenc_hevc" | "nvh265enc" => {
            format!("nvh265enc bitrate={}", bitrate / 1_000)
        }
        "h265_qsv" | "hevc_qsv" | "qsv_h265" | "qsv_hevc" | "qsvh265enc" => {
            format!("qsvh265enc bitrate={}", bitrate / 1_000)
        }
        "hevc_libav" | "h265_libav" | "avenc_hevc" => format!("avenc_hevc bitrate={bitrate}"),
        "mpeg4" | "mpeg4video" | "avenc_mpeg4" => format!("avenc_mpeg4 bitrate={bitrate}"),
        actual if actual.starts_with("avenc_") => format!("{actual} bitrate={bitrate}"),
        actual if actual.ends_with("enc") => generic_video_encoder_fragment(actual, bitrate),
        _ => format!(
            "x264enc tune=zerolatency speed-preset=ultrafast bitrate={} qos=true",
            bitrate / 1_000
        ),
    }
}

fn generic_video_encoder_fragment(element: &str, bitrate: u32) -> String {
    match element {
        name if name.starts_with("amf") || name.starts_with("qsv") || name.starts_with("nv") => {
            format!("{name} bitrate={}", bitrate / 1_000)
        }
        name if name.starts_with("mf") || name.starts_with("d3d12") => {
            format!("{name} bitrate={}", bitrate / 1_000)
        }
        name => name.to_string(),
    }
}

fn video_encoder_preset(value: &str, fallback: &'static str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "ultrafast" | "superfast" | "veryfast" | "faster" | "fast" | "medium" | "slow"
        | "slower" | "veryslow" => value.trim().to_ascii_lowercase(),
        _ => fallback.to_string(),
    }
}

fn video_encoder_tune(value: &str, fallback: &'static str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "zerolatency" | "film" | "animation" | "grain" | "stillimage" | "fastdecode" | "psnr"
        | "ssim" => value.trim().to_ascii_lowercase(),
        _ => fallback.to_string(),
    }
}

fn top_field_first(field_order: &str) -> bool {
    !field_order.eq_ignore_ascii_case("bff")
        && !field_order.eq_ignore_ascii_case("bb")
        && !field_order.eq_ignore_ascii_case("bottom-field-first")
}

fn doubled_gst_fraction(value: &str) -> Option<String> {
    let (num, den) = value.split_once('/')?;
    let num = num.trim().parse::<u32>().ok()?;
    let den = den.trim().parse::<u32>().ok()?;
    Some(format!("{}/{}", num.saturating_mul(2), den.max(1)))
}

fn video_encoder_element(codec: &str) -> String {
    match codec.trim().to_ascii_lowercase().as_str() {
        "mpeg2video" | "mpeg2" | "avenc_mpeg2video" => "avenc_mpeg2video".to_string(),
        "h264" | "avc" | "x264" | "libx264" | "x264enc" => "x264enc".to_string(),
        "avenc_h264" | "h264_libav" => "avenc_h264".to_string(),
        "h264_amf" | "amf_h264" | "amfh264enc" => "amfh264enc".to_string(),
        "h264_nvenc" | "nvenc_h264" | "nvh264enc" => "nvh264enc".to_string(),
        "h264_qsv" | "qsv_h264" | "qsvh264enc" => "qsvh264enc".to_string(),
        "mpeg2_qsv" | "qsv_mpeg2" | "qsvmpeg2enc" => "qsvmpeg2enc".to_string(),
        "openh264" | "h264_openh264" | "openh264enc" => "openh264enc".to_string(),
        "h265" | "hevc" | "x265" | "libx265" | "x265enc" => "x265enc".to_string(),
        "h265_amf" | "hevc_amf" | "amf_h265" | "amf_hevc" | "amfh265enc" => {
            "amfh265enc".to_string()
        }
        "h265_nvenc" | "hevc_nvenc" | "nvenc_h265" | "nvenc_hevc" | "nvh265enc" => {
            "nvh265enc".to_string()
        }
        "h265_qsv" | "hevc_qsv" | "qsv_h265" | "qsv_hevc" | "qsvh265enc" => {
            "qsvh265enc".to_string()
        }
        "hevc_libav" | "h265_libav" | "avenc_hevc" => "avenc_hevc".to_string(),
        "mpeg4" | "mpeg4video" | "avenc_mpeg4" => "avenc_mpeg4".to_string(),
        actual if actual.starts_with("avenc_") || actual.ends_with("enc") => actual.to_string(),
        _ => "x264enc".to_string(),
    }
}

fn audio_encoder_fragment_bps(
    codec: &str,
    bitrate_bps: i64,
    settings: &EncoderCodecSettings,
) -> String {
    let bitrate = bitrate_bps.max(1);
    let bitrate = settings
        .bitrate_kbps
        .map(|kbps| i64::from(kbps.max(1)) * 1_000)
        .unwrap_or(bitrate);
    match codec.trim().to_ascii_lowercase().as_str() {
        "ac3" | "avenc_ac3" => format!("avenc_ac3 bitrate={bitrate}"),
        "eac3" | "e-ac3" | "e_ac3" | "enhanced_ac3" | "enhanced-ac3" | "avenc_eac3" => {
            format!("avenc_eac3 bitrate={bitrate}")
        }
        "mp2" | "mpeg_layer_2" | "mpeg-layer-2" | "avenc_mp2" => {
            format!("avenc_mp2 bitrate={bitrate}")
        }
        "mp3" | "lamemp3enc" => {
            format!("lamemp3enc target=bitrate bitrate={}", bitrate / 1_000)
        }
        "avenc_mp3" => format!("avenc_mp3 bitrate={bitrate}"),
        "opus" | "opusenc" => format!("opusenc bitrate={bitrate}"),
        "vorbis" | "vorbisenc" => format!("vorbisenc bitrate={bitrate}"),
        "flac" | "flacenc" => "flacenc".to_string(),
        "avenc_flac" => "avenc_flac".to_string(),
        "avenc_aac" | "aac" => format!("avenc_aac bitrate={bitrate}"),
        "faac" => format!("faac bitrate={bitrate}"),
        actual if actual.starts_with("avenc_") => format!("{actual} bitrate={bitrate}"),
        "fdkaacenc" | "mfaacenc" => format!("{} bitrate={bitrate}", codec.trim()),
        actual if actual.ends_with("enc") => actual.to_string(),
        _ => format!("avenc_aac bitrate={bitrate}"),
    }
}

fn audio_encoder_element(codec: &str) -> String {
    match codec.trim().to_ascii_lowercase().as_str() {
        "ac3" | "avenc_ac3" => "avenc_ac3".to_string(),
        "eac3" | "e-ac3" | "e_ac3" | "enhanced_ac3" | "enhanced-ac3" | "avenc_eac3" => {
            "avenc_eac3".to_string()
        }
        "mp2" | "mpeg_layer_2" | "mpeg-layer-2" | "avenc_mp2" => "avenc_mp2".to_string(),
        "mp3" | "lamemp3enc" => "lamemp3enc".to_string(),
        "avenc_mp3" => "avenc_mp3".to_string(),
        "opus" | "opusenc" => "opusenc".to_string(),
        "vorbis" | "vorbisenc" => "vorbisenc".to_string(),
        "flac" | "flacenc" => "flacenc".to_string(),
        "avenc_flac" => "avenc_flac".to_string(),
        "aac" | "avenc_aac" => "avenc_aac".to_string(),
        "faac" => "faac".to_string(),
        actual if actual.starts_with("avenc_") || actual.ends_with("enc") => actual.to_string(),
        _ => "avenc_aac".to_string(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UdpEndpoint {
    host: String,
    port: u16,
    buffer_size: Option<u32>,
    reuse: bool,
}

impl UdpEndpoint {
    fn parse(url: &str) -> Option<Self> {
        let url = url.trim();
        if !url.get(..6)?.eq_ignore_ascii_case("udp://") {
            return None;
        }
        let rest = &url[6..];
        let authority = rest.split(['/', '?']).next().unwrap_or(rest);
        let (host, port) = authority.rsplit_once(':')?;
        let port = port.parse::<u16>().ok()?;
        let query = rest.split_once('?').map(|(_, query)| query).unwrap_or("");
        Some(Self {
            host: host.trim_matches(['[', ']']).to_string(),
            port,
            buffer_size: udp_query_u32(query, &["buffer_size", "buffer-size", "fifo_size"])
                .map(|value| value.max(1316)),
            reuse: udp_query_bool(query, "reuse").unwrap_or(true),
        })
    }
}

fn udp_query_u32(query: &str, keys: &[&str]) -> Option<u32> {
    query.split('&').find_map(|pair| {
        let (key, value) = pair.split_once('=')?;
        keys.iter()
            .any(|candidate| key.eq_ignore_ascii_case(candidate))
            .then(|| value.parse::<u32>().ok())
            .flatten()
    })
}

fn udp_query_bool(query: &str, key: &str) -> Option<bool> {
    query.split('&').find_map(|pair| {
        let (candidate, value) = pair.split_once('=')?;
        candidate.eq_ignore_ascii_case(key).then(|| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    })
}

fn gst_fraction(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("source") {
        return None;
    }
    if value.contains('/') {
        return Some(value.to_string());
    }
    value.parse::<u32>().ok().map(|fps| format!("{fps}/1"))
}

fn gst_quote(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AudioConfig, AudioRenditionConfig, BannerConfig, ClockConfig, EncoderCodecSettings,
        EndpointConfig, FeedConfig, GraphicsConfig, LadderConfig, PriorityInputConfig,
        StandbyConfig, StateConfig, SyncConfig, TextConfig, VideoConfig, VideoRenditionConfig,
    };

    #[test]
    fn parses_udp_sink_url() {
        let endpoint = UdpEndpoint::parse("udp://239.0.0.2:9001?pkt_size=1316&buffer_size=1048576")
            .expect("udp endpoint");
        assert_eq!(endpoint.host, "239.0.0.2");
        assert_eq!(endpoint.port, 9001);
        assert_eq!(endpoint.buffer_size, Some(1_048_576));
        assert!(endpoint.reuse);
    }

    #[test]
    fn gstreamer_plan_uses_ts_demux_and_mpegts_mux() {
        let feed = test_feed();
        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");
        assert!(plan.description.contains("udpsrc address="));
        assert!(plan.description.contains("retrieve-sender-address=false"));
        assert!(plan.description.contains("buffer-size=2000000"));
        assert!(!plan.description.contains("timeout="));
        assert!(plan.description.contains("tsdemux name=src"));
        assert!(plan.description.contains("src. ! parsebin ! decodebin"));
        assert!(plan.description.contains("tee name=video_tee"));
        assert!(plan.description.contains("tee name=audio_tee"));
        assert!(plan
            .description
            .contains("input-selector name=video_selector"));
        assert!(plan.description.contains(
            "input-selector name=video_selector sync-streams=true sync-mode=clock cache-buffers=false drop-backwards=true"
        ));
        assert!(plan.description.contains("video_selector.sink_0"));
        assert!(plan
            .description
            .contains("videotestsrc name=standby_video_src pattern=black"));
        assert!(plan.description.contains("video_selector.sink_1"));
        assert!(plan
            .description
            .contains("videotestsrc name=smpte_video_src pattern=smpte"));
        assert!(plan.description.contains("video_selector.sink_2"));
        assert!(plan
            .description
            .contains("identity name=cgen_overlay_hd silent=true"));
        assert!(plan
            .description
            .contains("input-selector name=audio_selector"));
        assert!(plan.description.contains(
            "input-selector name=audio_selector sync-streams=true sync-mode=clock cache-buffers=false drop-backwards=true"
        ));
        assert!(plan.description.contains("audiotestsrc wave=silence"));
        assert!(plan.description.contains("appsrc name=priority_audio_src"));
        assert!(plan.description.contains(
            "appsrc name=priority_audio_src is-live=true block=false leaky-type=downstream stream-type=stream format=time do-timestamp=true min-latency=0 max-latency=240000000"
        ));
        assert!(plan.description.contains("audio_selector.sink_0"));
        assert!(plan.description.contains("audio_selector.sink_1"));
        assert!(plan.description.contains("audio_selector.sink_2"));
        assert!(plan.description.contains("video_tee."));
        assert!(plan.description.contains("audio_tee."));
        assert!(plan.description.contains(
            "audio_tee. ! queue max-size-time=240000000 max-size-buffers=0 max-size-bytes=0 flush-on-eos=true leaky=downstream ! audioconvert"
        ));
        assert!(plan.description.contains("mux.sink_256"));
        assert!(plan.description.contains("mux.sink_257"));
        assert!(plan.description.contains("mpegtsmux name=mux"));
        assert!(plan.description.contains("max-size-time=240000000"));
        assert!(plan.description.contains("flush-on-eos=true"));
        assert!(plan.description.contains("audiorate"));
        assert!(plan
            .description
            .contains("audio/x-raw,rate=48000,channels=2,layout=interleaved"));
        assert!(plan.description.contains("latency=120000000"));
        assert!(plan.description.contains("pat-interval=4500"));
        assert!(plan.description.contains("pcr-interval=1800"));
        assert!(plan.description.contains("max-bytes=138240"));
        assert!(!plan.description.contains("max-size-time=300000000"));
        assert!(plan
            .description
            .contains("udpsink host=\"239.0.0.2\" port=9001"));
        assert!(plan.description.contains("qos=true max-lateness=240000000"));
        assert!(plan
            .description
            .contains("avenc_mpeg2video bitrate=12000000 gop-size=15 qos=true flags=+ildct+ilme alternate-scan=true field-order=tt seq-disp-ext=always"));
        assert!(plan.description.contains("videorate ! video/x-raw,width=1920,height=1080,framerate=60000/1001,format=BGRx ! identity name=cgen_overlay_hd silent=true"));
        assert!(plan.description.contains(
            "videoconvert ! video/x-raw,format=I420 ! interlace field-pattern=1:1 top-field-first=true"
        ));
        assert!(plan.required_elements.contains(&"identity".to_string()));
        assert!(!plan.description.contains("interlace-mode=interleaved"));
        assert!(plan.description.contains("buffer-size=4194304"));
        assert!(plan.required_elements.contains(&"udpsrc".to_string()));
        assert!(plan.required_elements.contains(&"parsebin".to_string()));
        assert!(plan.required_elements.contains(&"tsdemux".to_string()));
        assert!(plan.required_elements.contains(&"decodebin".to_string()));
        assert!(plan.required_elements.contains(&"interlace".to_string()));
        assert!(plan.required_elements.contains(&"mpegtsmux".to_string()));
        assert!(plan
            .required_elements
            .contains(&"avenc_mpeg2video".to_string()));
        assert!(plan.required_elements.contains(&"avenc_ac3".to_string()));
        assert!(plan.required_elements.contains(&"videotestsrc".to_string()));
        assert!(plan.required_elements.contains(&"udpsink".to_string()));
    }

    #[test]
    fn gstreamer_plan_uses_sync_buffer_for_queues_and_mux_latency() {
        let mut feed = test_feed();
        feed.sync.source_buffer_ms = 640;

        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");

        assert!(plan.description.contains("max-size-time=640000000"));
        assert!(plan.description.contains(
            "src. ! parsebin ! decodebin ! audio/x-raw ! queue max-size-time=640000000 max-size-buffers=0 max-size-bytes=0 flush-on-eos=true leaky=downstream ! audioconvert"
        ));
        assert!(plan.description.contains("latency=320000000"));
        assert!(plan.description.contains("max-bytes=368640"));
    }

    #[test]
    fn source_fps_omits_forced_ntsc_framerate_caps() {
        let mut feed = test_feed();
        feed.video.fps = "source".to_string();
        feed.video.interlaced = false;
        feed.ladder.videos[0].fps = "source".to_string();
        feed.ladder.videos[0].interlaced = false;

        assert_eq!(
            source_video_caps_fragment(&feed),
            "video/x-raw,width=1920,height=1080"
        );
        assert_eq!(
            video_caps_fragment(&feed, &feed.ladder.videos[0]),
            "video/x-raw,width=1920,height=1080"
        );
        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");

        assert!(!plan.description.contains("framerate=30000/1001"));
        assert!(!plan.description.contains("framerate=60000/1001"));
    }

    #[test]
    fn gstreamer_plan_fans_out_enabled_ladder_renditions() {
        let mut feed = test_feed();
        feed.ladder.videos = vec![
            VideoRenditionConfig {
                id: "hd".to_string(),
                enabled: "true".to_string(),
                width: 1920,
                height: 1080,
                fps: "30000/1001".to_string(),
                interlaced: true,
                field_order: "tff".to_string(),
                standard: "atsc".to_string(),
                vcodec: "mpeg2video".to_string(),
                bitrate_kbps: Some(14_000),
                program: Some(1),
                video_pid: Some(0x100),
                pmt_pid: Some(0x1000),
            },
            VideoRenditionConfig {
                id: "p720".to_string(),
                enabled: "true".to_string(),
                width: 1280,
                height: 720,
                fps: "60000/1001".to_string(),
                interlaced: false,
                field_order: "progressive".to_string(),
                standard: "atsc".to_string(),
                vcodec: "h264".to_string(),
                bitrate_kbps: Some(6_000),
                program: Some(2),
                video_pid: Some(0x120),
                pmt_pid: Some(0x1001),
            },
        ];
        feed.ladder.audios = vec![
            AudioRenditionConfig {
                id: "surround_51".to_string(),
                enabled: "true".to_string(),
                channels: 6,
                acodec: "ac3".to_string(),
                bitrate_kbps: Some(384),
                language: "eng".to_string(),
            },
            AudioRenditionConfig {
                id: "stereo".to_string(),
                enabled: "true".to_string(),
                channels: 2,
                acodec: "ac3".to_string(),
                bitrate_kbps: Some(192),
                language: "eng".to_string(),
            },
        ];

        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");

        assert!(plan
            .description
            .contains("video/x-raw,width=1920,height=1080"));
        assert!(plan
            .description
            .contains("video/x-raw,width=1280,height=720"));
        assert!(plan
            .description
            .contains("identity name=cgen_overlay_hd silent=true"));
        assert!(plan
            .description
            .contains("identity name=cgen_overlay_p720 silent=true"));
        assert!(plan
            .description
            .contains("x264enc tune=zerolatency speed-preset=ultrafast bitrate=6000 qos=true"));
        assert!(plan
            .description
            .contains("audio/x-raw,rate=48000,channels=6,layout=interleaved"));
        assert!(plan
            .description
            .contains("audio/x-raw,rate=48000,channels=2,layout=interleaved"));

        let status = plan.status_value();
        assert_eq!(status["video_count"], 2);
        assert_eq!(status["audio_count"], 4);
        assert_eq!(status["program_map"], "program_map,sink_256=(int)1,sink_288=(int)2,sink_257=(int)1,sink_258=(int)1,sink_289=(int)2,sink_290=(int)2");
        assert_eq!(status["videos"][0]["program"], 1);
        assert_eq!(status["videos"][0]["video_pid"], 0x100);
        assert_eq!(status["videos"][0]["pmt_pid"], 0x1000);
        assert_eq!(status["videos"][1]["program"], 2);
        assert_eq!(status["audios"][0]["program"], 1);
        assert_eq!(status["audios"][0]["audio_pid"], 0x101);
        assert_eq!(status["audios"][0]["channels"], 6);
        assert_eq!(status["audios"][0]["bitrate_bps"], 384_000);
        assert_eq!(status["audios"][1]["language"], "eng");
    }

    #[test]
    fn gstreamer_plan_uses_selected_hevc_and_eac3_encoders() {
        let mut feed = test_feed();
        feed.program_output.vcodec = "h265".to_string();
        feed.program_output.acodec = "e-ac3".to_string();
        for video in &mut feed.ladder.videos {
            video.vcodec = "h265".to_string();
        }
        for audio in &mut feed.ladder.audios {
            audio.acodec = "e-ac3".to_string();
        }

        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");

        assert!(
            plan.description
                .contains("x265enc speed-preset=ultrafast tune=zerolatency bitrate=12000"),
            "HEVC selection should not fall back to x264: {}",
            plan.description
        );
        assert!(plan.description.contains("avenc_eac3 bitrate=192000"));
        assert!(plan.required_elements.contains(&"x265enc".to_string()));
        assert!(plan.required_elements.contains(&"avenc_eac3".to_string()));
        assert!(!plan.required_elements.contains(&"x264enc".to_string()));
    }

    #[test]
    fn gstreamer_plan_applies_companion_encoder_settings() {
        let mut feed = test_feed();
        feed.program_output.vcodec = "x264enc".to_string();
        feed.program_output.acodec = "avenc_ac3".to_string();
        feed.ladder.videos[0].vcodec = "x264enc".to_string();
        feed.ladder.audios[0].acodec = "avenc_ac3".to_string();
        feed.encoder.video = EncoderCodecSettings {
            codec: "x264enc".to_string(),
            bitrate_kbps: Some(7_000),
            gop: Some(30),
            bframes: Some(2),
            preset: "veryfast".to_string(),
            tune: "zerolatency".to_string(),
            ..Default::default()
        };
        feed.encoder.audio = EncoderCodecSettings {
            codec: "avenc_ac3".to_string(),
            bitrate_kbps: Some(256),
            ..Default::default()
        };

        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");

        assert!(plan.description.contains(
            "x264enc tune=zerolatency speed-preset=veryfast bitrate=7000 qos=true key-int-max=30 bframes=2"
        ));
        assert!(plan.description.contains("avenc_ac3 bitrate=256000"));
    }

    #[test]
    fn gstreamer_plan_uses_uridecodebin_for_non_udp_inputs() {
        let mut feed = test_feed();
        feed.program_input.url = "http://172.16.1.31:8866/live?channel_id=10798".to_string();
        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");
        assert!(plan.description.contains("uridecodebin uri="));
        assert!(!plan.description.contains("souphttpsrc"));
        assert!(!plan.description.contains("tsdemux name=src"));
        assert!(plan.required_elements.contains(&"uridecodebin".to_string()));
        assert!(!plan.required_elements.contains(&"udpsrc".to_string()));
        assert!(!plan.required_elements.contains(&"tsdemux".to_string()));
    }

    #[test]
    fn gstreamer_preflight_reports_missing_elements() {
        gst::init().expect("gst init");
        let err =
            validate_gstreamer_elements(&["definitely_missing_haze_cgen_element".to_string()])
                .expect_err("missing element should fail");

        assert!(err
            .to_string()
            .contains("definitely_missing_haze_cgen_element"));
    }

    #[test]
    fn restart_backoff_uses_xml_bounds() {
        let mut feed = test_feed();
        feed.sync.reconnect_initial_ms = 2_000;
        feed.sync.reconnect_max_ms = 5_000;

        let backoff = restart_backoff_bounds(&feed);

        assert_eq!(backoff.initial, Duration::from_millis(2_000));
        assert_eq!(backoff.max, Duration::from_millis(5_000));
    }
    #[test]
    fn sync_status_reports_runtime_tuning() {
        let mut feed = test_feed();
        feed.sync.source_buffer_ms = 360;

        let status = sync_status_value(&feed);

        assert_eq!(status["source_buffer_ms"], 360);
        assert_eq!(status["reconnect_initial_ms"], 500);
        assert_eq!(status["status_interval_ms"], 750);
    }

    #[test]
    fn graphics_fatal_error_rejects_cairo_fallback() {
        let message = crate::wgpu_renderer::fatal_error_message();

        assert!(message.contains("HAZE CGEN FATAL GRAPHICS ERROR"));
        assert!(message.contains("WGPU"));
        assert!(message.contains("Cairo fallback is disabled"));
    }

    #[test]
    fn pipeline_diagnostics_status_reports_counts_and_latency_errors() {
        let mut diagnostics = PipelineDiagnostics::default();
        diagnostics.record_latency_recalculation(None);
        diagnostics.record_latency_recalculation(Some("latency query failed".to_string()));

        let status = diagnostics.status_value();

        assert_eq!(status["warning_count"], 0);
        assert_eq!(status["qos_count"], 0);
        assert_eq!(status["latency_recalculation_count"], 2);
        assert_eq!(status["last_latency_error"], "latency query failed");
        assert!(status["last_qos"].is_null());
    }

    #[test]
    fn gstreamer_plan_parses_and_selector_modes_are_settable() {
        gst::init().expect("gst init");
        let feed = test_feed();
        let plan = GstPipelinePlan::from_feed(&feed).expect("plan");
        let element = gst::parse::launch(&plan.description).expect("parse launch");
        let pipeline = element.downcast::<gst::Pipeline>().expect("pipeline");

        set_audio_selector_mode(&pipeline, AudioSelectorMode::Silence).expect("silence pad");
        set_audio_selector_mode(&pipeline, AudioSelectorMode::Program).expect("program pad");
        set_audio_selector_mode(&pipeline, AudioSelectorMode::Priority).expect("priority pad");
        set_video_selector_mode(&pipeline, VideoSelectorMode::Program).expect("program video pad");
        set_video_selector_mode(&pipeline, VideoSelectorMode::Standby).expect("standby video pad");
        set_video_selector_mode(&pipeline, VideoSelectorMode::Smpte).expect("smpte video pad");
        apply_mpegts_program_map(&pipeline, &plan).expect("program map");
        priority_audio_appsrc(&pipeline).expect("priority appsrc");
        pipeline
            .by_name("cgen_overlay_hd")
            .expect("graphics handoff");
        pipeline.set_state(gst::State::Null).expect("pipeline stop");
    }

    #[test]
    fn routine_and_priority_audio_source_are_independent_flags() {
        let mut input = PriorityInputConfig {
            audio_source: "both".to_string(),
            ..Default::default()
        };
        assert!(input.priority_audio_enabled());
        assert!(input.routine_audio_enabled());
        input.audio_source = "routine".to_string();
        assert!(!input.priority_audio_enabled());
        assert!(input.routine_audio_enabled());
        input.audio_source = "priority".to_string();
        assert!(input.priority_audio_enabled());
        assert!(!input.routine_audio_enabled());
    }
    #[test]
    fn standby_mute_does_not_mute_external_source_audio() {
        let feed = test_feed();
        let state = RuntimeState::default();
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, true),
            AudioSelectorMode::Program
        );
    }

    #[test]
    fn standby_mute_can_mute_routine_idle_audio() {
        let mut feed = test_feed();
        feed.audio.idle = "routine".to_string();
        let state = RuntimeState::default();
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, true),
            AudioSelectorMode::Silence
        );
    }

    #[test]
    fn program_audio_can_be_selected_when_standby_mute_is_disabled() {
        let mut feed = test_feed();
        feed.audio.mute_standby_routine = false;
        let state = RuntimeState::default();
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, true),
            AudioSelectorMode::Program
        );
    }

    #[test]
    fn video_lifecycle_is_independent_from_alert_audio() {
        let mut feed = test_feed();
        let mut state = RuntimeState::default();

        assert_eq!(
            desired_video_selector_mode(&feed, &state, true),
            VideoSelectorMode::Program
        );
        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["*"],
            "queue_id": "q1",
            "data": {
                "queue_id": "q1",
                "audio_path": "runtime/audio/alerts/q1.pcm16le",
                "duration_ms": 60000
            }
        })));
        assert_eq!(
            desired_video_selector_mode(&feed, &state, true),
            VideoSelectorMode::Program
        );
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, true),
            AudioSelectorMode::Priority
        );

        feed.state.mode = "standby".to_string();
        assert_eq!(
            desired_video_selector_mode(&feed, &state, true),
            VideoSelectorMode::Standby
        );
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, true),
            AudioSelectorMode::Priority
        );
        feed.state.smpte_bars = true;
        assert_eq!(
            desired_video_selector_mode(&feed, &state, true),
            VideoSelectorMode::Smpte
        );
    }

    #[test]
    fn live_cgen_control_can_switch_video_selector_without_static_config_change() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        assert_eq!(
            desired_video_selector_mode(&feed, &state, true),
            VideoSelectorMode::Program
        );
        assert!(state.apply_event(&json!({
            "type": "cgen.control",
            "subject": feed.id.clone(),
            "data": {
                "feed_id": feed.id.clone(),
                "action": "smpte_bars",
                "enabled": true,
                "smpte_bars": true
            }
        })));
        assert_eq!(
            desired_video_selector_mode(&feed, &state, true),
            VideoSelectorMode::Smpte
        );
    }

    #[test]
    fn input_health_drives_standby_without_touching_priority_audio() {
        let mut feed = test_feed();
        feed.audio.mute_standby_routine = false;
        let mut state = RuntimeState::default();

        assert_eq!(
            desired_video_selector_mode(&feed, &state, false),
            VideoSelectorMode::Standby
        );
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, false),
            AudioSelectorMode::Silence
        );

        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["*"],
            "queue_id": "q1",
            "data": {
                "queue_id": "q1",
                "audio_path": "runtime/audio/alerts/q1.pcm16le",
                "duration_ms": 60000
            }
        })));
        assert_eq!(
            desired_video_selector_mode(&feed, &state, false),
            VideoSelectorMode::Standby
        );
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, false),
            AudioSelectorMode::Priority
        );
    }

    #[test]
    fn input_health_marks_frames_and_timeouts() {
        let feed = test_feed();
        let mut health = InputHealth::default();

        assert!(!health.video_connected(&feed));
        assert!(!health.audio_connected(&feed));
        health.mark_video_frame();
        assert!(health.video_connected(&feed));
        assert!(!health.audio_connected(&feed));
        health.mark_audio_frame();
        assert!(health.video_connected(&feed));
        assert!(health.audio_connected(&feed));
        health.mark_timeout();
        assert!(!health.video_connected(&feed));
        assert!(!health.audio_connected(&feed));
    }

    #[test]
    fn source_audio_health_is_independent_from_video_health() {
        let mut feed = test_feed();
        feed.audio.mute_standby_routine = false;
        let state = RuntimeState::default();
        let mut health = InputHealth::default();

        health.mark_video_frame();
        assert!(health.video_connected(&feed));
        assert!(!health.audio_connected(&feed));
        assert_eq!(
            desired_video_selector_mode(&feed, &state, health.video_connected(&feed)),
            VideoSelectorMode::Program
        );
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, health.audio_connected(&feed)),
            AudioSelectorMode::Silence
        );

        health.mark_audio_frame();
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, health.audio_connected(&feed)),
            AudioSelectorMode::Program
        );
    }

    #[test]
    fn priority_audio_forces_program_audio_off() {
        let mut feed = test_feed();
        feed.audio.mute_standby_routine = false;
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["*"],
            "queue_id": "q1",
            "data": {
                "queue_id": "q1",
                "audio_path": "runtime/audio/alerts/q1.pcm16le",
                "duration_ms": 60000
            }
        })));
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, true),
            AudioSelectorMode::Priority
        );
        assert!(priority_audio_active(&feed, &state));
    }

    #[test]
    fn priority_audio_feeder_status_reports_padding_and_errors() {
        gst::init().expect("gst init");
        let appsrc = gst::ElementFactory::make("appsrc")
            .build()
            .expect("appsrc")
            .downcast::<gst_app::AppSrc>()
            .expect("appsrc type");
        let reached_eof = Arc::new(AtomicBool::new(false));
        let reached_eof_at = Arc::new(Mutex::new(None));
        let error = Arc::new(Mutex::new(None));
        let feeder = PriorityAudioFeeder {
            appsrc,
            base_dir: PathBuf::new(),
            feed_id: "CAP-IT-ALL".to_string(),
            eof_drain: Duration::from_millis(1_000),
            active: Some(ActivePriorityAudioFeeder {
                queue_id: "q1".to_string(),
                source_status: "active",
                started_at: Instant::now(),
                release_after: Duration::from_secs(60),
                stop: Arc::new(AtomicBool::new(false)),
                reached_eof: Arc::clone(&reached_eof),
                reached_eof_at,
                finished: Arc::new(AtomicBool::new(false)),
                error: Arc::clone(&error),
                worker: None,
            }),
            completed: BTreeMap::new(),
        };

        assert_eq!(feeder.active_status().status, "active");
        reached_eof.store(true, Ordering::Relaxed);
        assert_eq!(feeder.active_status().status, "padding");
        *error.lock().expect("error lock") = Some("push failed".to_string());
        let status = feeder.active_status();
        assert_eq!(status.status, "error");
        assert_eq!(status.error.as_deref(), Some("push failed"));
    }

    #[test]
    fn completed_priority_audio_restores_program_audio() {
        let feed = test_feed();
        let status = PriorityAudioSourceStatus {
            status: "idle",
            queue_id: None,
            error: None,
        };

        assert_eq!(
            desired_audio_selector_mode_with_status(&feed, true, &status),
            AudioSelectorMode::Program
        );
    }

    #[test]
    fn active_priority_padding_keeps_priority_selected_for_drain() {
        let feed = test_feed();
        let status = PriorityAudioSourceStatus {
            status: "padding",
            queue_id: Some("q1".to_string()),
            error: None,
        };

        assert_eq!(
            desired_audio_selector_mode_with_status(&feed, true, &status),
            AudioSelectorMode::Priority
        );
    }

    #[test]
    fn priority_only_input_silences_when_no_priority_audio_is_live() {
        let mut feed = test_feed();
        feed.audio.mute_standby_routine = false;
        feed.priority_input.audio_source = "priority".to_string();
        let status = PriorityAudioSourceStatus {
            status: "idle",
            queue_id: None,
            error: None,
        };

        assert_eq!(
            desired_audio_selector_mode_with_status(&feed, true, &status),
            AudioSelectorMode::Silence
        );
    }

    #[test]
    fn priority_audio_without_path_forces_silence() {
        let mut feed = test_feed();
        feed.audio.mute_standby_routine = false;
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["*"],
            "queue_id": "q1",
            "data": {
                "queue_id": "q1",
                "duration_ms": 60000
            }
        })));
        assert_eq!(
            desired_audio_selector_mode(&feed, &state, true),
            AudioSelectorMode::Silence
        );
        assert!(priority_audio_active(&feed, &state));
    }

    #[test]
    fn priority_audio_missing_path_holds_silence_until_state_clears() {
        gst::init().expect("gst init");
        let appsrc = gst::ElementFactory::make("appsrc")
            .build()
            .expect("appsrc")
            .downcast::<gst_app::AppSrc>()
            .expect("appsrc type");
        let mut feeder = PriorityAudioFeeder::new(
            appsrc,
            PathBuf::new(),
            "CAP-IT-ALL".to_string(),
            Duration::from_millis(250),
        );
        let audio = crate::state::PriorityAudio {
            queue_id: "q-missing".to_string(),
            audio_path: None,
            duration_ms: Some(60_000),
            sample_rate: 48_000,
            channels: 2,
            alert_packet: None,
            banner_text: None,
            background_color: None,
            priority: None,
            started_at: chrono::Utc::now(),
        };

        let first = feeder.update(Some(&audio));
        assert_eq!(first.status, "silence");
        let held = feeder.update(Some(&audio));
        assert_eq!(held.status, "silence");
        let cleared = feeder.update(None);
        assert_eq!(cleared.status, "idle");
    }

    #[test]
    fn pcm_timing_helpers_make_realtime_chunks() {
        assert_eq!(pcm_chunk_bytes(48_000, 2, 20), 3840);
        assert_eq!(
            pcm_duration(48_000, 2, 3840),
            gst::ClockTime::from_mseconds(20)
        );
    }
    #[test]
    fn text_overlay_state_tracks_banner_presentation() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        let idle = text_overlay_state(&feed, &state);
        assert!(idle.silent);

        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Native CGEN crawl"}]
            }
        })));
        let active = text_overlay_state(&feed, &state);
        assert!(!active.silent);
        assert_eq!(active.text, "Native CGEN crawl");
        assert!(active.font_desc.starts_with("Arial "));
        assert!((active.ypos - 0.08).abs() < 0.01);
        assert!(active.x_absolute > 1.0);
    }

    #[test]
    fn text_overlay_state_does_not_start_visual_from_priority_audio_alone() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["CAP-IT-ALL"],
            "queue_id": "q-alert",
            "data": {
                "queue_id": "q-alert",
                "audio_path": "runtime/audio/alerts/q-alert.pcm16le",
                "duration_ms": 60000,
                "banner_text": "Priority alert crawl"
            }
        })));

        let active = text_overlay_state(&feed, &state);
        assert!(active.silent);
        assert_eq!(active.text, "");
    }

    #[test]
    fn ticker_visual_finishes_after_banner_clears() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        let mut controller = TextOverlayController::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "signature": "crawl-1",
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Native CGEN crawl should finish"}]
            }
        })));
        let active = controller.next_state(&feed, &state, false, true);
        assert!(!active.silent);
        assert_eq!(active.visual_id, "crawl-1");
        controller.last = Some(active);

        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": false,
                "signature": "crawl-1",
                "feed_id": "CAP-IT-ALL",
                "alerts": []
            }
        })));
        let retained = controller.next_state(&feed, &state, false, true);
        assert!(!retained.silent);
        assert_eq!(retained.source_text, "Native CGEN crawl should finish");
        assert_eq!(retained.visual_id, "crawl-1");
    }

    #[test]
    fn reissued_same_banner_restarts_after_retained_clear() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        let mut controller = TextOverlayController::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "signature": "crawl-1",
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Same crawl should restart"}]
            }
        })));
        let mut active = controller.next_state(&feed, &state, false, true);
        let old_start = Instant::now() - Duration::from_secs(12);
        controller.started_at = Some(old_start);
        active.started_at = old_start;
        controller.last = Some(active);

        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": false,
                "signature": "crawl-1",
                "feed_id": "CAP-IT-ALL",
                "alerts": []
            }
        })));
        let retained = controller.next_state(&feed, &state, false, true);
        assert!(!retained.silent);
        controller.last = Some(retained);

        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "signature": "crawl-1",
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Same crawl should restart"}]
            }
        })));
        let reissued = controller.next_state(&feed, &state, false, true);

        assert!(!reissued.silent);
        assert!(reissued.x_absolute > 1.0);
        assert_eq!(reissued.visual_id, "crawl-1");
    }

    #[test]
    fn operator_visual_override_clears_stale_ticker() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        let mut controller = TextOverlayController::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "signature": "crawl-1",
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Native CGEN crawl should not linger"}]
            }
        })));
        let active = controller.next_state(&feed, &state, false, true);
        controller.last = Some(active);
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": false,
                "signature": "crawl-1",
                "feed_id": "CAP-IT-ALL",
                "alerts": []
            }
        })));
        assert!(state.apply_event(&json!({
            "type": "cgen.control",
            "subject": "CAP-IT-ALL",
            "data": {
                "feed_id": "CAP-IT-ALL",
                "action": "smpte_bars",
                "smpte_bars": true
            }
        })));

        let cleared = controller.next_state(&feed, &state, false, true);

        assert!(cleared.silent);
        assert_eq!(cleared.visual_mode, "smpte");
        assert_eq!(cleared.text, "");
    }

    #[test]
    fn ticker_repeats_while_priority_audio_is_live() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        let mut controller = TextOverlayController::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"queue_id": "q1", "message": "Repeat crawl"}]
            }
        })));

        let mut active = controller.next_state(&feed, &state, false, true);
        active.started_at = Instant::now() - Duration::from_secs(120);
        controller.last = Some(active);
        let repeated = controller.next_state(&feed, &state, false, true);

        assert!(!repeated.silent);
        assert!(repeated.visual_id.ends_with("#pass2"));
        assert_eq!(repeated.source_text, "Repeat crawl");
    }

    #[test]
    fn ticker_runs_configured_extra_passes_after_audio_release() {
        let mut feed = test_feed();
        feed.banner.after_eom_repeats = 1;
        let mut state = RuntimeState::default();
        let mut controller = TextOverlayController::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"queue_id": "q1", "message": "Post EOM crawl"}]
            }
        })));

        let mut active = controller.next_state(&feed, &state, false, true);
        active.started_at = Instant::now() - Duration::from_secs(120);
        controller.last = Some(active);
        let post_eom = controller.next_state(&feed, &state, false, false);

        assert!(!post_eom.silent);
        assert!(post_eom.visual_id.ends_with("#pass2"));
    }

    #[test]
    fn fixed_ticker_repeats_ignore_audio_lifetime() {
        let mut feed = test_feed();
        feed.banner.scroll_repeat_mode = "fixed".to_string();
        feed.banner.fixed_repeats = 1;
        let mut state = RuntimeState::default();
        let mut controller = TextOverlayController::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"queue_id": "q1", "message": "One pass only"}]
            }
        })));

        let mut active = controller.next_state(&feed, &state, false, true);
        active.started_at = Instant::now() - Duration::from_secs(120);
        controller.last = Some(active);
        let cleared = controller.next_state(&feed, &state, false, true);

        assert!(cleared.silent);
    }

    #[test]
    fn ticker_overlay_clears_after_exiting_left_edge() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Short crawl"}]
            }
        })));
        let snapshot = crate::graphics::presentation_snapshot(&feed, &state, false);
        let (start_x, start_silent) =
            ticker_x_absolute(&feed, &snapshot, &snapshot.overlay_text, Instant::now());
        let (end_x, end_silent) = ticker_x_absolute(
            &feed,
            &snapshot,
            &snapshot.overlay_text,
            Instant::now() - Duration::from_secs(30),
        );

        assert!(start_x > 1.0);
        assert!(!start_silent);
        assert!(end_x < 0.0);
        assert!(end_silent);
    }

    #[test]
    fn ticker_scroll_speed_ignores_output_fps() {
        let mut feed = test_feed();
        feed.video.fps = "60000/1001".to_string();
        feed.banner.scroll_speed = 12;
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Wallclock crawl"}]
            }
        })));
        let snapshot = crate::graphics::presentation_snapshot(&feed, &state, false);

        let (x_absolute, silent) = ticker_x_absolute(
            &feed,
            &snapshot,
            &snapshot.overlay_text,
            Instant::now() - Duration::from_secs(1),
        );

        let x_px = x_absolute * f64::from(feed.video.width);
        let expected = f64::from(feed.video.width) + 1.0 - (12.0 * 30.0);
        assert!(!silent);
        assert!((x_px - expected).abs() < 2.0);
    }

    #[test]
    fn ticker_done_state_is_stable_after_full_exit() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Long enough crawl text to fully exit"}]
            }
        })));
        let snapshot = crate::graphics::presentation_snapshot(&feed, &state, false);
        let (x1, silent1) = ticker_x_absolute(
            &feed,
            &snapshot,
            &snapshot.overlay_text,
            Instant::now() - Duration::from_secs(120),
        );
        let (x2, silent2) = ticker_x_absolute(
            &feed,
            &snapshot,
            &snapshot.overlay_text,
            Instant::now() - Duration::from_secs(180),
        );

        assert_eq!((x1, silent1), (-1.0, true));
        assert_eq!((x2, silent2), (-1.0, true));
    }

    #[test]
    fn publish_status_uses_no_signal_compositor_state() {
        let mut feed = test_feed();
        feed.standby.mode = "banner".to_string();
        feed.standby.text = "Standby Details Channel".to_string();
        let (_state_tx, state_rx) = watch::channel(RuntimeState::default());
        let (status_tx, mut status_rx) = mpsc::unbounded_channel();

        publish_status(
            &Some(status_tx),
            &feed,
            &state_rx,
            true,
            json!({"media_backend": "test"}),
        );

        let status = status_rx.try_recv().expect("status payload");
        assert_eq!(status["no_signal"], true);
        assert_eq!(status["visual_mode"], "banner");
        assert_eq!(status["overlay_text"], "Standby Details Channel");
    }

    #[test]
    fn idle_overlay_state_does_not_require_frame_mutation() {
        let feed = test_feed();
        let state = RuntimeState::default();
        let idle = text_overlay_state(&feed, &state);

        assert!(!overlay_render_state_needs_frame_mutation(None));
        assert!(!overlay_render_state_needs_frame_mutation(Some(&idle)));

        let mut clock = idle.clone();
        clock.clock_text = "12:34".to_string();
        assert!(overlay_render_state_needs_frame_mutation(Some(&clock)));
    }

    #[test]
    fn preview_writer_due_check_rate_limits_attempts() {
        let mut writer = FramePreviewWriter::new(PathBuf::from("unused.preview.jpg"), 1, 1);
        let now = Instant::now();

        assert!(writer.write_due_at(now));
        writer.last_write = Some(now);
        assert!(!writer.write_due_at(now + FRAME_PREVIEW_INTERVAL / 2));
        assert!(writer.write_due_at(now + FRAME_PREVIEW_INTERVAL));
    }

    fn test_feed() -> FeedConfig {
        FeedConfig {
            id: "CAP-IT-ALL".to_string(),
            name: "CAP CGEN".to_string(),
            enabled: true,
            program_input: EndpointConfig {
                url: "udp://172.16.1.30:9098?fifo_size=2000000&overrun_nonfatal=1&reuse=1"
                    .to_string(),
                format: "mpegts".to_string(),
                ..Default::default()
            },
            priority_input: PriorityInputConfig {
                feed_id: "*".to_string(),
                audio_source: "both".to_string(),
                format: "priority-audio".to_string(),
            },
            program_output: EndpointConfig {
                url: "udp://239.0.0.2:9001?pkt_size=1316".to_string(),
                format: "mpegts".to_string(),
                vcodec: "mpeg2video".to_string(),
                acodec: "ac3".to_string(),
                video_bitrate_kbps: Some(12_000),
                audio_bitrate_kbps: Some(192),
            },
            program: Default::default(),
            priority: Default::default(),
            media: Default::default(),
            presentation: Default::default(),
            video: VideoConfig {
                width: 1920,
                height: 1080,
                fps: "30000/1001".to_string(),
                interlaced: true,
                field_order: "tff".to_string(),
                standard: "atsc".to_string(),
            },
            audio: AudioConfig {
                idle: "source".to_string(),
                alert_mode: "replace".to_string(),
                mute_standby_routine: true,
            },
            ladder: LadderConfig {
                videos: vec![VideoRenditionConfig {
                    id: "hd".to_string(),
                    enabled: "auto".to_string(),
                    width: 1920,
                    height: 1080,
                    fps: "30000/1001".to_string(),
                    interlaced: true,
                    field_order: "tff".to_string(),
                    standard: "atsc".to_string(),
                    vcodec: "mpeg2video".to_string(),
                    bitrate_kbps: Some(12_000),
                    program: Some(1),
                    video_pid: Some(0x100),
                    pmt_pid: Some(0x1000),
                }],
                audios: vec![
                    AudioRenditionConfig {
                        id: "stereo".to_string(),
                        enabled: "true".to_string(),
                        channels: 2,
                        acodec: "ac3".to_string(),
                        bitrate_kbps: Some(192),
                        language: "eng".to_string(),
                    },
                    AudioRenditionConfig {
                        id: "surround_51".to_string(),
                        enabled: "true".to_string(),
                        channels: 6,
                        acodec: "ac3".to_string(),
                        bitrate_kbps: Some(384),
                        language: "eng".to_string(),
                    },
                ],
            },
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
            standby: StandbyConfig::default(),
            sync: SyncConfig::default(),
            encoder: Default::default(),
        }
    }
}
