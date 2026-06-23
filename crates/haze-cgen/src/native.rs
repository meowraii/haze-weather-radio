use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use haze_media::{normalize_pcm, Pcm};
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecParameters, AVPacket};
use rsmpeg::avformat::{AVFormatContextInput, AVFormatContextOutput};
use rsmpeg::avutil::{AVChannelLayout, AVDictionary, AVFrame, AVRational};
use rsmpeg::swresample::SwrContext;
use rsmpeg::swscale::SwsContext;
use rsmpeg::{error::RsmpegError, ffi};
use serde_json::{json, Value};
use tokio::sync::{mpsc, watch};
use tokio::time::sleep;
use tracing::{info, warn};

use crate::config::{AudioRenditionConfig, FeedConfig, SyncConfig, VideoRenditionConfig};
use crate::graphics::NativeGraphicsRenderer;
use crate::state::{PriorityAudio, RuntimeState};

const ALERT_SAMPLE_RATE: u32 = 48_000;
const ALERT_CHANNELS: u16 = 2;
const MAX_OUTPUT_AUDIO_CHANNELS: u16 = 8;
const AC3_FRAME_SAMPLES: usize = 1536;
const DEFAULT_AC3_BITRATE: i64 = 192_000;
#[allow(dead_code)]
const DEFAULT_VIDEO_BITRATE: i64 = 12_000_000;
#[cfg(test)]
const CLOCK_CORRECTION_GAIN: f64 = 0.02;

#[derive(Debug, Clone)]
struct StreamSpec {
    input_index: usize,
    output_index: i32,
    input_time_base: AVRational,
    output_time_base: AVRational,
}

#[derive(Debug, Clone)]
struct InputStreamSpec {
    input_index: usize,
    time_base: AVRational,
    start_time: i64,
    #[allow(dead_code)]
    frame_rate: AVRational,
    codecpar: AVCodecParameters,
}

#[derive(Debug, Clone)]
struct OutputProgram {
    program_number: i32,
    pmt_pid: i32,
    pcr_pid: i32,
    stream_indexes: Vec<i32>,
}

struct VideoOutput {
    rendition: VideoRenditionConfig,
    processor: VideoProcessor,
    spec: StreamSpec,
}

struct AudioOutput {
    rendition: AudioRenditionConfig,
    processor: AudioProcessor,
    spec: StreamSpec,
    program_number: i32,
}

struct AuxiliaryOutput {
    input_index: usize,
    spec: StreamSpec,
}

#[derive(Debug, Clone)]
struct CgenClock {
    sync: SyncConfig,
    audio_next_pts: Option<i64>,
    audio_drift_ms: f64,
}

impl CgenClock {
    fn new(sync: SyncConfig) -> Self {
        Self {
            sync,
            audio_next_pts: None,
            audio_drift_ms: 0.0,
        }
    }

    fn prepare_audio_target(&mut self, target_pts: i64, time_base: AVRational) -> bool {
        let current = self.audio_next_pts.unwrap_or_default();
        self.audio_drift_ms = pts_delta_ms(current.saturating_sub(target_pts), time_base);
        if self.audio_drift_ms.abs() > f64::from(self.sync.hard_reset_ms.max(1)) {
            self.audio_next_pts = Some(target_pts.max(0));
            self.audio_drift_ms = 0.0;
            return true;
        }
        if self.audio_next_pts.is_none() {
            self.audio_next_pts = Some(target_pts.max(0));
        }
        false
    }

    fn audio_pts(&self, target_pts: i64) -> i64 {
        self.audio_next_pts.unwrap_or(target_pts)
    }

    fn ensure_audio_started_at(&mut self, pts: i64) {
        if self.audio_next_pts.is_none() {
            self.audio_next_pts = Some(pts);
        }
    }

    fn reset_audio(&mut self) {
        self.audio_next_pts = None;
        self.audio_drift_ms = 0.0;
    }

    fn reset_audio_to(&mut self, pts: i64) {
        self.audio_next_pts = Some(pts);
        self.audio_drift_ms = 0.0;
    }

    fn advance_audio(&mut self, duration: i64) {
        let pts = self.audio_next_pts.unwrap_or_default();
        self.audio_next_pts = Some(pts.saturating_add(duration));
    }

    fn finish_audio_target(&mut self, target_pts: i64, time_base: AVRational) -> bool {
        let final_pts = self.audio_next_pts.unwrap_or(target_pts);
        self.audio_drift_ms = pts_delta_ms(final_pts.saturating_sub(target_pts), time_base);
        if self.audio_drift_ms < -(f64::from(self.sync.hard_reset_ms.max(1))) {
            self.audio_next_pts = Some(target_pts);
            self.audio_drift_ms = 0.0;
            return true;
        }
        false
    }

    fn max_audio_frames_per_video(&self) -> usize {
        usize::try_from(self.sync.max_audio_frames_per_video.max(1))
            .unwrap_or(12)
            .clamp(1, 64)
    }

    fn drift_ms(&self) -> f64 {
        self.audio_drift_ms
    }
}

pub(crate) async fn run_remux_supervised(
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    base_dir: PathBuf,
    status_tx: Option<mpsc::UnboundedSender<Value>>,
) -> Result<()> {
    let reconnect_initial = feed.sync.reconnect_initial_ms.max(50);
    let reconnect_max = feed.sync.reconnect_max_ms.max(reconnect_initial).max(100);
    let mut restart_delay = Duration::from_millis(u64::from(reconnect_initial));
    loop {
        let worker_feed = feed.clone();
        let worker_state_rx = state_rx.clone();
        let worker_base_dir = base_dir.clone();
        let worker_status_tx = status_tx.clone();
        info!(
            feed_id = %feed.id,
            input = %feed.program_input_url(),
            output = %feed.program_output_url(),
            "starting native rsmpeg cgen constant encoder"
        );
        let result = tokio::task::spawn_blocking(move || {
            encode_once(
                &worker_feed,
                worker_state_rx,
                &worker_base_dir,
                worker_status_tx,
            )
        })
        .await
        .context("native cgen encoder worker panicked")?;
        match result {
            Ok(()) => {
                info!(feed_id = %feed.id, "native rsmpeg cgen encoder exited cleanly");
                restart_delay = Duration::from_millis(500);
            }
            Err(err) => {
                warn!(feed_id = %feed.id, "native rsmpeg cgen encoder failed: {err:#}");
                publish_status(
                    &status_tx,
                    &feed,
                    json!({
                        "feed_id": feed.id.as_str(),
                        "input_connected": false,
                        "output_active": false,
                        "no_signal": true,
                        "last_error": err.to_string(),
                        "reconnect_delay_ms": restart_delay.as_millis(),
                    }),
                );
            }
        }
        sleep(restart_delay).await;
        restart_delay = (restart_delay * 2).min(Duration::from_millis(u64::from(reconnect_max)));
    }
}

fn encode_once(
    feed: &FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    base_dir: &Path,
    status_tx: Option<mpsc::UnboundedSender<Value>>,
) -> Result<()> {
    let input_url = cstring_arg(feed.program_input_url(), "program input url")?;
    let output_url = cstring_arg(feed.program_output_url(), "program output url")?;
    let output_format = cstring_arg(
        non_empty(feed.output().format.as_str()).unwrap_or("mpegts"),
        "output format",
    )?;

    let mut input = AVFormatContextInput::open(&input_url).with_context(|| {
        format!(
            "failed to open native cgen input {}",
            feed.program_input_url()
        )
    })?;
    publish_status(
        &status_tx,
        feed,
        json!({
            "feed_id": feed.id.as_str(),
            "input_connected": true,
            "output_active": false,
            "input_url": feed.program_input_url(),
            "renderer_backend": "rsmpeg/wgpu-cpu-fallback",
        }),
    );
    let input_specs = collect_input_specs(&input)?;
    if input_specs.is_empty() {
        bail!("native cgen input has no streams");
    }

    let video_input_index = input_specs
        .iter()
        .find(|spec| spec.codecpar.codec_type == ffi::AVMEDIA_TYPE_VIDEO)
        .map(|spec| spec.input_index)
        .context("native cgen input has no video stream")?;
    let video_spec = input_specs
        .iter()
        .find(|spec| spec.input_index == video_input_index)
        .context("native cgen video stream disappeared")?;
    let source_width = u32::try_from(video_spec.codecpar.width.max(0)).unwrap_or_default();
    let source_height = u32::try_from(video_spec.codecpar.height.max(0)).unwrap_or_default();
    let video_renditions = feed.enabled_video_renditions(source_width, source_height);
    let audio_renditions = feed.enabled_audio_renditions();
    if video_renditions.is_empty() {
        bail!("native cgen has no enabled video renditions");
    }
    if audio_renditions.is_empty() {
        bail!("native cgen has no enabled audio renditions");
    }
    let mut output = AVFormatContextOutput::builder()
        .format_name(output_format.as_c_str())
        .filename(&output_url)
        .build()
        .with_context(|| {
            format!(
                "failed to open native cgen output {} as {}",
                feed.program_output_url(),
                output_format.to_string_lossy()
            )
        })?;

    let mut video_outputs = Vec::with_capacity(video_renditions.len());
    let mut programs = Vec::with_capacity(video_renditions.len());
    for (index, rendition) in video_renditions.into_iter().enumerate() {
        let processor = VideoProcessor::new(feed, video_spec, &rendition, state_rx.clone())?;
        let video_pid = rendition.video_pid(index);
        let spec = add_output_stream(
            &mut output,
            video_spec,
            processor.output_codecpar(),
            processor.output_time_base(),
            video_pid,
        );
        let program_number = rendition.program_number(index);
        let pmt_pid = rendition.pmt_pid(index);
        programs.push(OutputProgram {
            program_number,
            pmt_pid,
            pcr_pid: video_pid,
            stream_indexes: vec![spec.output_index],
        });
        video_outputs.push(VideoOutput {
            rendition,
            processor,
            spec,
        });
    }

    let audio_input_indices = input_specs
        .iter()
        .filter(|spec| spec.codecpar.codec_type == ffi::AVMEDIA_TYPE_AUDIO)
        .map(|spec| spec.input_index)
        .collect::<Vec<_>>();
    if audio_input_indices.is_empty() {
        warn!(
            feed_id = %feed.id,
            "native cgen input has no audio stream; CGEN will output silence until priority alert audio appears"
        );
    }
    let mut audio_outputs = Vec::new();
    for (program_index, program) in programs.iter_mut().enumerate() {
        for (audio_index, rendition) in audio_renditions.iter().enumerate() {
            let Some(audio_spec) = select_audio_source(&input_specs, rendition.channels()) else {
                continue;
            };
            let processor = AudioProcessor::new(
                feed,
                audio_spec,
                rendition,
                video_spec,
                state_rx.clone(),
                base_dir.to_path_buf(),
            )?;
            let spec = add_output_stream(
                &mut output,
                audio_spec,
                processor.output_codecpar(),
                AVRational {
                    num: 1,
                    den: ALERT_SAMPLE_RATE as i32,
                },
                audio_pid(program_index, audio_index),
            );
            program.stream_indexes.push(spec.output_index);
            audio_outputs.push(AudioOutput {
                rendition: rendition.clone(),
                processor,
                spec,
                program_number: program.program_number,
            });
        }
    }
    let mut auxiliary_outputs = Vec::new();
    for spec in input_specs.iter().filter(|spec| {
        spec.input_index != video_input_index
            && spec.codecpar.codec_type != ffi::AVMEDIA_TYPE_AUDIO
            && is_auxiliary_passthrough_stream(spec.codecpar.codec_type)
    }) {
        let mut codecpar = AVCodecParameters::new();
        codecpar.copy(&spec.codecpar);
        let out_spec = add_output_stream(
            &mut output,
            spec,
            codecpar,
            spec.time_base,
            auxiliary_pid(auxiliary_outputs.len()),
        );
        for program in &mut programs {
            program.stream_indexes.push(out_spec.output_index);
        }
        auxiliary_outputs.push(AuxiliaryOutput {
            input_index: spec.input_index,
            spec: out_spec,
        });
    }
    create_output_programs(&mut output, &programs)?;
    let mut header_options = None;
    output
        .write_header(&mut header_options)
        .context("failed to write native cgen output header")?;
    refresh_output_specs(
        &output,
        &mut video_outputs,
        &mut audio_outputs,
        &mut auxiliary_outputs,
    );
    publish_status(
        &status_tx,
        feed,
        json!({
            "feed_id": feed.id.as_str(),
            "input_connected": true,
            "output_active": true,
            "no_signal": false,
            "programs": programs.iter().map(|program| program.program_number).collect::<Vec<_>>(),
            "video_streams": video_outputs.len(),
            "audio_streams": audio_outputs.len(),
            "auxiliary_streams": auxiliary_outputs.len(),
            "active_video_renditions": video_outputs.iter().map(|output| output.rendition.id.as_str()).collect::<Vec<_>>(),
            "active_audio_renditions": audio_outputs.iter().map(|output| output.rendition.id.as_str()).collect::<Vec<_>>(),
            "width": video_outputs.first().map(|output| output.rendition.width).unwrap_or(feed.video.width),
            "height": video_outputs.first().map(|output| output.rendition.height).unwrap_or(feed.video.height),
            "renderer_backend": "rsmpeg/wgpu-cpu-fallback",
        }),
    );
    let mut status_clock = Instant::now();
    let mut last_status_frames = 0u64;
    let status_interval = Duration::from_millis(u64::from(feed.sync.status_interval_ms.max(100)));
    while let Some(mut packet) = input
        .read_packet()
        .context("native cgen read packet failed")?
    {
        let input_index = packet.stream_index as usize;
        if input_index == video_input_index {
            for video_output in &mut video_outputs {
                video_output
                    .processor
                    .process_packet(&packet, &video_output.spec, &mut output)?;
            }
            let reference_next_pts = video_outputs
                .first()
                .map(|output| {
                    (
                        output.processor.next_pts,
                        output.processor.output_time_base(),
                    )
                })
                .unwrap_or((0, AVRational { num: 1, den: 30 }));
            for audio_output in &mut audio_outputs {
                let target_audio_pts = rescale_q(
                    reference_next_pts.0,
                    reference_next_pts.1,
                    audio_output.spec.output_time_base,
                );
                audio_output.processor.write_mixed_until(
                    target_audio_pts,
                    &audio_output.spec,
                    &mut output,
                )?;
            }
            if status_clock.elapsed() >= status_interval {
                let elapsed = status_clock.elapsed().as_secs_f64().max(0.001);
                let encoded_frames = video_outputs
                    .first()
                    .map(|output| output.processor.encoded_frames)
                    .unwrap_or(0);
                let frames = encoded_frames.saturating_sub(last_status_frames);
                let fps = frames as f64 / elapsed;
                publish_status(
                    &status_tx,
                    feed,
                    json!({
                        "feed_id": feed.id.as_str(),
                        "input_connected": true,
                        "output_active": true,
                        "current_fps": fps,
                        "audio_video_drift_ms": audio_outputs.iter().map(|output| output.processor.drift_ms()).fold(0.0_f64, |acc, value| acc.max(value.abs())),
                        "active_alert_queue_id": audio_outputs.iter().find_map(|output| output.processor.active_queue_id()),
                        "programs": programs.iter().map(|program| program.program_number).collect::<Vec<_>>(),
                        "renderer_backend": "rsmpeg/wgpu-cpu-fallback",
                    }),
                );
                status_clock = Instant::now();
                last_status_frames = encoded_frames;
            }
            continue;
        }
        let mut handled_audio = false;
        for audio_output in audio_outputs
            .iter_mut()
            .filter(|output| output.processor.input_index == input_index)
        {
            audio_output
                .processor
                .process_packet(&packet, &audio_output.spec, &mut output)?;
            handled_audio = true;
        }
        if handled_audio {
            continue;
        }
        if let Some(auxiliary) = auxiliary_outputs
            .iter()
            .find(|output| output.input_index == input_index)
        {
            packet.rescale_ts(
                auxiliary.spec.input_time_base,
                auxiliary.spec.output_time_base,
            );
            packet.set_stream_index(auxiliary.spec.output_index);
            packet.set_pos(-1);
            output
                .interleaved_write_frame(&mut packet)
                .context("native cgen write auxiliary packet failed")?;
        }
    }
    output
        .write_trailer()
        .context("failed to write native cgen trailer")
}

fn publish_status(
    status_tx: &Option<mpsc::UnboundedSender<Value>>,
    feed: &FeedConfig,
    mut data: Value,
) {
    if let Some(object) = data.as_object_mut() {
        object
            .entry("feed_id")
            .or_insert_with(|| Value::String(feed.id.clone()));
    }
    if let Some(tx) = status_tx {
        let _ = tx.send(data);
    }
}

fn collect_input_specs(input: &AVFormatContextInput) -> Result<Vec<InputStreamSpec>> {
    let mut out = Vec::with_capacity(input.streams().len());
    for stream in input.streams() {
        let mut codecpar = AVCodecParameters::new();
        codecpar.copy(&stream.codecpar());
        out.push(InputStreamSpec {
            input_index: stream.index as usize,
            time_base: stream.time_base,
            start_time: stream.start_time,
            frame_rate: stream
                .guess_framerate()
                .unwrap_or(AVRational { num: 0, den: 1 }),
            codecpar,
        });
    }
    Ok(out)
}

fn add_output_stream(
    output: &mut AVFormatContextOutput,
    input_spec: &InputStreamSpec,
    codecpar: AVCodecParameters,
    time_base: AVRational,
    pid: i32,
) -> StreamSpec {
    let mut stream = output.new_stream();
    stream.set_codecpar(codecpar);
    stream.set_time_base(time_base);
    // SAFETY: `stream` is newly allocated on this muxer and uniquely borrowed
    // here. `id` is the MPEG-TS stream PID field FFmpeg expects callers to set
    // before the muxer header is written.
    unsafe {
        (*stream.as_mut_ptr()).id = pid;
    }
    StreamSpec {
        input_index: input_spec.input_index,
        output_index: stream.index,
        input_time_base: input_spec.time_base,
        output_time_base: stream.time_base,
    }
}

fn select_audio_source(
    input_specs: &[InputStreamSpec],
    desired_channels: u16,
) -> Option<&InputStreamSpec> {
    let audio_specs = input_specs
        .iter()
        .filter(|spec| spec.codecpar.codec_type == ffi::AVMEDIA_TYPE_AUDIO)
        .collect::<Vec<_>>();
    if desired_channels > 2 {
        audio_specs
            .iter()
            .copied()
            .find(|spec| audio_output_channels(&spec.codecpar) >= desired_channels)
            .or_else(|| audio_specs.first().copied())
    } else {
        audio_specs
            .iter()
            .copied()
            .find(|spec| audio_output_channels(&spec.codecpar) <= 2)
            .or_else(|| audio_specs.first().copied())
    }
}

fn audio_pid(program_index: usize, audio_index: usize) -> i32 {
    0x101
        + (i32::try_from(program_index).unwrap_or(0) * 0x20)
        + i32::try_from(audio_index).unwrap_or(0)
}

fn auxiliary_pid(index: usize) -> i32 {
    0x1f0 + i32::try_from(index).unwrap_or(0)
}

fn create_output_programs(
    output: &mut AVFormatContextOutput,
    programs: &[OutputProgram],
) -> Result<()> {
    for program in programs {
        // SAFETY: the output context is alive and not yet header-written.
        // FFmpeg owns the returned AVProgram, and we only fill public program
        // metadata plus stream indexes that were just allocated on this muxer.
        unsafe {
            let raw = ffi::av_new_program(output.as_mut_ptr(), program.program_number);
            if raw.is_null() {
                bail!(
                    "failed to create native cgen output program {}",
                    program.program_number
                );
            }
            (*raw).program_num = program.program_number;
            (*raw).pmt_pid = program.pmt_pid;
            (*raw).pcr_pid = program.pcr_pid;
            for index in &program.stream_indexes {
                let index = u32::try_from(*index).context("native cgen stream index overflow")?;
                ffi::av_program_add_stream_index(
                    output.as_mut_ptr(),
                    program.program_number,
                    index,
                );
            }
        }
    }
    Ok(())
}

fn is_auxiliary_passthrough_stream(codec_type: ffi::AVMediaType) -> bool {
    matches!(
        codec_type,
        ffi::AVMEDIA_TYPE_DATA
            | ffi::AVMEDIA_TYPE_SUBTITLE
            | ffi::AVMEDIA_TYPE_ATTACHMENT
            | ffi::AVMEDIA_TYPE_UNKNOWN
    )
}

fn refresh_stream_spec(output: &AVFormatContextOutput, spec: &mut StreamSpec) {
    if let Some(stream) = output
        .streams()
        .iter()
        .find(|stream| stream.index == spec.output_index)
    {
        spec.output_time_base = stream.time_base;
    }
}

fn refresh_output_specs(
    output: &AVFormatContextOutput,
    video_outputs: &mut [VideoOutput],
    audio_outputs: &mut [AudioOutput],
    auxiliary_outputs: &mut [AuxiliaryOutput],
) {
    for video in video_outputs {
        refresh_stream_spec(output, &mut video.spec);
    }
    for audio in audio_outputs {
        refresh_stream_spec(output, &mut audio.spec);
    }
    for auxiliary in auxiliary_outputs {
        refresh_stream_spec(output, &mut auxiliary.spec);
    }
}

#[cfg(test)]
#[derive(Debug, Default)]
struct CorrectedWallClock {
    started_at: Option<Instant>,
    anchor_pts: Option<i64>,
    correction_ticks: f64,
}

#[cfg(test)]
impl CorrectedWallClock {
    fn observe_media_pts(
        &mut self,
        media_pts: i64,
        media_time_base: AVRational,
        output_time_base: AVRational,
    ) {
        let observed = rescale_q(media_pts, media_time_base, output_time_base);
        if self.anchor_pts.is_none() {
            self.started_at = Some(Instant::now());
            self.anchor_pts = Some(observed);
            self.correction_ticks = 0.0;
            return;
        }

        let Some(target) = self.target_pts(output_time_base) else {
            return;
        };
        let ticks_per_second = ticks_per_second(output_time_base);
        let error = observed.saturating_sub(target);
        if error.unsigned_abs() > (ticks_per_second * 5.0) as u64 {
            self.started_at = Some(Instant::now());
            self.anchor_pts = Some(observed);
            self.correction_ticks = 0.0;
            return;
        }
        if error.unsigned_abs() > (ticks_per_second * 0.05) as u64 {
            let max_correction = ticks_per_second * 0.25;
            self.correction_ticks = (self.correction_ticks
                + (error as f64 * CLOCK_CORRECTION_GAIN))
                .clamp(-max_correction, max_correction);
        }
    }

    fn target_pts(&self, output_time_base: AVRational) -> Option<i64> {
        let started_at = self.started_at?;
        let anchor_pts = self.anchor_pts?;
        Some(
            anchor_pts
                .saturating_add(duration_to_pts(started_at.elapsed(), output_time_base))
                .saturating_add(self.correction_ticks.round() as i64),
        )
    }
}

#[cfg(test)]
fn duration_to_pts(duration: Duration, output_time_base: AVRational) -> i64 {
    (duration.as_secs_f64() * ticks_per_second(output_time_base)).round() as i64
}

#[cfg(test)]
fn ticks_per_second(time_base: AVRational) -> f64 {
    f64::from(time_base.den.max(1)) / f64::from(time_base.num.max(1))
}

struct VideoProcessor {
    feed_id: String,
    decoder: AVCodecContext,
    encoder: AVCodecContext,
    scaler: Option<SwsContext>,
    encoder_width: i32,
    encoder_height: i32,
    encoder_pix_fmt: i32,
    source_frame_rate: AVRational,
    output_frame_rate: AVRational,
    decoded_frames: u64,
    encoded_frames: u64,
    convert_frame_rate: bool,
    interlaced: bool,
    field_order: ffi::AVFieldOrder,
    graphics: NativeGraphicsRenderer,
    pending_field_frame: Option<AVFrame>,
    next_pts: i64,
}

impl VideoProcessor {
    fn new(
        feed: &FeedConfig,
        video_spec: &InputStreamSpec,
        rendition: &VideoRenditionConfig,
        state_rx: watch::Receiver<RuntimeState>,
    ) -> Result<Self> {
        let decoder = create_video_decoder(&video_spec.codecpar)?;
        let encoder = create_video_encoder(feed, video_spec, rendition)?;
        let encoder_pix_fmt = encoder.pix_fmt;
        let output_frame_rate = encoder.framerate;
        let source_frame_rate = if valid_rational(&video_spec.frame_rate) {
            video_spec.frame_rate
        } else {
            output_frame_rate
        };
        let output_fps = rendition.frame_rate_text(&feed.video.fps);
        let convert_frame_rate = parse_rational(output_fps).is_some()
            && valid_rational(&source_frame_rate)
            && valid_rational(&output_frame_rate)
            && rational_cmp(source_frame_rate, output_frame_rate) != std::cmp::Ordering::Equal;
        let mut render_feed = feed.clone();
        render_feed.video.width = rendition.width;
        render_feed.video.height = rendition.height;
        render_feed.video.fps = output_fps.to_string();
        render_feed.video.interlaced = rendition.interlaced;
        render_feed.video.field_order = rendition.field_order.clone();
        render_feed.video.standard = rendition.standard.clone();
        if render_feed.graphics.banner_width == 0
            || render_feed.graphics.banner_width == feed.video.width
        {
            render_feed.graphics.banner_width = rendition.width;
        }
        Ok(Self {
            feed_id: feed.id.clone(),
            decoder,
            encoder,
            scaler: None,
            encoder_width: rendition.width as i32,
            encoder_height: rendition.height as i32,
            encoder_pix_fmt,
            source_frame_rate,
            output_frame_rate,
            decoded_frames: 0,
            encoded_frames: 0,
            convert_frame_rate,
            interlaced: rendition.interlaced,
            field_order: video_field_order(&rendition.field_order),
            graphics: NativeGraphicsRenderer::new(&render_feed, state_rx),
            pending_field_frame: None,
            next_pts: 0,
        })
    }

    fn output_codecpar(&self) -> AVCodecParameters {
        let mut codecpar = self.encoder.extract_codecpar();
        // SAFETY: `extract_codecpar` returns an owned AVCodecParameters for
        // this output stream. We only normalize plain stream metadata before
        // passing ownership to libavformat.
        unsafe {
            let raw = codecpar.as_mut_ptr();
            (*raw).codec_type = ffi::AVMEDIA_TYPE_VIDEO;
            (*raw).codec_tag = 0;
            (*raw).field_order = if self.interlaced {
                self.field_order
            } else {
                ffi::AV_FIELD_PROGRESSIVE
            };
        }
        codecpar
    }

    fn output_time_base(&self) -> AVRational {
        self.encoder.time_base
    }

    fn process_packet(
        &mut self,
        source_packet: &AVPacket,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        self.decoder
            .send_packet(Some(source_packet))
            .context("failed to send native cgen source video packet to decoder")?;
        loop {
            match self.decoder.receive_frame() {
                Ok(frame) => self.process_frame(&frame, spec, output)?,
                Err(err) if is_again(&err) => return Ok(()),
                Err(err) => return Err(err).context("failed to decode native cgen source video"),
            }
        }
    }

    fn process_frame(
        &mut self,
        source_frame: &AVFrame,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        if self.should_weave_fields() {
            let frame = self.normalize_frame(source_frame)?;
            if let Some(first_field) = self.pending_field_frame.take() {
                let mut woven = self.weave_field_pair(&first_field, &frame)?;
                copy_a53_side_data(source_frame, &mut woven)
                    .context("failed to preserve native cgen caption side data")?;
                self.stamp_frame_metadata(&mut woven);
                woven.set_pts(self.frame_pts(source_frame, spec));
                return self.encode_frame(&mut woven, spec, output);
            }
            self.pending_field_frame = Some(frame);
            self.decoded_frames = self.decoded_frames.saturating_add(1);
            return Ok(());
        }

        if !self.should_emit_source_frame() {
            return Ok(());
        }
        let mut frame = self.normalize_frame(source_frame)?;
        self.stamp_frame_metadata(&mut frame);
        frame.set_pts(self.frame_pts(source_frame, spec));
        self.encode_frame(&mut frame, spec, output)
    }

    fn encode_frame(
        &mut self,
        frame: &mut AVFrame,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        self.graphics.render_frame(frame, self.encoded_frames);
        self.encoder
            .send_frame(Some(frame))
            .context("failed to send native cgen video frame to encoder")?;
        self.encoded_frames = self.encoded_frames.saturating_add(1);
        self.drain_packets(spec, output)
    }

    fn should_weave_fields(&self) -> bool {
        self.interlaced
            && self.convert_frame_rate
            && valid_rational(&self.source_frame_rate)
            && valid_rational(&self.output_frame_rate)
            && rational_cmp(
                AVRational {
                    num: self.output_frame_rate.num.saturating_mul(2),
                    den: self.output_frame_rate.den,
                },
                self.source_frame_rate,
            ) == std::cmp::Ordering::Equal
    }

    fn should_emit_source_frame(&mut self) -> bool {
        let decoded_index = self.decoded_frames;
        self.decoded_frames = self.decoded_frames.saturating_add(1);
        if !self.convert_frame_rate || self.encoded_frames == 0 {
            return true;
        }
        let lhs = u128::from(decoded_index)
            * self.output_frame_rate.num as u128
            * self.source_frame_rate.den as u128;
        let rhs = u128::from(self.encoded_frames)
            * self.source_frame_rate.num as u128
            * self.output_frame_rate.den as u128;
        lhs >= rhs
    }

    fn normalize_frame(&mut self, source_frame: &AVFrame) -> Result<AVFrame> {
        if source_frame.width == self.encoder_width
            && source_frame.height == self.encoder_height
            && source_frame.format == self.encoder_pix_fmt
        {
            let mut frame = source_frame.clone();
            frame
                .make_writable()
                .context("failed to make native cgen video frame writable")?;
            return Ok(frame);
        }

        let mut frame = AVFrame::new();
        frame.set_width(self.encoder_width);
        frame.set_height(self.encoder_height);
        frame.set_format(self.encoder_pix_fmt);
        frame
            .alloc_buffer()
            .context("failed to allocate native cgen video frame")?;
        let scaler = self.scaler.take();
        self.scaler = Some(if let Some(scaler) = scaler {
            scaler
                .get_cached_context(
                    source_frame.width,
                    source_frame.height,
                    source_frame.format,
                    self.encoder_width,
                    self.encoder_height,
                    self.encoder_pix_fmt,
                    ffi::SWS_BICUBIC,
                    None,
                    None,
                    None,
                )
                .context("failed to reuse native cgen video scaler")?
        } else {
            SwsContext::get_context(
                source_frame.width,
                source_frame.height,
                source_frame.format,
                self.encoder_width,
                self.encoder_height,
                self.encoder_pix_fmt,
                ffi::SWS_BICUBIC,
                None,
                None,
                None,
            )
            .context("failed to create native cgen video scaler")?
        });
        self.scaler
            .as_mut()
            .context("native cgen video scaler is unavailable")?
            .scale_frame(source_frame, 0, source_frame.height, &mut frame)
            .context("failed to scale native cgen video frame")?;
        copy_a53_side_data(source_frame, &mut frame)
            .context("failed to preserve native cgen caption side data")?;
        Ok(frame)
    }

    fn frame_pts(&mut self, source_frame: &AVFrame, spec: &StreamSpec) -> i64 {
        let _ = (source_frame, spec);
        let pts = self.next_pts;
        self.next_pts = pts.saturating_add(1);
        pts
    }

    fn stamp_frame_metadata(&self, frame: &mut AVFrame) {
        // SAFETY: `frame` is uniquely borrowed and points to a valid AVFrame
        // owned by rsmpeg. We only update FFmpeg's per-frame metadata flags.
        unsafe {
            let raw = frame.as_mut_ptr();
            (*raw).pkt_dts = ffi::AV_NOPTS_VALUE;
            (*raw).duration = 1;
            (*raw).time_base = self.encoder.time_base;
            if self.interlaced {
                (*raw).flags |= ffi::AV_FRAME_FLAG_INTERLACED as i32;
                if self.field_order == ffi::AV_FIELD_TT || self.field_order == ffi::AV_FIELD_TB {
                    (*raw).flags |= ffi::AV_FRAME_FLAG_TOP_FIELD_FIRST as i32;
                } else {
                    (*raw).flags &= !(ffi::AV_FRAME_FLAG_TOP_FIELD_FIRST as i32);
                }
            } else {
                (*raw).flags &= !(ffi::AV_FRAME_FLAG_INTERLACED as i32);
                (*raw).flags &= !(ffi::AV_FRAME_FLAG_TOP_FIELD_FIRST as i32);
            }
        }
    }

    fn weave_field_pair(&self, first: &AVFrame, second: &AVFrame) -> Result<AVFrame> {
        let mut frame = AVFrame::new();
        frame.set_width(self.encoder_width);
        frame.set_height(self.encoder_height);
        frame.set_format(self.encoder_pix_fmt);
        frame
            .alloc_buffer()
            .context("failed to allocate native cgen interlaced video frame")?;
        frame
            .make_writable()
            .context("failed to make native cgen interlaced video frame writable")?;
        weave_frame_planes(first, second, &mut frame, self.field_order)
            .context("failed to weave native cgen interlaced video frame")?;
        Ok(frame)
    }

    fn drain_packets(
        &mut self,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        loop {
            match self.encoder.receive_packet() {
                Ok(mut packet) => {
                    packet.rescale_ts(self.encoder.time_base, spec.output_time_base);
                    packet.set_stream_index(spec.output_index);
                    packet.set_pos(-1);
                    output
                        .interleaved_write_frame(&mut packet)
                        .with_context(|| {
                            format!(
                                "native cgen write encoded video packet failed for {}",
                                self.feed_id
                            )
                        })?;
                }
                Err(err) if is_again(&err) => return Ok(()),
                Err(err) => return Err(err).context("failed to receive native cgen video packet"),
            }
        }
    }
}

fn weave_frame_planes(
    first: &AVFrame,
    second: &AVFrame,
    out: &mut AVFrame,
    field_order: ffi::AVFieldOrder,
) -> Result<()> {
    let top_field_first = field_order != ffi::AV_FIELD_BB && field_order != ffi::AV_FIELD_BT;
    for plane in 0..3 {
        let Some((rows, row_bytes)) = frame_plane_layout(out.format, out.width, out.height, plane)
        else {
            continue;
        };
        if out.data[plane].is_null() || first.data[plane].is_null() || second.data[plane].is_null()
        {
            continue;
        }
        let dst_stride = usize::try_from(out.linesize[plane])
            .context("native cgen output video plane has negative stride")?;
        let first_stride = usize::try_from(first.linesize[plane])
            .context("native cgen first field video plane has negative stride")?;
        let second_stride = usize::try_from(second.linesize[plane])
            .context("native cgen second field video plane has negative stride")?;
        if row_bytes > dst_stride || row_bytes > first_stride || row_bytes > second_stride {
            bail!("native cgen video plane stride is too small for field weaving");
        }
        for row in 0..rows {
            let first_has_row = if top_field_first {
                row % 2 == 0
            } else {
                row % 2 != 0
            };
            let src_base = if first_has_row {
                first.data[plane]
            } else {
                second.data[plane]
            };
            let src_stride = if first_has_row {
                first_stride
            } else {
                second_stride
            };
            // SAFETY: all three frames are allocated AVFrames with positive
            // strides verified above. `row_bytes` is bounded by each stride,
            // and every row index is within the active plane height.
            unsafe {
                ptr::copy_nonoverlapping(
                    src_base.add(row * src_stride),
                    out.data[plane].add(row * dst_stride),
                    row_bytes,
                );
            }
        }
    }
    Ok(())
}

fn copy_a53_side_data(source: &AVFrame, target: &mut AVFrame) -> Result<bool> {
    let Some(side_data) = source.get_side_data(ffi::AV_FRAME_DATA_A53_CC) else {
        return Ok(false);
    };
    if side_data.data.is_null() || side_data.size == 0 {
        return Ok(false);
    }
    // SAFETY: both frames are valid AVFrames. FFmpeg allocates `dst` owned by
    // `target`; source side-data size and pointer are checked above before the
    // bounded byte copy.
    unsafe {
        let dst = ffi::av_frame_new_side_data(
            target.as_mut_ptr(),
            ffi::AV_FRAME_DATA_A53_CC,
            side_data.size,
        );
        if dst.is_null() {
            bail!("failed to allocate native cgen A53 caption side data");
        }
        ptr::copy_nonoverlapping(side_data.data, (*dst).data, side_data.size);
    }
    Ok(true)
}

fn frame_plane_layout(
    format: i32,
    width: i32,
    height: i32,
    plane: usize,
) -> Option<(usize, usize)> {
    let width = usize::try_from(width.max(0)).ok()?;
    let height = usize::try_from(height.max(0)).ok()?;
    match format {
        ffi::AV_PIX_FMT_YUV420P => match plane {
            0 => Some((height, width)),
            1 | 2 => Some((height.div_ceil(2), width.div_ceil(2))),
            _ => None,
        },
        ffi::AV_PIX_FMT_NV12 => match plane {
            0 => Some((height, width)),
            1 => Some((height.div_ceil(2), width)),
            _ => None,
        },
        ffi::AV_PIX_FMT_YUV422P => match plane {
            0 => Some((height, width)),
            1 | 2 => Some((height, width.div_ceil(2))),
            _ => None,
        },
        _ => {
            if plane == 0 {
                Some((height, width))
            } else {
                None
            }
        }
    }
}

struct AudioProcessor {
    input_index: usize,
    feed_id: String,
    priority_feed_id: String,
    state_rx: watch::Receiver<RuntimeState>,
    base_dir: PathBuf,
    input_time_base: AVRational,
    input_start_time: i64,
    video_time_base: AVRational,
    video_start_time: i64,
    decoder: AVCodecContext,
    encoder: AVCodecContext,
    resampler: Option<AudioResampler>,
    output_channels: u16,
    output_bitrate: i64,
    active_queue_id: Option<String>,
    active_path: Option<PathBuf>,
    pcm: Vec<u8>,
    cursor: usize,
    source_pcm: VecDeque<i16>,
    clock: CgenClock,
    source_max_samples: usize,
    was_alert_active: bool,
    warned_unsupported_source_format: bool,
}

impl AudioProcessor {
    fn new(
        feed: &FeedConfig,
        audio_spec: &InputStreamSpec,
        rendition: &AudioRenditionConfig,
        video_spec: &InputStreamSpec,
        state_rx: watch::Receiver<RuntimeState>,
        base_dir: PathBuf,
    ) -> Result<Self> {
        let decoder = create_audio_decoder(&audio_spec.codecpar)?;
        let output_channels = rendition.channels();
        let output_bitrate = rendition.bitrate_bps();
        let encoder = create_ac3_encoder(output_channels, output_bitrate)?;
        let priority_feed_id = if feed.priority_input.feed_id.trim().is_empty() {
            feed.id.clone()
        } else {
            feed.priority_input.feed_id.clone()
        };
        Ok(Self {
            input_index: audio_spec.input_index,
            feed_id: feed.id.clone(),
            priority_feed_id,
            state_rx,
            base_dir,
            input_time_base: audio_spec.time_base,
            input_start_time: audio_spec.start_time,
            video_time_base: video_spec.time_base,
            video_start_time: video_spec.start_time,
            decoder,
            encoder,
            resampler: None,
            output_channels,
            output_bitrate,
            active_queue_id: None,
            active_path: None,
            pcm: Vec::new(),
            cursor: 0,
            source_pcm: VecDeque::with_capacity(source_buffer_samples(&feed.sync, output_channels)),
            clock: CgenClock::new(feed.sync.clone()),
            source_max_samples: source_buffer_samples(&feed.sync, output_channels),
            was_alert_active: false,
            warned_unsupported_source_format: false,
        })
    }

    fn output_codecpar(&self) -> AVCodecParameters {
        let mut codecpar = self.encoder.extract_codecpar();
        // SAFETY: `extract_codecpar` returns an owned AVCodecParameters for
        // this output stream. These fields are plain AC-3 stream metadata
        // needed by MPEG-TS readers before we give the parameters to the muxer.
        unsafe {
            let raw = codecpar.as_mut_ptr();
            (*raw).codec_type = ffi::AVMEDIA_TYPE_AUDIO;
            (*raw).codec_id = ffi::AV_CODEC_ID_AC3;
            (*raw).codec_tag = 0;
            (*raw).format = ffi::AV_SAMPLE_FMT_FLTP;
            (*raw).sample_rate = ALERT_SAMPLE_RATE as i32;
            (*raw).ch_layout =
                AVChannelLayout::from_nb_channels(self.output_channels.into()).into_inner();
            (*raw).bit_rate = self.output_bitrate;
            (*raw).frame_size = AC3_FRAME_SAMPLES as i32;
        }
        codecpar
    }

    fn process_packet(
        &mut self,
        source_packet: &AVPacket,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        if self.alert_active()? {
            self.source_pcm.clear();
            return Ok(());
        }
        self.decoder
            .send_packet(Some(source_packet))
            .context("failed to send native cgen source audio packet to decoder")?;
        loop {
            match self.decoder.receive_frame() {
                Ok(frame) => {
                    self.enqueue_source_frame(&frame, spec)?;
                    self.write_available_source_audio(spec, output)?;
                }
                Err(err) if is_again(&err) => return Ok(()),
                Err(err) => return Err(err).context("failed to decode native cgen source audio"),
            }
        }
    }

    fn enqueue_source_frame(&mut self, source_frame: &AVFrame, spec: &StreamSpec) -> Result<()> {
        let source_pts = self.source_frame_output_pts(source_frame, spec.output_time_base);
        if let Some(source_pts) = source_pts {
            if self.clock.audio_next_pts.is_none() {
                self.clock.ensure_audio_started_at(source_pts);
            } else if self.source_pcm.is_empty() {
                let current = self.clock.audio_next_pts.unwrap_or(source_pts);
                let drift_ms =
                    pts_delta_ms(current.saturating_sub(source_pts), spec.output_time_base);
                if drift_ms.abs() > f64::from(self.clock.sync.hard_reset_ms.max(1)) {
                    warn!(
                        feed_id = %self.feed_id,
                        input_index = self.input_index,
                        drift_ms,
                        source_pts,
                        "native cgen source audio clock reset after source timestamp discontinuity"
                    );
                    self.clock.reset_audio_to(source_pts);
                }
            }
        }
        let source_pcm = match self.frame_to_output_pcm16(source_frame) {
            Ok(samples) => samples,
            Err(err) => {
                if !self.warned_unsupported_source_format {
                    warn!(
                        feed_id = %self.feed_id,
                        "native cgen cannot convert source audio frame; outputting silence until supported frame arrives: {err:#}"
                    );
                    self.warned_unsupported_source_format = true;
                }
                Vec::new()
            }
        };
        if source_pcm.is_empty() {
            return Ok(());
        }
        self.source_pcm.extend(source_pcm);
        self.drop_excess_source_audio(spec.output_time_base);
        Ok(())
    }

    fn source_frame_output_pts(
        &self,
        source_frame: &AVFrame,
        output_time_base: AVRational,
    ) -> Option<i64> {
        let audio_pts = if valid_media_pts(source_frame.pts) {
            source_frame.pts
        } else if valid_media_pts(self.input_start_time) {
            self.input_start_time
        } else {
            return None;
        };
        source_audio_output_pts(
            audio_pts,
            self.input_time_base,
            self.video_start_time,
            self.video_time_base,
            output_time_base,
        )
    }

    fn drop_excess_source_audio(&mut self, output_time_base: AVRational) {
        let channels = usize::from(self.output_channels.max(1));
        while self.source_pcm.len() > self.source_max_samples {
            let excess = self
                .source_pcm
                .len()
                .saturating_sub(self.source_max_samples);
            let sample_frames = (excess / channels).max(1);
            let drop_samples = sample_frames.saturating_mul(channels);
            for _ in 0..drop_samples {
                if self.source_pcm.pop_front().is_none() {
                    break;
                }
            }
            let duration = rescale_q(
                sample_frames as i64,
                AVRational {
                    num: 1,
                    den: ALERT_SAMPLE_RATE as i32,
                },
                output_time_base,
            );
            self.clock.advance_audio(duration.max(0));
        }
    }

    fn frame_to_output_pcm16(&mut self, frame: &AVFrame) -> Result<Vec<i16>> {
        self.resampler = Some(AudioResampler::for_frame(
            self.resampler.take(),
            frame,
            self.output_channels,
        )?);
        match self
            .resampler
            .as_mut()
            .context("native cgen audio resampler is unavailable")?
            .convert(frame)
        {
            Ok(samples) => Ok(samples),
            Err(_err) => {
                self.resampler = Some(AudioResampler::for_frame(
                    None,
                    frame,
                    self.output_channels,
                )?);
                self.resampler
                    .as_mut()
                    .context("native cgen audio resampler is unavailable after rebuild")?
                    .convert(frame)
            }
        }
    }

    fn write_mixed_until(
        &mut self,
        target_pts: i64,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        if !self.alert_active()? {
            return Ok(());
        }
        let duration = audio_frame_duration(spec.output_time_base);
        if self
            .clock
            .prepare_audio_target(target_pts, spec.output_time_base)
        {
            warn!(
                feed_id = %self.feed_id,
                drift_ms = self.clock.drift_ms(),
                target_pts,
                "native cgen audio clock hard reset after drift limit"
            );
            self.source_pcm.clear();
        }
        let max_frames = self.clock.max_audio_frames_per_video();
        for _ in 0..max_frames {
            let pts = self.clock.audio_pts(target_pts);
            if pts.saturating_add(duration) > target_pts {
                break;
            }
            let mut encoded = self
                .encode_next_alert_frame()
                .context("failed to encode native cgen alert audio")?;
            encoded.set_pts(pts);
            encoded.set_dts(pts);
            encoded.set_duration(duration);
            encoded.set_stream_index(spec.output_index);
            encoded.set_pos(-1);
            output
                .interleaved_write_frame(&mut encoded)
                .context("native cgen write mixed audio packet failed")?;
            self.clock.advance_audio(duration);
        }
        if self
            .clock
            .finish_audio_target(target_pts, spec.output_time_base)
        {
            warn!(
                feed_id = %self.feed_id,
                drift_ms = self.clock.drift_ms(),
                "native cgen dropped stale audio backlog after frame cap"
            );
            self.source_pcm.clear();
        }
        Ok(())
    }

    fn write_available_source_audio(
        &mut self,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        let samples_per_packet = AC3_FRAME_SAMPLES * usize::from(self.output_channels);
        if samples_per_packet == 0 {
            return Ok(());
        }
        let duration = audio_frame_duration(spec.output_time_base);
        self.clock.ensure_audio_started_at(0);
        let mut frames_written = 0usize;
        while self.source_pcm.len() >= samples_per_packet && frames_written < 32 {
            let pts = self.clock.audio_pts(0);
            let mut encoded = self
                .encode_next_source_frame()
                .context("failed to encode native cgen source audio")?;
            encoded.set_pts(pts);
            encoded.set_dts(pts);
            encoded.set_duration(duration);
            encoded.set_stream_index(spec.output_index);
            encoded.set_pos(-1);
            output
                .interleaved_write_frame(&mut encoded)
                .context("native cgen write source audio packet failed")?;
            self.clock.advance_audio(duration);
            frames_written = frames_written.saturating_add(1);
        }
        Ok(())
    }

    fn drift_ms(&self) -> f64 {
        self.clock.drift_ms()
    }

    fn active_queue_id(&self) -> Option<String> {
        self.active_queue_id.clone()
    }

    fn alert_active(&mut self) -> Result<bool> {
        self.active_audio()
    }

    fn active_audio(&mut self) -> Result<bool> {
        let audio = self
            .state_rx
            .borrow()
            .priority_audio_for(&self.priority_feed_id)
            .cloned();
        let Some(audio) = audio else {
            if self.was_alert_active {
                self.source_pcm.clear();
                self.clock.reset_audio();
            }
            self.active_queue_id = None;
            self.active_path = None;
            self.pcm.clear();
            self.cursor = 0;
            self.was_alert_active = false;
            return Ok(false);
        };
        if !self.was_alert_active {
            self.source_pcm.clear();
            self.cursor = 0;
            self.clock.reset_audio();
        }
        self.was_alert_active = true;
        self.ensure_loaded(&audio)?;
        Ok(true)
    }

    fn ensure_loaded(&mut self, audio: &PriorityAudio) -> Result<()> {
        let path = audio
            .audio_path
            .as_deref()
            .map(|path| resolve_path(&self.base_dir, path));
        if self.active_queue_id.as_deref() == Some(audio.queue_id.as_str())
            && self.active_path == path
        {
            return Ok(());
        }
        self.active_queue_id = Some(audio.queue_id.clone());
        self.active_path = path.clone();
        self.cursor = 0;
        self.pcm.clear();
        if let Some(path) = path {
            let raw = fs::read(&path).with_context(|| {
                format!(
                    "failed to read native cgen priority audio {}",
                    path.display()
                )
            })?;
            let pcm = normalize_pcm(
                Pcm {
                    sample_rate: audio.sample_rate,
                    channels: audio.channels,
                    data: raw,
                },
                ALERT_SAMPLE_RATE,
                ALERT_CHANNELS,
            );
            self.pcm = pcm.data;
            info!(
                feed_id = %self.feed_id,
                priority_feed = %self.priority_feed_id,
                queue_id = %audio.queue_id,
                samples = self.pcm.len() / usize::from(ALERT_CHANNELS) / 2,
                "native cgen loaded priority alert audio"
            );
        } else {
            warn!(
                feed_id = %self.feed_id,
                priority_feed = %self.priority_feed_id,
                queue_id = %audio.queue_id,
                "native cgen alert is active but has no audio_path; outputting alert silence"
            );
        }
        Ok(())
    }

    fn encode_next_alert_frame(&mut self) -> Result<AVPacket> {
        let pcm = self.next_alert_samples();
        self.encode_frame_from_i16(&pcm)
    }

    fn encode_next_source_frame(&mut self) -> Result<AVPacket> {
        let mut chunk = Vec::with_capacity(AC3_FRAME_SAMPLES * usize::from(self.output_channels));
        for _ in 0..(AC3_FRAME_SAMPLES * usize::from(self.output_channels)) {
            chunk.push(self.source_pcm.pop_front().unwrap_or_default());
        }
        self.encode_frame_from_i16(&chunk)
    }

    fn next_alert_samples(&mut self) -> Vec<i16> {
        let channels = usize::from(self.output_channels);
        let mut out = Vec::with_capacity(AC3_FRAME_SAMPLES * channels);
        for _ in 0..AC3_FRAME_SAMPLES {
            let (left, right) = next_stereo_sample(&self.pcm, &mut self.cursor);
            append_alert_sample_for_layout(&mut out, left, right, channels);
        }
        out
    }

    fn encode_frame_from_i16(&mut self, samples: &[i16]) -> Result<AVPacket> {
        let mut frame = AVFrame::new();
        frame.set_nb_samples(AC3_FRAME_SAMPLES as i32);
        frame.set_format(ffi::AV_SAMPLE_FMT_FLTP);
        frame.set_sample_rate(ALERT_SAMPLE_RATE as i32);
        frame.set_ch_layout(
            AVChannelLayout::from_nb_channels(self.output_channels.into()).into_inner(),
        );
        frame
            .alloc_buffer()
            .context("failed to allocate native cgen alert audio frame")?;
        frame
            .make_writable()
            .context("failed to make native cgen alert audio frame writable")?;
        fill_fltp_frame(
            &mut frame,
            samples,
            AC3_FRAME_SAMPLES,
            usize::from(self.output_channels),
        );
        self.encoder
            .send_frame(Some(&frame))
            .context("failed to send native cgen alert audio frame to encoder")?;
        match self.encoder.receive_packet() {
            Ok(packet) => Ok(packet),
            Err(err) if is_again(&err) => {
                bail!("native cgen AC-3 encoder accepted a frame but produced no packet")
            }
            Err(err) => Err(err).context("failed to receive native cgen alert audio packet"),
        }
    }
}

struct AudioResampler {
    context: SwrContext,
    input: AudioResamplerInput,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AudioResamplerInput {
    layout: String,
    format: i32,
    sample_rate: i32,
    output_channels: u16,
}

impl AudioResampler {
    fn for_frame(existing: Option<Self>, frame: &AVFrame, output_channels: u16) -> Result<Self> {
        let input = AudioResamplerInput::from_frame(frame, output_channels);
        if let Some(existing) = existing {
            if existing.input == input {
                return Ok(existing);
            }
        }

        let input_layout = frame_channel_layout(frame);
        let output_layout = AVChannelLayout::from_nb_channels(output_channels.into());
        let mut context = SwrContext::new(
            &output_layout.into_inner(),
            ffi::AV_SAMPLE_FMT_S16,
            ALERT_SAMPLE_RATE as i32,
            &input_layout.into_inner(),
            frame.format,
            frame.sample_rate.max(1),
        )
        .context("failed to create native cgen audio resampler")?;
        context
            .init()
            .context("failed to initialize native cgen audio resampler")?;
        Ok(Self { context, input })
    }

    fn convert(&mut self, frame: &AVFrame) -> Result<Vec<i16>> {
        let input_samples = frame.nb_samples.max(0);
        if input_samples == 0 {
            return Ok(Vec::new());
        }
        let out_samples = self.context.get_out_samples(input_samples).max(1);
        let mut output = AVFrame::new();
        output.set_nb_samples(out_samples);
        output.set_format(ffi::AV_SAMPLE_FMT_S16);
        output.set_sample_rate(ALERT_SAMPLE_RATE as i32);
        output.set_ch_layout(
            AVChannelLayout::from_nb_channels(self.input.output_channels.into()).into_inner(),
        );
        output
            .alloc_buffer()
            .context("failed to allocate native cgen resampled audio frame")?;
        let input_data = unsafe {
            let extended = (*frame.as_ptr()).extended_data as *const *const u8;
            if extended.is_null() {
                frame.data.as_ptr() as *const *const u8
            } else {
                extended
            }
        };
        let produced = unsafe {
            self.context.convert(
                output.data_mut().as_mut_ptr(),
                out_samples,
                input_data,
                input_samples,
            )
        }
        .context("failed to resample native cgen source audio")?;
        if produced <= 0 {
            return Ok(Vec::new());
        }
        output.set_nb_samples(produced);
        packed_s16_frame_to_vec(&output, usize::from(self.input.output_channels))
    }
}

impl AudioResamplerInput {
    fn from_frame(frame: &AVFrame, output_channels: u16) -> Self {
        Self {
            layout: frame
                .ch_layout()
                .describe()
                .ok()
                .and_then(|value| value.into_string().ok())
                .unwrap_or_else(|| format!("{}c", frame.ch_layout().nb_channels.max(1))),
            format: frame.format,
            sample_rate: frame.sample_rate.max(1),
            output_channels,
        }
    }
}

fn create_ac3_encoder(channels: u16, bitrate: i64) -> Result<AVCodecContext> {
    let codec = AVCodec::find_encoder(ffi::AV_CODEC_ID_AC3)
        .context("native cgen AC-3 encoder is unavailable")?;
    let sample_fmt = if codec
        .sample_fmts()
        .map(|fmts| fmts.contains(&ffi::AV_SAMPLE_FMT_FLTP))
        .unwrap_or(false)
    {
        ffi::AV_SAMPLE_FMT_FLTP
    } else {
        bail!("native cgen AC-3 encoder does not support fltp sample format");
    };
    let mut encoder = AVCodecContext::new(&codec);
    encoder.set_sample_rate(ALERT_SAMPLE_RATE as i32);
    encoder.set_sample_fmt(sample_fmt);
    encoder.set_ch_layout(AVChannelLayout::from_nb_channels(channels.into()).into_inner());
    encoder.set_time_base(AVRational {
        num: 1,
        den: ALERT_SAMPLE_RATE as i32,
    });
    encoder.set_bit_rate(bitrate);
    encoder
        .open(None)
        .context("failed to open native cgen AC-3 encoder")?;
    Ok(encoder)
}

fn create_audio_decoder(codecpar: &AVCodecParameters) -> Result<AVCodecContext> {
    let codec = AVCodec::find_decoder(codecpar.codec_id)
        .context("native cgen source audio decoder is unavailable")?;
    let mut decoder = AVCodecContext::new(&codec);
    decoder
        .apply_codecpar(codecpar)
        .context("failed to apply native cgen source audio codec parameters")?;
    decoder
        .open(None)
        .context("failed to open native cgen source audio decoder")?;
    Ok(decoder)
}

#[allow(dead_code)]
fn create_video_decoder(codecpar: &AVCodecParameters) -> Result<AVCodecContext> {
    let codec = AVCodec::find_decoder(codecpar.codec_id)
        .context("native cgen source video decoder is unavailable")?;
    let mut decoder = AVCodecContext::new(&codec);
    decoder
        .apply_codecpar(codecpar)
        .context("failed to apply native cgen source video codec parameters")?;
    decoder
        .open(None)
        .context("failed to open native cgen source video decoder")?;
    Ok(decoder)
}

#[allow(dead_code)]
fn create_video_encoder(
    feed: &FeedConfig,
    video_spec: &InputStreamSpec,
    rendition: &VideoRenditionConfig,
) -> Result<AVCodecContext> {
    let output = feed.output();
    let codec_name = output_codec_name(rendition.codec_name(&output.vcodec));
    let codec_name_c = cstring_arg(&codec_name, "video codec name")?;
    let codec = AVCodec::find_encoder_by_name(codec_name_c.as_c_str())
        .with_context(|| format!("native cgen video encoder {codec_name} is unavailable"))?;
    let pix_fmt = select_video_pix_fmt(&codec, &codec_name);
    let frame_rate = output_frame_rate(feed, video_spec, rendition);
    let mut encoder = AVCodecContext::new(&codec);
    let width = if rendition.width == 0 {
        video_spec.codecpar.width
    } else {
        rendition.width as i32
    };
    let height = if rendition.height == 0 {
        video_spec.codecpar.height
    } else {
        rendition.height as i32
    };
    encoder.set_width(width);
    encoder.set_height(height);
    encoder.set_pix_fmt(pix_fmt);
    encoder.set_time_base(AVRational {
        num: frame_rate.den.max(1),
        den: frame_rate.num.max(1),
    });
    encoder.set_framerate(frame_rate);
    encoder.set_pkt_timebase(encoder.time_base);
    encoder.set_gop_size(if rendition.interlaced { 15 } else { 30 });
    encoder.set_max_b_frames(0);
    encoder.set_bit_rate(
        rendition
            .bitrate_kbps
            .or(output.video_bitrate_kbps)
            .map(|kbps| i64::from(kbps) * 1000)
            .unwrap_or(DEFAULT_VIDEO_BITRATE),
    );
    if rendition.interlaced {
        encoder.set_flags(
            encoder.flags
                | ffi::AV_CODEC_FLAG_INTERLACED_DCT as i32
                | ffi::AV_CODEC_FLAG_INTERLACED_ME as i32,
        );
        // SAFETY: the encoder context is owned here and not yet opened. Field
        // order is a plain AVCodecContext metadata field used by encoders and
        // muxers to advertise interlaced output.
        unsafe {
            (*encoder.as_mut_ptr()).field_order = video_field_order(&rendition.field_order);
        }
    } else {
        // SAFETY: same ownership as above; explicitly mark non-interlaced
        // output progressive so stale input codec parameters cannot leak.
        unsafe {
            (*encoder.as_mut_ptr()).field_order = ffi::AV_FIELD_PROGRESSIVE;
        }
    }
    let options = video_encoder_options(&codec_name, rendition.interlaced, &rendition.field_order);
    encoder
        .open(options)
        .with_context(|| format!("failed to open native cgen video encoder {codec_name}"))?;
    Ok(encoder)
}

#[allow(dead_code)]
fn output_codec_name(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        return "mpeg2video".to_string();
    }
    if value.eq_ignore_ascii_case("h264") {
        return "libx264".to_string();
    }
    if value.eq_ignore_ascii_case("hevc") || value.eq_ignore_ascii_case("h265") {
        return "libx265".to_string();
    }
    value.to_string()
}

#[allow(dead_code)]
fn output_frame_rate(
    feed: &FeedConfig,
    video_spec: &InputStreamSpec,
    rendition: &VideoRenditionConfig,
) -> AVRational {
    parse_rational(rendition.frame_rate_text(&feed.video.fps))
        .filter(valid_rational)
        .or_else(|| valid_rational(&video_spec.frame_rate).then_some(video_spec.frame_rate))
        .unwrap_or(AVRational {
            num: 30_000,
            den: 1001,
        })
}

fn video_field_order(value: &str) -> ffi::AVFieldOrder {
    if value.eq_ignore_ascii_case("bff")
        || value.eq_ignore_ascii_case("bottom")
        || value.eq_ignore_ascii_case("bottom_first")
    {
        ffi::AV_FIELD_BB
    } else {
        ffi::AV_FIELD_TT
    }
}

fn parse_rational(value: &str) -> Option<AVRational> {
    let value = value.trim();
    if value.is_empty() || value.eq_ignore_ascii_case("source") {
        return None;
    }
    if value.eq_ignore_ascii_case("ntsc") {
        return Some(AVRational {
            num: 30_000,
            den: 1001,
        });
    }
    if value.eq_ignore_ascii_case("pal") || value.eq_ignore_ascii_case("secam") {
        return Some(AVRational { num: 25, den: 1 });
    }
    if let Some((num, den)) = value.split_once('/') {
        let num = num.trim().parse::<i32>().ok()?;
        let den = den.trim().parse::<i32>().ok()?;
        return Some(AVRational { num, den });
    }
    value
        .parse::<i32>()
        .ok()
        .map(|num| AVRational { num, den: 1 })
}

#[allow(dead_code)]
fn valid_rational(value: &AVRational) -> bool {
    value.num > 0 && value.den > 0
}

fn rational_cmp(left: AVRational, right: AVRational) -> std::cmp::Ordering {
    (i64::from(left.num) * i64::from(right.den)).cmp(&(i64::from(right.num) * i64::from(left.den)))
}

#[allow(dead_code)]
fn select_video_pix_fmt(codec: &AVCodec, codec_name: &str) -> i32 {
    let desired = if prefers_nv12(codec_name) {
        ffi::AV_PIX_FMT_NV12
    } else {
        ffi::AV_PIX_FMT_YUV420P
    };
    let Some(supported) = codec.pix_fmts() else {
        return desired;
    };
    if supported.is_empty() || supported.contains(&desired) {
        return desired;
    }
    warn!(
        codec = %codec_name,
        desired_pix_fmt = desired,
        fallback_pix_fmt = supported[0],
        "native cgen video encoder does not support preferred pixel format; using first supported format"
    );
    supported[0]
}

fn prefers_nv12(codec_name: &str) -> bool {
    let codec_name = codec_name.to_ascii_lowercase();
    codec_name.contains("nvenc")
        || codec_name.contains("qsv")
        || codec_name.contains("amf")
        || codec_name.contains("vaapi")
}

#[allow(dead_code)]
fn video_encoder_options(
    codec_name: &str,
    interlaced: bool,
    field_order: &str,
) -> Option<AVDictionary> {
    let lower = codec_name.to_ascii_lowercase();
    if lower.contains("libx264") || lower.contains("libx265") {
        return Some(AVDictionary::new(c"preset", c"ultrafast", 0).set(c"tune", c"zerolatency", 0));
    }
    if lower.contains("nvenc") {
        return Some(AVDictionary::new(c"preset", c"ll", 0).set(c"tune", c"ull", 0));
    }
    if interlaced && lower.contains("mpeg2") {
        let top = if video_field_order(field_order) == ffi::AV_FIELD_BB {
            c"0"
        } else {
            c"1"
        };
        return Some(
            AVDictionary::new(c"flags", c"+ildct+ilme", 0)
                .set(c"top", top, 0)
                .set(c"alternate_scan", c"1", 0)
                .set(c"seq_disp_ext", c"always", 0)
                .set(c"video_format", c"ntsc", 0),
        );
    }
    None
}

fn fill_fltp_frame(frame: &mut AVFrame, pcm: &[i16], samples: usize, channels: usize) {
    let planes = frame.data_mut();
    for channel in 0..channels {
        let plane = planes[channel] as *mut f32;
        for sample_index in 0..samples {
            let value = pcm
                .get(sample_index * channels + channel)
                .copied()
                .unwrap_or_default();
            // SAFETY: AVFrame::alloc_buffer allocated one FLTP plane per
            // channel with at least `samples` f32 values each.
            unsafe {
                *plane.add(sample_index) = pcm_i16_to_f32(value);
            }
        }
    }
}

fn packed_s16_frame_to_vec(frame: &AVFrame, channels: usize) -> Result<Vec<i16>> {
    let samples = usize::try_from(frame.nb_samples.max(0)).unwrap_or_default();
    let total = samples.saturating_mul(channels);
    let data = frame.data[0] as *const i16;
    if data.is_null() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(total);
    for index in 0..total {
        // SAFETY: the output frame is allocated as packed S16 with
        // `nb_samples * channels` samples in data[0].
        out.push(unsafe { *data.add(index) });
    }
    Ok(out)
}

fn append_alert_sample_for_layout(out: &mut Vec<i16>, left: i16, right: i16, channels: usize) {
    match channels {
        0 => {}
        1 => out.push(((i32::from(left) + i32::from(right)) / 2) as i16),
        2 => {
            out.push(left);
            out.push(right);
        }
        _ => {
            out.push(left);
            out.push(right);
            let center = ((i32::from(left) + i32::from(right)) / 2) as i16;
            out.push(center);
            if channels > 3 {
                out.push(0);
            }
            for channel in 4..channels {
                let value = if channel % 2 == 0 { left } else { right };
                out.push(value);
            }
        }
    }
}

fn next_stereo_sample(pcm: &[u8], cursor: &mut usize) -> (i16, i16) {
    if *cursor + 4 > pcm.len() {
        return (0, 0);
    }
    let left = i16::from_le_bytes([pcm[*cursor], pcm[*cursor + 1]]);
    let right = i16::from_le_bytes([pcm[*cursor + 2], pcm[*cursor + 3]]);
    *cursor += 4;
    (left, right)
}

fn pcm_i16_to_f32(value: i16) -> f32 {
    f32::from(value) / 32768.0
}

fn audio_frame_duration(output_time_base: AVRational) -> i64 {
    rescale_q(
        AC3_FRAME_SAMPLES as i64,
        AVRational {
            num: 1,
            den: ALERT_SAMPLE_RATE as i32,
        },
        output_time_base,
    )
    .max(1)
}

fn audio_output_channels(codecpar: &AVCodecParameters) -> u16 {
    let channels = codecpar
        .ch_layout()
        .nb_channels
        .clamp(1, i32::from(MAX_OUTPUT_AUDIO_CHANNELS));
    u16::try_from(channels).unwrap_or(ALERT_CHANNELS)
}

fn frame_channel_layout(frame: &AVFrame) -> AVChannelLayout {
    frame
        .ch_layout()
        .describe()
        .ok()
        .and_then(|value| AVChannelLayout::from_string(value.as_c_str()))
        .unwrap_or_else(|| AVChannelLayout::from_nb_channels(frame.ch_layout().nb_channels.max(1)))
}

fn ac3_bitrate_for_channels(channels: u16) -> i64 {
    if channels > 2 {
        384_000
    } else {
        DEFAULT_AC3_BITRATE
    }
}

fn source_buffer_samples(sync: &SyncConfig, channels: u16) -> usize {
    let buffer_ms = sync.source_buffer_ms.clamp(32, 250);
    let samples =
        (u64::from(ALERT_SAMPLE_RATE) * u64::from(channels.max(1)) * u64::from(buffer_ms)) / 1_000;
    usize::try_from(samples)
        .unwrap_or(AC3_FRAME_SAMPLES * usize::from(channels.max(1)) * 4)
        .max(AC3_FRAME_SAMPLES * usize::from(channels.max(1)) * 2)
}

fn pts_delta_ms(delta: i64, time_base: AVRational) -> f64 {
    let num = f64::from(time_base.num.max(1));
    let den = f64::from(time_base.den.max(1));
    (delta as f64) * num * 1000.0 / den
}

fn source_audio_output_pts(
    audio_pts: i64,
    audio_time_base: AVRational,
    video_start_pts: i64,
    video_time_base: AVRational,
    output_time_base: AVRational,
) -> Option<i64> {
    if !valid_media_pts(audio_pts)
        || !valid_media_pts(video_start_pts)
        || !valid_rational(&audio_time_base)
        || !valid_rational(&video_time_base)
        || !valid_rational(&output_time_base)
    {
        return None;
    }
    let audio = rescale_q(audio_pts, audio_time_base, output_time_base);
    let video = rescale_q(video_start_pts, video_time_base, output_time_base);
    Some(audio.saturating_sub(video))
}

fn valid_media_pts(value: i64) -> bool {
    value != ffi::AV_NOPTS_VALUE
}

fn rescale_q(value: i64, source: AVRational, target: AVRational) -> i64 {
    unsafe { ffi::av_rescale_q(value, source, target) }
}

fn is_again(err: &RsmpegError) -> bool {
    err.raw_error() == Some(-11)
}

fn resolve_path(base_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn cstring_arg(value: &str, label: &str) -> Result<CString> {
    let value = value.trim();
    if value.is_empty() {
        bail!("{label} is empty");
    }
    CString::new(value).with_context(|| format!("{label} contains an embedded NUL byte"))
}

fn non_empty(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

#[allow(dead_code)]
fn cstr_debug(value: &CStr) -> String {
    value.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cstring_arg_rejects_empty_and_nul() {
        assert!(cstring_arg("", "url").is_err());
        assert!(cstring_arg("udp://ok", "url").is_ok());
        assert!(cstring_arg("bad\0url", "url").is_err());
    }

    #[test]
    fn parses_video_frame_rates_for_output_clock() {
        assert_eq!(
            rational_tuple(parse_rational("30000/1001")),
            Some((30_000, 1001))
        );
        assert_eq!(rational_tuple(parse_rational("pal")), Some((25, 1)));
        assert!(parse_rational("source").is_none());
    }

    #[test]
    fn hardware_video_codecs_prefer_nv12() {
        assert!(prefers_nv12("h264_nvenc"));
        assert!(prefers_nv12("hevc_qsv"));
        assert!(prefers_nv12("h264_amf"));
        assert!(!prefers_nv12("mpeg2video"));
        assert!(!prefers_nv12("libx264"));
    }

    #[test]
    fn corrected_wall_clock_anchors_to_media_pts() {
        let mut clock = CorrectedWallClock::default();
        let tb = AVRational {
            num: 1,
            den: 48_000,
        };
        clock.observe_media_pts(
            90_000,
            AVRational {
                num: 1,
                den: 90_000,
            },
            tb,
        );
        let target = clock.target_pts(tb).expect("clock target");
        assert!(target >= 48_000);
        assert!(target < 49_000);
    }

    #[test]
    fn sync_source_buffer_uses_configured_milliseconds() {
        let sync = SyncConfig {
            source_buffer_ms: 250,
            ..SyncConfig::default()
        };
        assert_eq!(
            source_buffer_samples(&sync, ALERT_CHANNELS),
            (ALERT_SAMPLE_RATE as usize * ALERT_CHANNELS as usize) / 4
        );
    }

    #[test]
    fn pts_delta_ms_uses_time_base() {
        let time_base = AVRational {
            num: 1,
            den: ALERT_SAMPLE_RATE as i32,
        };
        let delta = (ALERT_SAMPLE_RATE / 10) as i64;
        assert!((pts_delta_ms(delta, time_base) - 100.0).abs() < 0.01);
    }

    #[test]
    fn source_audio_output_pts_preserves_audio_lead() {
        let ninety_k = AVRational {
            num: 1,
            den: 90_000,
        };
        let audio_tb = AVRational {
            num: 1,
            den: ALERT_SAMPLE_RATE as i32,
        };
        let output = source_audio_output_pts(4_800, audio_tb, 9_000, ninety_k, audio_tb)
            .expect("valid source offset");
        assert_eq!(output, 0);

        let leading_audio = source_audio_output_pts(0, audio_tb, 9_000, ninety_k, audio_tb)
            .expect("valid leading source offset");
        assert_eq!(leading_audio, -4_800);
    }

    #[test]
    fn cgen_clock_hard_resets_large_audio_drift() {
        let mut clock = CgenClock::new(SyncConfig {
            hard_reset_ms: 100,
            ..SyncConfig::default()
        });
        let time_base = AVRational {
            num: 1,
            den: ALERT_SAMPLE_RATE as i32,
        };
        clock.prepare_audio_target(0, time_base);
        clock.advance_audio((ALERT_SAMPLE_RATE as i64) * 2);
        assert!(clock.prepare_audio_target(0, time_base));
        assert_eq!(clock.audio_pts(0), 0);
    }

    fn rational_tuple(value: Option<AVRational>) -> Option<(i32, i32)> {
        value.map(|value| (value.num, value.den))
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    #[test]
    fn ac3_encoder_emits_packet_for_one_frame() {
        let channels = ALERT_CHANNELS;
        let mut encoder =
            create_ac3_encoder(channels, ac3_bitrate_for_channels(channels)).expect("AC-3 encoder");
        let silence = vec![0i16; AC3_FRAME_SAMPLES * usize::from(channels)];
        let mut frame = AVFrame::new();
        frame.set_nb_samples(AC3_FRAME_SAMPLES as i32);
        frame.set_format(ffi::AV_SAMPLE_FMT_FLTP);
        frame.set_sample_rate(ALERT_SAMPLE_RATE as i32);
        frame.set_ch_layout(AVChannelLayout::from_nb_channels(channels.into()).into_inner());
        frame.alloc_buffer().expect("audio frame buffer");
        fill_fltp_frame(
            &mut frame,
            &silence,
            AC3_FRAME_SAMPLES,
            usize::from(channels),
        );
        encoder.send_frame(Some(&frame)).expect("send frame");
        let packet = encoder.receive_packet().expect("receive packet");
        assert!(packet.size > 0);
    }

    #[test]
    fn alert_samples_expand_to_surround_layout() {
        let mut out = Vec::new();
        append_alert_sample_for_layout(&mut out, 1000, -1000, 6);
        assert_eq!(out, vec![1000, -1000, 0, 0, 1000, -1000]);
    }
}
