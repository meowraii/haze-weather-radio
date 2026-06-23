use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::Duration;
#[cfg(test)]
use std::time::Instant;

use anyhow::{bail, Context, Result};
use haze_media::{normalize_pcm, Pcm};
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecParameters, AVPacket};
use rsmpeg::avformat::{AVFormatContextInput, AVFormatContextOutput};
use rsmpeg::avutil::{AVChannelLayout, AVDictionary, AVFrame, AVRational};
use rsmpeg::swscale::SwsContext;
use rsmpeg::{error::RsmpegError, ffi};
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::config::FeedConfig;
use crate::graphics::NativeGraphicsRenderer;
use crate::state::{PriorityAudio, RuntimeState};

const ALERT_SAMPLE_RATE: u32 = 48_000;
const ALERT_CHANNELS: u16 = 2;
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
    #[allow(dead_code)]
    frame_rate: AVRational,
    codecpar: AVCodecParameters,
}

pub(crate) async fn run_remux_supervised(
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    base_dir: PathBuf,
) -> Result<()> {
    let mut restart_delay = Duration::from_millis(500);
    loop {
        let worker_feed = feed.clone();
        let worker_state_rx = state_rx.clone();
        let worker_base_dir = base_dir.clone();
        info!(
            feed_id = %feed.id,
            input = %feed.program_input_url(),
            output = %feed.program_output_url(),
            "starting native rsmpeg cgen constant encoder"
        );
        let result = tokio::task::spawn_blocking(move || {
            encode_once(&worker_feed, worker_state_rx, &worker_base_dir)
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
            }
        }
        sleep(restart_delay).await;
        restart_delay = (restart_delay * 2).min(Duration::from_secs(10));
    }
}

fn encode_once(
    feed: &FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    base_dir: &Path,
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
    let input_specs = collect_input_specs(&input)?;
    if input_specs.is_empty() {
        bail!("native cgen input has no streams");
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
    let video_input_index = input_specs
        .iter()
        .find(|spec| spec.codecpar.codec_type == ffi::AVMEDIA_TYPE_VIDEO)
        .map(|spec| spec.input_index)
        .context("native cgen input has no video stream")?;
    let video_spec = input_specs
        .iter()
        .find(|spec| spec.input_index == video_input_index)
        .context("native cgen video stream disappeared")?;
    let mut video_processor = VideoProcessor::new(feed, video_spec, state_rx.clone())?;

    let audio_input_index = input_specs
        .iter()
        .find(|spec| spec.codecpar.codec_type == ffi::AVMEDIA_TYPE_AUDIO)
        .map(|spec| spec.input_index);
    let mut audio_processor = if let Some(input_index) = audio_input_index {
        let audio_spec = input_specs
            .iter()
            .find(|spec| spec.input_index == input_index)
            .context("native cgen audio stream disappeared")?;
        Some(AudioProcessor::new(
            feed,
            audio_spec,
            state_rx,
            base_dir.to_path_buf(),
        )?)
    } else {
        warn!(
            feed_id = %feed.id,
            "native cgen input has no audio stream; priority alert audio cannot be injected"
        );
        None
    };
    let mut stream_map = create_output_streams(
        &mut output,
        &input_specs,
        video_input_index,
        video_processor.output_codecpar(),
        video_processor.output_time_base(),
        audio_input_index,
        audio_processor
            .as_ref()
            .map(|processor| processor.output_codecpar()),
    )?;
    let audio_stream_spec = audio_input_index.and_then(|input_index| {
        stream_map
            .iter()
            .find(|spec| spec.input_index == input_index)
            .cloned()
    });
    let mut header_options = None;
    output
        .write_header(&mut header_options)
        .context("failed to write native cgen output header")?;
    refresh_output_time_bases(&output, &mut stream_map);
    while let Some(mut packet) = input
        .read_packet()
        .context("native cgen read packet failed")?
    {
        let input_index = packet.stream_index as usize;
        let Some(spec) = stream_map
            .iter()
            .find(|spec| spec.input_index == input_index)
        else {
            continue;
        };
        if input_index == video_input_index {
            video_processor.process_packet(&packet, spec, &mut output)?;
            if let (Some(processor), Some(audio_spec)) =
                (audio_processor.as_mut(), audio_stream_spec.as_ref())
            {
                let target_audio_pts = rescale_q(
                    video_processor.next_pts,
                    video_processor.output_time_base(),
                    audio_spec.output_time_base,
                );
                processor.write_mixed_until(target_audio_pts, audio_spec, &mut output)?;
            }
            continue;
        }
        if Some(input_index) == audio_input_index {
            if let Some(processor) = audio_processor.as_mut() {
                processor.process_packet(&mut packet, spec, &mut output)?;
                continue;
            }
        }
        packet.rescale_ts(spec.input_time_base, spec.output_time_base);
        packet.set_stream_index(spec.output_index);
        packet.set_pos(-1);
        output
            .interleaved_write_frame(&mut packet)
            .context("native cgen write packet failed")?;
    }
    output
        .write_trailer()
        .context("failed to write native cgen trailer")
}

fn collect_input_specs(input: &AVFormatContextInput) -> Result<Vec<InputStreamSpec>> {
    let mut out = Vec::with_capacity(input.streams().len());
    for stream in input.streams() {
        let mut codecpar = AVCodecParameters::new();
        codecpar.copy(&stream.codecpar());
        out.push(InputStreamSpec {
            input_index: stream.index as usize,
            time_base: stream.time_base,
            frame_rate: stream
                .guess_framerate()
                .unwrap_or(AVRational { num: 0, den: 1 }),
            codecpar,
        });
    }
    Ok(out)
}

fn create_output_streams(
    output: &mut AVFormatContextOutput,
    input_specs: &[InputStreamSpec],
    video_input_index: usize,
    video_codecpar: AVCodecParameters,
    video_time_base: AVRational,
    audio_input_index: Option<usize>,
    audio_codecpar: Option<AVCodecParameters>,
) -> Result<Vec<StreamSpec>> {
    let mut out = Vec::with_capacity(2);
    for spec in input_specs {
        if spec.input_index != video_input_index && Some(spec.input_index) != audio_input_index {
            continue;
        }
        let mut stream = output.new_stream();
        let output_time_base = if spec.input_index == video_input_index {
            stream.set_codecpar(video_codecpar.clone());
            stream.set_time_base(video_time_base);
            stream.time_base
        } else if Some(spec.input_index) == audio_input_index {
            let codecpar = audio_codecpar
                .as_ref()
                .context("native cgen audio output codec parameters are missing")?;
            stream.set_codecpar(codecpar.clone());
            stream.set_time_base(AVRational {
                num: 1,
                den: ALERT_SAMPLE_RATE as i32,
            });
            stream.time_base
        } else {
            unreachable!("non-selected native cgen stream was filtered before stream creation")
        };
        out.push(StreamSpec {
            input_index: spec.input_index,
            output_index: stream.index,
            input_time_base: spec.time_base,
            output_time_base,
        });
    }
    Ok(out)
}

fn refresh_output_time_bases(output: &AVFormatContextOutput, stream_map: &mut [StreamSpec]) {
    for spec in stream_map {
        if let Some(stream) = output
            .streams()
            .iter()
            .find(|stream| stream.index == spec.output_index)
        {
            spec.output_time_base = stream.time_base;
        }
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
        state_rx: watch::Receiver<RuntimeState>,
    ) -> Result<Self> {
        let decoder = create_video_decoder(&video_spec.codecpar)?;
        let encoder = create_video_encoder(feed, video_spec)?;
        let encoder_pix_fmt = encoder.pix_fmt;
        let output_frame_rate = encoder.framerate;
        let source_frame_rate = if valid_rational(&video_spec.frame_rate) {
            video_spec.frame_rate
        } else {
            output_frame_rate
        };
        let convert_frame_rate = parse_rational(&feed.video.fps).is_some()
            && valid_rational(&source_frame_rate)
            && valid_rational(&output_frame_rate)
            && rational_cmp(source_frame_rate, output_frame_rate) != std::cmp::Ordering::Equal;
        Ok(Self {
            feed_id: feed.id.clone(),
            decoder,
            encoder,
            scaler: None,
            encoder_width: feed.video.width as i32,
            encoder_height: feed.video.height as i32,
            encoder_pix_fmt,
            source_frame_rate,
            output_frame_rate,
            decoded_frames: 0,
            encoded_frames: 0,
            convert_frame_rate,
            interlaced: feed.video.interlaced,
            field_order: video_field_order(feed),
            graphics: NativeGraphicsRenderer::new(feed, state_rx),
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
    feed_id: String,
    priority_feed_id: String,
    state_rx: watch::Receiver<RuntimeState>,
    base_dir: PathBuf,
    decoder: AVCodecContext,
    encoder: AVCodecContext,
    active_queue_id: Option<String>,
    active_path: Option<PathBuf>,
    pcm: Vec<u8>,
    cursor: usize,
    source_pcm: VecDeque<i16>,
    next_pts: Option<i64>,
    warned_unsupported_source_format: bool,
}

impl AudioProcessor {
    fn new(
        feed: &FeedConfig,
        audio_spec: &InputStreamSpec,
        state_rx: watch::Receiver<RuntimeState>,
        base_dir: PathBuf,
    ) -> Result<Self> {
        let decoder = create_audio_decoder(&audio_spec.codecpar)?;
        let encoder = create_ac3_encoder()?;
        let priority_feed_id = if feed.priority_input.feed_id.trim().is_empty() {
            feed.id.clone()
        } else {
            feed.priority_input.feed_id.clone()
        };
        Ok(Self {
            feed_id: feed.id.clone(),
            priority_feed_id,
            state_rx,
            base_dir,
            decoder,
            encoder,
            active_queue_id: None,
            active_path: None,
            pcm: Vec::new(),
            cursor: 0,
            source_pcm: VecDeque::with_capacity(
                AC3_FRAME_SAMPLES * usize::from(ALERT_CHANNELS) * 8,
            ),
            next_pts: Some(0),
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
                AVChannelLayout::from_nb_channels(ALERT_CHANNELS.into()).into_inner();
            (*raw).bit_rate = DEFAULT_AC3_BITRATE;
            (*raw).frame_size = AC3_FRAME_SAMPLES as i32;
        }
        codecpar
    }

    fn process_packet(
        &mut self,
        source_packet: &mut AVPacket,
        _spec: &StreamSpec,
        _output: &mut AVFormatContextOutput,
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
                Ok(frame) => self.enqueue_source_frame(&frame)?,
                Err(err) if is_again(&err) => return Ok(()),
                Err(err) => return Err(err).context("failed to decode native cgen source audio"),
            }
        }
    }

    fn enqueue_source_frame(&mut self, source_frame: &AVFrame) -> Result<()> {
        let source_pcm = match frame_to_stereo_pcm16(source_frame) {
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
        let max_samples = AC3_FRAME_SAMPLES * usize::from(ALERT_CHANNELS) * 64;
        while self.source_pcm.len() > max_samples {
            self.source_pcm.pop_front();
        }
        Ok(())
    }

    fn write_mixed_until(
        &mut self,
        target_pts: i64,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        let duration = audio_frame_duration(spec.output_time_base);
        if self.next_pts.is_none() {
            self.next_pts = Some(target_pts);
        }
        let max_frames = 24usize;
        for _ in 0..max_frames {
            let pts = self.next_pts.unwrap_or(target_pts);
            if pts >= target_pts {
                break;
            }
            let mut encoded = if self.alert_active()? {
                self.encode_next_alert_frame()
            } else {
                self.encode_next_source_frame()
            }
            .context("failed to encode native cgen mixed audio")?;
            encoded.set_pts(pts);
            encoded.set_dts(pts);
            encoded.set_duration(duration);
            encoded.set_stream_index(spec.output_index);
            encoded.set_pos(-1);
            output
                .interleaved_write_frame(&mut encoded)
                .context("native cgen write mixed audio packet failed")?;
            self.next_pts = Some(pts.saturating_add(duration));
        }
        Ok(())
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
            self.active_queue_id = None;
            self.active_path = None;
            self.pcm.clear();
            self.cursor = 0;
            return Ok(false);
        };
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
        let mut chunk = Vec::with_capacity(AC3_FRAME_SAMPLES * usize::from(ALERT_CHANNELS));
        for _ in 0..(AC3_FRAME_SAMPLES * usize::from(ALERT_CHANNELS)) {
            chunk.push(self.source_pcm.pop_front().unwrap_or_default());
        }
        self.encode_frame_from_i16(&chunk)
    }

    fn next_alert_samples(&mut self) -> Vec<i16> {
        let mut out = Vec::with_capacity(AC3_FRAME_SAMPLES * usize::from(ALERT_CHANNELS));
        for _ in 0..AC3_FRAME_SAMPLES {
            let (left, right) = next_stereo_sample(&self.pcm, &mut self.cursor);
            out.push(left);
            out.push(right);
        }
        out
    }

    fn encode_frame_from_i16(&mut self, samples: &[i16]) -> Result<AVPacket> {
        let mut frame = AVFrame::new();
        frame.set_nb_samples(AC3_FRAME_SAMPLES as i32);
        frame.set_format(ffi::AV_SAMPLE_FMT_FLTP);
        frame.set_sample_rate(ALERT_SAMPLE_RATE as i32);
        frame.set_ch_layout(AVChannelLayout::from_nb_channels(ALERT_CHANNELS.into()).into_inner());
        frame
            .alloc_buffer()
            .context("failed to allocate native cgen alert audio frame")?;
        frame
            .make_writable()
            .context("failed to make native cgen alert audio frame writable")?;
        fill_fltp_stereo_frame(&mut frame, samples, AC3_FRAME_SAMPLES);
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

fn create_ac3_encoder() -> Result<AVCodecContext> {
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
    encoder.set_ch_layout(AVChannelLayout::from_nb_channels(ALERT_CHANNELS.into()).into_inner());
    encoder.set_time_base(AVRational {
        num: 1,
        den: ALERT_SAMPLE_RATE as i32,
    });
    encoder.set_bit_rate(DEFAULT_AC3_BITRATE);
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
fn create_video_encoder(feed: &FeedConfig, video_spec: &InputStreamSpec) -> Result<AVCodecContext> {
    let output = feed.output();
    let codec_name = output_codec_name(&output.vcodec);
    let codec_name_c = cstring_arg(&codec_name, "video codec name")?;
    let codec = AVCodec::find_encoder_by_name(codec_name_c.as_c_str())
        .with_context(|| format!("native cgen video encoder {codec_name} is unavailable"))?;
    let pix_fmt = select_video_pix_fmt(&codec, &codec_name);
    let frame_rate = output_frame_rate(feed, video_spec);
    let mut encoder = AVCodecContext::new(&codec);
    let width = if feed.video.width == 0 {
        video_spec.codecpar.width
    } else {
        feed.video.width as i32
    };
    let height = if feed.video.height == 0 {
        video_spec.codecpar.height
    } else {
        feed.video.height as i32
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
    encoder.set_gop_size(if feed.video.interlaced { 15 } else { 30 });
    encoder.set_max_b_frames(0);
    encoder.set_bit_rate(
        output
            .video_bitrate_kbps
            .map(|kbps| i64::from(kbps) * 1000)
            .unwrap_or(DEFAULT_VIDEO_BITRATE),
    );
    if feed.video.interlaced {
        encoder.set_flags(
            encoder.flags
                | ffi::AV_CODEC_FLAG_INTERLACED_DCT as i32
                | ffi::AV_CODEC_FLAG_INTERLACED_ME as i32,
        );
        // SAFETY: the encoder context is owned here and not yet opened. Field
        // order is a plain AVCodecContext metadata field used by encoders and
        // muxers to advertise interlaced output.
        unsafe {
            (*encoder.as_mut_ptr()).field_order = video_field_order(feed);
        }
    } else {
        // SAFETY: same ownership as above; explicitly mark non-interlaced
        // output progressive so stale input codec parameters cannot leak.
        unsafe {
            (*encoder.as_mut_ptr()).field_order = ffi::AV_FIELD_PROGRESSIVE;
        }
    }
    let options = video_encoder_options(&codec_name, feed);
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
fn output_frame_rate(feed: &FeedConfig, video_spec: &InputStreamSpec) -> AVRational {
    parse_rational(&feed.video.fps)
        .filter(valid_rational)
        .or_else(|| valid_rational(&video_spec.frame_rate).then_some(video_spec.frame_rate))
        .unwrap_or(AVRational {
            num: 30_000,
            den: 1001,
        })
}

fn video_field_order(feed: &FeedConfig) -> ffi::AVFieldOrder {
    if feed.video.field_order.eq_ignore_ascii_case("bff")
        || feed.video.field_order.eq_ignore_ascii_case("bottom")
        || feed.video.field_order.eq_ignore_ascii_case("bottom_first")
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
fn video_encoder_options(codec_name: &str, feed: &FeedConfig) -> Option<AVDictionary> {
    let lower = codec_name.to_ascii_lowercase();
    if lower.contains("libx264") || lower.contains("libx265") {
        return Some(AVDictionary::new(c"preset", c"ultrafast", 0).set(c"tune", c"zerolatency", 0));
    }
    if lower.contains("nvenc") {
        return Some(AVDictionary::new(c"preset", c"ll", 0).set(c"tune", c"ull", 0));
    }
    if feed.video.interlaced && lower.contains("mpeg2") {
        let top = if video_field_order(feed) == ffi::AV_FIELD_BB {
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

fn frame_to_stereo_pcm16(frame: &AVFrame) -> Result<Vec<i16>> {
    let channels = usize::try_from(frame.ch_layout().nb_channels.max(1)).unwrap_or(1);
    let samples = usize::try_from(frame.nb_samples.max(0)).unwrap_or_default();
    let mut out = Vec::with_capacity(samples * usize::from(ALERT_CHANNELS));
    match frame.format {
        ffi::AV_SAMPLE_FMT_S16 => {
            let data = frame.data[0] as *const i16;
            for sample_index in 0..samples {
                // SAFETY: decoded audio frames expose at least nb_samples *
                // channels packed samples in data[0] for AV_SAMPLE_FMT_S16.
                let (left, right) =
                    unsafe { packed_i16_stereo_sample(data, sample_index, channels) };
                out.push(left);
                out.push(right);
            }
        }
        ffi::AV_SAMPLE_FMT_S16P => {
            let left_plane = frame.data[0] as *const i16;
            let right_plane = if channels > 1 {
                frame.data[1] as *const i16
            } else {
                left_plane
            };
            for sample_index in 0..samples {
                // SAFETY: decoded planar frames expose nb_samples entries in
                // each channel plane used by the channel layout.
                let (left, right) = unsafe {
                    (
                        *left_plane.add(sample_index),
                        *right_plane.add(sample_index),
                    )
                };
                out.push(left);
                out.push(right);
            }
        }
        ffi::AV_SAMPLE_FMT_FLT => {
            let data = frame.data[0] as *const f32;
            for sample_index in 0..samples {
                // SAFETY: decoded audio frames expose at least nb_samples *
                // channels packed f32 samples in data[0] for AV_SAMPLE_FMT_FLT.
                let (left, right) =
                    unsafe { packed_f32_stereo_sample(data, sample_index, channels) };
                out.push(pcm_f32_to_i16(left));
                out.push(pcm_f32_to_i16(right));
            }
        }
        ffi::AV_SAMPLE_FMT_FLTP => {
            let left_plane = frame.data[0] as *const f32;
            let right_plane = if channels > 1 {
                frame.data[1] as *const f32
            } else {
                left_plane
            };
            for sample_index in 0..samples {
                // SAFETY: decoded planar frames expose nb_samples entries in
                // each channel plane used by the channel layout.
                let (left, right) = unsafe {
                    (
                        *left_plane.add(sample_index),
                        *right_plane.add(sample_index),
                    )
                };
                out.push(pcm_f32_to_i16(left));
                out.push(pcm_f32_to_i16(right));
            }
        }
        other => bail!("unsupported source audio sample format {other}"),
    }
    Ok(out)
}

unsafe fn packed_i16_stereo_sample(
    data: *const i16,
    sample_index: usize,
    channels: usize,
) -> (i16, i16) {
    let base = sample_index * channels;
    let left = unsafe { *data.add(base) };
    let right = if channels > 1 {
        unsafe { *data.add(base + 1) }
    } else {
        left
    };
    (left, right)
}

unsafe fn packed_f32_stereo_sample(
    data: *const f32,
    sample_index: usize,
    channels: usize,
) -> (f32, f32) {
    let base = sample_index * channels;
    let left = unsafe { *data.add(base) };
    let right = if channels > 1 {
        unsafe { *data.add(base + 1) }
    } else {
        left
    };
    (left, right)
}

fn fill_fltp_stereo_frame(frame: &mut AVFrame, pcm: &[i16], samples: usize) {
    let planes = frame.data_mut();
    let left = planes[0] as *mut f32;
    let right = planes[1] as *mut f32;
    for sample_index in 0..samples {
        let base = sample_index * usize::from(ALERT_CHANNELS);
        let l = pcm.get(base).copied().unwrap_or_default();
        let r = pcm.get(base + 1).copied().unwrap_or_default();
        // SAFETY: AVFrame::alloc_buffer allocated two FLTP planes with at least
        // `samples` f32 values each, and the loop writes within that range.
        unsafe {
            *left.add(sample_index) = pcm_i16_to_f32(l);
            *right.add(sample_index) = pcm_i16_to_f32(r);
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

fn pcm_f32_to_i16(value: f32) -> i16 {
    (value.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
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

    fn rational_tuple(value: Option<AVRational>) -> Option<(i32, i32)> {
        value.map(|value| (value.num, value.den))
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    #[test]
    fn ac3_encoder_emits_packet_for_one_frame() {
        let mut encoder = create_ac3_encoder().expect("AC-3 encoder");
        let silence = vec![0i16; AC3_FRAME_SAMPLES * usize::from(ALERT_CHANNELS)];
        let mut frame = AVFrame::new();
        frame.set_nb_samples(AC3_FRAME_SAMPLES as i32);
        frame.set_format(ffi::AV_SAMPLE_FMT_FLTP);
        frame.set_sample_rate(ALERT_SAMPLE_RATE as i32);
        frame.set_ch_layout(AVChannelLayout::from_nb_channels(ALERT_CHANNELS.into()).into_inner());
        frame.alloc_buffer().expect("audio frame buffer");
        fill_fltp_stereo_frame(&mut frame, &silence, AC3_FRAME_SAMPLES);
        encoder.send_frame(Some(&frame)).expect("send frame");
        let packet = encoder.receive_packet().expect("receive packet");
        assert!(packet.size > 0);
    }
}
