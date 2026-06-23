use std::ffi::{CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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
use crate::state::{PriorityAudio, RuntimeState};

const ALERT_SAMPLE_RATE: u32 = 48_000;
const ALERT_CHANNELS: u16 = 2;
const AC3_FRAME_SAMPLES: usize = 1536;
const DEFAULT_AC3_BITRATE: i64 = 192_000;
const DEFAULT_VIDEO_BITRATE: i64 = 12_000_000;
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
    let mut video_processor = VideoProcessor::new(feed, video_spec)?;

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
        video_processor.output_time_base(),
        video_processor.output_codecpar(),
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
    let mut output_clock = CorrectedWallClock::default();
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
            if let (Some(processor), Some(audio_spec)) =
                (audio_processor.as_mut(), audio_stream_spec.as_ref())
            {
                if packet.pts != ffi::AV_NOPTS_VALUE {
                    output_clock.observe_media_pts(
                        packet.pts,
                        spec.input_time_base,
                        audio_spec.output_time_base,
                    );
                }
                if processor.alert_active()? {
                    let target_audio_pts = output_clock
                        .target_pts(audio_spec.output_time_base)
                        .unwrap_or_else(|| {
                            packet_pts_or_next_audio_pts(
                                &packet,
                                spec.input_time_base,
                                audio_spec.output_time_base,
                                processor.next_pts,
                            )
                        });
                    processor.write_alert_until(target_audio_pts, audio_spec, &mut output)?;
                }
            }
            video_processor.process_packet(&packet, spec, &mut output)?;
            continue;
        }
        if Some(input_index) == audio_input_index {
            if let Some(processor) = audio_processor.as_mut() {
                processor.process_packet(&packet, spec, &mut output)?;
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
    video_time_base: AVRational,
    video_codecpar: AVCodecParameters,
    audio_input_index: Option<usize>,
    audio_codecpar: Option<AVCodecParameters>,
) -> Result<Vec<StreamSpec>> {
    let mut out = Vec::with_capacity(input_specs.len());
    for spec in input_specs {
        let mut stream = output.new_stream();
        let output_time_base = if spec.input_index == video_input_index {
            stream.set_codecpar(video_codecpar.clone());
            stream.set_time_base(video_time_base);
            video_time_base
        } else if Some(spec.input_index) == audio_input_index {
            let codecpar = audio_codecpar
                .as_ref()
                .context("native cgen audio output codec parameters are missing")?;
            stream.set_codecpar(codecpar.clone());
            stream.set_time_base(spec.time_base);
            stream.time_base
        } else {
            stream.set_codecpar(spec.codecpar.clone());
            stream.set_time_base(spec.time_base);
            stream.time_base
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

#[derive(Debug, Default)]
struct CorrectedWallClock {
    started_at: Option<Instant>,
    anchor_pts: Option<i64>,
    correction_ticks: f64,
}

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

fn packet_pts_or_next_audio_pts(
    packet: &AVPacket,
    packet_time_base: AVRational,
    audio_time_base: AVRational,
    next_audio_pts: Option<i64>,
) -> i64 {
    if packet.pts != ffi::AV_NOPTS_VALUE {
        return rescale_q(packet.pts, packet_time_base, audio_time_base);
    }
    next_audio_pts.unwrap_or(0)
}

fn duration_to_pts(duration: Duration, output_time_base: AVRational) -> i64 {
    (duration.as_secs_f64() * ticks_per_second(output_time_base)).round() as i64
}

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
    next_pts: i64,
}

impl VideoProcessor {
    fn new(feed: &FeedConfig, video_spec: &InputStreamSpec) -> Result<Self> {
        let decoder = create_video_decoder(&video_spec.codecpar)?;
        let encoder = create_video_encoder(feed, video_spec)?;
        let encoder_pix_fmt = encoder.pix_fmt;
        Ok(Self {
            feed_id: feed.id.clone(),
            decoder,
            encoder,
            scaler: None,
            encoder_width: feed.video.width as i32,
            encoder_height: feed.video.height as i32,
            encoder_pix_fmt,
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
        let mut frame = self.normalize_frame(source_frame)?;
        frame.set_pts(self.frame_pts(source_frame, spec));
        self.encoder
            .send_frame(Some(&frame))
            .context("failed to send native cgen video frame to encoder")?;
        self.drain_packets(spec, output)
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
        let pts = if source_frame.pts != ffi::AV_NOPTS_VALUE {
            rescale_q(
                source_frame.pts,
                spec.input_time_base,
                self.encoder.time_base,
            )
        } else {
            self.next_pts
        };
        self.next_pts = pts.saturating_add(1);
        pts
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
            next_pts: None,
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
        source_packet: &AVPacket,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        if self.alert_active()? {
            return Ok(());
        }
        if self.next_pts.is_none() && source_packet.pts != ffi::AV_NOPTS_VALUE {
            self.next_pts = Some(rescale_q(
                source_packet.pts,
                spec.input_time_base,
                spec.output_time_base,
            ));
        }
        self.decoder
            .send_packet(Some(source_packet))
            .context("failed to send native cgen source audio packet to decoder")?;
        loop {
            match self.decoder.receive_frame() {
                Ok(frame) => self.process_frame(&frame, spec, output)?,
                Err(err) if is_again(&err) => return Ok(()),
                Err(err) => return Err(err).context("failed to decode native cgen source audio"),
            }
        }
    }

    fn process_frame(
        &mut self,
        source_frame: &AVFrame,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        let active_audio = self.active_audio()?;
        let source_pcm = if active_audio {
            Vec::new()
        } else {
            match frame_to_stereo_pcm16(source_frame) {
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
            }
        };
        let frame_samples = usize::try_from(source_frame.nb_samples.max(0)).unwrap_or_default();
        let chunks = frame_samples.div_ceil(AC3_FRAME_SAMPLES).max(1);
        for chunk in 0..chunks {
            let mut encoded = if active_audio {
                self.encode_next_alert_frame()
            } else {
                self.encode_source_frame(&source_pcm, chunk * AC3_FRAME_SAMPLES)
            }
            .context("failed to encode native cgen mixed audio")?;
            let duration = audio_frame_duration(spec.output_time_base);
            let pts = self.next_pts.unwrap_or(0);
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

    fn write_alert_until(
        &mut self,
        target_pts: i64,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
        if !self.alert_active()? {
            return Ok(());
        }
        let duration = audio_frame_duration(spec.output_time_base);
        if self.next_pts.is_none() {
            self.next_pts = Some(target_pts);
        }
        let max_frames = 12usize;
        for _ in 0..max_frames {
            let pts = self.next_pts.unwrap_or(target_pts);
            if pts > target_pts.saturating_add(duration) {
                break;
            }
            let mut encoded = self
                .encode_next_alert_frame()
                .context("failed to encode native cgen alert break-in audio")?;
            encoded.set_pts(pts);
            encoded.set_dts(pts);
            encoded.set_duration(duration);
            encoded.set_stream_index(spec.output_index);
            encoded.set_pos(-1);
            output
                .interleaved_write_frame(&mut encoded)
                .context("native cgen write alert break-in audio packet failed")?;
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

    fn encode_source_frame(
        &mut self,
        source_pcm: &[i16],
        offset_frames: usize,
    ) -> Result<AVPacket> {
        let mut chunk = vec![0i16; AC3_FRAME_SAMPLES * usize::from(ALERT_CHANNELS)];
        let offset = offset_frames * usize::from(ALERT_CHANNELS);
        let available = source_pcm.len().saturating_sub(offset).min(chunk.len());
        if available > 0 {
            chunk[..available].copy_from_slice(&source_pcm[offset..offset + available]);
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

fn create_video_encoder(feed: &FeedConfig, video_spec: &InputStreamSpec) -> Result<AVCodecContext> {
    let output = feed.output();
    let codec_name = output_codec_name(&output.vcodec);
    let codec_name_c = cstring_arg(&codec_name, "video codec name")?;
    let codec = AVCodec::find_encoder_by_name(codec_name_c.as_c_str())
        .with_context(|| format!("native cgen video encoder {codec_name} is unavailable"))?;
    let pix_fmt = select_video_pix_fmt(&codec, &codec_name);
    let frame_rate = output_frame_rate(feed, video_spec);
    let mut encoder = AVCodecContext::new(&codec);
    encoder.set_width(feed.video.width as i32);
    encoder.set_height(feed.video.height as i32);
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
        encoder.set_flags(encoder.flags | ffi::AV_CODEC_FLAG_INTERLACED_DCT as i32);
    }
    let options = video_encoder_options(&codec_name);
    encoder
        .open(options)
        .with_context(|| format!("failed to open native cgen video encoder {codec_name}"))?;
    Ok(encoder)
}

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

fn output_frame_rate(feed: &FeedConfig, video_spec: &InputStreamSpec) -> AVRational {
    parse_rational(&feed.video.fps)
        .filter(valid_rational)
        .or_else(|| valid_rational(&video_spec.frame_rate).then_some(video_spec.frame_rate))
        .unwrap_or(AVRational {
            num: 30_000,
            den: 1001,
        })
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

fn valid_rational(value: &AVRational) -> bool {
    value.num > 0 && value.den > 0
}

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

fn video_encoder_options(codec_name: &str) -> Option<AVDictionary> {
    let lower = codec_name.to_ascii_lowercase();
    if lower.contains("libx264") || lower.contains("libx265") {
        return Some(AVDictionary::new(c"preset", c"ultrafast", 0).set(c"tune", c"zerolatency", 0));
    }
    if lower.contains("nvenc") {
        return Some(AVDictionary::new(c"preset", c"ll", 0).set(c"tune", c"ull", 0));
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
