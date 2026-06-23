use std::ffi::{CStr, CString};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use haze_media::{normalize_pcm, Pcm};
use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecParameters, AVPacket};
use rsmpeg::avformat::{AVFormatContextInput, AVFormatContextOutput};
use rsmpeg::avutil::{AVChannelLayout, AVFrame, AVRational};
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
            "starting native rsmpeg cgen transport"
        );
        let result = tokio::task::spawn_blocking(move || {
            remux_once(&worker_feed, worker_state_rx, &worker_base_dir)
        })
        .await
        .context("native cgen remux worker panicked")?;
        match result {
            Ok(()) => {
                info!(feed_id = %feed.id, "native rsmpeg cgen transport exited cleanly");
                restart_delay = Duration::from_millis(500);
            }
            Err(err) => {
                warn!(feed_id = %feed.id, "native rsmpeg cgen transport failed: {err:#}");
            }
        }
        sleep(restart_delay).await;
        restart_delay = (restart_delay * 2).min(Duration::from_secs(10));
    }
}

fn remux_once(
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
    let stream_map = create_output_streams(
        &mut output,
        &input_specs,
        audio_input_index,
        audio_processor
            .as_ref()
            .map(|processor| processor.output_codecpar()),
    )?;
    let mut header_options = None;
    output
        .write_header(&mut header_options)
        .context("failed to write native cgen output header")?;
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
            codecpar,
        });
    }
    Ok(out)
}

fn create_output_streams(
    output: &mut AVFormatContextOutput,
    input_specs: &[InputStreamSpec],
    audio_input_index: Option<usize>,
    audio_codecpar: Option<AVCodecParameters>,
) -> Result<Vec<StreamSpec>> {
    let mut out = Vec::with_capacity(input_specs.len());
    for spec in input_specs {
        let mut stream = output.new_stream();
        if Some(spec.input_index) == audio_input_index {
            let codecpar = audio_codecpar
                .as_ref()
                .context("native cgen audio output codec parameters are missing")?;
            stream.set_codecpar(codecpar.clone());
        } else {
            stream.set_codecpar(spec.codecpar.clone());
        }
        stream.set_time_base(spec.time_base);
        out.push(StreamSpec {
            input_index: spec.input_index,
            output_index: stream.index,
            input_time_base: spec.time_base,
            output_time_base: stream.time_base,
        });
    }
    Ok(out)
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
        self.encoder.extract_codecpar()
    }

    fn process_packet(
        &mut self,
        source_packet: &AVPacket,
        spec: &StreamSpec,
        output: &mut AVFormatContextOutput,
    ) -> Result<()> {
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
