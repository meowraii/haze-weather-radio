use std::io::Cursor;

use anyhow::{bail, Context, Result};

/// Interleaved signed 16-bit little-endian PCM audio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcm {
    pub sample_rate: u32,
    pub channels: u16,
    pub data: Vec<u8>,
}

/// Describes PCM stream shape.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioFormat {
    /// Creates a normalized audio format with at least one channel.
    #[must_use]
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        Self {
            sample_rate,
            channels: channels.max(1),
        }
    }

    /// Returns the number of bytes in one interleaved PCM frame.
    #[must_use]
    pub fn frame_bytes(self) -> usize {
        usize::from(self.channels.max(1)) * 2
    }
}

/// Generates signed 16-bit little-endian silence.
#[must_use]
pub fn silence_chunk(sample_rate: u32, channels: u16, millis: u32) -> Vec<u8> {
    let samples = (sample_rate.saturating_mul(millis) / 1000).max(1);
    vec![0; samples as usize * channels.max(1) as usize * 2]
}

/// Decodes a WAV payload into interleaved signed 16-bit little-endian PCM.
///
/// # Errors
///
/// Returns an error when the WAV header is invalid, the stream has no audio, or
/// the bit depth cannot be converted safely into 16-bit PCM.
pub fn decode_wav(data: &[u8]) -> Result<Pcm> {
    let cursor = Cursor::new(data);
    let mut reader = hound::WavReader::new(cursor).context("failed to parse WAV")?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        bail!("WAV format is missing sample rate or channel count");
    }
    let mut out = Vec::with_capacity(reader.duration() as usize * spec.channels as usize * 2);
    match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 8 {
                for sample in reader.samples::<i8>() {
                    push_i16(&mut out, i16::from(sample?) << 8);
                }
            } else if spec.bits_per_sample <= 16 {
                for sample in reader.samples::<i16>() {
                    push_i16(&mut out, sample?);
                }
            } else if spec.bits_per_sample <= 32 {
                let shift = u32::from(spec.bits_per_sample.saturating_sub(16));
                for sample in reader.samples::<i32>() {
                    let scaled = sample? >> shift;
                    push_i16(
                        &mut out,
                        scaled.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
                    );
                }
            } else {
                bail!("unsupported WAV bit depth {}", spec.bits_per_sample);
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let scaled = (sample?.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round();
                push_i16(&mut out, scaled as i16);
            }
        }
    }
    if out.is_empty() {
        bail!("WAV contains no audio data");
    }
    Ok(Pcm {
        sample_rate: spec.sample_rate,
        channels: spec.channels,
        data: out,
    })
}

/// Remixes and resamples PCM into the requested format.
#[must_use]
pub fn normalize_pcm(mut pcm: Pcm, sample_rate: u32, channels: u16) -> Pcm {
    let format = AudioFormat::new(sample_rate, channels);
    if pcm.channels == 0 {
        pcm.channels = format.channels;
    }
    if pcm.sample_rate == 0 {
        pcm.sample_rate = format.sample_rate;
    }
    if pcm.channels != format.channels {
        pcm.data = remix_pcm16(&pcm.data, pcm.channels, format.channels);
        pcm.channels = format.channels;
    }
    if pcm.sample_rate != format.sample_rate {
        pcm.data = resample_pcm16(&pcm.data, pcm.sample_rate, format.sample_rate, pcm.channels);
        pcm.sample_rate = format.sample_rate;
    }
    let frame_bytes = AudioFormat::new(pcm.sample_rate, pcm.channels).frame_bytes();
    if frame_bytes > 0 {
        pcm.data
            .truncate(pcm.data.len() - pcm.data.len() % frame_bytes);
    }
    pcm
}

/// Changes channel count by averaging input channels for downmixes and
/// duplicating mono content for upmixes.
#[must_use]
pub fn remix_pcm16(data: &[u8], in_channels: u16, out_channels: u16) -> Vec<u8> {
    if in_channels == out_channels || in_channels == 0 || out_channels == 0 {
        return data.to_vec();
    }
    let in_channels = usize::from(in_channels);
    let out_channels = usize::from(out_channels);
    let frames = data.len() / (in_channels * 2);
    let mut out = Vec::with_capacity(frames * out_channels * 2);
    for frame in 0..frames {
        let mut mono = 0i32;
        for channel in 0..in_channels {
            let offset = (frame * in_channels + channel) * 2;
            mono += i32::from(read_i16(&data[offset..offset + 2]));
        }
        mono /= in_channels as i32;
        let sample = mono.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
        for _ in 0..out_channels {
            push_i16(&mut out, sample);
        }
    }
    out
}

/// Resamples interleaved PCM16 using deterministic linear interpolation.
///
/// This is the built-in fallback. The rsmpeg backend will replace this for
/// production codec paths as that migration lands.
#[must_use]
pub fn resample_pcm16(data: &[u8], in_rate: u32, out_rate: u32, channels: u16) -> Vec<u8> {
    if in_rate == out_rate || in_rate == 0 || out_rate == 0 || channels == 0 || data.is_empty() {
        return data.to_vec();
    }
    let channels = usize::from(channels);
    let in_frames = data.len() / (channels * 2);
    if in_frames == 0 {
        return Vec::new();
    }
    let out_frames =
        ((in_frames as f64 * f64::from(out_rate) / f64::from(in_rate)).round() as usize).max(1);
    let mut out = Vec::with_capacity(out_frames * channels * 2);
    for out_frame in 0..out_frames {
        let source = out_frame as f64 * f64::from(in_rate) / f64::from(out_rate);
        let left = source.floor() as usize;
        let right = (left + 1).min(in_frames - 1);
        let fraction = source - left as f64;
        for channel in 0..channels {
            let left_offset = (left * channels + channel) * 2;
            let right_offset = (right * channels + channel) * 2;
            let a = f64::from(read_i16(&data[left_offset..left_offset + 2]));
            let b = f64::from(read_i16(&data[right_offset..right_offset + 2]));
            let sample = (a + (b - a) * fraction).round() as i32;
            push_i16(
                &mut out,
                sample.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
            );
        }
    }
    out
}

#[inline]
#[must_use]
pub fn read_i16(bytes: &[u8]) -> i16 {
    i16::from_le_bytes([bytes[0], bytes[1]])
}

#[inline]
pub fn push_i16(out: &mut Vec<u8>, sample: i16) {
    out.extend_from_slice(&sample.to_le_bytes());
}

pub fn pcm16_samples(data: &[u8]) -> impl Iterator<Item = i16> + '_ {
    data.chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_24_bit_wav_to_16_bit_pcm() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 48_000,
                bits_per_sample: 24,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("writer");
            writer.write_sample::<i32>(0x12_3400).expect("sample");
            writer.write_sample::<i32>(-0x12_3400).expect("sample");
            writer.finalize().expect("finalize");
        }
        let pcm = decode_wav(&cursor.into_inner()).expect("decode");
        assert_eq!(pcm.sample_rate, 48_000);
        assert_eq!(pcm.channels, 1);
        let samples: Vec<i16> = pcm16_samples(&pcm.data).collect();
        assert_eq!(samples, vec![0x1234, -0x1234]);
    }

    #[test]
    fn resamples_and_remixes() {
        let mut stereo = Vec::new();
        push_i16(&mut stereo, 100);
        push_i16(&mut stereo, 300);
        push_i16(&mut stereo, 500);
        push_i16(&mut stereo, 700);
        let mono = remix_pcm16(&stereo, 2, 1);
        let samples: Vec<i16> = pcm16_samples(&mono).collect();
        assert_eq!(samples, vec![200, 600]);
        let resampled = resample_pcm16(&mono, 2, 4, 1);
        assert_eq!(resampled.len(), 8);
    }
}
