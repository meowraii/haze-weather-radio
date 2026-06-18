use std::str::FromStr;

use anyhow::{bail, Context, Result};
use chrono::{Datelike, Timelike, Utc};
use clap::{Args, Subcommand};
use serde::Serialize;

use crate::same_core::{
    eom_sequence, generate_same_header_attention_sequence, generate_same_header_sequence,
    SameHeader, ToneType,
};

#[derive(Debug, Subcommand)]
pub enum SameCommand {
    /// Generate SAME header/tone/EOM audio.
    Generate(SameGenerateArgs),
}

#[derive(Debug, Args)]
pub struct SameGenerateArgs {
    /// SAME originator code.
    #[arg(long, default_value = "WXR")]
    originator: String,

    /// SAME event code.
    #[arg(long)]
    event: String,

    /// Comma-separated SAME location codes.
    #[arg(long, value_delimiter = ',')]
    locations: Vec<String>,

    /// SAME duration as HHMM.
    #[arg(long, default_value = "0015")]
    duration: String,

    /// Station callsign / sender field.
    #[arg(long, default_value = "HAZE")]
    callsign: String,

    /// SAME issue time as JJJHHMM. Defaults to current UTC.
    #[arg(long)]
    issue_time: Option<String>,

    /// Attention tone type: WXR, EAS, NPAS, EGG_TIMER, QUEBEC, or NONE.
    #[arg(long, default_value = "WXR")]
    tone: ToneArg,

    /// Audio segment to generate: full, header, or eom.
    #[arg(long, default_value = "full")]
    sequence: SameSequenceArg,

    /// Emit JSON instead of a plain header string.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Clone, Copy)]
struct ToneArg(Option<ToneType>);

impl FromStr for ToneArg {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let tone = match value.trim().to_ascii_uppercase().as_str() {
            "" | "NONE" | "NO" | "OFF" => None,
            "WXR" | "WEATHER" => Some(ToneType::Wxr),
            "EAS" => Some(ToneType::Eas),
            "NPAS" | "ALERT_READY" | "ALERT-READY" => Some(ToneType::Npas),
            "EGG_TIMER" | "EGG-TIMER" | "EGG" => Some(ToneType::EggTimer),
            "QUEBEC" | "QC" => Some(ToneType::Quebec),
            other => bail!("unsupported attention tone {other:?}"),
        };
        Ok(Self(tone))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SameSequenceArg {
    Full,
    Header,
    Eom,
}

impl FromStr for SameSequenceArg {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "full" | "all" | "complete" => Ok(Self::Full),
            "header" | "start" | "attention" | "head" => Ok(Self::Header),
            "eom" | "end" | "tail" => Ok(Self::Eom),
            other => bail!("unsupported SAME sequence {other:?}"),
        }
    }
}

#[derive(Debug, Serialize)]
struct SameGenerateOutput {
    ok: bool,
    header: String,
    format: &'static str,
    sample_rate: u32,
    channels: u8,
    audio_base64: String,
    duration_seconds: f32,
}

pub(crate) fn run(command: SameCommand) -> Result<()> {
    match command {
        SameCommand::Generate(args) => run_generate(args),
    }
}

fn run_generate(args: SameGenerateArgs) -> Result<()> {
    let header = SameHeader::new(
        args.originator.trim().to_ascii_uppercase(),
        args.event.trim().to_ascii_uppercase(),
        args.locations
            .iter()
            .map(|location| location.trim().to_string())
            .filter(|location| !location.is_empty()),
        args.duration.trim().to_string(),
        args.callsign.trim().to_string(),
        args.issue_time
            .unwrap_or_else(current_same_issue_time)
            .trim()
            .to_string(),
    )
    .context("failed to build SAME header")?;
    let audio = match args.sequence {
        SameSequenceArg::Full => generate_same_header_sequence(&header, args.tone.0),
        SameSequenceArg::Header => generate_same_header_attention_sequence(&header, args.tone.0),
        SameSequenceArg::Eom => eom_sequence(),
    };
    let pcm = audio.to_pcm16le();
    let output = SameGenerateOutput {
        ok: true,
        header: header.encoded(),
        format: "raw",
        sample_rate: audio.sample_rate,
        channels: 1,
        audio_base64: encode_base64(&pcm),
        duration_seconds: audio.samples.len() as f32 / audio.sample_rate as f32,
    };
    if args.json {
        println!("{}", serde_json::to_string(&output)?);
    } else {
        println!("{}", output.header);
    }
    Ok(())
}

fn current_same_issue_time() -> String {
    let now = Utc::now();
    format!("{:03}{:02}{:02}", now.ordinal(), now.hour(), now.minute())
}

fn encode_base64(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut output = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | b2 as u32;
        output.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        output.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            output.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
        } else {
            output.push('=');
        }
        if chunk.len() > 2 {
            output.push(TABLE[(n & 0x3f) as usize] as char);
        } else {
            output.push('=');
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_panel_tone_names() {
        assert_eq!("WXR".parse::<ToneArg>().unwrap().0, Some(ToneType::Wxr));
        assert_eq!("NPAS".parse::<ToneArg>().unwrap().0, Some(ToneType::Npas));
        assert_eq!(
            "EGG_TIMER".parse::<ToneArg>().unwrap().0,
            Some(ToneType::EggTimer)
        );
        assert_eq!(
            "QUEBEC".parse::<ToneArg>().unwrap().0,
            Some(ToneType::Quebec)
        );
        assert_eq!("NONE".parse::<ToneArg>().unwrap().0, None);
    }

    #[test]
    fn parses_same_sequence_names() {
        assert_eq!(
            "full".parse::<SameSequenceArg>().unwrap(),
            SameSequenceArg::Full
        );
        assert_eq!(
            "header".parse::<SameSequenceArg>().unwrap(),
            SameSequenceArg::Header
        );
        assert_eq!(
            "eom".parse::<SameSequenceArg>().unwrap(),
            SameSequenceArg::Eom
        );
    }

    #[test]
    fn issue_time_has_same_shape() {
        let issue_time = current_same_issue_time();

        assert_eq!(issue_time.len(), 7);
        assert!(issue_time.bytes().all(|byte| byte.is_ascii_digit()));
    }

    #[test]
    fn encodes_base64_without_external_dependency() {
        assert_eq!(encode_base64(b""), "");
        assert_eq!(encode_base64(b"f"), "Zg==");
        assert_eq!(encode_base64(b"fo"), "Zm8=");
        assert_eq!(encode_base64(b"foo"), "Zm9v");
    }
}
