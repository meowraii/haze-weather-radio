use std::ffi::{CStr, CString};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use rsmpeg::avcodec::AVCodecParameters;
use rsmpeg::avformat::{AVFormatContextInput, AVFormatContextOutput};
use rsmpeg::avutil::AVRational;
use tokio::time::sleep;
use tracing::{info, warn};

use crate::config::FeedConfig;

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

pub(crate) async fn run_remux_supervised(feed: FeedConfig) -> Result<()> {
    let mut restart_delay = Duration::from_millis(500);
    loop {
        let worker_feed = feed.clone();
        info!(
            feed_id = %feed.id,
            input = %feed.program_input_url(),
            output = %feed.program_output_url(),
            "starting native rsmpeg cgen transport"
        );
        let result = tokio::task::spawn_blocking(move || remux_once(&worker_feed))
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

fn remux_once(feed: &FeedConfig) -> Result<()> {
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
    let stream_map = create_output_streams(&mut output, &input_specs)?;
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
) -> Result<Vec<StreamSpec>> {
    let mut out = Vec::with_capacity(input_specs.len());
    for spec in input_specs {
        let mut stream = output.new_stream();
        stream.set_codecpar(spec.codecpar.clone());
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
}
