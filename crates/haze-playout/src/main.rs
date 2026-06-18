mod bridge;
mod config;
mod engine;
mod sinks;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "haze-playout",
    version,
    about = "Rust playout engine for Haze Weather Radio"
)]
struct Args {
    #[arg(long, default_value = "config.yaml")]
    config: PathBuf,

    #[arg(long, env = "HAZE_HOST_BRIDGE_ADDR")]
    bridge: String,

    #[arg(long, default_value = "500ms")]
    alert_poll: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing();
    engine::run(engine::Options {
        config_path: args.config,
        bridge_addr: args.bridge,
        alert_poll: parse_duration_ms(&args.alert_poll, Duration::from_millis(500)),
    })
    .await
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(false)
        .try_init();
    let media = haze_media::backend_status();
    tracing::info!(
        backend = media.name,
        available = media.available,
        version = media.version.as_deref().unwrap_or("n/a"),
        "media backend ready"
    );
}

fn parse_duration_ms(raw: &str, fallback: Duration) -> Duration {
    let value = raw.trim();
    if value.is_empty() {
        return fallback;
    }
    if let Some(ms) = value.strip_suffix("ms") {
        return ms
            .trim()
            .parse::<u64>()
            .map(Duration::from_millis)
            .unwrap_or(fallback);
    }
    if let Some(seconds) = value.strip_suffix('s') {
        return seconds
            .trim()
            .parse::<u64>()
            .map(Duration::from_secs)
            .unwrap_or(fallback);
    }
    value
        .parse::<u64>()
        .map(Duration::from_millis)
        .unwrap_or(fallback)
}
