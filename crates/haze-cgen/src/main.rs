mod bridge;
mod config;
mod graphics;
#[cfg(feature = "ffmpeg-rsmpeg")]
mod native;
mod pipeline;
mod state;

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use serde_json::json;
use tokio::sync::watch;
use tracing::{info, warn};

const SOURCE_ID: &str = "haze-cgen";

#[derive(Debug, Parser)]
#[command(name = "haze-cgen", about = "Haze managed character generator service")]
struct Args {
    #[arg(long, default_value = "config.yaml")]
    config: PathBuf,
    #[arg(long, default_value = "managed/configs/cgen.xml")]
    cgen: PathBuf,
    #[arg(long, env = "HAZE_HOST_BRIDGE_ADDR")]
    bridge: String,
    #[arg(long)]
    ffmpeg: Option<String>,
    #[arg(long, default_value = "auto")]
    graphics_backend: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "haze_cgen=info,warn".into()),
        )
        .init();

    let args = Args::parse();
    spawn_parent_watcher();

    let base_dir = args
        .config
        .parent()
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("."));
    let cgen_path = config::resolve_path(&base_dir, &args.cgen);
    let root = config::load_config(&cgen_path)
        .with_context(|| format!("failed to load cgen config {}", cgen_path.display()))?;
    let feeds = root.enabled_feeds()?;
    info!(
        feeds = feeds.len(),
        config = %cgen_path.display(),
        graphics_backend = %args.graphics_backend,
        "haze-cgen config loaded"
    );

    let bridge = bridge::connect_retry(&args.bridge).await?;
    bridge
        .client
        .publish(json!({
            "type": "bridge.client",
            "source": SOURCE_ID,
            "data": { "receive_events": true }
        }))
        .await?;
    bridge
        .client
        .publish(json!({
            "type": "service.ready",
            "source": SOURCE_ID,
            "data": {
                "service": SOURCE_ID,
                "feeds": feeds.len(),
                "config": cgen_path,
                "graphics_backend": args.graphics_backend,
                "media_pipeline": "supervised",
                "priority_audio_only": true
            }
        }))
        .await?;

    let (state_tx, state_rx) = watch::channel(state::RuntimeState::default());
    for feed in feeds {
        let worker = pipeline::PipelineWorker::new(
            feed,
            state_rx.clone(),
            args.ffmpeg.clone(),
            args.graphics_backend.clone(),
            base_dir.clone(),
            Some(bridge.client.clone()),
        );
        tokio::spawn(async move {
            if let Err(err) = worker.run().await {
                warn!("cgen feed worker exited: {err}");
            }
        });
    }

    run_event_loop(bridge.events, state_tx).await
}

async fn run_event_loop(
    mut events: tokio::sync::mpsc::Receiver<serde_json::Value>,
    state_tx: watch::Sender<state::RuntimeState>,
) -> Result<()> {
    let mut runtime = state::RuntimeState::default();
    loop {
        tokio::select! {
            event = events.recv() => {
                let Some(event) = event else {
                    anyhow::bail!("event bridge closed");
                };
                if runtime.apply_event(&event) {
                    let _ = state_tx.send(runtime.clone());
                } else if event.get("type").and_then(serde_json::Value::as_str) == Some("cgen.config.updated") {
                    info!("cgen config update received; exiting for daemon restart");
                    return Ok(());
                }
            }
            result = tokio::signal::ctrl_c() => {
                result.context("failed waiting for Ctrl-C")?;
                info!("haze-cgen shutting down");
                return Ok(());
            }
        }
    }
}

fn spawn_parent_watcher() {
    let Ok(raw) = std::env::var("HAZE_PARENT_PID") else {
        return;
    };
    let Ok(parent_pid) = raw.parse::<u32>() else {
        return;
    };
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(1));
        loop {
            ticker.tick().await;
            if !process_alive(parent_pid) {
                warn!("parent process exited; haze-cgen exiting");
                std::process::exit(0);
            }
        }
    });
}

#[cfg(windows)]
fn process_alive(pid: u32) -> bool {
    use windows_sys::Win32::Foundation::{CloseHandle, FALSE};
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    unsafe {
        let handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
        if handle.is_null() {
            return false;
        }
        CloseHandle(handle);
        true
    }
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(any(unix, windows)))]
fn process_alive(_pid: u32) -> bool {
    true
}
