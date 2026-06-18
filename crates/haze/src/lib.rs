mod daemon_services;
mod go_services;
mod host_bridge;
mod runtime_dir;
mod same_cli;
#[allow(dead_code)]
mod same_core;
mod signals;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use tracing::info;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Command line options for the Haze daemon host.
#[derive(Debug, Parser)]
#[command(name = "haze", version, about = "Haze Weather Radio host")]
pub struct DaemonArgs {
    #[command(subcommand)]
    pub command: Option<DaemonCommand>,

    /// Path to the Haze YAML configuration file.
    #[arg(short, long, default_value = "config.yaml")]
    pub config: PathBuf,

    /// Runtime directory containing config, logs, data, audio, and managed state.
    #[arg(long)]
    pub workdir: Option<PathBuf>,

    /// Override the Haze log level.
    #[arg(short = 'l', long)]
    pub log_level: Option<String>,

    /// Initialize the host runtime and exit without starting long-running services.
    #[arg(long = "host-smoke", alias = "runtime-smoke")]
    pub host_smoke: bool,
}

#[derive(Debug, Subcommand)]
pub enum DaemonCommand {
    /// SAME tools.
    Same {
        #[command(subcommand)]
        command: same_cli::SameCommand,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct ServiceHostConfig {
    pub app_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub config_path: PathBuf,
}

/// Run the Haze daemon host.
///
/// # Errors
///
/// Returns an error when the working directory cannot be selected, the bundled
/// runtime cannot be initialized, or Haze exits with an exception.
pub fn run(args: DaemonArgs) -> Result<()> {
    if let Some(command) = args.command {
        return match command {
            DaemonCommand::Same { command } => same_cli::run(command),
        };
    }

    let layout = runtime_dir::resolve(args.workdir.as_deref())?;
    let config_path = normalize_config_path(&layout.runtime_dir, &args.config);
    let host_exe = env::current_exe().context("failed to resolve Haze host executable")?;

    env::set_current_dir(&layout.runtime_dir).with_context(|| {
        format!(
            "failed to enter Haze runtime directory {}",
            layout.runtime_dir.display()
        )
    })?;
    env::set_var("CONFIG_PATH", &config_path);
    env::set_var("HAZE_HOST_RUNTIME", "haze");
    env::set_var("HAZE_HOST_EXE", &host_exe);
    env::set_var("HAZE_APP_DIR", &layout.app_dir);
    env::set_var("HAZE_RUNTIME_DIR", &layout.runtime_dir);
    if let Some(level) = args.log_level.as_deref() {
        env::set_var("LOG_LEVEL", level.to_uppercase());
    }

    let _log_guard = init_tracing(args.log_level.as_deref(), &layout.runtime_dir);
    let host_bridge = host_bridge::HostBridge::start()?;
    env::set_var("HAZE_HOST_BRIDGE_ADDR", host_bridge.addr());
    env::set_var("HAZE_LOG_BRIDGE_ADDR", host_bridge.addr());

    let host = ServiceHostConfig {
        app_dir: layout.app_dir,
        runtime_dir: layout.runtime_dir,
        config_path,
    };

    info!(
        "starting Haze services: config={}, app={}, runtime={}",
        host.config_path.display(),
        host.app_dir.display(),
        host.runtime_dir.display()
    );

    if args.host_smoke {
        info!("host smoke check completed");
        return Ok(());
    }
    signals::install_shutdown_handler()?;
    let mut daemon_services =
        daemon_services::DaemonServices::start(&host, host_bridge.publisher())?;
    let mut go_services = go_services::GoServiceSupervisor::start(&host)?;
    wait_for_shutdown(
        &mut daemon_services,
        &mut go_services,
        host_bridge.publisher(),
    )
}

fn wait_for_shutdown(
    daemon_services: &mut daemon_services::DaemonServices,
    go_services: &mut go_services::GoServiceSupervisor,
    publisher: Sender<Value>,
) -> Result<()> {
    info!("Haze host services running");
    while !signals::shutdown_requested() {
        go_services.poll_children();
        thread::sleep(Duration::from_millis(250));
    }
    info!("shutdown requested; stopping managed services");
    let _ = publisher.send(json!({
        "type": "system.shutdown",
        "source": "haze",
    }));
    daemon_services.shutdown();
    go_services.shutdown();
    Ok(())
}

fn init_tracing(log_level: Option<&str>, runtime_dir: &Path) -> Option<WorkerGuard> {
    let fallback = log_level.unwrap_or("info");
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(fallback));
    let log_dir = runtime_dir.join("logs");
    let file_appender = tracing_appender::rolling::daily(log_dir, "haze.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

    let console_layer = tracing_subscriber::fmt::layer()
        .with_ansi(true)
        .with_target(false)
        .compact();
    let file_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_target(true)
        .with_writer(file_writer);

    match tracing_subscriber::registry()
        .with(filter)
        .with(console_layer)
        .with(file_layer)
        .try_init()
    {
        Ok(()) => Some(guard),
        Err(_) => None,
    }
}

fn normalize_config_path(workdir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        dunce::simplified(path).to_path_buf()
    } else {
        dunce::simplified(&workdir.join(path)).to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_config_is_resolved_against_workdir() {
        let workdir = PathBuf::from("haze-root");
        let config = normalize_config_path(&workdir, Path::new("config.yaml"));

        assert_eq!(config, PathBuf::from("haze-root").join("config.yaml"));
    }

    #[test]
    fn absolute_config_is_preserved() {
        let absolute = std::env::current_dir()
            .expect("test process should have a current directory")
            .join("prod.yaml");
        let config = normalize_config_path(Path::new("."), &absolute);

        assert_eq!(config, absolute);
    }
}
