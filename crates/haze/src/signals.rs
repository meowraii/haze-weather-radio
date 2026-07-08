use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use tracing::{info, warn};

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

pub(crate) fn install_shutdown_handler() -> Result<()> {
    ctrlc::set_handler(move || {
        if SHUTDOWN_REQUESTED.swap(true, Ordering::SeqCst) {
            warn!("second shutdown signal received; exiting immediately");
            std::process::exit(130);
        }
        info!("shutdown signal received; asking managed services to stop");
    })
    .context("failed to install shutdown handler")
}

pub(crate) fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::SeqCst)
}

#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn request_shutdown() {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}
