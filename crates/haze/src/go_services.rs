use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, error, info, warn};

use crate::ServiceHostConfig;

const DEFAULT_SETTINGS_FILE: &str = "runtime/state/daemonSettings.json";
const STATUS_FILE: &str = "runtime/state/goServiceRuntime.json";
const EMBEDDED_BIN_DIR: &str = "bin";
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/go_assets.rs"));
}

#[derive(Debug, Default, Deserialize)]
struct RootConfig {
    services: Option<ServicesConfig>,
    webpanel: Option<WebPanelConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct ServicesConfig {
    go: Option<GoServicesConfig>,
    rust: Option<RustServicesConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct RustServicesConfig {
    playout: Option<RustPlayoutConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct RustPlayoutConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    alert_poll: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct GoServicesConfig {
    enabled: Option<bool>,
    web_gateway: Option<WebGatewayConfig>,
    data_ingest: Option<DataIngestConfig>,
    cap_ingest: Option<CapIngestConfig>,
    tts: Option<TtsConfig>,
    product_render: Option<ProductRenderConfig>,
    playlist: Option<PlaylistConfig>,
    ivr: Option<IvrConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct WebGatewayConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    addr: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct WebPanelConfig {
    host: Option<String>,
    port: Option<u16>,
    public: Option<WebPanelSurfaceConfig>,
    admin: Option<WebPanelSurfaceConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct WebPanelSurfaceConfig {
    enabled: Option<bool>,
    host: Option<String>,
    port: Option<u16>,
}

#[derive(Debug, Default, Deserialize)]
struct CapIngestConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    source_id: Option<String>,
    source: Option<String>,
    url: Option<String>,
    interval: Option<String>,
    timeout: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct DataIngestConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    interval: Option<String>,
    timeout: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct TtsConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    readers: Option<String>,
    dictionary: Option<String>,
    provider: Option<String>,
    language: Option<String>,
    timezone: Option<String>,
    out_dir: Option<String>,
    timeout: Option<String>,
    piper_executable: Option<String>,
    piper_voices_dir: Option<String>,
    piper_mode: Option<String>,
    piper_workers: Option<usize>,
    piper_prewarm: Option<bool>,
    piper_cuda: Option<bool>,
    kokoro_model_dir: Option<String>,
    kokoro_runtime_provider: Option<String>,
    kokoro_threads: Option<usize>,
    kokoro_speed: Option<f32>,
    kokoro_length_scale: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
struct ProductRenderConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    refresh: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct PlaylistConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    tick: Option<String>,
    lookahead: Option<String>,
    out_dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct IvrConfig {
    enabled: Option<bool>,
    executable: Option<PathBuf>,
    http: Option<IvrHttpConfig>,
    sip: Option<IvrSipConfig>,
    cache: Option<IvrCacheConfig>,
}

#[derive(Debug, Default, Deserialize)]
struct IvrHttpConfig {
    addr: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct IvrSipConfig {
    listen: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct IvrCacheConfig {
    dir: Option<String>,
}

#[derive(Debug)]
struct ServiceSpec {
    id: &'static str,
    kind: &'static str,
    binary: &'static str,
    configured_executable: Option<PathBuf>,
    args: Vec<String>,
}

#[derive(Debug)]
struct ManagedChild {
    id: &'static str,
    child: Child,
}

/// Owns managed service child processes for the lifetime of the daemon.
pub(crate) struct GoServiceSupervisor {
    children: Vec<ManagedChild>,
    status_path: PathBuf,
    statuses: BTreeMap<String, serde_json::Value>,
}

impl GoServiceSupervisor {
    /// Starts enabled managed services from the loaded daemon configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the main YAML configuration cannot be read or
    /// parsed. Individual optional service spawn failures are recorded in the
    /// runtime status file and do not abort the Haze daemon.
    pub(crate) fn start(host: &ServiceHostConfig) -> Result<Self> {
        let root = load_config_with_overlay(&host.config_path, &host.runtime_dir)?;
        let status_path = host.runtime_dir.join(STATUS_FILE);
        if let Some(parent) = status_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create managed service status directory {}",
                    parent.display()
                )
            })?;
        }
        stop_previous_managed_services(&status_path);

        let mut supervisor = Self {
            children: Vec::new(),
            status_path,
            statuses: BTreeMap::new(),
        };

        let specs = service_specs(&root, host);
        for spec in specs {
            supervisor.start_one(spec, host);
        }
        supervisor.write_status();
        Ok(supervisor)
    }

    fn start_one(&mut self, spec: ServiceSpec, host: &ServiceHostConfig) {
        let Some(executable) =
            resolve_executable(host, spec.configured_executable.as_deref(), spec.binary).or_else(
                || {
                    extract_embedded_executable(host, spec.binary)
                        .ok()
                        .flatten()
                },
            )
        else {
            let error = format!("managed service binary not found: {}", spec.binary);
            warn!("[{}] {error}", service_label(spec.id));
            self.set_status(spec.id, "missing", None, Some(error));
            return;
        };

        let mut command = Command::new(&executable);
        configure_managed_process(&mut command);
        command
            .args(&spec.args)
            .current_dir(&host.runtime_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        match command.spawn() {
            Ok(mut child) => {
                let pid = child.id();
                if let Some(stdout) = child.stdout.take() {
                    pipe_service_output(spec.id, "stdout", stdout, false);
                }
                if let Some(stderr) = child.stderr.take() {
                    pipe_service_output(spec.id, "stderr", stderr, true);
                }
                info!("started {} (pid {pid})", service_label(spec.id));
                debug!(
                    "[{}] executable {}",
                    service_label(spec.id),
                    executable.display()
                );
                self.statuses.insert(
                    spec.id.to_string(),
                    json!({
                        "id": spec.id,
                        "kind": spec.kind,
                        "status": "running",
                        "pid": pid,
                        "executable": executable,
                        "args": spec.args,
                        "embedded": embedded_binary(spec.binary).is_some(),
                        "started_at_unix": unix_now(),
                    }),
                );
                self.children.push(ManagedChild { id: spec.id, child });
            }
            Err(err) => {
                let detail = format!("failed to start {}: {err}", service_label(spec.id));
                warn!("{detail}");
                self.set_status(spec.id, "failed", None, Some(detail));
            }
        }
    }

    fn set_status(&mut self, id: &str, status: &str, pid: Option<u32>, last_error: Option<String>) {
        self.statuses.insert(
            id.to_string(),
            json!({
                "id": id,
                "kind": "managed",
                "status": status,
                "pid": pid,
                "last_error": last_error,
                "updated_at_unix": unix_now(),
            }),
        );
    }

    pub(crate) fn poll_children(&mut self) {
        let mut changed = false;
        let mut running = Vec::with_capacity(self.children.len());
        for mut child in self.children.drain(..) {
            match child.child.try_wait() {
                Ok(Some(status)) => {
                    let state = if status.success() { "exited" } else { "failed" };
                    let detail = format!("{} exited with {status}", service_label(child.id));
                    warn!("[{}] exited with {status}", service_label(child.id));
                    self.statuses.insert(
                        child.id.to_string(),
                        json!({
                            "id": child.id,
                            "kind": "managed",
                            "status": state,
                            "pid": child.child.id(),
                            "last_error": if status.success() { None::<String> } else { Some(detail) },
                            "exited_at_unix": unix_now(),
                        }),
                    );
                    changed = true;
                }
                Ok(None) => running.push(child),
                Err(err) => {
                    let detail = format!("failed to inspect {}: {err}", service_label(child.id));
                    warn!(
                        "[{}] failed to inspect service: {err}",
                        service_label(child.id)
                    );
                    self.statuses.insert(
                        child.id.to_string(),
                        json!({
                            "id": child.id,
                            "kind": "managed",
                            "status": "unknown",
                            "pid": child.child.id(),
                            "last_error": detail,
                            "updated_at_unix": unix_now(),
                        }),
                    );
                    running.push(child);
                    changed = true;
                }
            }
        }
        self.children = running;
        if changed {
            self.write_status();
        }
    }

    pub(crate) fn shutdown(&mut self) {
        self.poll_children();
        let mut changed = false;
        for mut child in self.children.drain(..) {
            match child.child.try_wait() {
                Ok(Some(status)) => {
                    info!(
                        "[{}] exited during shutdown with {status}",
                        service_label(child.id)
                    );
                    self.statuses.insert(
                        child.id.to_string(),
                        json!({
                            "id": child.id,
                            "kind": "managed",
                            "status": if status.success() { "exited" } else { "failed" },
                            "pid": child.child.id(),
                            "exited_at_unix": unix_now(),
                        }),
                    );
                }
                Ok(None) => {
                    info!("stopping {}", service_label(child.id));
                    if let Err(err) = terminate_child_tree(&mut child.child) {
                        warn!(
                            "[{}] failed to stop service: {err}",
                            service_label(child.id)
                        );
                    }
                    let status = wait_for_child_exit(&mut child.child, SHUTDOWN_GRACE);
                    self.statuses.insert(
                        child.id.to_string(),
                        json!({
                            "id": child.id,
                            "kind": "managed",
                            "status": "stopped",
                            "pid": child.child.id(),
                            "exit_status": status.map(|status| status.to_string()),
                            "stopped_at_unix": unix_now(),
                        }),
                    );
                }
                Err(err) => {
                    warn!(
                        "[{}] failed to inspect service: {err}",
                        service_label(child.id)
                    );
                    self.statuses.insert(
                        child.id.to_string(),
                        json!({
                            "id": child.id,
                            "kind": "managed",
                            "status": "unknown",
                            "pid": child.child.id(),
                            "last_error": format!("failed to inspect managed service: {err}"),
                            "updated_at_unix": unix_now(),
                        }),
                    );
                }
            }
            changed = true;
        }
        if changed {
            self.write_status();
        }
    }

    fn write_status(&self) {
        let payload = json!({
            "updated_at_unix": unix_now(),
            "services": self.statuses,
        });
        if let Ok(raw) = serde_json::to_vec_pretty(&payload) {
            let tmp = self.status_path.with_extension("json.tmp");
            if fs::write(&tmp, raw).is_ok() {
                let _ = fs::rename(tmp, &self.status_path);
            }
        }
    }
}

fn wait_for_child_exit(child: &mut Child, timeout: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(100)),
            Ok(None) => {
                let _ = child.kill();
                return child.wait().ok();
            }
            Err(_) => return None,
        }
    }
}

impl Drop for GoServiceSupervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn service_specs(root: &RootConfig, host: &ServiceHostConfig) -> Vec<ServiceSpec> {
    let rust_playout = root
        .services
        .as_ref()
        .and_then(|services| services.rust.as_ref())
        .and_then(|services| services.playout.as_ref());
    let mut specs = Vec::new();
    let mut deferred_cap_specs = Vec::new();

    if let Some(go) = root
        .services
        .as_ref()
        .and_then(|services| services.go.as_ref())
        .filter(|go| go.enabled.unwrap_or(false))
    {
        if let Some(web) = &go.web_gateway {
            if web.enabled.unwrap_or(false) {
                specs.extend(web_gateway_specs(web, root.webpanel.as_ref(), host));
            }
        }

        if let Some(data_ingest) = &go.data_ingest {
            if data_ingest.enabled.unwrap_or(false) {
                let mut args = vec![
                    "--config".to_string(),
                    host.config_path.to_string_lossy().into_owned(),
                ];
                if let Some(interval) = data_ingest
                    .interval
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--interval".to_string(), interval.to_string()]);
                }
                if let Some(timeout) = data_ingest
                    .timeout
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--timeout".to_string(), timeout.to_string()]);
                }
                specs.push(ServiceSpec {
                    id: "svc:data_ingest",
                    kind: "managed",
                    binary: executable_name("haze-data-ingest"),
                    configured_executable: data_ingest.executable.clone(),
                    args,
                });
            }
        }

        if let Some(cap) = &go.cap_ingest {
            if cap.enabled.unwrap_or(false) {
                let mut args = vec![
                    "--source-id".to_string(),
                    cap.source_id
                        .clone()
                        .unwrap_or_else(|| "go-cap".to_string()),
                    "--source".to_string(),
                    cap.source.clone().unwrap_or_else(|| "naads".to_string()),
                    "--interval".to_string(),
                    cap.interval.clone().unwrap_or_else(|| "30s".to_string()),
                    "--timeout".to_string(),
                    cap.timeout.clone().unwrap_or_else(|| "15s".to_string()),
                ];
                if let Some(url) = &cap.url {
                    if !url.trim().is_empty() {
                        args.extend(["--url".to_string(), url.to_string()]);
                    }
                }
                deferred_cap_specs.push(ServiceSpec {
                    id: "go:cap_ingest",
                    kind: "managed",
                    binary: executable_name("haze-cap-ingest"),
                    configured_executable: cap.executable.clone(),
                    args,
                });
            }
        }

        if let Some(tts) = &go.tts {
            if tts.enabled.unwrap_or(false) {
                let mut args = vec![
                    "--service".to_string(),
                    "--readers".to_string(),
                    tts.readers
                        .clone()
                        .unwrap_or_else(|| "managed/configs/readers.xml".to_string()),
                    "--dictionary".to_string(),
                    tts.dictionary
                        .clone()
                        .unwrap_or_else(|| "managed/dictionary.json".to_string()),
                    "--provider".to_string(),
                    tts.provider.clone().unwrap_or_else(|| "auto".to_string()),
                    "--lang".to_string(),
                    tts.language.clone().unwrap_or_else(|| "en-CA".to_string()),
                    "--timezone".to_string(),
                    tts.timezone.clone().unwrap_or_else(|| "Local".to_string()),
                    "--out-dir".to_string(),
                    tts.out_dir
                        .clone()
                        .unwrap_or_else(|| "runtime/audio/tts".to_string()),
                    "--timeout".to_string(),
                    tts.timeout.clone().unwrap_or_else(|| "60s".to_string()),
                ];
                if let Some(executable) = tts
                    .piper_executable
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--piper-exe".to_string(), executable.to_string()]);
                }
                if let Some(voices_dir) = tts
                    .piper_voices_dir
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--piper-voices-dir".to_string(), voices_dir.to_string()]);
                }
                if let Some(mode) = tts
                    .piper_mode
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--piper-mode".to_string(), mode.to_string()]);
                }
                if let Some(workers) = tts.piper_workers {
                    args.extend(["--piper-workers".to_string(), workers.to_string()]);
                }
                if let Some(prewarm) = tts.piper_prewarm {
                    args.push(format!("--piper-prewarm={prewarm}"));
                }
                if let Some(cuda) = tts.piper_cuda {
                    args.push(format!("--piper-cuda={cuda}"));
                }
                if let Some(model_dir) = tts
                    .kokoro_model_dir
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--kokoro-model-dir".to_string(), model_dir.to_string()]);
                }
                if let Some(provider) = tts
                    .kokoro_runtime_provider
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend([
                        "--kokoro-runtime-provider".to_string(),
                        provider.to_string(),
                    ]);
                }
                if let Some(threads) = tts.kokoro_threads {
                    args.extend(["--kokoro-threads".to_string(), threads.to_string()]);
                }
                if let Some(speed) = tts.kokoro_speed {
                    args.extend(["--kokoro-speed".to_string(), speed.to_string()]);
                }
                if let Some(length_scale) = tts.kokoro_length_scale {
                    args.extend([
                        "--kokoro-length-scale".to_string(),
                        length_scale.to_string(),
                    ]);
                }
                specs.push(ServiceSpec {
                    id: "aux:tts",
                    kind: "managed",
                    binary: executable_name("haze-tts"),
                    configured_executable: tts.executable.clone(),
                    args,
                });
            }
        }

        if let Some(product_render) = &go.product_render {
            if product_render.enabled.unwrap_or(false) {
                let mut args = vec![
                    "--config".to_string(),
                    host.config_path.to_string_lossy().into_owned(),
                    "--bridge".to_string(),
                    std::env::var("HAZE_HOST_BRIDGE_ADDR").unwrap_or_default(),
                ];
                if let Some(refresh) = product_render
                    .refresh
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--refresh".to_string(), refresh.to_string()]);
                }
                specs.push(ServiceSpec {
                    id: "svc:product_render",
                    kind: "managed",
                    binary: executable_name("haze-product-render"),
                    configured_executable: product_render.executable.clone(),
                    args,
                });
            }
        }

        if let Some(playlist) = &go.playlist {
            if playlist.enabled.unwrap_or(false) {
                let mut args = vec![
                    "--config".to_string(),
                    host.config_path.to_string_lossy().into_owned(),
                    "--bridge".to_string(),
                    std::env::var("HAZE_HOST_BRIDGE_ADDR").unwrap_or_default(),
                    "--out-dir".to_string(),
                    playlist
                        .out_dir
                        .clone()
                        .unwrap_or_else(|| "runtime/audio/playlist".to_string()),
                ];
                if let Some(tick) = playlist
                    .tick
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--tick".to_string(), tick.to_string()]);
                }
                if let Some(lookahead) = playlist
                    .lookahead
                    .as_ref()
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--lookahead".to_string(), lookahead.to_string()]);
                }
                specs.push(ServiceSpec {
                    id: "svc:playlist",
                    kind: "managed",
                    binary: executable_name("haze-playlist"),
                    configured_executable: playlist.executable.clone(),
                    args,
                });
            }
        }

        if let Some(ivr) = &go.ivr {
            if ivr.enabled.unwrap_or(false) {
                let mut args = vec![
                    "--config".to_string(),
                    host.config_path.to_string_lossy().into_owned(),
                    "--bridge".to_string(),
                    std::env::var("HAZE_HOST_BRIDGE_ADDR").unwrap_or_default(),
                ];
                if let Some(addr) = ivr
                    .http
                    .as_ref()
                    .and_then(|http| http.addr.as_ref())
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--http-addr".to_string(), addr.to_string()]);
                }
                if let Some(addr) = ivr
                    .sip
                    .as_ref()
                    .and_then(|sip| sip.listen.as_ref())
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--sip-addr".to_string(), addr.to_string()]);
                }
                if let Some(dir) = ivr
                    .cache
                    .as_ref()
                    .and_then(|cache| cache.dir.as_ref())
                    .filter(|value| !value.trim().is_empty())
                {
                    args.extend(["--cache-dir".to_string(), dir.to_string()]);
                }
                specs.push(ServiceSpec {
                    id: "svc:ivr",
                    kind: "managed",
                    binary: executable_name("haze-ivr"),
                    configured_executable: ivr.executable.clone(),
                    args,
                });
            }
        }
    }

    if let Some(playout) = rust_playout.filter(|playout| playout.enabled.unwrap_or(false)) {
        let mut args = vec![
            "--config".to_string(),
            host.config_path.to_string_lossy().into_owned(),
            "--bridge".to_string(),
            std::env::var("HAZE_HOST_BRIDGE_ADDR").unwrap_or_default(),
        ];
        if let Some(alert_poll) = playout
            .alert_poll
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            args.extend(["--alert-poll".to_string(), alert_poll.to_string()]);
        }
        specs.push(ServiceSpec {
            id: "svc:playout",
            kind: "managed",
            binary: executable_name("haze-playout-rs"),
            configured_executable: playout.executable.clone(),
            args,
        });
    }
    specs.extend(deferred_cap_specs);
    specs
}

fn web_gateway_specs(
    web: &WebGatewayConfig,
    webpanel: Option<&WebPanelConfig>,
    host: &ServiceHostConfig,
) -> Vec<ServiceSpec> {
    let mut specs = Vec::new();
    let Some(panel) = webpanel else {
        specs.push(web_gateway_spec(
            "go:web_gateway",
            "combined",
            web.addr
                .clone()
                .unwrap_or_else(|| "0.0.0.0:8086".to_string()),
            web.executable.clone(),
            host,
        ));
        return specs;
    };

    let public_enabled = panel
        .public
        .as_ref()
        .and_then(|surface| surface.enabled)
        .unwrap_or(true);
    let admin_enabled = panel
        .admin
        .as_ref()
        .and_then(|surface| surface.enabled)
        .unwrap_or(true);
    let public_addr = webpanel_addr(
        panel.public.as_ref(),
        panel.host.as_deref().unwrap_or("0.0.0.0"),
        Some(8086),
    );
    let admin_addr = webpanel_addr(
        panel.admin.as_ref(),
        panel.host.as_deref().unwrap_or("0.0.0.0"),
        panel.port.or(Some(8086)),
    );
    if public_enabled && admin_enabled && public_addr == admin_addr {
        specs.push(web_gateway_spec(
            "go:web_gateway",
            "combined",
            public_addr,
            web.executable.clone(),
            host,
        ));
    } else {
        if public_enabled {
            specs.push(web_gateway_spec(
                "go:web_public",
                "public",
                public_addr,
                web.executable.clone(),
                host,
            ));
        }
        if admin_enabled {
            specs.push(web_gateway_spec(
                "go:web_admin",
                "admin",
                admin_addr,
                web.executable.clone(),
                host,
            ));
        }
    }

    if specs.is_empty() {
        specs.push(web_gateway_spec(
            "go:web_gateway",
            "combined",
            web.addr
                .clone()
                .unwrap_or_else(|| "0.0.0.0:8086".to_string()),
            web.executable.clone(),
            host,
        ));
    }
    specs
}

fn web_gateway_spec(
    id: &'static str,
    surface: &str,
    addr: String,
    configured_executable: Option<PathBuf>,
    host: &ServiceHostConfig,
) -> ServiceSpec {
    ServiceSpec {
        id,
        kind: "managed",
        binary: executable_name("haze-web"),
        configured_executable,
        args: vec![
            "--addr".to_string(),
            addr,
            "--surface".to_string(),
            surface.to_string(),
            "--webroot".to_string(),
            host.app_dir.join("webroot").to_string_lossy().into_owned(),
            "--config".to_string(),
            host.config_path.to_string_lossy().into_owned(),
        ],
    }
}

fn webpanel_addr(
    surface: Option<&WebPanelSurfaceConfig>,
    fallback_host: &str,
    fallback_port: Option<u16>,
) -> String {
    let host = surface
        .and_then(|surface| surface.host.as_deref())
        .filter(|host| !host.trim().is_empty())
        .unwrap_or(fallback_host)
        .trim();
    let port = surface
        .and_then(|surface| surface.port)
        .or(fallback_port)
        .unwrap_or(8086);
    format!("{host}:{port}")
}

fn executable_name(base: &'static str) -> &'static str {
    #[cfg(windows)]
    {
        match base {
            "haze-web" => "haze-web.exe",
            "haze-data-ingest" => "haze-data-ingest.exe",
            "haze-cap-ingest" => "haze-cap-ingest.exe",
            "haze-tts" => "haze-tts.exe",
            "haze-product-render" => "haze-product-render.exe",
            "haze-playlist" => "haze-playlist.exe",
            "haze-ivr" => "haze-ivr.exe",
            "haze-playout-rs" => "haze-playout-rs.exe",
            _ => base,
        }
    }
    #[cfg(not(windows))]
    {
        base
    }
}

fn resolve_executable(
    host: &ServiceHostConfig,
    configured: Option<&Path>,
    binary: &str,
) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = configured {
        candidates.push(resolve_path(&host.runtime_dir, path));
        candidates.push(resolve_path(&host.app_dir, path));
    }
    candidates.extend([
        host.app_dir.join("bin").join(binary),
        host.runtime_dir.join("bin").join(binary),
        host.app_dir.join(binary),
        host.runtime_dir.join(binary),
        host.app_dir.join("services").join(binary),
        host.runtime_dir.join("services").join(binary),
        host.app_dir.join("go-services").join(binary),
        host.runtime_dir.join("go-services").join(binary),
    ]);

    candidates
        .into_iter()
        .map(|candidate| dunce::simplified(&candidate).to_path_buf())
        .find(|candidate| candidate.is_file())
}

fn embedded_binary(binary: &str) -> Option<&'static [u8]> {
    embedded::EMBEDDED_GO_BINARIES
        .iter()
        .find_map(|(name, bytes)| (*name == binary).then_some(*bytes))
}

fn extract_embedded_executable(host: &ServiceHostConfig, binary: &str) -> Result<Option<PathBuf>> {
    let Some(bytes) = embedded_binary(binary) else {
        return Ok(None);
    };
    if bytes.is_empty() {
        return Ok(None);
    }

    let bin_dir = host.runtime_dir.join(EMBEDDED_BIN_DIR);
    fs::create_dir_all(&bin_dir).with_context(|| {
        format!(
            "failed to create service binary directory {}",
            bin_dir.display()
        )
    })?;
    extract_embedded_support_libraries(&bin_dir)?;
    let target = bin_dir.join(binary);
    let should_write = fs::read(&target)
        .map(|existing| existing.as_slice() != bytes)
        .unwrap_or(true);
    if should_write {
        fs::write(&target, bytes)
            .with_context(|| format!("failed to extract service binary {}", target.display()))?;
        make_executable(&target)?;
    }
    Ok(Some(target))
}

fn extract_embedded_support_libraries(bin_dir: &Path) -> Result<()> {
    for name in embedded_support_library_names() {
        let Some(bytes) = embedded_binary(name) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }
        let target = bin_dir.join(name);
        let should_write = fs::read(&target)
            .map(|existing| existing.as_slice() != bytes)
            .unwrap_or(true);
        if should_write {
            fs::write(&target, bytes).with_context(|| {
                format!("failed to extract support library {}", target.display())
            })?;
        }
    }
    Ok(())
}

fn embedded_support_library_names() -> &'static [&'static str] {
    #[cfg(windows)]
    {
        &["libopus-0.dll", "libopusfile-0.dll", "libogg-0.dll"]
    }
    #[cfg(not(windows))]
    {
        &[]
    }
}

fn stop_previous_managed_services(status_path: &Path) {
    let Ok(raw) = fs::read_to_string(status_path) else {
        return;
    };
    let Ok(payload) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return;
    };
    let Some(services) = payload
        .get("services")
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };
    for (id, service) in services {
        let status = service
            .get("status")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if !matches!(status, "running" | "unknown") {
            continue;
        }
        let pid = service.get("pid").and_then(serde_json::Value::as_u64);
        let executable = service
            .get("executable")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let Some(pid) = pid.and_then(|pid| u32::try_from(pid).ok()) else {
            continue;
        };
        if pid == std::process::id() || !is_known_managed_executable(executable) {
            continue;
        }
        match terminate_pid_tree(pid) {
            Ok(()) => info!(
                "stopped stale {} from previous run (pid {pid})",
                service_label(id)
            ),
            Err(err) => warn!(
                "failed to stop stale {} from previous run (pid {pid}): {err}",
                service_label(id)
            ),
        }
    }
}

fn is_known_managed_executable(path: &str) -> bool {
    let Some(file_name) = Path::new(path).file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let file_name = file_name.to_ascii_lowercase();
    [
        executable_name("haze-web"),
        executable_name("haze-data-ingest"),
        executable_name("haze-cap-ingest"),
        executable_name("haze-tts"),
        executable_name("haze-product-render"),
        executable_name("haze-playlist"),
        executable_name("haze-ivr"),
        executable_name("haze-playout-rs"),
    ]
    .into_iter()
    .any(|known| known.eq_ignore_ascii_case(&file_name))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<()> {
    Ok(())
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[cfg(windows)]
fn configure_managed_process(command: &mut Command) {
    use std::os::windows::process::CommandExt;

    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    command.creation_flags(CREATE_NEW_PROCESS_GROUP);
}

#[cfg(not(windows))]
fn configure_managed_process(_command: &mut Command) {}

#[cfg(windows)]
fn terminate_child_tree(child: &mut Child) -> std::io::Result<()> {
    terminate_pid_tree(child.id()).or_else(|_| child.kill())
}

#[cfg(not(windows))]
fn terminate_child_tree(child: &mut Child) -> std::io::Result<()> {
    child.kill()
}

#[cfg(windows)]
fn terminate_pid_tree(pid: u32) -> std::io::Result<()> {
    let status = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() || status.code() == Some(128) {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "taskkill exited with {status}"
        )))
    }
}

#[cfg(not(windows))]
fn terminate_pid_tree(pid: u32) -> std::io::Result<()> {
    let status = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("kill exited with {status}"),
        ))
    }
}

fn load_config_with_overlay(path: &Path, runtime_dir: &Path) -> Result<RootConfig> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read daemon config {}", path.display()))?;
    let mut value: serde_yaml::Value = serde_yaml::from_str(&raw)
        .with_context(|| format!("failed to parse daemon config {}", path.display()))?;

    let settings_file = value
        .get("daemon_settings_file")
        .and_then(serde_yaml::Value::as_str)
        .filter(|raw| !raw.trim().is_empty())
        .unwrap_or(DEFAULT_SETTINGS_FILE);
    let settings_path = resolve_path(runtime_dir, Path::new(settings_file));
    if settings_path.is_file() {
        let overlay_raw = fs::read_to_string(&settings_path).with_context(|| {
            format!(
                "failed to read daemon settings overlay {}",
                settings_path.display()
            )
        })?;
        let overlay_json: serde_json::Value =
            serde_json::from_str(&overlay_raw).with_context(|| {
                format!(
                    "failed to parse daemon settings overlay {}",
                    settings_path.display()
                )
            })?;
        let overlay_yaml = serde_yaml::to_value(overlay_json)
            .context("failed to convert daemon settings overlay")?;
        merge_yaml(&mut value, overlay_yaml);
    }

    serde_yaml::from_value(value).context("failed to decode daemon service configuration")
}

fn merge_yaml(base: &mut serde_yaml::Value, overlay: serde_yaml::Value) {
    match (base, overlay) {
        (serde_yaml::Value::Mapping(base_map), serde_yaml::Value::Mapping(overlay_map)) => {
            for (key, value) in overlay_map {
                match base_map.get_mut(&key) {
                    Some(existing) => merge_yaml(existing, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (base_slot, overlay_value) => {
            *base_slot = overlay_value;
        }
    }
}

fn pipe_service_output<R>(
    service_id: &'static str,
    stream: &'static str,
    reader: R,
    warn_lines: bool,
) where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(std::result::Result::ok) {
            log_service_line(service_id, stream, warn_lines, &line);
        }
    });
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ServiceLogLevel {
    Info,
    Warn,
    Error,
}

fn log_service_line(service_id: &str, stream: &str, warn_lines: bool, raw: &str) {
    let Some((mut level, line)) = normalize_service_log_line(raw, warn_lines) else {
        return;
    };
    if stream == "stderr" && level == ServiceLogLevel::Info {
        level = ServiceLogLevel::Warn;
    }
    let label = service_label(service_id);
    match level {
        ServiceLogLevel::Info => info!("[{label}] {line}"),
        ServiceLogLevel::Warn => warn!("[{label}] {line}"),
        ServiceLogLevel::Error => error!("[{label}] {line}"),
    }
}

fn normalize_service_log_line(raw: &str, warn_lines: bool) -> Option<(ServiceLogLevel, String)> {
    let cleaned = strip_ansi(raw);
    let mut line = cleaned.trim();
    if line.is_empty() {
        return None;
    }

    let mut level = if warn_lines {
        ServiceLogLevel::Warn
    } else {
        ServiceLogLevel::Info
    };

    line = strip_go_log_timestamp(line);
    line = strip_iso_log_timestamp(line);
    if let Some((parsed_level, rest)) = strip_log_level(line) {
        level = parsed_level;
        line = rest;
    }
    line = strip_rust_target_prefix(line);
    line = strip_redundant_service_prefix(line);
    if let Some((parsed_level, rest)) = strip_log_level(line) {
        level = parsed_level;
        line = rest;
    }

    let line = line.trim();
    (!line.is_empty()).then(|| (level, line.to_string()))
}

fn service_label(service_id: &str) -> &str {
    match service_id {
        "go:web_gateway" => "Web panel",
        "go:web_public" => "Public web",
        "go:web_admin" => "Admin web",
        "svc:data_ingest" => "Data ingest",
        "go:cap_ingest" => "CAP ingest",
        "aux:tts" => "TTS",
        "svc:product_render" => "Product render",
        "svc:playlist" => "Playlist",
        "svc:ivr" => "IVR",
        "svc:playout" => "Playout",
        _ => service_id,
    }
}

fn strip_ansi(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for code in chars.by_ref() {
                if ('@'..='~').contains(&code) {
                    break;
                }
            }
            continue;
        }
        output.push(ch);
    }
    output
}

fn strip_go_log_timestamp(line: &str) -> &str {
    let bytes = line.as_bytes();
    if bytes.len() >= 20
        && digits(bytes, 0..4)
        && bytes[4] == b'/'
        && digits(bytes, 5..7)
        && bytes[7] == b'/'
        && digits(bytes, 8..10)
        && bytes[10] == b' '
        && digits(bytes, 11..13)
        && bytes[13] == b':'
        && digits(bytes, 14..16)
        && bytes[16] == b':'
        && digits(bytes, 17..19)
        && bytes[19].is_ascii_whitespace()
    {
        line[20..].trim_start()
    } else {
        line
    }
}

fn strip_iso_log_timestamp(line: &str) -> &str {
    let bytes = line.as_bytes();
    if bytes.len() < 20
        || !digits(bytes, 0..4)
        || bytes[4] != b'-'
        || !digits(bytes, 5..7)
        || bytes[7] != b'-'
        || !digits(bytes, 8..10)
        || bytes[10] != b'T'
        || !digits(bytes, 11..13)
        || bytes[13] != b':'
        || !digits(bytes, 14..16)
        || bytes[16] != b':'
        || !digits(bytes, 17..19)
    {
        return line;
    }
    let Some(end) = line.find(char::is_whitespace) else {
        return line;
    };
    line[end..].trim_start()
}

fn digits(bytes: &[u8], range: std::ops::Range<usize>) -> bool {
    range
        .filter_map(|index| bytes.get(index))
        .all(u8::is_ascii_digit)
}

fn strip_log_level(line: &str) -> Option<(ServiceLogLevel, &str)> {
    const LEVELS: [(&str, ServiceLogLevel); 6] = [
        ("ERROR", ServiceLogLevel::Error),
        ("CRITICAL", ServiceLogLevel::Error),
        ("WARNING", ServiceLogLevel::Warn),
        ("WARN", ServiceLogLevel::Warn),
        ("INFO", ServiceLogLevel::Info),
        ("DEBUG", ServiceLogLevel::Info),
    ];

    let trimmed = line.trim_start();
    for (prefix, level) in LEVELS {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest
                .strip_prefix(':')
                .or_else(|| rest.strip_prefix(' '))
                .or_else(|| rest.strip_prefix('\t'))?;
            return Some((level, rest.trim_start()));
        }
    }
    None
}

fn strip_rust_target_prefix(line: &str) -> &str {
    let Some((target, rest)) = line.split_once(": ") else {
        return line;
    };
    if target.len() > 96 {
        return line;
    }
    let rust_like_target = target.contains("::") || target.starts_with("haze_") || target == "haze";
    if rust_like_target
        && target
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | ':' | '-'))
    {
        rest.trim_start()
    } else {
        line
    }
}

fn strip_redundant_service_prefix(line: &str) -> &str {
    const PREFIXES: [&str; 14] = [
        "haze-web: ",
        "haze-web ",
        "haze-data-ingest: ",
        "haze-data-ingest ",
        "haze-cap-ingest: ",
        "haze-cap-ingest ",
        "haze-tts: ",
        "haze-tts ",
        "haze-product-render: ",
        "haze-product-render ",
        "haze-playlist: ",
        "haze-playlist ",
        "haze-ivr: ",
        "haze-ivr ",
    ];
    PREFIXES
        .iter()
        .find_map(|prefix| line.strip_prefix(prefix))
        .unwrap_or(line)
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_host() -> ServiceHostConfig {
        ServiceHostConfig {
            app_dir: PathBuf::from("app"),
            runtime_dir: PathBuf::from("runtime"),
            config_path: PathBuf::from("config.yaml"),
        }
    }

    #[test]
    fn web_gateway_specs_combines_public_and_admin_on_same_port() {
        let root: RootConfig = serde_yaml::from_str(
            r#"
services:
  go:
    enabled: true
    web_gateway:
      enabled: true
      addr: 0.0.0.0:8086
webpanel:
  host: 0.0.0.0
  port: 8086
  public:
    enabled: true
    host: 0.0.0.0
    port: 8086
  admin:
    enabled: true
    host: 0.0.0.0
    port: 8086
"#,
        )
        .expect("config");

        let specs = service_specs(&root, &test_host());

        let web_specs: Vec<&ServiceSpec> = specs
            .iter()
            .filter(|spec| spec.binary == executable_name("haze-web"))
            .collect();
        assert_eq!(web_specs.len(), 1);
        assert_eq!(web_specs[0].id, "go:web_gateway");
        assert!(web_specs[0].args.contains(&"0.0.0.0:8086".to_string()));
        assert!(web_specs[0].args.contains(&"combined".to_string()));
    }

    #[test]
    fn web_gateway_specs_fall_back_to_combined_legacy_addr_without_webpanel() {
        let root: RootConfig = serde_yaml::from_str(
            r#"
services:
  go:
    enabled: true
    web_gateway:
      enabled: true
      addr: 127.0.0.1:9999
"#,
        )
        .expect("config");

        let specs = service_specs(&root, &test_host());

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].id, "go:web_gateway");
        assert!(specs[0].args.contains(&"127.0.0.1:9999".to_string()));
        assert!(specs[0].args.contains(&"combined".to_string()));
    }

    #[test]
    fn rust_playout_is_the_only_playout_service() {
        let root: RootConfig = serde_yaml::from_str(
            r#"
services:
  rust:
    playout:
      enabled: true
      alert_poll: 250ms
  go:
    enabled: true
"#,
        )
        .expect("config");

        let specs = service_specs(&root, &test_host());

        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].id, "svc:playout");
        assert_eq!(specs[0].binary, executable_name("haze-playout-rs"));
        assert!(specs[0].args.contains(&"250ms".to_string()));
    }

    #[test]
    fn service_log_normalizer_strips_go_timestamp_and_binary_name() {
        let (level, line) = normalize_service_log_line(
            "2026/06/16 21:13:04 haze-web combined listening on 0.0.0.0:8086",
            false,
        )
        .expect("normalized line");

        assert_eq!(level, ServiceLogLevel::Info);
        assert_eq!(line, "combined listening on 0.0.0.0:8086");
    }

    #[test]
    fn service_log_normalizer_strips_ansi_rust_prefix() {
        let (level, line) = normalize_service_log_line(
            "\u{1b}[2m2026-06-16T01:35:27.612989Z\u{1b}[0m \u{1b}[33mWARN\u{1b}[0m haze_playout_rs::sinks: UDP sink warning",
            false,
        )
        .expect("normalized line");

        assert_eq!(level, ServiceLogLevel::Warn);
        assert_eq!(line, "UDP sink warning");
    }

    #[test]
    fn service_labels_are_operator_readable() {
        assert_eq!(service_label("go:web_gateway"), "Web panel");
        assert_eq!(service_label("svc:product_render"), "Product render");
        assert_eq!(service_label("unknown-service"), "unknown-service");
    }
}
