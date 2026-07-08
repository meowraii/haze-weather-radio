use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};
use windows_service::define_windows_service;
use windows_service::service::{
    ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
use windows_service::service_dispatcher;

use crate::{signals, DaemonArgs};

struct ServiceRuntime {
    name: String,
    args: DaemonArgs,
}

static SERVICE_RUNTIME: OnceLock<Mutex<Option<ServiceRuntime>>> = OnceLock::new();

define_windows_service!(ffi_service_main, service_main);

pub(crate) fn run(args: DaemonArgs) -> Result<()> {
    let service_name = args.service_name.clone();
    SERVICE_RUNTIME
        .set(Mutex::new(Some(ServiceRuntime {
            name: service_name.clone(),
            args,
        })))
        .map_err(|_| anyhow::anyhow!("Windows service runtime was already initialized"))?;
    service_dispatcher::start(service_name, ffi_service_main)
        .context("failed to connect to the Windows Service Control Manager")
}

fn service_main(_arguments: Vec<std::ffi::OsString>) {
    if let Err(err) = run_service() {
        eprintln!("haze Windows service failed: {err:?}");
    }
}

fn run_service() -> Result<()> {
    let runtime = SERVICE_RUNTIME
        .get()
        .context("Windows service runtime was not initialized")?
        .lock()
        .map_err(|_| anyhow::anyhow!("Windows service runtime lock was poisoned"))?
        .take()
        .context("Windows service runtime was already consumed")?;
    let service_name = runtime.name.clone();

    let status_handle =
        service_control_handler::register(
            &service_name,
            move |control_event| match control_event {
                ServiceControl::Stop | ServiceControl::Shutdown => {
                    signals::request_shutdown();
                    ServiceControlHandlerResult::NoError
                }
                ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
                _ => ServiceControlHandlerResult::NotImplemented,
            },
        )
        .with_context(|| {
            format!("failed to register Windows service handler for {service_name}")
        })?;

    set_service_status(
        &status_handle,
        ServiceState::Running,
        ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN,
        ServiceExitCode::Win32(0),
    )?;

    let result = crate::run(runtime.args);
    let exit_code = if result.is_ok() {
        ServiceExitCode::Win32(0)
    } else {
        ServiceExitCode::ServiceSpecific(1)
    };
    let _ = set_service_status(
        &status_handle,
        ServiceState::Stopped,
        ServiceControlAccept::empty(),
        exit_code,
    );
    result
}

fn set_service_status(
    status_handle: &service_control_handler::ServiceStatusHandle,
    current_state: ServiceState,
    controls_accepted: ServiceControlAccept,
    exit_code: ServiceExitCode,
) -> Result<()> {
    status_handle
        .set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state,
            controls_accepted,
            exit_code,
            checkpoint: 0,
            wait_hint: Duration::from_secs(10),
            process_id: None,
        })
        .context("failed to update Windows service status")
}
