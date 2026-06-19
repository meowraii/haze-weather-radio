use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TrySendError};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::Value;
use tracing::{debug, error, info, warn};

const CAP_REPLAY_WINDOW: Duration = Duration::from_secs(15 * 60);
const CAP_REPLAY_LIMIT: usize = 512;

/// Local event bridge used by daemon services and playout.
pub(crate) struct HostBridge {
    addr: String,
    sender: Sender<Value>,
    events: Option<Receiver<Value>>,
}

impl HostBridge {
    /// Starts a loopback JSONL event bridge.
    ///
    /// # Errors
    ///
    /// Returns an error if the daemon cannot bind a local TCP listener.
    pub(crate) fn start() -> Result<Self> {
        let listener =
            TcpListener::bind(("127.0.0.1", 0)).context("failed to bind host event bridge")?;
        let addr = listener
            .local_addr()
            .context("failed to inspect host event bridge address")?
            .to_string();
        let (sender, receiver) = mpsc::channel::<Value>();
        let (client_sender, client_receiver) = mpsc::channel::<SyncSender<Vec<u8>>>();
        let (event_sender, event_receiver) = mpsc::channel::<Value>();

        thread::spawn(move || {
            let mut clients: Vec<SyncSender<Vec<u8>>> = Vec::new();
            let mut cap_replay = VecDeque::<(Instant, Vec<u8>)>::new();
            loop {
                while let Ok(client) = client_receiver.try_recv() {
                    prune_replay(&mut cap_replay);
                    for (_, raw) in &cap_replay {
                        let _ = client.try_send(raw.clone());
                    }
                    clients.push(client);
                }
                match receiver.recv() {
                    Ok(message) => {
                        let _ = event_sender.send(message.clone());
                        let Ok(mut raw) = serde_json::to_vec(&message) else {
                            continue;
                        };
                        raw.push(b'\n');
                        if replayable_event(&message) {
                            cap_replay.push_back((Instant::now(), raw.clone()));
                            prune_replay(&mut cap_replay);
                            while cap_replay.len() > CAP_REPLAY_LIMIT {
                                cap_replay.pop_front();
                            }
                        }
                        clients.retain(|client| match client.try_send(raw.clone()) {
                            Ok(()) => true,
                            Err(TrySendError::Full(_)) => {
                                warn!("dropped slow host bridge client");
                                false
                            }
                            Err(TrySendError::Disconnected(_)) => false,
                        });
                    }
                    Err(_) => break,
                }
            }
        });

        thread::spawn({
            let addr_for_log = addr.clone();
            let accept_publisher = sender.clone();
            move || {
                info!("event bridge listening on {addr_for_log}");
                for accepted in listener.incoming() {
                    match accepted {
                        Ok(stream) => {
                            let peer = stream
                                .peer_addr()
                                .map(|addr| addr.to_string())
                                .unwrap_or_default();
                            debug!(peer, "host bridge client connected");
                            match stream.try_clone() {
                                Ok(writer) => {
                                    let (tx, rx) = mpsc::sync_channel::<Vec<u8>>(256);
                                    if client_sender.send(tx).is_err() {
                                        break;
                                    }
                                    spawn_client_writer(writer, rx);
                                }
                                Err(err) => warn!("failed to clone host bridge stream: {err}"),
                            }
                            spawn_client_reader(stream, accept_publisher.clone());
                        }
                        Err(err) => {
                            warn!("host event bridge accept failed: {err}");
                            break;
                        }
                    }
                }
            }
        });

        Ok(Self {
            addr,
            sender,
            events: Some(event_receiver),
        })
    }

    pub(crate) fn addr(&self) -> &str {
        &self.addr
    }

    pub(crate) fn publisher(&self) -> Sender<Value> {
        self.sender.clone()
    }

    pub(crate) fn take_events(&mut self) -> Receiver<Value> {
        self.events
            .take()
            .expect("host bridge events can only be taken once")
    }
}

fn spawn_client_reader<R>(reader: R, publisher: Sender<Value>)
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for line in reader.lines().map_while(std::result::Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(&line) {
                Ok(value) => {
                    if let Some(event) = handle_client_message(value) {
                        let _ = publisher.send(event);
                    }
                }
                Err(err) => warn!("host bridge received invalid JSON: {err}"),
            }
        }
    });
}

fn spawn_client_writer(mut writer: TcpStream, receiver: Receiver<Vec<u8>>) {
    thread::spawn(move || {
        for raw in receiver {
            if writer.write_all(&raw).is_err() {
                let _ = writer.shutdown(Shutdown::Both);
                return;
            }
        }
        let _ = writer.shutdown(Shutdown::Both);
    });
}

fn handle_client_message(value: Value) -> Option<Value> {
    let msg_type = value.get("type").and_then(Value::as_str).unwrap_or("");
    if msg_type == "log_record" {
        let level = value
            .get("level")
            .and_then(Value::as_str)
            .unwrap_or("INFO")
            .to_ascii_uppercase();
        let logger = value
            .get("logger")
            .and_then(Value::as_str)
            .unwrap_or("haze");
        let message = value.get("message").and_then(Value::as_str).unwrap_or("");
        match level.as_str() {
            "ERROR" | "CRITICAL" => error!("[{}] {message}", logger_label(logger)),
            "WARNING" | "WARN" => warn!("[{}] {message}", logger_label(logger)),
            "DEBUG" => debug!("[{}] {message}", logger_label(logger)),
            _ => info!("[{}] {message}", logger_label(logger)),
        }
        return None;
    }
    Some(value)
}

fn logger_label(logger: &str) -> &str {
    logger
        .strip_prefix("module.")
        .or_else(|| logger.strip_prefix("haze."))
        .unwrap_or(logger)
}

fn replayable_event(value: &Value) -> bool {
    matches!(
        value.get("type").and_then(Value::as_str).unwrap_or(""),
        "cap.alert.received"
    )
}

fn prune_replay(replay: &mut VecDeque<(Instant, Vec<u8>)>) {
    let now = Instant::now();
    while replay
        .front()
        .is_some_and(|(inserted_at, _)| now.duration_since(*inserted_at) > CAP_REPLAY_WINDOW)
    {
        replay.pop_front();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn log_records_are_consumed_by_logger() {
        let result = handle_client_message(json!({
            "type": "log_record",
            "level": "INFO",
            "logger": "test",
            "message": "hello",
        }));

        assert!(result.is_none());
    }

    #[test]
    fn service_events_are_republished() {
        let event = json!({
            "type": "cap.alert.received",
            "source": "go-cap",
            "subject": "abc",
        });
        let result = handle_client_message(event.clone());

        assert_eq!(result, Some(event));
    }
}
