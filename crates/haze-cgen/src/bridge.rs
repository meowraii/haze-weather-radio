use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Mutex};
use tokio::time::sleep;

#[derive(Clone)]
pub(crate) struct BridgeClient {
    writer: Arc<Mutex<OwnedWriteHalf>>,
}

pub(crate) struct BridgeConnection {
    pub(crate) client: BridgeClient,
    pub(crate) events: mpsc::Receiver<Value>,
}

pub(crate) async fn connect(addr: &str) -> Result<BridgeConnection> {
    let stream = TcpStream::connect(addr)
        .await
        .with_context(|| format!("failed to connect to host bridge at {addr}"))?;
    let (reader, writer) = stream.into_split();
    let (tx, rx) = mpsc::channel(512);
    tokio::spawn(read_loop(reader, tx));
    Ok(BridgeConnection {
        client: BridgeClient {
            writer: Arc::new(Mutex::new(writer)),
        },
        events: rx,
    })
}

pub(crate) async fn connect_retry(addr: &str) -> Result<BridgeConnection> {
    loop {
        match connect(addr).await {
            Ok(connection) => return Ok(connection),
            Err(err) => {
                tracing::warn!("waiting for host event bridge at {addr}: {err}");
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

impl BridgeClient {
    pub(crate) async fn publish(&self, mut value: Value) -> Result<()> {
        if value.get("timestamp").is_none() {
            value["timestamp"] = serde_json::json!(chrono::Utc::now().to_rfc3339());
        }
        let mut raw = serde_json::to_vec(&value)?;
        raw.push(b'\n');
        let mut writer = self.writer.lock().await;
        writer
            .write_all(&raw)
            .await
            .context("failed to write host bridge event")
    }
}

async fn read_loop(reader: OwnedReadHalf, tx: mpsc::Sender<Value>) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => {
                if tx.send(value).await.is_err() {
                    return;
                }
            }
            Err(err) => tracing::warn!("ignored malformed bridge event: {err}"),
        }
    }
}
