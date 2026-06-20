use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::time::{sleep, timeout};

type BridgeResult<T> = std::result::Result<T, String>;
type PendingReply<T> = oneshot::Sender<BridgeResult<T>>;
type PendingMap<T> = Arc<Mutex<HashMap<String, PendingReply<T>>>>;
type PendingSynth = PendingMap<String>;
type PendingProducts = PendingMap<RenderedProduct>;

const SYNTH_REPLY_TIMEOUT: Duration = Duration::from_secs(90);
const PRODUCT_REPLY_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub(crate) struct BridgeClient {
    writer: Arc<Mutex<OwnedWriteHalf>>,
    pending_synth: PendingSynth,
    pending_products: PendingProducts,
}

pub(crate) struct BridgeConnection {
    pub(crate) client: BridgeClient,
    pub(crate) events: mpsc::Receiver<Value>,
}

#[derive(Debug, Clone)]
pub(crate) struct SynthJob {
    pub(crate) id: String,
    pub(crate) text: String,
    pub(crate) reader_id: String,
    pub(crate) language: String,
    pub(crate) output_path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ProductRenderRequest {
    pub(crate) request_id: String,
    pub(crate) feed_id: String,
    pub(crate) package_id: String,
    pub(crate) force: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RenderedProduct {
    #[serde(default)]
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) feed_id: String,
    #[serde(default)]
    pub(crate) package_id: String,
    #[serde(default)]
    pub(crate) title: String,
    #[serde(default)]
    pub(crate) text: String,
    #[serde(default)]
    pub(crate) reader_id: String,
    #[serde(default)]
    pub(crate) language: String,
    #[serde(default)]
    pub(crate) metadata: HashMap<String, String>,
}

pub(crate) async fn connect(addr: &str) -> Result<BridgeConnection> {
    let stream = TcpStream::connect(addr)
        .await
        .with_context(|| format!("failed to connect to host bridge at {addr}"))?;
    let (reader, writer) = stream.into_split();
    let (tx, rx) = mpsc::channel(256);
    let pending_synth = Arc::new(Mutex::new(HashMap::new()));
    let pending_products = Arc::new(Mutex::new(HashMap::new()));
    tokio::spawn(read_loop(
        reader,
        tx,
        Arc::clone(&pending_synth),
        Arc::clone(&pending_products),
    ));
    Ok(BridgeConnection {
        client: BridgeClient {
            writer: Arc::new(Mutex::new(writer)),
            pending_synth,
            pending_products,
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
            value["timestamp"] = json!(chrono::Utc::now().to_rfc3339());
        }
        let mut raw = serde_json::to_vec(&value)?;
        raw.push(b'\n');
        let mut writer = self.writer.lock().await;
        writer
            .write_all(&raw)
            .await
            .context("failed to write host event bridge event")
    }

    pub(crate) async fn service_ready(&self, feeds: usize) {
        let _ = self
            .publish(json!({
                "type": "service.ready",
                "source": "haze-playout",
                "data": {
                    "service": "haze-playout",
                    "feeds": feeds,
                }
            }))
            .await;
    }

    pub(crate) async fn synthesize(&self, job: SynthJob) -> Result<String> {
        let job_id = fallback_text(
            &job.id,
            &format!(
                "tts-{}",
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ),
        );
        let (tx, rx) = oneshot::channel();
        self.pending_synth.lock().await.insert(job_id.clone(), tx);
        if let Some(parent) = job.output_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let publish_result = self
            .publish(json!({
                "type": "tts.synthesize",
                "source": "haze-playout",
                "subject": job_id,
                "data": {
                    "job_id": job_id,
                    "text": job.text,
                    "reader_id": job.reader_id,
                    "language": job.language,
                    "output_path": job.output_path,
                }
            }))
            .await;
        if let Err(err) = publish_result {
            self.pending_synth.lock().await.remove(&job_id);
            return Err(err);
        }
        let result = match timeout(SYNTH_REPLY_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                self.pending_synth.lock().await.remove(&job_id);
                anyhow::bail!("event bridge closed while waiting for TTS");
            }
            Err(_) => {
                self.pending_synth.lock().await.remove(&job_id);
                anyhow::bail!("TTS request timed out");
            }
        };
        match result {
            Ok(path) => Ok(path),
            Err(err) => anyhow::bail!("{err}"),
        }
    }

    pub(crate) async fn render_product(
        &self,
        request: ProductRenderRequest,
    ) -> Result<RenderedProduct> {
        let request_id = fallback_text(
            &request.request_id,
            &format!(
                "product-{}-{}-{}",
                request.feed_id,
                request.package_id,
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ),
        );
        let (tx, rx) = oneshot::channel();
        self.pending_products
            .lock()
            .await
            .insert(request_id.clone(), tx);
        let publish_result = self
            .publish(json!({
                "type": "product.render.request",
                "source": "haze-playout",
                "subject": request_id,
                "data": {
                    "request_id": request_id,
                    "feed_id": request.feed_id,
                    "package_id": request.package_id,
                    "pkg_id": request.package_id,
                    "force": request.force,
                }
            }))
            .await;
        if let Err(err) = publish_result {
            self.pending_products.lock().await.remove(&request_id);
            return Err(err);
        }
        let result = match timeout(PRODUCT_REPLY_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => {
                self.pending_products.lock().await.remove(&request_id);
                anyhow::bail!("event bridge closed while waiting for product render");
            }
            Err(_) => {
                self.pending_products.lock().await.remove(&request_id);
                anyhow::bail!("product render request timed out");
            }
        };
        match result {
            Ok(product) => Ok(product),
            Err(err) => anyhow::bail!("{err}"),
        }
    }
}

async fn read_loop(
    reader: OwnedReadHalf,
    tx: mpsc::Sender<Value>,
    pending_synth: PendingSynth,
    pending_products: PendingProducts,
) {
    let mut lines = BufReader::new(reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if handle_synth_result(&value, &pending_synth).await
            || handle_product_result(&value, &pending_products).await
        {
            continue;
        }
        if tx.send(value).await.is_err() {
            break;
        }
    }
    fail_pending(&pending_synth, "event bridge closed while waiting for TTS").await;
    fail_pending(
        &pending_products,
        "event bridge closed while waiting for product render",
    )
    .await;
}

async fn handle_synth_result(value: &Value, pending: &PendingSynth) -> bool {
    let msg_type = string_at(value, "type");
    if msg_type != "tts.synthesized" && msg_type != "tts.failed" {
        return false;
    }
    let data = data(value);
    let job_id = first_text(value, data, &["job_id", "subject"]);
    if job_id.is_empty() {
        return true;
    }
    let Some(ch) = pending.lock().await.remove(job_id) else {
        return true;
    };
    let result = if msg_type == "tts.failed" {
        Err(format!(
            "TTS failed for {job_id}: {}",
            string_at(data, "error")
        ))
    } else {
        Ok(string_at(data, "output_path").to_string())
    };
    let _ = ch.send(result);
    true
}

async fn handle_product_result(value: &Value, pending: &PendingProducts) -> bool {
    let msg_type = string_at(value, "type");
    if msg_type != "product.rendered" && msg_type != "product.render.failed" {
        return false;
    }
    let data = data(value);
    let request_id = first_text(value, data, &["request_id", "subject"]);
    if request_id.is_empty() {
        return true;
    }
    let Some(ch) = pending.lock().await.remove(request_id) else {
        return true;
    };
    let result = if msg_type == "product.render.failed" {
        Err(format!(
            "product render failed for {request_id}: {}",
            string_at(data, "error")
        ))
    } else {
        serde_json::from_value::<RenderedProduct>(
            data.get("product").cloned().unwrap_or(Value::Null),
        )
        .map_err(|err| err.to_string())
    };
    let _ = ch.send(result);
    true
}

async fn fail_pending<T>(pending: &PendingMap<T>, message: &str) {
    let mut pending = pending.lock().await;
    for (_, ch) in pending.drain() {
        let _ = ch.send(Err(message.to_string()));
    }
}

pub(crate) fn string_at<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("").trim()
}

pub(crate) fn data(value: &Value) -> &Value {
    value.get("data").unwrap_or(&Value::Null)
}

pub(crate) fn first_text<'a>(message: &'a Value, data: &'a Value, keys: &[&str]) -> &'a str {
    for key in keys {
        let from_data = string_at(data, key);
        if !from_data.is_empty() {
            return from_data;
        }
        let from_message = string_at(message, key);
        if !from_message.is_empty() {
            return from_message;
        }
    }
    ""
}

fn fallback_text(value: &str, fallback: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        fallback.to_string()
    } else {
        value.to_string()
    }
}
