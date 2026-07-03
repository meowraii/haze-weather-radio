use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use futures::{stream, StreamExt};
use reqwest::header::{HeaderMap, ACCEPT, ETAG, IF_MODIFIED_SINCE, IF_NONE_MATCH, LAST_MODIFIED};
use reqwest::{Client, StatusCode};
use serde_json::json;
use tokio::time::sleep;
use tracing::{debug, info, warn};

use crate::bridge::{EventEnvelope, EventPublisher};
use crate::model::{Alert, AtomEntry};
use crate::parse::{parse_atom_entries, parse_cap};

const MAX_BODY_BYTES: usize = 5 << 20;

#[derive(Clone, Debug)]
pub struct SourceConfig {
    pub id: String,
    pub source: SourceKind,
    pub urls: Vec<String>,
    pub interval: Duration,
    pub timeout: Duration,
    pub user_agent: String,
    pub shadow: bool,
    pub startup_seed: bool,
    pub concurrency: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceKind {
    Naads,
    Nws,
    Custom(String),
}

impl SourceKind {
    pub fn from_raw(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "naads" | "cap-cp" | "cap_cp" => Self::Naads,
            "nws" | "nws-cap" | "nws_cap" => Self::Nws,
            other => Self::Custom(other.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Naads => "naads",
            Self::Nws => "nws",
            Self::Custom(value) => value.as_str(),
        }
    }
}

#[derive(Clone, Debug, Default)]
struct EndpointCache {
    etag: Option<String>,
    last_modified: Option<String>,
}

#[derive(Clone, Debug)]
struct AtomFetch {
    url: String,
    status: FetchStatus,
    entries: Vec<AtomEntry>,
    headers: HeaderMap,
    fetch_ms: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum FetchStatus {
    Modified,
    NotModified,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct AlertKey {
    identifier: String,
    updated: String,
    sent: String,
    message_type: String,
    references: String,
}

pub struct Poller {
    source: SourceConfig,
    client: Client,
    publisher: EventPublisher,
    endpoint_cache: HashMap<String, EndpointCache>,
    seen_entries: HashSet<(String, String)>,
    seen_alerts: HashSet<AlertKey>,
    seeded_startup: bool,
}

impl Poller {
    pub fn new(source: SourceConfig, publisher: EventPublisher) -> Result<Self> {
        let client = Client::builder()
            .timeout(source.timeout)
            .user_agent(source.user_agent.clone())
            .build()
            .context("failed to create CAP HTTP client")?;
        Ok(Self {
            source,
            client,
            publisher,
            endpoint_cache: HashMap::new(),
            seen_entries: HashSet::new(),
            seen_alerts: HashSet::new(),
            seeded_startup: false,
        })
    }

    pub async fn run(&mut self, once: bool) -> Result<()> {
        loop {
            if let Err(err) = self.poll_once().await {
                warn!(
                    source = self.source.id,
                    source_kind = self.source.source.as_str(),
                    "CAP poll failed: {err:#}"
                );
                self.publish_status(json!({
                    "source_id": self.source.id,
                    "source": self.source.source.as_str(),
                    "shadow": self.source.shadow,
                    "status": "error",
                    "error": err.to_string(),
                    "timestamp_unix_ms": unix_ms(),
                }))
                .await
                .ok();
            }
            if once {
                return Ok(());
            }
            sleep(self.source.interval).await;
        }
    }

    async fn poll_once(&mut self) -> Result<()> {
        let poll_started = Instant::now();
        let fetch = self.fetch_atom().await?;
        if fetch.status == FetchStatus::NotModified {
            self.publish_status(json!({
                "source_id": self.source.id,
                "source": self.source.source.as_str(),
                "shadow": self.source.shadow,
                "status": "not_modified",
                "url": fetch.url,
                "fetch_ms": fetch.fetch_ms,
                "poll_ms": poll_started.elapsed().as_millis(),
                "timestamp_unix_ms": unix_ms(),
            }))
            .await?;
            return Ok(());
        }

        let mut entries = fetch.entries;
        entries.sort_by(|left, right| {
            right
                .updated
                .cmp(&left.updated)
                .then(right.id.cmp(&left.id))
        });
        entries.dedup_by(|left, right| left.id == right.id && left.updated == right.updated);

        if self.source.startup_seed && !self.seeded_startup {
            let count = entries
                .iter()
                .filter(|entry| {
                    self.seen_entries
                        .insert((entry.id.clone(), entry.updated.clone()))
                })
                .count();
            self.seeded_startup = true;
            info!(source = self.source.id, count, "seeded CAP startup entries");
            self.publish_status(json!({
                "source_id": self.source.id,
                "source": self.source.source.as_str(),
                "shadow": self.source.shadow,
                "status": "seeded",
                "url": fetch.url,
                "entries": entries.len(),
                "seeded": count,
                "fetch_ms": fetch.fetch_ms,
                "poll_ms": poll_started.elapsed().as_millis(),
                "timestamp_unix_ms": unix_ms(),
            }))
            .await?;
            return Ok(());
        }
        self.seeded_startup = true;

        let fresh_entries = entries
            .into_iter()
            .filter(|entry| {
                self.seen_entries
                    .insert((entry.id.clone(), entry.updated.clone()))
            })
            .collect::<Vec<_>>();
        let fresh_entry_count = fresh_entries.len();

        let cap_started = Instant::now();
        let cap_results = stream::iter(fresh_entries.iter().cloned())
            .map(|entry| self.fetch_first_cap(entry))
            .buffered(self.source.concurrency.max(1))
            .collect::<Vec<_>>()
            .await;

        let mut published = 0usize;
        let mut deduped = 0usize;
        for fetched in cap_results {
            let fetched = match fetched {
                Ok(fetched) => fetched,
                Err(err) => {
                    warn!(source = self.source.id, "CAP fetch skipped: {err:#}");
                    continue;
                }
            };
            let key = AlertKey {
                identifier: fetched.alert.identifier.clone(),
                updated: fetched.entry.updated.clone(),
                sent: fetched.alert.sent.clone(),
                message_type: fetched.alert.message_type.clone(),
                references: fetched.alert.references.clone(),
            };
            if !self.seen_alerts.insert(key) {
                deduped += 1;
                debug!(
                    source = self.source.id,
                    identifier = fetched.alert.identifier,
                    updated = fetched.entry.updated,
                    "deduped CAP alert"
                );
                continue;
            }

            let alert_value = serde_json::to_value(&fetched.alert)?;
            let ingest = json!({
                "source": self.source.source.as_str(),
                "source_id": self.source.id,
                "shadow": self.source.shadow,
                "atom_id": fetched.entry.id,
                "atom_updated": fetched.entry.updated,
                "atom_url": fetch.url,
                "cap_url": fetched.url,
                "fetch_latency_ms": fetched.fetch_ms,
                "parse_latency_ms": fetched.parse_ms,
                "publish_started_unix_ms": unix_ms(),
            });
            if self.source.shadow {
                self.publish_status(json!({
                    "source_id": self.source.id,
                    "source": self.source.source.as_str(),
                    "shadow": true,
                    "status": "would_publish",
                    "identifier": fetched.alert.identifier,
                    "message_type": fetched.alert.message_type,
                    "atom_updated": fetched.entry.updated,
                    "cap_url": fetched.url,
                    "fetch_ms": fetched.fetch_ms,
                    "parse_ms": fetched.parse_ms,
                    "timestamp_unix_ms": unix_ms(),
                }))
                .await?;
            } else {
                self.publisher
                    .publish(&EventEnvelope::cap_alert(
                        &self.source.id,
                        alert_value,
                        ingest,
                    ))
                    .await?;
                published += 1;
            }
        }

        self.publish_status(json!({
            "source_id": self.source.id,
            "source": self.source.source.as_str(),
            "shadow": self.source.shadow,
            "status": "ok",
            "url": fetch.url,
            "entries": self.seen_entries.len(),
            "fresh_entries": fresh_entry_count,
            "published": published,
            "deduped": deduped,
            "fetch_ms": fetch.fetch_ms,
            "cap_fetch_parse_ms": cap_started.elapsed().as_millis(),
            "poll_ms": poll_started.elapsed().as_millis(),
            "timestamp_unix_ms": unix_ms(),
        }))
        .await?;
        Ok(())
    }

    async fn fetch_atom(&mut self) -> Result<AtomFetch> {
        if self.source.urls.is_empty() {
            return Err(anyhow!("no Atom URLs configured"));
        }
        let client = self.client.clone();
        let fetches = self
            .source
            .urls
            .iter()
            .cloned()
            .map(|url| {
                let cache = self.endpoint_cache.get(&url).cloned().unwrap_or_default();
                fetch_atom_url(client.clone(), url, cache)
            })
            .collect::<Vec<_>>();

        let mut last_error = None;
        let mut not_modified = None;
        let mut stream = stream::iter(fetches).buffer_unordered(self.source.urls.len());
        while let Some(result) = stream.next().await {
            match result {
                Ok(fetch) if fetch.status == FetchStatus::Modified => {
                    self.store_endpoint_headers(&fetch.url, &fetch.headers);
                    return Ok(fetch);
                }
                Ok(fetch) => {
                    self.store_endpoint_headers(&fetch.url, &fetch.headers);
                    not_modified = Some(fetch);
                }
                Err(err) => last_error = Some(err),
            }
        }
        if let Some(fetch) = not_modified {
            return Ok(fetch);
        }
        Err(last_error.unwrap_or_else(|| anyhow!("all Atom endpoints failed")))
    }

    fn store_endpoint_headers(&mut self, url: &str, headers: &HeaderMap) {
        let cache = self.endpoint_cache.entry(url.to_string()).or_default();
        if let Some(value) = headers.get(ETAG).and_then(|value| value.to_str().ok()) {
            cache.etag = Some(value.to_string());
        }
        if let Some(value) = headers
            .get(LAST_MODIFIED)
            .and_then(|value| value.to_str().ok())
        {
            cache.last_modified = Some(value.to_string());
        }
    }

    async fn fetch_first_cap(&self, entry: AtomEntry) -> Result<FetchedAlert> {
        let mut last_error = None;
        for link in entry.links.clone() {
            let started = Instant::now();
            let result = self
                .client
                .get(&link)
                .header(
                    ACCEPT,
                    "application/cap+xml, application/xml;q=0.9, */*;q=0.1",
                )
                .send()
                .await
                .with_context(|| format!("failed to fetch CAP from {link}"));
            let Ok(response) = result else {
                last_error = result.err();
                continue;
            };
            if !response.status().is_success() {
                last_error = Some(anyhow!(
                    "unexpected HTTP status {} from {link}",
                    response.status()
                ));
                continue;
            }
            let body = match read_limited(response).await {
                Ok(body) => body,
                Err(err) => {
                    last_error = Some(err);
                    continue;
                }
            };
            let fetch_ms = started.elapsed().as_millis();
            let parse_started = Instant::now();
            match parse_cap(&body) {
                Ok(alert) if !alert.identifier.is_empty() => {
                    return Ok(FetchedAlert {
                        entry,
                        alert,
                        url: link,
                        fetch_ms,
                        parse_ms: parse_started.elapsed().as_millis(),
                    });
                }
                Ok(_) => last_error = Some(anyhow!("CAP alert from {link} had empty identifier")),
                Err(err) => last_error = Some(err.into()),
            }
        }
        Err(last_error.unwrap_or_else(|| anyhow!("no parseable CAP alert found")))
    }

    async fn publish_status(&self, data: serde_json::Value) -> Result<()> {
        self.publisher
            .publish(&EventEnvelope::status(
                &self.source.id,
                &format!("{}:{}", self.source.source.as_str(), self.source.id),
                data,
            ))
            .await
    }
}

#[derive(Debug)]
struct FetchedAlert {
    entry: AtomEntry,
    alert: Alert,
    url: String,
    fetch_ms: u128,
    parse_ms: u128,
}

async fn fetch_atom_url(client: Client, url: String, cache: EndpointCache) -> Result<AtomFetch> {
    let started = Instant::now();
    let mut request = client.get(&url).header(
        ACCEPT,
        "application/atom+xml, application/xml;q=0.9, */*;q=0.1",
    );
    if let Some(etag) = cache.etag {
        request = request.header(IF_NONE_MATCH, etag);
    }
    if let Some(last_modified) = cache.last_modified {
        request = request.header(IF_MODIFIED_SINCE, last_modified);
    }
    let response = request
        .send()
        .await
        .with_context(|| format!("failed to fetch Atom from {url}"))?;
    let status = response.status();
    let headers = response.headers().clone();
    if status == StatusCode::NOT_MODIFIED {
        return Ok(AtomFetch {
            url,
            status: FetchStatus::NotModified,
            entries: Vec::new(),
            headers,
            fetch_ms: started.elapsed().as_millis(),
        });
    }
    if !status.is_success() {
        return Err(anyhow!("unexpected HTTP status {status} from {url}"));
    }
    let body = read_limited(response).await?;
    let entries = parse_atom_entries(&body)?;
    Ok(AtomFetch {
        url,
        status: FetchStatus::Modified,
        entries,
        headers,
        fetch_ms: started.elapsed().as_millis(),
    })
}

async fn read_limited(response: reqwest::Response) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > MAX_BODY_BYTES as u64)
    {
        return Err(anyhow!("response exceeds {MAX_BODY_BYTES} bytes"));
    }
    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        if body.len().saturating_add(chunk.len()) > MAX_BODY_BYTES {
            return Err(anyhow!("response exceeds {MAX_BODY_BYTES} bytes"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn unix_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or_default()
}

pub fn default_atom_urls(kind: &SourceKind) -> Vec<String> {
    match kind {
        SourceKind::Naads => Vec::new(),
        SourceKind::Nws => vec!["https://api.weather.gov/alerts/active.atom".to_string()],
        SourceKind::Custom(_) => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_kind_atom_defaults_are_source_specific() {
        assert!(default_atom_urls(&SourceKind::from_raw("naads")).is_empty());
        assert_eq!(
            default_atom_urls(&SourceKind::from_raw("nws"))[0],
            "https://api.weather.gov/alerts/active.atom"
        );
    }
}
