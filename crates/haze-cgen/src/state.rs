use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct RuntimeState {
    pub(crate) banners: BTreeMap<String, BannerPayload>,
    pub(crate) active_audio: BTreeMap<String, PriorityAudio>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct BannerPayload {
    #[serde(default)]
    pub(crate) active: bool,
    #[serde(default)]
    pub(crate) signature: String,
    #[serde(default)]
    pub(crate) feed_id: String,
    #[serde(default)]
    pub(crate) feed_name: String,
    #[serde(default)]
    pub(crate) primary_color: String,
    #[serde(default)]
    pub(crate) primary_gradient: Vec<String>,
    #[serde(default)]
    pub(crate) alerts: Vec<SerializedAlert>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct SerializedAlert {
    #[serde(flatten)]
    pub(crate) fields: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PriorityAudio {
    pub(crate) queue_id: String,
    pub(crate) audio_path: Option<PathBuf>,
    pub(crate) duration_ms: Option<u64>,
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) alert_packet: Option<Value>,
    pub(crate) banner_text: Option<String>,
    pub(crate) background_color: Option<String>,
    pub(crate) priority: Option<String>,
    pub(crate) started_at: DateTime<Utc>,
}

impl RuntimeState {
    pub(crate) fn apply_event(&mut self, event: &Value) -> bool {
        let event_type = text_at(event, &["type"]);
        match event_type.as_str() {
            "banner.state.updated" => self.apply_banner_state(event),
            "alert.playout.started" | "cap.alert.audio.ready" => self.apply_priority_audio(event),
            "alert.playout.completed" | "playout.interrupted" => self.clear_priority_audio(event),
            _ => false,
        }
    }

    pub(crate) fn banner_for(&self, feed_id: &str) -> Option<&BannerPayload> {
        self.banners
            .get(feed_id)
            .or_else(|| self.banners.get("*"))
            .filter(|payload| payload.active)
    }

    pub(crate) fn priority_audio_for(&self, feed_id: &str) -> Option<&PriorityAudio> {
        if feed_id.trim() == "*" {
            return self
                .active_audio
                .values()
                .max_by_key(|audio| audio.started_at);
        }
        self.active_audio
            .get(feed_id)
            .or_else(|| self.active_audio.get("*"))
    }

    fn apply_banner_state(&mut self, event: &Value) -> bool {
        let payload_value = event.get("data").cloned().unwrap_or(Value::Null);
        let Ok(payload) = serde_json::from_value::<BannerPayload>(payload_value) else {
            return false;
        };
        let feed_id = fallback_text(&payload.feed_id, &text_at(event, &["subject"]), "*");
        let changed = self.banners.get(&feed_id) != Some(&payload);
        if changed {
            self.banners.insert(feed_id, payload);
        }
        changed
    }

    fn apply_priority_audio(&mut self, event: &Value) -> bool {
        let data = event.get("data").unwrap_or(event);
        let queue_id = fallback_text(
            &text_at(data, &["queue_id"]),
            &text_at(event, &["queue_id"]),
            "priority-alert",
        );
        let audio = PriorityAudio {
            queue_id,
            audio_path: non_empty(text_at(data, &["audio_path"])).map(PathBuf::from),
            duration_ms: number_at(data, &["duration_ms"]),
            sample_rate: number_at(data, &["sample_rate"])
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(48_000),
            channels: number_at(data, &["channels"])
                .and_then(|value| u16::try_from(value).ok())
                .filter(|value| *value > 0)
                .unwrap_or(1),
            alert_packet: data.get("alert_packet").cloned(),
            banner_text: non_empty(text_at(data, &["banner_text"])),
            background_color: non_empty(text_at(data, &["background_color"])),
            priority: non_empty(text_at(data, &["priority"])),
            started_at: Utc::now(),
        };
        let mut changed = false;
        for feed_id in feed_ids(event) {
            changed |= self.active_audio.get(&feed_id) != Some(&audio);
            self.active_audio.insert(feed_id, audio.clone());
        }
        changed
    }

    fn clear_priority_audio(&mut self, event: &Value) -> bool {
        let queue_id = fallback_text(
            &text_at(event.get("data").unwrap_or(event), &["queue_id"]),
            &text_at(event, &["queue_id"]),
            "",
        );
        let mut changed = false;
        for feed_id in feed_ids(event) {
            if queue_id.is_empty() {
                changed |= self.active_audio.remove(&feed_id).is_some();
                continue;
            }
            let matches = self
                .active_audio
                .get(&feed_id)
                .map(|audio| audio.queue_id == queue_id)
                .unwrap_or(false);
            if matches {
                self.active_audio.remove(&feed_id);
                changed = true;
            }
            if self.banners.remove(&feed_id).is_some() {
                changed = true;
            }
        }
        changed
    }
}

fn feed_ids(event: &Value) -> Vec<String> {
    let mut out = BTreeSet::<String>::new();
    add_feed_id(&mut out, text_at(event, &["feed_id"]));
    add_feed_id(
        &mut out,
        text_at(event.get("data").unwrap_or(event), &["feed_id"]),
    );
    for path in [&["feed_ids"][..], &["data", "feed_ids"][..]] {
        if let Some(values) = value_at(event, path).and_then(Value::as_array) {
            for value in values {
                add_feed_id(&mut out, value.as_str().unwrap_or_default().to_string());
            }
        }
    }
    if out.is_empty() {
        out.insert("*".to_string());
    }
    out.into_iter().collect()
}

fn add_feed_id(out: &mut BTreeSet<String>, value: String) {
    let value = value.trim();
    if !value.is_empty() {
        out.insert(value.to_string());
    }
}

fn text_at(value: &Value, path: &[&str]) -> String {
    value_at(value, path)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn number_at(value: &Value, path: &[&str]) -> Option<u64> {
    value_at(value, path).and_then(Value::as_u64)
}

fn value_at<'a>(mut value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    for part in path {
        value = value.get(*part)?;
    }
    Some(value)
}

fn fallback_text(values: &str, fallback: &str, default: &str) -> String {
    for value in [values, fallback, default] {
        let value = value.trim();
        if !value.is_empty() {
            return value.to_string();
        }
    }
    String::new()
}

fn non_empty(value: String) -> Option<String> {
    (!value.trim().is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn consumes_canonical_banner_state() {
        let mut state = RuntimeState::default();
        let changed = state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "sk-0001",
            "data": {
                "active": true,
                "signature": "abc",
                "feed_id": "sk-0001",
                "primary_color": "#b91c1c",
                "primary_gradient": ["#b91c1c", "#7f1d1d"],
                "alerts": [{"message": "Alert text"}]
            }
        }));
        assert!(changed);
        let banner = state.banner_for("sk-0001").expect("banner");
        assert_eq!(banner.signature, "abc");
        assert_eq!(banner.alerts.len(), 1);
    }

    #[test]
    fn priority_audio_is_per_feed_and_cleared_by_completion() {
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["sk-0001"],
            "queue_id": "alert-1",
            "data": {
                "queue_id": "alert-1",
                "audio_path": "runtime/audio/alerts/alert.raw",
                "duration_ms": 42000,
                "banner_text": "banner"
            }
        })));
        assert!(state.priority_audio_for("sk-0001").is_some());
        assert!(state.apply_event(&json!({
            "type": "alert.playout.completed",
            "feed_ids": ["sk-0001"],
            "queue_id": "alert-1",
            "data": {"queue_id": "alert-1"}
        })));
        assert!(state.priority_audio_for("sk-0001").is_none());
    }

    #[test]
    fn wildcard_priority_audio_applies_to_any_feed() {
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "cap.alert.audio.ready",
            "data": {
                "feed_ids": ["*"],
                "queue_id": "alert-all",
                "audio_path": "runtime/audio/alerts/all.raw"
            }
        })));
        assert!(state.priority_audio_for("CAP-IT-ALL").is_some());
    }

    #[test]
    fn wildcard_priority_query_uses_any_active_feed() {
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "alert.playout.started",
            "feed_ids": ["sk-0001"],
            "queue_id": "alert-1",
            "data": {
                "queue_id": "alert-1",
                "audio_path": "runtime/audio/alerts/sk.raw"
            }
        })));

        let audio = state.priority_audio_for("*").expect("wildcard audio");
        assert_eq!(audio.queue_id, "alert-1");
    }

    #[test]
    fn completion_clears_visual_banner_for_feed() {
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP-IT-ALL",
            "data": {
                "active": true,
                "feed_id": "CAP-IT-ALL",
                "alerts": [{"message": "Alert text"}]
            }
        })));
        assert!(state.banner_for("CAP-IT-ALL").is_some());
        assert!(state.apply_event(&json!({
            "type": "alert.playout.completed",
            "feed_ids": ["CAP-IT-ALL"],
            "queue_id": "alert-1",
            "data": {"queue_id": "alert-1"}
        })));
        assert!(state.banner_for("CAP-IT-ALL").is_none());
    }
}
