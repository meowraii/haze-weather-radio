use serde_json::{json, Value};

use crate::scene::{
    RgbaColor, SceneCatalog, SceneId, PROGRAM_PASSTHROUGH_ID, STANDARD_CRAWL_ID, STANDBY_ID,
};
use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AlertClass {
    Warning,
    Watch,
    Advisory,
    Unknown,
}

impl AlertClass {
    fn from_text(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "warning" | "extreme" | "severe" => Self::Warning,
            "watch" | "moderate" => Self::Watch,
            "advisory" | "minor" => Self::Advisory,
            _ => Self::Unknown,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Warning => "warning",
            Self::Watch => "watch",
            Self::Advisory => "advisory",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AlertBindingSnapshot {
    pub(crate) queue_id: String,
    pub(crate) organization: String,
    pub(crate) action: String,
    pub(crate) event: String,
    pub(crate) full_text: String,
    pub(crate) alert_class: AlertClass,
    pub(crate) warning_gradient: [RgbaColor; 3],
    pub(crate) watch_gradient: [RgbaColor; 3],
    pub(crate) advisory_gradient: [RgbaColor; 3],
}

impl AlertBindingSnapshot {
    fn from_state(audio: &PriorityAudio, banner: Option<&BannerPayload>) -> Self {
        let banner_alert = banner.and_then(|payload| payload.alerts.first());
        let organization = first_non_empty([
            audio.presentation.organization.as_str(),
            alert_text(banner_alert, &["organization", "issuer", "sender_name"]),
        ]);
        let action = first_non_empty([
            audio.presentation.action.as_str(),
            alert_text(banner_alert, &["action", "action_phrase"]),
        ]);
        let event = first_non_empty([
            audio.presentation.event.as_str(),
            alert_text(banner_alert, &["event_name", "event", "headline", "title"]),
        ]);
        let full_text = first_non_empty([
            audio.presentation.full_text.as_str(),
            audio.banner_text.as_deref().unwrap_or_default(),
            alert_text(
                banner_alert,
                &["banner_text", "message", "scroll_text", "description"],
            ),
        ]);
        let class_text = first_non_empty([
            audio.presentation.alert_class.as_str(),
            audio.priority.as_deref().unwrap_or_default(),
        ]);
        let alert_class = AlertClass::from_text(&class_text);
        let mut warning_gradient = default_warning_gradient();
        let mut watch_gradient = default_watch_gradient();
        let mut advisory_gradient = default_advisory_gradient();
        if let Some(primary) = banner.and_then(parse_banner_gradient) {
            match alert_class {
                AlertClass::Warning | AlertClass::Unknown => warning_gradient = primary,
                AlertClass::Watch => watch_gradient = primary,
                AlertClass::Advisory => advisory_gradient = primary,
            }
        }
        Self {
            queue_id: audio.queue_id.clone(),
            organization,
            action,
            event,
            full_text,
            alert_class,
            warning_gradient,
            watch_gradient,
            advisory_gradient,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CompositorMode {
    ProgramPassthrough,
    Standby,
    Alert { scene_id: SceneId, queue_id: String },
}

impl CompositorMode {
    pub(crate) fn scene_id(&self) -> &str {
        match self {
            Self::ProgramPassthrough => PROGRAM_PASSTHROUGH_ID,
            Self::Standby => STANDBY_ID,
            Self::Alert { scene_id, .. } => scene_id.as_str(),
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::ProgramPassthrough => "program_passthrough",
            Self::Standby => "standby",
            Self::Alert { .. } => "alert",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResolvedSceneState {
    pub(crate) mode: CompositorMode,
    pub(crate) binding: Option<AlertBindingSnapshot>,
    pub(crate) catalog_degraded: bool,
    pub(crate) selected_scene_available: bool,
}

impl ResolvedSceneState {
    pub(crate) fn status_value(&self) -> Value {
        json!({
            "mode": self.mode.name(),
            "active_scene_id": self.mode.scene_id(),
            "active_queue_id": self.binding.as_ref().map(|binding| binding.queue_id.as_str()),
            "alert_class": self.binding.as_ref().map(|binding| binding.alert_class.as_str()),
            "scene_catalog_degraded": self.catalog_degraded,
            "selected_scene_available": self.selected_scene_available,
        })
    }
}

pub(crate) fn resolve_scene_state(
    runtime: &RuntimeState,
    alert_feed_id: &str,
    alert_scene_id: &SceneId,
    input_healthy: bool,
    catalog: &SceneCatalog,
) -> ResolvedSceneState {
    let banner = runtime.banner_for(alert_feed_id);
    if let Some(audio) = runtime.priority_audio_for(alert_feed_id) {
        let selected_scene_available = catalog.alert_scene(alert_scene_id).is_ok();
        let scene_id = if selected_scene_available {
            alert_scene_id.clone()
        } else {
            SceneId::new(STANDARD_CRAWL_ID).expect("protected scene ID is valid")
        };
        return ResolvedSceneState {
            mode: CompositorMode::Alert {
                scene_id,
                queue_id: audio.queue_id.clone(),
            },
            binding: Some(AlertBindingSnapshot::from_state(audio, banner)),
            catalog_degraded: catalog.is_degraded(),
            selected_scene_available,
        };
    }

    ResolvedSceneState {
        mode: if input_healthy {
            CompositorMode::ProgramPassthrough
        } else {
            CompositorMode::Standby
        },
        binding: None,
        catalog_degraded: catalog.is_degraded(),
        selected_scene_available: true,
    }
}

fn alert_text<'a>(alert: Option<&'a SerializedAlert>, keys: &[&str]) -> &'a str {
    let Some(alert) = alert else {
        return "";
    };
    keys.iter()
        .find_map(|key| {
            alert
                .fields
                .get(*key)
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|text| !text.is_empty())
        })
        .unwrap_or_default()
}

fn first_non_empty<const N: usize>(values: [&str; N]) -> String {
    values
        .into_iter()
        .map(str::trim)
        .find(|value| !value.is_empty())
        .unwrap_or_default()
        .to_string()
}

fn parse_banner_gradient(payload: &BannerPayload) -> Option<[RgbaColor; 3]> {
    if payload.primary_gradient.len() < 3 {
        return None;
    }
    Some([
        payload.primary_gradient[0].parse().ok()?,
        payload.primary_gradient[1].parse().ok()?,
        payload.primary_gradient[2].parse().ok()?,
    ])
}

fn default_warning_gradient() -> [RgbaColor; 3] {
    [
        RgbaColor::new(94, 0, 0, 255),
        RgbaColor::new(220, 20, 20, 255),
        RgbaColor::new(94, 0, 0, 255),
    ]
}

fn default_watch_gradient() -> [RgbaColor; 3] {
    [
        RgbaColor::new(128, 64, 0, 255),
        RgbaColor::new(255, 166, 0, 255),
        RgbaColor::new(128, 64, 0, 255),
    ]
}

fn default_advisory_gradient() -> [RgbaColor; 3] {
    [
        RgbaColor::new(0, 54, 112, 255),
        RgbaColor::new(0, 133, 202, 255),
        RgbaColor::new(0, 54, 112, 255),
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::Utc;

    use super::*;
    use crate::scene::{load_scene_directory, SceneId};
    use crate::state::{AlertPresentation, PriorityAudio};

    fn active_alert() -> PriorityAudio {
        PriorityAudio {
            queue_id: "queue-1".to_string(),
            audio_path: None,
            duration_ms: None,
            sample_rate: 48_000,
            channels: 1,
            banner_text: Some("Full alert text".to_string()),
            background_color: None,
            priority: Some("warning".to_string()),
            presentation: AlertPresentation {
                organization: "Civil Authorities".to_string(),
                action: "Issued a".to_string(),
                event: "Severe Thunderstorm Warning".to_string(),
                full_text: "Full alert text".to_string(),
                alert_class: "warning".to_string(),
            },
            started_at: Utc::now(),
        }
    }

    #[test]
    fn healthy_input_without_alert_uses_zero_draw_passthrough() {
        let catalog = load_scene_directory(std::path::Path::new("missing-scenes"));
        let state = resolve_scene_state(
            &RuntimeState::default(),
            "CAP",
            &SceneId::new(STANDARD_CRAWL_ID).expect("scene ID"),
            true,
            &catalog,
        );
        assert_eq!(state.mode, CompositorMode::ProgramPassthrough);
        assert!(state.binding.is_none());
    }

    #[test]
    fn unhealthy_input_without_alert_uses_standby() {
        let catalog = load_scene_directory(std::path::Path::new("missing-scenes"));
        let state = resolve_scene_state(
            &RuntimeState::default(),
            "CAP",
            &SceneId::new(STANDARD_CRAWL_ID).expect("scene ID"),
            false,
            &catalog,
        );
        assert_eq!(state.mode, CompositorMode::Standby);
    }

    #[test]
    fn active_alert_overrides_standby_and_keeps_binding_identity() {
        let catalog = load_scene_directory(std::path::Path::new("missing-scenes"));
        let mut runtime = RuntimeState::default();
        runtime.active_audio = BTreeMap::from([("CAP".to_string(), active_alert())]);
        let state = resolve_scene_state(
            &runtime,
            "CAP",
            &SceneId::new("Fullscreen_Takeover").expect("scene ID"),
            false,
            &catalog,
        );
        assert!(matches!(state.mode, CompositorMode::Alert { .. }));
        let binding = state.binding.expect("binding");
        assert_eq!(binding.queue_id, "queue-1");
        assert_eq!(binding.organization, "Civil Authorities");
        assert_eq!(binding.action, "Issued a");
        assert_eq!(binding.event, "Severe Thunderstorm Warning");
    }
}
