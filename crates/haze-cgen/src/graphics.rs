use serde_json::{json, Value};

use crate::config::FeedConfig;
use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PresentationSnapshot {
    pub(crate) visual_mode: String,
    pub(crate) overlay_active: bool,
    pub(crate) overlay_text: String,
    pub(crate) ticker_color: String,
    pub(crate) ticker_gradient: [String; 3],
    pub(crate) ticker_y: i32,
    pub(crate) ticker_height: u32,
    pub(crate) ticker_speed_px_per_frame: u32,
    pub(crate) font: String,
    pub(crate) font_weight: String,
    pub(crate) font_size: u32,
    pub(crate) visual_id: String,
    pub(crate) active_alert_queue_id: Option<String>,
}

pub(crate) fn presentation_snapshot(
    feed: &FeedConfig,
    state: &RuntimeState,
    no_signal: bool,
) -> PresentationSnapshot {
    let priority_feed_id = priority_feed_id(feed);
    let banner_feed_id = banner_feed_id(feed, priority_feed_id);
    let banner = state.banner_for(banner_feed_id);
    let audio = state.priority_audio_for(priority_feed_id);
    let has_visual_input = banner.is_some();
    let overlay_text = if has_visual_input {
        overlay_text(feed, banner, audio)
    } else if no_signal && feed.standby.mode.eq_ignore_ascii_case("banner") {
        feed.standby.text.trim().to_string()
    } else if feed.text.enabled {
        feed.text.content.trim().to_string()
    } else {
        String::new()
    };
    let ticker_color = ticker_color(banner, audio).unwrap_or_else(|| "#111827".to_string());
    let gradient = ticker_gradient(banner, audio, &ticker_color);
    PresentationSnapshot {
        visual_mode: visual_mode(feed, has_visual_input, no_signal).to_string(),
        overlay_active: !overlay_text.trim().is_empty()
            || feed.clock.enabled
            || feed.state.smpte_bars,
        overlay_text,
        ticker_color,
        ticker_gradient: [
            rgb_to_hex(gradient.top),
            rgb_to_hex(gradient.middle),
            rgb_to_hex(gradient.bottom),
        ],
        ticker_y: ticker_y(
            feed,
            i32::try_from(feed.video.height).unwrap_or(1080),
            no_signal,
        ),
        ticker_height: feed
            .banner
            .ticker_height
            .max(feed.graphics.banner_height)
            .max(48),
        ticker_speed_px_per_frame: feed.banner.scroll_speed.max(1),
        font: ticker_font_family(feed).to_string(),
        font_weight: ticker_font_weight(feed).to_string(),
        font_size: feed.banner.font_size.max(feed.graphics.font_size).max(16),
        visual_id: banner
            .map(banner_visual_id)
            .or_else(|| audio.map(|audio| audio.queue_id.clone()))
            .unwrap_or_default(),
        active_alert_queue_id: audio.map(|audio| audio.queue_id.clone()),
    }
}

pub(crate) fn compositor_status(feed: &FeedConfig, state: &RuntimeState, no_signal: bool) -> Value {
    let snapshot = presentation_snapshot(feed, state, no_signal);
    json!({
        "visual_mode": snapshot.visual_mode,
        "state_mode": feed.state.mode,
        "smpte_bars": feed.state.smpte_bars,
        "overlay_active": snapshot.overlay_active,
        "overlay_text": snapshot.overlay_text,
        "ticker_color": snapshot.ticker_color,
        "ticker_gradient": snapshot.ticker_gradient,
        "ticker_y": snapshot.ticker_y,
        "ticker_height": snapshot.ticker_height,
        "ticker_speed_px_per_frame": snapshot.ticker_speed_px_per_frame,
        "scroll_repeat_mode": feed.banner.scroll_repeat_mode,
        "after_eom_repeats": feed.banner.after_eom_repeats,
        "fixed_repeats": feed.banner.fixed_repeats,
        "font": snapshot.font,
        "font_weight": snapshot.font_weight,
        "font_size": snapshot.font_size,
        "visual_id": snapshot.visual_id,
        "clock_enabled": feed.clock.enabled,
        "clock_format": feed.clock.format,
        "clock_x": feed.clock.x,
        "clock_y": feed.clock.y,
        "clock_font_size": feed.clock.font_size,
        "clock_color": feed.clock.color,
        "text_enabled": feed.text.enabled,
        "text_x": feed.text.x,
        "text_y": feed.text.y,
        "text_font_size": feed.text.font_size,
        "text_color": feed.text.color,
        "standby_mode": feed.standby.mode,
        "standby_text": feed.standby.text,
        "standby_font_size": feed.standby.font_size,
        "standby_y_percent": feed.standby.y_percent,
        "state_updated_at": feed.state.updated_at,
        "no_signal": no_signal,
        "active_alert_queue_id": snapshot.active_alert_queue_id,
    })
}

fn priority_feed_id(feed: &FeedConfig) -> &str {
    if feed.priority_input.feed_id.trim().is_empty() {
        feed.id.as_str()
    } else {
        feed.priority_input.feed_id.as_str()
    }
}

fn banner_feed_id<'a>(feed: &'a FeedConfig, priority_feed_id: &'a str) -> &'a str {
    if feed.id.trim().is_empty() {
        priority_feed_id
    } else {
        feed.id.as_str()
    }
}

fn banner_visual_id(banner: &BannerPayload) -> String {
    let mut parts = Vec::new();
    for alert in &banner.alerts {
        for key in [
            "queue_id",
            "identifier",
            "id",
            "alert_id",
            "event",
            "message",
        ] {
            if let Some(text) = alert.fields.get(key).and_then(Value::as_str) {
                let text = text.trim();
                if !text.is_empty() {
                    parts.push(text.to_string());
                    break;
                }
            }
        }
    }
    if !parts.is_empty() {
        return parts.join("|");
    }
    banner.signature.trim().to_string()
}

fn visual_mode(feed: &FeedConfig, has_visual_input: bool, no_signal: bool) -> &str {
    if feed.state.smpte_bars {
        "smpte"
    } else if has_visual_input {
        if feed.banner.mode.eq_ignore_ascii_case("fullscreen") {
            "fullscreen_alert"
        } else {
            "ticker_alert"
        }
    } else if no_signal {
        feed.standby.mode.trim()
    } else if feed.text.enabled || feed.clock.enabled {
        "overlay"
    } else {
        "release"
    }
}

fn overlay_text(
    feed: &FeedConfig,
    banner: Option<&BannerPayload>,
    audio: Option<&PriorityAudio>,
) -> String {
    let mut parts = Vec::new();
    if let Some(text) = audio.and_then(|audio| audio.banner_text.as_deref()) {
        push_text_part(&mut parts, text);
    }
    if let Some(banner) = banner {
        for alert in &banner.alerts {
            if let Some(text) = alert_text(alert) {
                push_text_part(&mut parts, text);
            }
        }
    }
    if feed.text.enabled {
        push_text_part(&mut parts, &feed.text.content);
    }
    parts.join("     ").chars().take(1800).collect()
}

fn alert_text(alert: &SerializedAlert) -> Option<&str> {
    for key in [
        "banner_text",
        "message",
        "scroll_text",
        "headline",
        "title",
        "event_name",
        "event",
        "description",
        "identifier",
    ] {
        if let Some(text) = alert.fields.get(key).and_then(Value::as_str) {
            let text = text.trim();
            if !text.is_empty() {
                return Some(text);
            }
        }
    }
    None
}

fn push_text_part(parts: &mut Vec<String>, text: &str) {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if !text.is_empty() && !parts.iter().any(|part| part == &text) {
        parts.push(text);
    }
}

fn ticker_font_family(feed: &FeedConfig) -> &str {
    non_empty_ref(&feed.banner.font)
        .or_else(|| non_empty_ref(&feed.graphics.font))
        .unwrap_or("Arial")
}

fn ticker_font_weight(feed: &FeedConfig) -> &str {
    non_empty_ref(&feed.banner.font_weight)
        .or_else(|| non_empty_ref(&feed.graphics.font_weight))
        .unwrap_or("regular")
}

fn ticker_y(feed: &FeedConfig, frame_height: i32, no_signal: bool) -> i32 {
    if no_signal {
        let percent = feed.standby.y_percent.min(100) as f32 / 100.0;
        return ((frame_height.max(1) as f32) * percent).round() as i32;
    }
    ((frame_height.max(1) as f32) * 0.08).round() as i32
}

#[cfg(test)]
fn ticker_text_x(frame_width: i32, _text_width: i32, frame_index: u64, scroll_speed: u32) -> i32 {
    let start_x = frame_width.saturating_add(1).max(1);
    let speed = scroll_speed.max(1) as i64;
    let scroll = (frame_index as i64).saturating_mul(speed);
    i64::from(start_x).saturating_sub(scroll) as i32
}

#[cfg(test)]
fn ticker_text_x_field(
    frame_width: i32,
    text_width: i32,
    field_index: u64,
    scroll_speed: u32,
) -> i32 {
    let start_x = i128::from(frame_width.saturating_add(1).max(1));
    let scroll = i128::from(field_index).saturating_mul(i128::from(scroll_speed.max(1))) / 2;
    let min_x = -i128::from(text_width.max(1));
    let value = start_x.saturating_sub(scroll).max(min_x);
    i32::try_from(value).unwrap_or(if value < 0 { i32::MIN } else { i32::MAX })
}

#[cfg(test)]
fn ticker_scroll_frames(frame_width: i32, text_width: i32, scroll_speed: u32) -> u64 {
    let start_x = frame_width.saturating_add(1).max(1);
    let travel = start_x.saturating_add(text_width.max(1)).saturating_add(1);
    let speed = scroll_speed.max(1);
    u64::from((travel as u32).div_ceil(speed))
}

#[cfg(test)]
fn field_parities(feed: &FeedConfig) -> [i32; 2] {
    if feed.video.field_order.eq_ignore_ascii_case("bff")
        || feed.video.field_order.eq_ignore_ascii_case("bottom")
        || feed.video.field_order.eq_ignore_ascii_case("bottom_first")
    {
        [1, 0]
    } else {
        [0, 1]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TickerGradient {
    top: [f32; 3],
    middle: [f32; 3],
    bottom: [f32; 3],
}

fn ticker_gradient(
    banner: Option<&BannerPayload>,
    audio: Option<&PriorityAudio>,
    base_color: &str,
) -> TickerGradient {
    if let Some(colors) = banner_gradient_colors(banner) {
        return colors;
    }
    if let Some(color) = audio
        .and_then(|audio| audio.background_color.as_deref())
        .and_then(non_empty_ref)
    {
        let rgb = rgb_from_hex_loose(color);
        return TickerGradient {
            top: rgb,
            middle: rgb,
            bottom: rgb,
        };
    }
    let middle = rgb_from_hex_loose(base_color);
    let edge = darken_rgb(middle, 0.62);
    TickerGradient {
        top: edge,
        middle,
        bottom: edge,
    }
}

fn ticker_color<'a>(
    banner: Option<&'a BannerPayload>,
    audio: Option<&'a PriorityAudio>,
) -> Option<String> {
    banner
        .and_then(|banner| non_empty_ref(&banner.primary_color))
        .or_else(|| {
            audio
                .and_then(|audio| audio.background_color.as_deref())
                .and_then(non_empty_ref)
        })
        .map(ToString::to_string)
}

fn banner_gradient_colors(banner: Option<&BannerPayload>) -> Option<TickerGradient> {
    let stops = &banner?.primary_gradient;
    let colors = stops
        .iter()
        .filter_map(|stop| non_empty_ref(stop))
        .map(rgb_from_hex_loose)
        .collect::<Vec<_>>();
    match colors.as_slice() {
        [] => None,
        [only] => Some(TickerGradient {
            top: *only,
            middle: *only,
            bottom: *only,
        }),
        [first, .., last] => Some(TickerGradient {
            top: *last,
            middle: *first,
            bottom: *last,
        }),
    }
}

fn rgb_from_hex_loose(value: &str) -> [f32; 3] {
    let value = value.trim().trim_start_matches('#');
    if value.len() != 6 {
        return [0.5, 0.5, 0.5];
    }
    let r = u8::from_str_radix(&value[0..2], 16).unwrap_or(128) as f32 / 255.0;
    let g = u8::from_str_radix(&value[2..4], 16).unwrap_or(128) as f32 / 255.0;
    let b = u8::from_str_radix(&value[4..6], 16).unwrap_or(128) as f32 / 255.0;
    [r, g, b]
}

fn darken_rgb(rgb: [f32; 3], amount: f32) -> [f32; 3] {
    [
        (rgb[0] * amount).clamp(0.0, 1.0),
        (rgb[1] * amount).clamp(0.0, 1.0),
        (rgb[2] * amount).clamp(0.0, 1.0),
    ]
}

fn rgb_to_hex(rgb: [f32; 3]) -> String {
    let r = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{r:02x}{g:02x}{b:02x}")
}

#[cfg(test)]
fn yuv_from_hex(value: &str) -> (u8, u8, u8) {
    let value = value.trim().trim_start_matches('#');
    if value.len() != 6 {
        return (128, 128, 128);
    }
    let r = u8::from_str_radix(&value[0..2], 16).unwrap_or(128) as f32;
    let g = u8::from_str_radix(&value[2..4], 16).unwrap_or(128) as f32;
    let b = u8::from_str_radix(&value[4..6], 16).unwrap_or(128) as f32;
    let y = (0.257 * r + 0.504 * g + 0.098 * b + 16.0)
        .round()
        .clamp(16.0, 235.0) as u8;
    let u = (-0.148 * r - 0.291 * g + 0.439 * b + 128.0)
        .round()
        .clamp(16.0, 240.0) as u8;
    let v = (0.439 * r - 0.368 * g - 0.071 * b + 128.0)
        .round()
        .clamp(16.0, 240.0) as u8;
    (y, u, v)
}

fn non_empty_ref(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AudioConfig, BannerConfig, ClockConfig, EndpointConfig, GraphicsConfig, LadderConfig,
        PriorityInputConfig, StandbyConfig, StateConfig, SyncConfig, TextConfig, VideoConfig,
    };

    #[test]
    fn overlay_text_prefers_priority_audio_then_banner() {
        let mut feed = test_feed();
        feed.text.enabled = true;
        feed.text.content = "Manual".to_string();
        let audio = PriorityAudio {
            queue_id: "q".to_string(),
            audio_path: None,
            duration_ms: None,
            sample_rate: 48_000,
            channels: 2,
            alert_packet: None,
            banner_text: Some("Audio banner".to_string()),
            background_color: None,
            priority: None,
            started_at: chrono::Utc::now(),
        };
        let banner = BannerPayload {
            active: true,
            alerts: vec![SerializedAlert {
                fields: [(
                    "message".to_string(),
                    Value::String("Visual banner".to_string()),
                )]
                .into_iter()
                .collect(),
            }],
            ..Default::default()
        };
        let text = overlay_text(&feed, Some(&banner), Some(&audio));
        assert!(text.starts_with("Audio banner"));
        assert!(text.contains("Visual banner"));
        assert!(text.contains("Manual"));
    }

    #[test]
    fn yuv_color_conversion_handles_hex() {
        let white = yuv_from_hex("#ffffff");
        let orange = yuv_from_hex("#b45309");
        assert!(white.0 > orange.0);
        assert_ne!(orange.1, 128);
    }

    #[test]
    fn ticker_scroll_starts_off_right_and_exits_left() {
        assert_eq!(ticker_text_x(1920, 320, 0, 4), 1921);
        let exit_frame = (1921 + 320) / 4;
        assert!(ticker_text_x(1920, 320, exit_frame as u64, 4) <= -319);
        assert_eq!(ticker_scroll_frames(1920, 320, 4), 561);
    }

    #[test]
    fn ticker_scroll_fields_advance_half_a_frame_apart() {
        assert_eq!(ticker_text_x_field(1920, 320, 0, 8), 1921);
        assert_eq!(ticker_text_x_field(1920, 320, 1, 8), 1917);
        assert_eq!(ticker_text_x_field(1920, 320, 2, 8), 1913);
    }

    #[test]
    fn interlaced_field_parities_follow_field_order() {
        let mut feed = test_feed();
        feed.video.interlaced = true;
        feed.video.field_order = "tff".to_string();
        assert_eq!(field_parities(&feed), [0, 1]);
        feed.video.field_order = "bff".to_string();
        assert_eq!(field_parities(&feed), [1, 0]);
    }

    #[test]
    fn default_ticker_y_is_eight_percent_down() {
        let feed = test_feed();
        assert_eq!(ticker_y(&feed, 1080, false), 86);
    }

    #[test]
    fn no_signal_banner_uses_standby_text_and_position() {
        let feed = test_feed();
        let state = RuntimeState::default();

        let snapshot = presentation_snapshot(&feed, &state, true);

        assert_eq!(snapshot.visual_mode, "banner");
        assert_eq!(snapshot.overlay_text, "EAS Details Channel");
        assert_eq!(snapshot.ticker_y, 72);
    }

    #[test]
    fn html_banner_gradient_order_is_preserved() {
        let banner = BannerPayload {
            primary_gradient: vec!["#ff0000".to_string(), "#220000".to_string()],
            ..Default::default()
        };
        let gradient = banner_gradient_colors(Some(&banner)).expect("gradient");
        assert_eq!(gradient.top, rgb_from_hex_loose("#220000"));
        assert_eq!(gradient.middle, rgb_from_hex_loose("#ff0000"));
        assert_eq!(gradient.bottom, rgb_from_hex_loose("#220000"));
    }

    #[test]
    fn presentation_snapshot_separates_audio_and_visual_sources() {
        let feed = test_feed();
        let mut state = RuntimeState::default();
        assert!(state.apply_event(&json!({
            "type": "banner.state.updated",
            "subject": "CAP",
            "data": {
                "active": true,
                "feed_id": "CAP",
                "primary_color": "#b91c1c",
                "primary_gradient": ["#b91c1c", "#7f1d1d"],
                "alerts": [{"message": "Visual only"}]
            }
        })));
        let snapshot = presentation_snapshot(&feed, &state, false);
        assert_eq!(snapshot.visual_mode, "ticker_alert");
        assert!(snapshot.overlay_active);
        assert_eq!(snapshot.overlay_text, "Visual only");
        assert_eq!(snapshot.active_alert_queue_id, None);
    }

    fn test_feed() -> FeedConfig {
        FeedConfig {
            id: "CAP".to_string(),
            name: "CAP CGEN".to_string(),
            enabled: true,
            program_input: EndpointConfig {
                url: "udp://239.0.0.1:9000".to_string(),
                format: "mpegts".to_string(),
                ..Default::default()
            },
            priority_input: PriorityInputConfig {
                feed_id: "CAP".to_string(),
                ..Default::default()
            },
            program_output: EndpointConfig {
                url: "udp://239.0.0.2:9001".to_string(),
                format: "mpegts".to_string(),
                ..Default::default()
            },
            program: Default::default(),
            priority: Default::default(),
            media: Default::default(),
            presentation: Default::default(),
            video: VideoConfig {
                width: 1280,
                height: 720,
                fps: "source".to_string(),
                interlaced: false,
                field_order: "tff".to_string(),
                standard: String::new(),
            },
            audio: AudioConfig::default(),
            ladder: LadderConfig::default(),
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
            standby: StandbyConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}
