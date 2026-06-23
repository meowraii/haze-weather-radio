use serde_json::Value;
use tokio::sync::watch;

#[cfg(feature = "ffmpeg-rsmpeg")]
use rsmpeg::{avutil::AVFrame, ffi};

use crate::config::FeedConfig;
use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

const TICKER_PIXELS_PER_FRAME_30: i32 = 4;

pub(crate) struct NativeGraphicsRenderer {
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    priority_feed_id: String,
    banner_feed_id: String,
    #[cfg(feature = "gpu-wgpu")]
    _gpu: Option<WgpuGraphicsContext>,
}

impl NativeGraphicsRenderer {
    pub(crate) fn new(feed: &FeedConfig, state_rx: watch::Receiver<RuntimeState>) -> Self {
        let priority_feed_id = if feed.priority_input.feed_id.trim().is_empty() {
            feed.id.clone()
        } else {
            feed.priority_input.feed_id.clone()
        };
        let banner_feed_id = if feed.id.trim().is_empty() {
            priority_feed_id.clone()
        } else {
            feed.id.clone()
        };
        Self {
            feed: feed.clone(),
            state_rx,
            priority_feed_id,
            banner_feed_id,
            #[cfg(feature = "gpu-wgpu")]
            _gpu: WgpuGraphicsContext::new().ok(),
        }
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    pub(crate) fn render_frame(&mut self, frame: &mut AVFrame, frame_index: u64) {
        if !is_supported_format(frame.format) {
            return;
        }
        if self.feed.state.smpte_bars {
            draw_smpte_bars(frame);
            return;
        }

        let state = self.state_rx.borrow();
        let audio = state.priority_audio_for(&self.priority_feed_id);
        let banner = state.banner_for(&self.banner_feed_id);
        let has_manual_text = self.feed.text.enabled && !self.feed.text.content.trim().is_empty();
        let has_clock = self.feed.clock.enabled;
        if audio.is_none() && banner.is_none() && !has_manual_text && !has_clock {
            return;
        }

        if audio.is_some() || banner.is_some() || has_manual_text {
            self.render_ticker(frame, frame_index, banner, audio);
        }
        if has_clock {
            self.render_clock(frame);
        }
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    fn render_ticker(
        &self,
        frame: &mut AVFrame,
        frame_index: u64,
        banner: Option<&BannerPayload>,
        audio: Option<&PriorityAudio>,
    ) {
        let text = overlay_text(&self.feed, banner, audio);
        if text.trim().is_empty() {
            return;
        }
        let color = audio
            .and_then(|audio| audio.background_color.as_deref())
            .or_else(|| banner.and_then(|banner| non_empty_ref(&banner.primary_color)))
            .or_else(|| non_empty_ref(&self.feed.banner.background_color))
            .unwrap_or("#b45309");
        let (y, u, v) = yuv_from_hex(color);
        let x = self.feed.banner.x.max(0);
        let box_y = self.feed.banner.y.max(0);
        let w = if self.feed.graphics.banner_width == 0 {
            frame.width
        } else {
            self.feed.graphics.banner_width as i32
        }
        .max(1);
        let h = self
            .feed
            .banner
            .ticker_height
            .max(self.feed.graphics.banner_height)
            .max(48) as i32;
        if self.feed.banner.background_enabled {
            fill_rect_yuv(frame, x, box_y, w, h, y, u, v);
        }

        let font_size = self.feed.banner.font_size.max(16) as i32;
        let scale = (font_size / 8).max(2);
        let text_width = text_width_px(&text, scale);
        let travel = frame.width.saturating_add(text_width).max(1);
        let scroll = ((frame_index as i32).saturating_mul(TICKER_PIXELS_PER_FRAME_30)) % travel;
        let text_x = frame.width.saturating_sub(scroll);
        let text_y = box_y + ((h - glyph_height_px(scale)) / 2).max(0);
        draw_text_yuv(frame, &text, text_x, text_y, scale, 235, 128, 128);
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    fn render_clock(&self, frame: &mut AVFrame) {
        let text = chrono::Local::now().format("%b %d %H:%M:%S").to_string();
        let scale = ((self.feed.clock.font_size.max(12) as i32) / 8).max(2);
        let (y, u, v) = yuv_from_hex(non_empty_ref(&self.feed.clock.color).unwrap_or("#ffffff"));
        draw_text_yuv(
            frame,
            &text,
            self.feed.clock.x,
            self.feed.clock.y,
            scale,
            y,
            u,
            v,
        );
    }
}

#[cfg(feature = "gpu-wgpu")]
struct WgpuGraphicsContext {
    _device: wgpu::Device,
    _queue: wgpu::Queue,
}

#[cfg(feature = "gpu-wgpu")]
impl WgpuGraphicsContext {
    fn new() -> anyhow::Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("haze-cgen-native-graphics"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;
        Ok(Self {
            _device: device,
            _queue: queue,
        })
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

fn non_empty_ref(value: &str) -> Option<&str> {
    let value = value.trim();
    (!value.is_empty()).then_some(value)
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn is_supported_format(format: i32) -> bool {
    matches!(
        format,
        ffi::AV_PIX_FMT_YUV420P | ffi::AV_PIX_FMT_NV12 | ffi::AV_PIX_FMT_YUV422P
    )
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn draw_smpte_bars(frame: &mut AVFrame) {
    let bars = [
        "#c0c0c0", "#c0c000", "#00c0c0", "#00c000", "#c000c0", "#c00000", "#0000c0",
    ];
    let bar_width = (frame.width / bars.len() as i32).max(1);
    for (index, color) in bars.iter().enumerate() {
        let (y, u, v) = yuv_from_hex(color);
        let x = index as i32 * bar_width;
        let width = if index + 1 == bars.len() {
            frame.width.saturating_sub(x)
        } else {
            bar_width
        };
        fill_rect_yuv(frame, x, 0, width, frame.height, y, u, v);
    }
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn fill_rect_yuv(frame: &mut AVFrame, x: i32, y: i32, w: i32, h: i32, yy: u8, u: u8, v: u8) {
    let x0 = x.clamp(0, frame.width);
    let y0 = y.clamp(0, frame.height);
    let x1 = x.saturating_add(w).clamp(0, frame.width);
    let y1 = y.saturating_add(h).clamp(0, frame.height);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    fill_plane_rect(frame, 0, x0, y0, x1 - x0, y1 - y0, yy);
    match frame.format {
        ffi::AV_PIX_FMT_YUV420P => {
            fill_plane_rect(
                frame,
                1,
                x0 / 2,
                y0 / 2,
                div_ceil_i32(x1 - x0, 2),
                div_ceil_i32(y1 - y0, 2),
                u,
            );
            fill_plane_rect(
                frame,
                2,
                x0 / 2,
                y0 / 2,
                div_ceil_i32(x1 - x0, 2),
                div_ceil_i32(y1 - y0, 2),
                v,
            );
        }
        ffi::AV_PIX_FMT_NV12 => fill_nv12_rect(
            frame,
            x0 / 2,
            y0 / 2,
            div_ceil_i32(x1 - x0, 2),
            div_ceil_i32(y1 - y0, 2),
            u,
            v,
        ),
        ffi::AV_PIX_FMT_YUV422P => {
            fill_plane_rect(frame, 1, x0 / 2, y0, div_ceil_i32(x1 - x0, 2), y1 - y0, u);
            fill_plane_rect(frame, 2, x0 / 2, y0, div_ceil_i32(x1 - x0, 2), y1 - y0, v);
        }
        _ => {}
    }
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn draw_text_yuv(
    frame: &mut AVFrame,
    text: &str,
    x: i32,
    y: i32,
    scale: i32,
    yy: u8,
    u: u8,
    v: u8,
) {
    let mut cursor = x;
    for ch in text.chars() {
        if ch == ' ' {
            cursor = cursor.saturating_add(4 * scale);
            continue;
        }
        if let Some(glyph) = glyph_rows(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) != 0 {
                        fill_rect_yuv(
                            frame,
                            cursor + col * scale,
                            y + row as i32 * scale,
                            scale,
                            scale,
                            yy,
                            u,
                            v,
                        );
                    }
                }
            }
        }
        cursor = cursor.saturating_add(6 * scale);
        if cursor > frame.width {
            break;
        }
    }
}

fn text_width_px(text: &str, scale: i32) -> i32 {
    text.chars()
        .map(|ch| if ch == ' ' { 4 * scale } else { 6 * scale })
        .sum()
}

fn glyph_height_px(scale: i32) -> i32 {
    7 * scale
}

fn div_ceil_i32(value: i32, divisor: i32) -> i32 {
    if divisor <= 0 || value <= 0 {
        return 0;
    }
    (value + divisor - 1) / divisor
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn fill_plane_rect(frame: &mut AVFrame, plane: usize, x: i32, y: i32, w: i32, h: i32, value: u8) {
    if frame.data[plane].is_null() {
        return;
    }
    let Ok(stride) = usize::try_from(frame.linesize[plane]) else {
        return;
    };
    for row in y.max(0)..y.saturating_add(h).max(0) {
        let Ok(offset) = usize::try_from(row) else {
            continue;
        };
        let Ok(start) = usize::try_from(x.max(0)) else {
            continue;
        };
        let Ok(len) = usize::try_from(w.max(0)) else {
            continue;
        };
        // SAFETY: frame planes are owned by FFmpeg. Callers clamp rectangles
        // to active frame bounds; this writes one row within the plane stride.
        unsafe {
            std::ptr::write_bytes(frame.data[plane].add(offset * stride + start), value, len);
        }
    }
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn fill_nv12_rect(frame: &mut AVFrame, x: i32, y: i32, w: i32, h: i32, u: u8, v: u8) {
    if frame.data[1].is_null() {
        return;
    }
    let Ok(stride) = usize::try_from(frame.linesize[1]) else {
        return;
    };
    for row in y.max(0)..y.saturating_add(h).max(0) {
        let Ok(row) = usize::try_from(row) else {
            continue;
        };
        for col in x.max(0)..x.saturating_add(w).max(0) {
            let Ok(col) = usize::try_from(col) else {
                continue;
            };
            // SAFETY: callers clamp the chroma rectangle. NV12 stores UV
            // pairs in plane 1, so each chroma column consumes two bytes.
            unsafe {
                let ptr = frame.data[1].add(row * stride + col * 2);
                *ptr = u;
                *ptr.add(1) = v;
            }
        }
    }
}

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

fn glyph_rows(ch: char) -> Option<[u8; 7]> {
    let ch = ch.to_ascii_uppercase();
    Some(match ch {
        'A' => [0x0e, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
        'B' => [0x1e, 0x11, 0x11, 0x1e, 0x11, 0x11, 0x1e],
        'C' => [0x0e, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0e],
        'D' => [0x1e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1e],
        'E' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x1f],
        'F' => [0x1f, 0x10, 0x10, 0x1e, 0x10, 0x10, 0x10],
        'G' => [0x0e, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0f],
        'H' => [0x11, 0x11, 0x11, 0x1f, 0x11, 0x11, 0x11],
        'I' => [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1f],
        'J' => [0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0e],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1f],
        'M' => [0x11, 0x1b, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11],
        'O' => [0x0e, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
        'P' => [0x1e, 0x11, 0x11, 0x1e, 0x10, 0x10, 0x10],
        'Q' => [0x0e, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0d],
        'R' => [0x1e, 0x11, 0x11, 0x1e, 0x14, 0x12, 0x11],
        'S' => [0x0f, 0x10, 0x10, 0x0e, 0x01, 0x01, 0x1e],
        'T' => [0x1f, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0e],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0a, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0a],
        'X' => [0x11, 0x11, 0x0a, 0x04, 0x0a, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0a, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1f],
        '0' => [0x0e, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0e],
        '1' => [0x04, 0x0c, 0x04, 0x04, 0x04, 0x04, 0x0e],
        '2' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1f],
        '3' => [0x1e, 0x01, 0x01, 0x0e, 0x01, 0x01, 0x1e],
        '4' => [0x02, 0x06, 0x0a, 0x12, 0x1f, 0x02, 0x02],
        '5' => [0x1f, 0x10, 0x10, 0x1e, 0x01, 0x01, 0x1e],
        '6' => [0x06, 0x08, 0x10, 0x1e, 0x11, 0x11, 0x0e],
        '7' => [0x1f, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0e, 0x11, 0x11, 0x0e, 0x11, 0x11, 0x0e],
        '9' => [0x0e, 0x11, 0x11, 0x0f, 0x01, 0x02, 0x0c],
        '-' => [0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00],
        '_' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1f],
        ':' => [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x00],
        ';' => [0x00, 0x04, 0x04, 0x00, 0x04, 0x04, 0x08],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c],
        ',' => [0x00, 0x00, 0x00, 0x00, 0x0c, 0x0c, 0x08],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '\'' => [0x04, 0x04, 0x08, 0x00, 0x00, 0x00, 0x00],
        '"' => [0x0a, 0x0a, 0x0a, 0x00, 0x00, 0x00, 0x00],
        '!' => [0x04, 0x04, 0x04, 0x04, 0x04, 0x00, 0x04],
        '?' => [0x0e, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04],
        '+' => [0x00, 0x04, 0x04, 0x1f, 0x04, 0x04, 0x00],
        '&' => [0x0c, 0x12, 0x14, 0x08, 0x15, 0x12, 0x0d],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        AudioConfig, BannerConfig, ClockConfig, EndpointConfig, GraphicsConfig,
        PriorityInputConfig, StateConfig, TextConfig, VideoConfig,
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

    fn test_feed() -> FeedConfig {
        FeedConfig {
            id: "CAP".to_string(),
            name: "CAP CGEN".to_string(),
            enabled: true,
            input: EndpointConfig::default(),
            output: EndpointConfig::default(),
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
            alert_output: EndpointConfig::default(),
            video: VideoConfig {
                width: 1280,
                height: 720,
                fps: "source".to_string(),
                interlaced: false,
                field_order: "tff".to_string(),
                standard: String::new(),
            },
            audio: AudioConfig::default(),
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
        }
    }
}
