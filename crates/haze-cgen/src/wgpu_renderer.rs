#[cfg(feature = "gpu-wgpu")]
use anyhow::Context;
use anyhow::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OverlayRenderState {
    pub(crate) visual_id: String,
    pub(crate) source_text: String,
    pub(crate) visual_mode: String,
    pub(crate) font_family: String,
    pub(crate) font_weight: String,
    pub(crate) font_size: u32,
    pub(crate) banner_height: u32,
    pub(crate) speed_px_per_frame: u32,
    pub(crate) frame_width: u32,
    pub(crate) draw_fps: f64,
    pub(crate) text_width_px: f64,
    pub(crate) started_at: std::time::Instant,
    pub(crate) text: String,
    pub(crate) font_desc: String,
    pub(crate) ypos: f64,
    pub(crate) x_absolute: f64,
    pub(crate) silent: bool,
    pub(crate) gradient: [String; 3],
}

pub(crate) fn fatal_error_message() -> &'static str {
    if cfg!(feature = "gpu-wgpu") {
        "HAZE CGEN FATAL GRAPHICS ERROR\n\nHaze CGEN could not create a valid WGPU compositor.\n\nPossible reasons:\n1. The WGPU renderer path failed to initialize.\n2. Your GPU driver is missing, too old, or currently crashed.\n3. The selected graphics backend is unavailable for this session.\n4. This system does not expose a compatible D3D12, Vulkan, Metal, or GL backend.\n\nCairo fallback is disabled on purpose because it causes unstable broadcast timing."
    } else {
        "HAZE CGEN FATAL GRAPHICS ERROR\n\nHaze CGEN could not create a valid WGPU compositor.\n\nPossible reasons:\n1. haze-cgen was built without the gpu-wgpu feature.\n2. Your GPU driver is missing, too old, or currently crashed.\n3. The selected graphics backend is unavailable for this session.\n4. This system does not expose a compatible D3D12, Vulkan, Metal, or GL backend.\n\nCairo fallback is disabled on purpose because it causes unstable broadcast timing."
    }
}

#[cfg(not(feature = "gpu-wgpu"))]
pub(crate) struct WgpuFrameRenderer;

#[cfg(not(feature = "gpu-wgpu"))]
impl WgpuFrameRenderer {
    pub(crate) fn new(_id: String, _width: u32, _height: u32, _interlaced: bool) -> Result<Self> {
        anyhow::bail!("{}", fatal_error_message())
    }

    pub(crate) fn composite_bgrx(
        &mut self,
        _frame: &mut [u8],
        _frame_pts_ns: Option<u64>,
        _state: Option<&OverlayRenderState>,
    ) -> Result<()> {
        Ok(())
    }

    pub(crate) fn status_value(&self) -> Value {
        json!({
            "graphics_backend": "wgpu",
            "fatal": true,
            "last_error": fatal_error_message(),
        })
    }
}

#[cfg(feature = "gpu-wgpu")]
pub(crate) struct WgpuFrameRenderer {
    id: String,
    width: u32,
    height: u32,
    interlaced: bool,
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    font_system: glyphon::FontSystem,
    swash_cache: glyphon::SwashCache,
    viewport: glyphon::Viewport,
    atlas: glyphon::TextAtlas,
    text_renderer: glyphon::TextRenderer,
    adapter_name: String,
    backend: String,
    ticker_key: Option<String>,
    ticker_start_pts_ns: Option<u64>,
    ticker_start_instant: std::time::Instant,
    ticker_start_frame: u64,
    rendered_frames: u64,
    dropped_frames: u64,
    last_error: Option<String>,
}

#[cfg(feature = "gpu-wgpu")]
#[derive(Debug, Clone)]
struct RenderedTextStrip {
    width: usize,
    height: usize,
    stride: usize,
    bgra: Vec<u8>,
}

#[cfg(feature = "gpu-wgpu")]
impl WgpuFrameRenderer {
    pub(crate) fn new(id: String, width: u32, height: u32, interlaced: bool) -> Result<Self> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = preferred_backends();
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context(fatal_error_message())?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("haze-cgen-wgpu-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .context(fatal_error_message())?;
        let font_system = glyphon::FontSystem::new();
        let swash_cache = glyphon::SwashCache::new();
        let cache = glyphon::Cache::new(&device);
        let viewport = glyphon::Viewport::new(&device, &cache);
        let mut atlas =
            glyphon::TextAtlas::new(&device, &queue, &cache, wgpu::TextureFormat::Bgra8UnormSrgb);
        let text_renderer = glyphon::TextRenderer::new(
            &mut atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );
        Ok(Self {
            id,
            width,
            height,
            interlaced,
            _instance: instance,
            _adapter: adapter,
            device,
            queue,
            font_system,
            swash_cache,
            viewport,
            atlas,
            text_renderer,
            adapter_name: info.name,
            backend: format!("{:?}", info.backend),
            ticker_key: None,
            ticker_start_pts_ns: None,
            ticker_start_instant: std::time::Instant::now(),
            ticker_start_frame: 0,
            rendered_frames: 0,
            dropped_frames: 0,
            last_error: None,
        })
    }

    pub(crate) fn composite_bgrx(
        &mut self,
        frame: &mut [u8],
        frame_pts_ns: Option<u64>,
        state: Option<&OverlayRenderState>,
    ) -> Result<()> {
        let expected = usize::try_from(self.width)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(self.height).unwrap_or(0))
            .saturating_mul(4);
        if frame.len() < expected {
            self.dropped_frames = self.dropped_frames.saturating_add(1);
            self.last_error = Some(format!(
                "short video frame: got {} bytes, expected {expected}",
                frame.len()
            ));
            return Ok(());
        }
        let Some(state) = state.filter(|state| !state.silent && !state.text.trim().is_empty())
        else {
            self.ticker_key = None;
            self.ticker_start_pts_ns = None;
            self.rendered_frames = self.rendered_frames.saturating_add(1);
            return Ok(());
        };
        self.touch_wgpu();
        if let Err(err) = self.composite_overlay(frame, frame_pts_ns, state) {
            self.dropped_frames = self.dropped_frames.saturating_add(1);
            self.last_error = Some(err.to_string());
        } else {
            self.rendered_frames = self.rendered_frames.saturating_add(1);
            self.last_error = None;
        }
        Ok(())
    }

    pub(crate) fn status_value(&self) -> Value {
        json!({
            "id": self.id,
            "graphics_backend": "wgpu",
            "adapter": self.adapter_name,
            "backend": self.backend,
            "text_renderer": "glyphon",
            "width": self.width,
            "height": self.height,
            "interlaced": self.interlaced,
            "rendered_frames": self.rendered_frames,
            "dropped_frames": self.dropped_frames,
            "last_error": self.last_error,
        })
    }

    fn composite_overlay(
        &mut self,
        frame: &mut [u8],
        frame_pts_ns: Option<u64>,
        state: &OverlayRenderState,
    ) -> Result<()> {
        let width = usize::try_from(self.width).unwrap_or(0).max(1);
        let height = usize::try_from(self.height).unwrap_or(0).max(1);
        let banner_height = usize::try_from(state.banner_height)
            .unwrap_or(72)
            .clamp(24, height);
        let y = ((state.ypos * height as f64).round() as isize)
            .clamp(0, height.saturating_sub(banner_height) as isize) as usize;
        let is_ticker = state.visual_mode == "ticker_alert";
        let mut render_state = state.clone();
        if is_ticker {
            let Some((text, x_absolute)) = self.ticker_window_for_frame(state, width, frame_pts_ns)
            else {
                return Ok(());
            };
            render_state.text = text;
            render_state.x_absolute = x_absolute;
        } else {
            self.ticker_key = None;
            self.ticker_start_pts_ns = None;
        }
        if is_ticker {
            self.draw_gradient(
                frame,
                width,
                height,
                y,
                banner_height,
                &render_state.gradient,
            );
        }
        let text_x = if is_ticker {
            (render_state.x_absolute * width as f64).round() as isize
        } else {
            let approx_text_width = render_state.text_width_px.round().max(0.0) as usize;
            ((width.saturating_sub(approx_text_width)) / 2) as isize
        };
        let text_y = if is_ticker { 0 } else { y as isize };
        let strip = self.render_text_strip(
            width as u32,
            banner_height as u32,
            text_x,
            text_y,
            &render_state,
        );
        match strip {
            Ok(strip) => self.blend_text_strip(frame, width, height, y, &strip),
            Err(err) => {
                self.last_error = Some(err.to_string());
            }
        }
        Ok(())
    }

    fn ticker_window_for_frame(
        &mut self,
        state: &OverlayRenderState,
        frame_width: usize,
        frame_pts_ns: Option<u64>,
    ) -> Option<(String, f64)> {
        let text = state.source_text.trim();
        if text.is_empty() {
            return None;
        }
        let key = format!(
            "{}|{}|{}|{}|{}|{}|{}",
            state.visual_mode,
            state.font_family,
            state.font_weight,
            state.font_size,
            state.banner_height,
            state.speed_px_per_frame,
            if state.visual_id.trim().is_empty() {
                text
            } else {
                state.visual_id.trim()
            }
        );
        if self.ticker_key.as_deref() != Some(key.as_str()) {
            self.ticker_key = Some(key);
            self.ticker_start_pts_ns = frame_pts_ns;
            self.ticker_start_instant = std::time::Instant::now();
            self.ticker_start_frame = self.rendered_frames;
        }
        let frame_width_f = frame_width.max(1) as f64;
        let elapsed = self.ticker_elapsed_seconds(frame_pts_ns);
        let pixels_per_second =
            f64::from(state.speed_px_per_frame.max(1)) * state.draw_fps.max(1.0);
        let x_px = frame_width_f + 1.0 - (elapsed * pixels_per_second);
        if x_px < -state.text_width_px.max(1.0) {
            return None;
        }
        Some((text.to_string(), x_px / frame_width_f))
    }

    fn ticker_elapsed_seconds(&self, frame_pts_ns: Option<u64>) -> f64 {
        match (frame_pts_ns, self.ticker_start_pts_ns) {
            (Some(now), Some(start)) if now >= start => {
                now.saturating_sub(start) as f64 / 1_000_000_000.0
            }
            _ => self.ticker_start_instant.elapsed().as_secs_f64(),
        }
    }

    fn draw_gradient(
        &self,
        frame: &mut [u8],
        width: usize,
        height: usize,
        y: usize,
        banner_height: usize,
        gradient: &[String; 3],
    ) {
        let top = rgb_from_hex(&gradient[0]);
        let mid = rgb_from_hex(&gradient[1]);
        let bottom = rgb_from_hex(&gradient[2]);
        for row in y..y.saturating_add(banner_height).min(height) {
            let local = (row - y) as f32 / banner_height.max(1) as f32;
            let rgb = if local <= 0.5 {
                lerp_rgb(top, mid, local * 2.0)
            } else {
                lerp_rgb(mid, bottom, (local - 0.5) * 2.0)
            };
            let start = row.saturating_mul(width).saturating_mul(4);
            for px in frame[start..start + width * 4].chunks_exact_mut(4) {
                px[0] = rgb[2];
                px[1] = rgb[1];
                px[2] = rgb[0];
            }
        }
    }

    fn render_text_strip(
        &mut self,
        strip_width: u32,
        strip_height: u32,
        x: isize,
        y: isize,
        state: &OverlayRenderState,
    ) -> Result<RenderedTextStrip> {
        let strip_width = strip_width.max(1);
        let strip_height = strip_height.max(1);
        let line_height = (state.font_size as f32 * 1.16).max(14.0);
        let font_size = state.font_size.max(12) as f32;
        let text_area_width = (state.text_width_px as f32 + strip_width as f32 * 2.0)
            .max(strip_width as f32)
            .min(65_535.0);
        let mut buffer = glyphon::Buffer::new(
            &mut self.font_system,
            glyphon::Metrics::new(font_size, line_height),
        );
        buffer.set_size(
            &mut self.font_system,
            Some(text_area_width),
            Some(strip_height as f32),
        );
        buffer.set_text(
            &mut self.font_system,
            &state.text,
            &glyphon::Attrs::new()
                .family(glyphon::Family::Name(&state.font_family))
                .weight(glyphon_weight(&state.font_weight)),
            glyphon::Shaping::Advanced,
            None,
        );
        buffer.shape_until_scroll(&mut self.font_system, false);

        self.viewport.update(
            &self.queue,
            glyphon::Resolution {
                width: strip_width,
                height: strip_height,
            },
        );
        let top = if state.visual_mode == "ticker_alert" {
            ((strip_height as f32 - line_height) * 0.5).max(0.0)
        } else {
            y.max(0) as f32
        };
        let left = x as f32;
        let bounds = glyphon::TextBounds {
            left: 0,
            top: 0,
            right: strip_width as i32,
            bottom: strip_height as i32,
        };
        let shadow_area = glyphon::TextArea {
            buffer: &buffer,
            left: left + 3.0,
            top: top + 3.0,
            scale: 1.0,
            bounds,
            default_color: glyphon::Color::rgba(0, 0, 0, 210),
            custom_glyphs: &[],
        };
        let text_area = glyphon::TextArea {
            buffer: &buffer,
            left,
            top,
            scale: 1.0,
            bounds,
            default_color: glyphon::Color::rgba(255, 255, 255, 255),
            custom_glyphs: &[],
        };
        self.text_renderer.prepare(
            &self.device,
            &self.queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            [shadow_area, text_area],
            &mut self.swash_cache,
        )?;

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("haze-cgen-glyphon-strip"),
            size: wgpu::Extent3d {
                width: strip_width,
                height: strip_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let row_bytes = strip_width.saturating_mul(4);
        let padded_row_bytes = align_to(row_bytes, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback_size = u64::from(padded_row_bytes).saturating_mul(u64::from(strip_height));
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("haze-cgen-glyphon-strip-readback"),
            size: readback_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("haze-cgen-glyphon-render"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("haze-cgen-glyphon-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.text_renderer
                .render(&self.atlas, &self.viewport, &mut pass)?;
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row_bytes),
                    rows_per_image: Some(strip_height),
                },
            },
            wgpu::Extent3d {
                width: strip_width,
                height: strip_height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()
            .context("glyphon readback channel closed")?
            .context("glyphon readback map failed")?;
        let mapped = slice.get_mapped_range();
        let mut bgra = vec![0u8; strip_width as usize * strip_height as usize * 4];
        let src_stride = padded_row_bytes as usize;
        let dst_stride = row_bytes as usize;
        for row in 0..strip_height as usize {
            let src = row * src_stride;
            let dst = row * dst_stride;
            bgra[dst..dst + dst_stride].copy_from_slice(&mapped[src..src + dst_stride]);
        }
        drop(mapped);
        readback.unmap();
        self.atlas.trim();
        Ok(RenderedTextStrip {
            width: strip_width as usize,
            height: strip_height as usize,
            stride: dst_stride,
            bgra,
        })
    }

    fn blend_text_strip(
        &self,
        frame: &mut [u8],
        width: usize,
        height: usize,
        y: usize,
        strip: &RenderedTextStrip,
    ) {
        for sy in 0..strip.height {
            let dy = y + sy;
            if dy >= height {
                continue;
            }
            let row_width = strip.width.min(width);
            for sx in 0..row_width {
                let src = sy * strip.stride + sx * 4;
                let alpha = strip.bgra[src + 3];
                if alpha == 0 {
                    continue;
                }
                let alpha = u16::from(alpha);
                let inv = 255u16.saturating_sub(alpha);
                let dst = (dy * width + sx) * 4;
                frame[dst] = blend(frame[dst], strip.bgra[src], alpha, inv);
                frame[dst + 1] = blend(frame[dst + 1], strip.bgra[src + 1], alpha, inv);
                frame[dst + 2] = blend(frame[dst + 2], strip.bgra[src + 2], alpha, inv);
            }
        }
    }

    fn touch_wgpu(&self) {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("haze-cgen-wgpu-frame"),
            });
        self.queue.submit(Some(encoder.finish()));
    }
}

#[cfg(feature = "gpu-wgpu")]
fn preferred_backends() -> wgpu::Backends {
    match std::env::var("HAZE_CGEN_WGPU_BACKEND")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "vulkan" => wgpu::Backends::VULKAN,
        "dx12" | "d3d12" | "direct3d12" => wgpu::Backends::DX12,
        "gl" | "opengl" => wgpu::Backends::GL,
        "all" => wgpu::Backends::all(),
        _ if cfg!(windows) => wgpu::Backends::VULKAN,
        _ => wgpu::Backends::all(),
    }
}

#[cfg(feature = "gpu-wgpu")]
fn glyphon_weight(value: &str) -> glyphon::Weight {
    match value.trim().to_ascii_lowercase().as_str() {
        "thin" | "100" => glyphon::Weight::THIN,
        "extra-light" | "extralight" | "ultra-light" | "ultralight" | "200" => {
            glyphon::Weight::EXTRA_LIGHT
        }
        "light" | "300" => glyphon::Weight::LIGHT,
        "medium" | "500" => glyphon::Weight::MEDIUM,
        "semi-bold" | "semibold" | "demi-bold" | "demibold" | "600" => glyphon::Weight::SEMIBOLD,
        "bold" | "700" => glyphon::Weight::BOLD,
        "extra-bold" | "extrabold" | "ultra-bold" | "ultrabold" | "800" => {
            glyphon::Weight::EXTRA_BOLD
        }
        "black" | "heavy" | "900" => glyphon::Weight::BLACK,
        _ => glyphon::Weight::NORMAL,
    }
}

#[cfg(feature = "gpu-wgpu")]
fn blend(dst: u8, src: u8, alpha: u16, inv: u16) -> u8 {
    (((u16::from(dst) * inv) + (u16::from(src) * alpha)) / 255) as u8
}

#[cfg(feature = "gpu-wgpu")]
fn align_to(value: u32, alignment: u32) -> u32 {
    value.div_ceil(alignment).saturating_mul(alignment)
}

#[cfg(feature = "gpu-wgpu")]
fn rgb_from_hex(value: &str) -> [u8; 3] {
    let value = value.trim().trim_start_matches('#');
    if value.len() != 6 {
        return [17, 24, 39];
    }
    let parse = |range: std::ops::Range<usize>| u8::from_str_radix(&value[range], 16).unwrap_or(17);
    [parse(0..2), parse(2..4), parse(4..6)]
}

#[cfg(feature = "gpu-wgpu")]
fn lerp_rgb(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[cfg(feature = "gpu-wgpu")]
fn lerp(a: u8, b: u8, t: f32) -> u8 {
    ((a as f32) + ((b as f32) - (a as f32)) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}
