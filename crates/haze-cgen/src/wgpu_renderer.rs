#[cfg(feature = "gpu-wgpu")]
use anyhow::Context;
use anyhow::Result;
use serde_json::{json, Value};

pub(crate) fn fatal_error_message() -> &'static str {
    "HAZE CGEN FATAL GRAPHICS ERROR: WGPU renderer initialization failed. Cairo fallback is disabled; install a working Vulkan/DX12/OpenGL-capable GPU driver or change the CGEN graphics backend."
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OverlayRenderState {
    pub(crate) visual_id: String,
    pub(crate) source_text: String,
    pub(crate) visual_mode: String,
    pub(crate) font_family: String,
    pub(crate) font_weight: String,
    pub(crate) font_size: u32,
    pub(crate) clock_text: String,
    pub(crate) clock_x: i32,
    pub(crate) clock_y: i32,
    pub(crate) clock_font_size: u32,
    pub(crate) clock_color: String,
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
    pub(crate) sunny_cat: bool,
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
    ticker_start_frame: u64,
    ticker_start_pts_ns: Option<u64>,
    text_strip_cache: std::collections::BTreeMap<String, RenderedTextStrip>,
    text_strip_cache_bytes: usize,
    rendered_frames: u64,
    dropped_frames: u64,
    last_error: Option<String>,
}

#[cfg(feature = "gpu-wgpu")]
const MAX_TEXT_STRIP_CACHE_ENTRIES: usize = 4;
#[cfg(feature = "gpu-wgpu")]
const MAX_TEXT_STRIP_CACHE_BYTES: usize = 24 * 1024 * 1024;

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
            ticker_start_frame: 0,
            ticker_start_pts_ns: None,
            text_strip_cache: std::collections::BTreeMap::new(),
            text_strip_cache_bytes: 0,
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
        let Some(state) = state.filter(|state| {
            state.sunny_cat
                || (!state.silent && !state.text.trim().is_empty())
                || !state.clock_text.trim().is_empty()
        }) else {
            self.ticker_key = None;
            self.ticker_start_pts_ns = None;
            self.clear_text_strip_cache();
            self.rendered_frames = self.rendered_frames.saturating_add(1);
            return Ok(());
        };
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
            "text_strip_cache_entries": self.text_strip_cache.len(),
            "text_strip_cache_bytes": self.text_strip_cache_bytes,
            "last_error": self.last_error,
            "sunny_cat_available": crate::sunny_cat::available(),
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
        if render_state.sunny_cat {
            draw_sunny_cat(frame, width, height);
        }
        if render_state.text.trim().is_empty() && render_state.clock_text.trim().is_empty() {
            return Ok(());
        }
        if is_ticker && !render_state.text.trim().is_empty() {
            let Some((text, x_absolute)) = self.ticker_window_for_frame(state, width, frame_pts_ns)
            else {
                self.draw_clock(frame, width, height, &render_state)?;
                return Ok(());
            };
            render_state.text = text;
            render_state.x_absolute = x_absolute;
        } else if !is_ticker {
            self.ticker_key = None;
            self.ticker_start_pts_ns = None;
            self.clear_text_strip_cache();
        }
        if is_ticker && !render_state.text.trim().is_empty() {
            self.draw_gradient(
                frame,
                width,
                height,
                y,
                banner_height,
                &render_state.gradient,
            );
            let text_x = (render_state.x_absolute * width as f64).round() as isize;
            let strip_width = ticker_strip_width(&render_state, width);
            let strip =
                self.ensure_ticker_strip(strip_width, banner_height as u32, &render_state)?;
            blend_text_strip_at(frame, width, height, text_x, y as isize, strip);
            self.draw_clock(frame, width, height, &render_state)?;
            return Ok(());
        }
        if !render_state.text.trim().is_empty() {
            let text_x = {
                let approx_text_width = render_state.text_width_px.round().max(0.0) as usize;
                ((width.saturating_sub(approx_text_width)) / 2) as isize
            };
            let text_y = y as isize;
            let strip = self.render_text_strip(
                width as u32,
                banner_height as u32,
                text_x,
                text_y,
                &render_state,
            );
            match strip {
                Ok(strip) => blend_text_strip_at(frame, width, height, 0, y as isize, &strip),
                Err(err) => {
                    self.last_error = Some(err.to_string());
                }
            }
        }
        self.draw_clock(frame, width, height, &render_state)?;
        Ok(())
    }

    fn draw_clock(
        &mut self,
        frame: &mut [u8],
        width: usize,
        height: usize,
        state: &OverlayRenderState,
    ) -> Result<()> {
        let clock_text = state.clock_text.trim();
        if clock_text.is_empty() {
            return Ok(());
        }
        let mut clock_state = state.clone();
        clock_state.text = clock_text.to_string();
        clock_state.source_text = clock_text.to_string();
        clock_state.font_size = state.clock_font_size.max(12);
        clock_state.text_width_px = estimated_text_width_px(clock_text, clock_state.font_size);
        clock_state.visual_mode = "clock".to_string();
        let strip_width = (clock_state.text_width_px.round().max(1.0) as u32)
            .saturating_add(clock_state.clock_font_size.saturating_mul(2))
            .min(self.width.max(1));
        let strip_height = clock_state
            .clock_font_size
            .saturating_mul(2)
            .max(24)
            .min(self.height.max(1));
        let strip = self.render_text_strip(strip_width, strip_height, 0, 0, &clock_state)?;
        blend_text_strip_at(
            frame,
            width,
            height,
            isize::try_from(state.clock_x).unwrap_or(0),
            isize::try_from(state.clock_y).unwrap_or(0),
            &strip,
        );
        Ok(())
    }

    fn ticker_window_for_frame(
        &mut self,
        state: &OverlayRenderState,
        frame_width: usize,
        frame_pts_ns: Option<u64>,
    ) -> Option<(String, f64)> {
        let source_text = state.source_text.trim();
        if source_text.is_empty() {
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
                source_text
            } else {
                state.visual_id.trim()
            }
        );
        if self.ticker_key.as_deref() != Some(key.as_str()) {
            self.ticker_key = Some(key);
            self.ticker_start_frame = self.rendered_frames;
            self.ticker_start_pts_ns = frame_pts_ns;
            self.clear_text_strip_cache();
        }
        let frame_width_f = frame_width.max(1) as f64;
        let frames_since_start = self.rendered_frames.saturating_sub(self.ticker_start_frame);
        let elapsed_seconds = ticker_elapsed_for_compositor_frame(
            frame_pts_ns,
            self.ticker_start_pts_ns,
            frames_since_start,
            state.draw_fps,
        );
        let x_px = ticker_x_for_compositor_time(
            frame_width_f,
            state.text_width_px,
            state.speed_px_per_frame,
            state.draw_fps,
            elapsed_seconds,
        );
        if x_px < -state.text_width_px.max(1.0) {
            return None;
        }
        Some(ticker_text_window(
            source_text,
            state.text_width_px,
            state.font_size,
            x_px,
            frame_width_f,
        ))
    }

    fn ensure_ticker_strip(
        &mut self,
        strip_width: u32,
        strip_height: u32,
        state: &OverlayRenderState,
    ) -> Result<&RenderedTextStrip> {
        let key = format!(
            "{}|{}|{}|{}|{}|{}",
            state.text,
            state.font_family,
            state.font_weight,
            state.font_size,
            strip_width,
            strip_height,
        );
        if !self.text_strip_cache.contains_key(&key) {
            let strip = self.render_text_strip(strip_width, strip_height, 0, 0, state)?;
            let strip_bytes = strip.bgra.len();
            if self.text_strip_cache.len() >= MAX_TEXT_STRIP_CACHE_ENTRIES
                || self.text_strip_cache_bytes.saturating_add(strip_bytes)
                    > MAX_TEXT_STRIP_CACHE_BYTES
            {
                self.clear_text_strip_cache();
            }
            self.text_strip_cache_bytes = self.text_strip_cache_bytes.saturating_add(strip_bytes);
            self.text_strip_cache.insert(key.clone(), strip);
        }
        self.text_strip_cache
            .get(&key)
            .context("ticker strip cache entry missing after render")
    }

    fn clear_text_strip_cache(&mut self) {
        self.text_strip_cache.clear();
        self.text_strip_cache_bytes = 0;
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
        let text_rgb = if state.visual_mode == "clock" {
            rgb_from_hex(&state.clock_color)
        } else {
            [255, 255, 255]
        };
        let text_area = glyphon::TextArea {
            buffer: &buffer,
            left,
            top,
            scale: 1.0,
            bounds,
            default_color: glyphon::Color::rgba(text_rgb[0], text_rgb[1], text_rgb[2], 255),
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
}

#[cfg(feature = "gpu-wgpu")]
fn ticker_text_window(
    text: &str,
    text_width_px: f64,
    font_size: u32,
    x_px: f64,
    frame_width: f64,
) -> (String, f64) {
    let frame_width = frame_width.max(1.0);
    let char_count = text.chars().count();
    if char_count == 0 {
        return (String::new(), 1.0);
    }
    let estimated_width = text_width_px
        .max(f64::from(font_size.max(16)) * char_count as f64 * 0.48)
        .max(1.0);
    let char_width = (estimated_width / char_count as f64).max(8.0);
    let first_visible_px = (-x_px).max(0.0);
    let page_width_px = (frame_width * 2.0).max(char_width * 32.0);
    let page_left_px = (first_visible_px / page_width_px).floor() * page_width_px;
    let first_char = ((page_left_px / char_width).floor() as usize).saturating_sub(8);
    let char_offset_px = first_char as f64 * char_width;
    let window_left_px = x_px + char_offset_px;
    let visible_width_px = (frame_width - window_left_px).max(frame_width) + page_width_px;
    let max_chars = ((visible_width_px / char_width).ceil() as usize)
        .saturating_add(24)
        .min(char_count.saturating_sub(first_char));
    let window_text = text.chars().skip(first_char).take(max_chars).collect();
    (window_text, window_left_px / frame_width)
}

#[cfg(feature = "gpu-wgpu")]
fn ticker_x_for_compositor_frame(
    frame_width: f64,
    text_width_px: f64,
    speed_px_per_frame: u32,
    interlaced: bool,
    frames_since_start: u64,
) -> f64 {
    let _ = interlaced;
    ticker_x_for_compositor_time(
        frame_width,
        text_width_px,
        speed_px_per_frame,
        1.0,
        frames_since_start as f64,
    )
}

#[cfg(feature = "gpu-wgpu")]
fn ticker_elapsed_for_compositor_frame(
    frame_pts_ns: Option<u64>,
    start_pts_ns: Option<u64>,
    frames_since_start: u64,
    draw_fps: f64,
) -> f64 {
    if let (Some(frame_pts_ns), Some(start_pts_ns)) = (frame_pts_ns, start_pts_ns) {
        if frame_pts_ns >= start_pts_ns {
            return frame_pts_ns.saturating_sub(start_pts_ns) as f64 / 1_000_000_000.0;
        }
    }
    frames_since_start as f64 / draw_fps.max(1.0)
}

#[cfg(feature = "gpu-wgpu")]
fn ticker_x_for_compositor_time(
    frame_width: f64,
    _text_width_px: f64,
    speed_px_per_frame: u32,
    draw_fps: f64,
    elapsed_seconds: f64,
) -> f64 {
    let start_x = frame_width.max(1.0) + 1.0;
    let pixels_per_second = f64::from(speed_px_per_frame.max(1)) * draw_fps.max(1.0);
    start_x - (elapsed_seconds.max(0.0) * pixels_per_second)
}

#[cfg(feature = "gpu-wgpu")]
fn ticker_strip_width(state: &OverlayRenderState, frame_width: usize) -> u32 {
    let chars = state.text.chars().count().max(1) as f64;
    let estimate = chars * f64::from(state.font_size.max(16)) * 0.92 + 32.0;
    let max_width = frame_width.saturating_mul(2).clamp(512, 8_192) as f64;
    estimate.ceil().clamp(64.0, max_width) as u32
}

#[cfg(feature = "gpu-wgpu")]
fn blend_text_strip_at(
    frame: &mut [u8],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    strip: &RenderedTextStrip,
) {
    let start_sx = if x < 0 {
        usize::try_from(-x).unwrap_or(strip.width).min(strip.width)
    } else {
        0
    };
    let end_sx = if x >= width as isize {
        0
    } else {
        let visible = width.saturating_sub(x.max(0) as usize);
        start_sx.saturating_add(visible).min(strip.width)
    };
    if start_sx >= end_sx {
        return;
    }
    for sy in 0..strip.height {
        let dy = y.saturating_add(sy as isize);
        if dy < 0 || dy >= height as isize {
            continue;
        }
        for sx in start_sx..end_sx {
            let dx = x.saturating_add(sx as isize);
            let src = sy * strip.stride + sx * 4;
            let alpha = strip.bgra[src + 3];
            if alpha == 0 {
                continue;
            }
            let alpha = u16::from(alpha);
            let inv = 255u16.saturating_sub(alpha);
            let dst = (dy as usize * width + dx as usize) * 4;
            frame[dst] = blend(frame[dst], strip.bgra[src], alpha, inv);
            frame[dst + 1] = blend(frame[dst + 1], strip.bgra[src + 1], alpha, inv);
            frame[dst + 2] = blend(frame[dst + 2], strip.bgra[src + 2], alpha, inv);
        }
    }
}

#[cfg(feature = "gpu-wgpu")]
fn draw_sunny_cat(frame: &mut [u8], width: usize, height: usize) {
    if !crate::sunny_cat::available() || width == 0 || height == 0 {
        return;
    }
    let src_width = crate::sunny_cat::WIDTH;
    let src_height = crate::sunny_cat::HEIGHT;
    let max_width = width.saturating_sub(32).max(1);
    let max_height = height.saturating_sub(32).max(1);
    let scale = (max_width as f64 / src_width as f64)
        .min(max_height as f64 / src_height as f64)
        .min(1.0)
        .max(0.1);
    let dst_width = (src_width as f64 * scale).round().max(1.0) as usize;
    let dst_height = (src_height as f64 * scale).round().max(1.0) as usize;
    let x0 = width.saturating_sub(dst_width).saturating_sub(36);
    let y0 = height.saturating_sub(dst_height).saturating_sub(36);
    let rgba = crate::sunny_cat::RGBA;
    for dy in 0..dst_height {
        let sy = ((dy as f64 / scale).floor() as usize).min(src_height.saturating_sub(1));
        let out_y = y0 + dy;
        if out_y >= height {
            continue;
        }
        for dx in 0..dst_width {
            let sx = ((dx as f64 / scale).floor() as usize).min(src_width.saturating_sub(1));
            let src = (sy * src_width + sx) * 4;
            if src + 3 >= rgba.len() {
                continue;
            }
            let alpha = rgba[src + 3];
            if alpha == 0 {
                continue;
            }
            let out_x = x0 + dx;
            if out_x >= width {
                continue;
            }
            let alpha = u16::from(alpha);
            let inv = 255u16.saturating_sub(alpha);
            let dst = (out_y * width + out_x) * 4;
            frame[dst] = blend(frame[dst], rgba[src + 2], alpha, inv);
            frame[dst + 1] = blend(frame[dst + 1], rgba[src + 1], alpha, inv);
            frame[dst + 2] = blend(frame[dst + 2], rgba[src], alpha, inv);
        }
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

#[cfg(feature = "gpu-wgpu")]
fn estimated_text_width_px(text: &str, font_size: u32) -> f64 {
    let font_size = f64::from(font_size.max(16));
    let units = text.chars().fold(0.0, |sum, ch| {
        sum + if ch.is_whitespace() {
            0.38
        } else if ch.is_ascii_punctuation() {
            0.42
        } else if ch.is_ascii_lowercase() {
            0.58
        } else if ch.is_ascii_digit() {
            0.60
        } else {
            0.74
        }
    });
    units * font_size
}

#[cfg(all(test, feature = "gpu-wgpu"))]
mod tests {
    use super::*;

    #[test]
    fn ticker_window_keeps_rendering_tail_when_leading_text_is_offscreen() {
        let text = "Environment Canada has issued a Severe Thunderstorm Warning. ".repeat(20);
        let text_width = text.chars().count() as f64 * 48.0 * 0.56;

        let (window, x_absolute) = ticker_text_window(&text, text_width, 48, -19_000.0, 1_920.0);

        assert!(!window.trim().is_empty());
        assert!(
            x_absolute > -1.0,
            "window x should keep the visible suffix on-screen"
        );
        assert!(
            window.len() < text.len(),
            "renderer should draw a bounded text window"
        );
    }

    #[test]
    fn ticker_window_uses_stable_pages_for_adjacent_frames() {
        let text = "Environment Canada has issued a Severe Thunderstorm Warning. ".repeat(20);
        let text_width = text.chars().count() as f64 * 48.0 * 0.56;

        let (first, first_x) = ticker_text_window(&text, text_width, 48, -320.0, 1_920.0);
        let (second, second_x) = ticker_text_window(&text, text_width, 48, -332.0, 1_920.0);

        assert_eq!(first, second);
        assert!(
            second_x < first_x,
            "the cached page should slide smoothly between frame positions"
        );
    }

    #[test]
    fn ticker_x_uses_exact_compositor_frame_steps() {
        assert_eq!(
            ticker_x_for_compositor_frame(1_920.0, 800.0, 12, false, 0),
            1_921.0
        );
        assert_eq!(
            ticker_x_for_compositor_frame(1_920.0, 800.0, 12, false, 1),
            1_909.0
        );
        assert_eq!(
            ticker_x_for_compositor_frame(1_920.0, 800.0, 12, false, 10),
            1_801.0
        );
    }

    #[test]
    fn ticker_x_uses_full_steps_for_interlaced_output() {
        assert_eq!(
            ticker_x_for_compositor_frame(1_920.0, 800.0, 12, true, 0),
            1_921.0
        );
        assert_eq!(
            ticker_x_for_compositor_frame(1_920.0, 800.0, 12, true, 1),
            1_909.0
        );
        assert_eq!(
            ticker_x_for_compositor_frame(1_920.0, 800.0, 12, true, 2),
            1_897.0
        );
    }

    #[test]
    fn ticker_elapsed_prefers_buffer_pts_over_callback_count() {
        let elapsed = ticker_elapsed_for_compositor_frame(
            Some(2_500_000_000),
            Some(1_000_000_000),
            120,
            60.0,
        );
        assert!((elapsed - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn ticker_elapsed_falls_back_to_draw_fps_without_pts() {
        let elapsed = ticker_elapsed_for_compositor_frame(None, None, 30, 30.0);
        assert!((elapsed - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn ticker_x_uses_media_time_and_configured_draw_fps() {
        let fps = 30_000.0 / 1_001.0;
        let x = ticker_x_for_compositor_time(1_920.0, 800.0, 12, fps, 1.0);
        let expected = 1_921.0 - (12.0 * fps);
        assert!((x - expected).abs() < 0.000_001);
    }
}
