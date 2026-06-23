#[cfg(feature = "gpu-wgpu")]
use std::borrow::Cow;
#[cfg(feature = "gpu-wgpu")]
use std::{collections::HashMap, sync::mpsc, sync::Arc};

use serde_json::Value;
use tokio::sync::watch;

#[cfg(feature = "gpu-wgpu")]
use fontdue::{
    layout::{CoordinateSystem, Layout, LayoutSettings, TextStyle},
    Font, FontSettings,
};
#[cfg(feature = "ffmpeg-rsmpeg")]
use rsmpeg::{avutil::AVFrame, ffi};
#[cfg(feature = "gpu-wgpu")]
use wgpu::util::DeviceExt;

use crate::config::FeedConfig;
use crate::state::{BannerPayload, PriorityAudio, RuntimeState, SerializedAlert};

pub(crate) struct NativeGraphicsRenderer {
    feed: FeedConfig,
    state_rx: watch::Receiver<RuntimeState>,
    priority_feed_id: String,
    banner_feed_id: String,
    ticker_key: String,
    ticker_start_frame: u64,
    ticker_visual: Option<TickerVisual>,
    #[cfg(feature = "gpu-wgpu")]
    gpu: Option<WgpuGraphicsContext>,
    #[cfg(feature = "gpu-wgpu")]
    cpu_fonts: SystemFontCache,
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
            ticker_key: String::new(),
            ticker_start_frame: 0,
            ticker_visual: None,
            #[cfg(feature = "gpu-wgpu")]
            gpu: WgpuGraphicsContext::new().ok(),
            #[cfg(feature = "gpu-wgpu")]
            cpu_fonts: SystemFontCache::new(),
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

        let (banner, audio) = {
            let state = self.state_rx.borrow();
            (
                state.banner_for(&self.banner_feed_id).cloned(),
                state.priority_audio_for(&self.priority_feed_id).cloned(),
            )
        };
        let has_clock = self.feed.clock.enabled;
        if audio.is_none() && !has_clock {
            self.render_lingering_ticker(frame, frame_index);
            return;
        }

        if audio.is_some() {
            self.render_ticker(frame, frame_index, banner.as_ref(), audio.as_ref());
        } else {
            self.render_lingering_ticker(frame, frame_index);
        }
        if has_clock {
            self.render_clock(frame);
        }
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    #[allow(dead_code)]
    pub(crate) fn render_no_signal(&mut self, frame: &mut AVFrame) {
        if !is_supported_format(frame.format) {
            return;
        }
        fill_rect_yuv(frame, 0, 0, frame.width, frame.height, 16, 128, 128);
        let banner_h = (frame.height / 5).clamp(72, 180);
        let banner_y = (frame.height - banner_h) / 2;
        fill_rect_yuv_gradient(
            frame,
            0,
            banner_y,
            frame.width,
            banner_h,
            TickerGradient {
                top: rgb_from_hex_loose("#111827"),
                middle: rgb_from_hex_loose("#000000"),
                bottom: rgb_from_hex_loose("#111827"),
            },
            (16, 128, 128),
        );
        let label = format!("HAZE CGEN NO SIGNAL - {}", self.feed.id);
        let scale = ((frame.height / 220).max(3)) as usize;
        draw_text_yuv(
            frame,
            &label,
            48,
            banner_y + banner_h / 2 - (8 * scale as i32) / 2,
            scale as i32,
            235,
            128,
            128,
        );
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    fn render_ticker(
        &mut self,
        frame: &mut AVFrame,
        frame_index: u64,
        banner: Option<&BannerPayload>,
        audio: Option<&PriorityAudio>,
    ) {
        let text = overlay_text(&self.feed, banner, audio);
        if text.trim().is_empty() {
            return;
        }
        let Some(color) = ticker_color(banner, audio) else {
            return;
        };
        let gradient = ticker_gradient(banner, audio, &color);
        let next_key = ticker_state_key(banner, audio);
        if self.ticker_key != next_key {
            self.ticker_key = next_key;
            self.ticker_start_frame = frame_index;
        }
        let relative_frame = frame_index.saturating_sub(self.ticker_start_frame);
        self.ticker_visual = Some(TickerVisual {
            text: text.clone(),
            color: color.clone(),
            gradient,
        });
        self.render_ticker_visual(frame, relative_frame, &text, &color, gradient);
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    fn render_lingering_ticker(&mut self, frame: &mut AVFrame, frame_index: u64) {
        let Some(visual) = self.ticker_visual.clone() else {
            self.ticker_key.clear();
            return;
        };
        let font_size = self.feed.banner.font_size.max(16);
        let font_family = ticker_font_family(&self.feed).to_string();
        let scale = ((font_size as i32) / 8).max(2);
        let text_width = self.ticker_text_width(&visual.text, scale, &font_family, font_size);
        let relative_frame = frame_index.saturating_sub(self.ticker_start_frame);
        if relative_frame
            > ticker_scroll_frames(frame.width, text_width, self.feed.banner.scroll_speed)
        {
            self.ticker_key.clear();
            self.ticker_visual = None;
            return;
        }
        self.render_ticker_visual(
            frame,
            relative_frame,
            &visual.text,
            &visual.color,
            visual.gradient,
        );
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    fn render_ticker_visual(
        &mut self,
        frame: &mut AVFrame,
        relative_frame: u64,
        text: &str,
        color: &str,
        gradient: TickerGradient,
    ) {
        let (y, u, v) = yuv_from_hex(color);
        let x = self.feed.banner.x.max(0);
        let box_y = ticker_y(&self.feed, frame.height).max(0);
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
        let font_size = self.feed.banner.font_size.max(16);
        let font_family = ticker_font_family(&self.feed).to_string();
        let scale = ((font_size as i32) / 8).max(2);
        let text_width = self.ticker_text_width(text, scale, &font_family, font_size);
        let text_x = ticker_text_x(
            frame.width,
            text_width,
            relative_frame,
            self.feed.banner.scroll_speed,
        );
        let text_y = box_y + ((h - glyph_height_px(scale)) / 2).max(0);

        if self.feed.video.interlaced {
            if self.feed.banner.background_enabled {
                fill_rect_yuv_gradient(frame, x, box_y, w, h, gradient, (y, u, v));
            }
            self.render_interlaced_ticker_text(
                frame,
                relative_frame,
                text,
                &font_family,
                font_size,
                scale,
                text_width,
                text_y,
            );
            return;
        }

        #[cfg(feature = "gpu-wgpu")]
        if let Some(gpu) = self.gpu.as_mut() {
            let text_rgb = rgb_from_hex(non_empty_ref(&self.feed.text.color).unwrap_or("#ffffff"));
            let target_x = x.clamp(0, frame.width);
            let target_y = box_y.clamp(0, frame.height);
            let target_w = w.min(frame.width.saturating_sub(target_x)).max(1);
            let target_h = h.min(frame.height.saturating_sub(target_y)).max(1);
            if target_w > 0 && target_h > 0 {
                let overlay = gpu.render_ticker(
                    target_w as u32,
                    target_h as u32,
                    text,
                    text_x.saturating_sub(target_x),
                    &font_family,
                    font_size,
                    scale,
                    self.feed.banner.background_enabled.then_some(gradient),
                    text_rgb,
                    true,
                );
                if let Ok(overlay) = overlay {
                    blend_rgba_overlay(
                        frame,
                        &overlay.pixels,
                        overlay.width,
                        overlay.height,
                        target_x,
                        target_y,
                    );
                    return;
                }
            }
        }

        if self.feed.banner.background_enabled {
            fill_rect_yuv_gradient(frame, x, box_y, w, h, gradient, (y, u, v));
        }
        #[cfg(feature = "gpu-wgpu")]
        if let Some((width, height, pixels)) = self
            .cpu_fonts
            .resolve(&font_family)
            .and_then(|font| raster_text_rgba_system(&font, text, font_size))
        {
            let shadow = tint_rgba_overlay(&pixels, [0, 0, 0], 220);
            blend_rgba_overlay(frame, &shadow, width, height, text_x, text_y);
            blend_rgba_overlay(frame, &shadow, width, height, text_x + 3, text_y + 4);
            blend_rgba_overlay(frame, &pixels, width, height, text_x, text_y);
            return;
        }
        draw_text_yuv(frame, text, text_x + 3, text_y + 4, scale, 16, 128, 128);
        draw_text_yuv(frame, text, text_x, text_y, scale, 235, 128, 128);
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    fn render_interlaced_ticker_text(
        &mut self,
        frame: &mut AVFrame,
        relative_frame: u64,
        text: &str,
        font_family: &str,
        font_size: u32,
        scale: i32,
        text_width: i32,
        text_y: i32,
    ) {
        let parities = field_parities(&self.feed);
        let (text_luma, _, _) =
            yuv_from_hex(non_empty_ref(&self.feed.text.color).unwrap_or("#ffffff"));
        #[cfg(feature = "gpu-wgpu")]
        if let Some((width, height, pixels)) = self
            .cpu_fonts
            .resolve(font_family)
            .and_then(|font| raster_text_rgba_system(&font, text, font_size))
        {
            let shadow = tint_rgba_overlay(&pixels, [0, 0, 0], 220);
            for (field_index, parity) in parities.into_iter().enumerate() {
                let text_x = ticker_text_x_field(
                    frame.width,
                    text_width,
                    relative_frame
                        .saturating_mul(2)
                        .saturating_add(field_index as u64),
                    self.feed.banner.scroll_speed,
                );
                blend_rgba_luma_overlay_field(
                    frame, &shadow, width, height, text_x, text_y, 16, parity,
                );
                blend_rgba_luma_overlay_field(
                    frame,
                    &shadow,
                    width,
                    height,
                    text_x + 3,
                    text_y + 4,
                    16,
                    parity,
                );
                blend_rgba_luma_overlay_field(
                    frame, &pixels, width, height, text_x, text_y, text_luma, parity,
                );
            }
            return;
        }
        for (field_index, parity) in parities.into_iter().enumerate() {
            let text_x = ticker_text_x_field(
                frame.width,
                text_width,
                relative_frame
                    .saturating_mul(2)
                    .saturating_add(field_index as u64),
                self.feed.banner.scroll_speed,
            );
            draw_text_yuv_field(frame, text, text_x + 3, text_y + 4, scale, 16, parity);
            draw_text_yuv_field(frame, text, text_x, text_y, scale, text_luma, parity);
        }
    }

    #[cfg(feature = "ffmpeg-rsmpeg")]
    fn ticker_text_width(
        &mut self,
        text: &str,
        scale: i32,
        font_family: &str,
        font_size: u32,
    ) -> i32 {
        let mut text_width = text_width_px(text, scale);
        #[cfg(feature = "gpu-wgpu")]
        {
            if let Some(width) =
                measure_text_width_with_cache(&mut self.cpu_fonts, font_family, font_size, text)
            {
                text_width = width;
            }
        }
        text_width
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
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    _white_texture: wgpu::Texture,
    white_view: wgpu::TextureView,
    target: Option<WgpuTarget>,
    text_cache: Option<WgpuTextCache>,
    fonts: SystemFontCache,
}

#[derive(Clone, Debug)]
struct TickerVisual {
    text: String,
    color: String,
    gradient: TickerGradient,
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
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("haze-cgen-ticker-shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(TICKER_SHADER)),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("haze-cgen-overlay-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("haze-cgen-overlay-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("haze-cgen-overlay-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[GpuVertex::layout()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("haze-cgen-overlay-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let white_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("haze-cgen-white-overlay-texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            white_texture.as_image_copy(),
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let white_view = white_texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            sampler,
            _white_texture: white_texture,
            white_view,
            target: None,
            text_cache: None,
            fonts: SystemFontCache::new(),
        })
    }

    fn render_ticker(
        &mut self,
        width: u32,
        height: u32,
        text: &str,
        text_x: i32,
        font_family: &str,
        font_size: u32,
        scale: i32,
        background: Option<TickerGradient>,
        text_rgb: [f32; 3],
        shadow: bool,
    ) -> anyhow::Result<RgbaOverlay> {
        self.ensure_target(width, height);
        self.ensure_text_texture(text, font_family, font_size, scale)?;
        let target = self.target.as_ref().expect("target was just ensured");
        let text_cache = self
            .text_cache
            .as_ref()
            .expect("text texture was just ensured");

        let text_bg = self.bind_group(&text_cache.view);
        let mut draw_items = Vec::with_capacity(3);
        if let Some(gradient) = background {
            let mid_y = (height as f32 / 2.0).max(1.0);
            draw_items.push((
                self.vertex_buffer(quad_vertices_px_gradient(
                    0.0,
                    0.0,
                    width as f32,
                    mid_y,
                    width as f32,
                    height as f32,
                    rgba(gradient.top, 0.92),
                    rgba(gradient.middle, 0.92),
                )),
                self.bind_group(&self.white_view),
            ));
            draw_items.push((
                self.vertex_buffer(quad_vertices_px_gradient(
                    0.0,
                    mid_y,
                    width as f32,
                    (height as f32 - mid_y).max(1.0),
                    width as f32,
                    height as f32,
                    rgba(gradient.middle, 0.92),
                    rgba(gradient.bottom, 0.92),
                )),
                self.bind_group(&self.white_view),
            ));
        }
        let text_y = ((height as i32 - text_cache.height as i32) / 2).max(0);
        if shadow {
            for (dx, dy, alpha) in [(0.0_f32, 0.0_f32, 0.75_f32), (3.0, 4.0, 0.90)] {
                draw_items.push((
                    self.vertex_buffer(quad_vertices_px(
                        text_x as f32 + dx,
                        text_y as f32 + dy,
                        text_cache.width as f32,
                        text_cache.height as f32,
                        width as f32,
                        height as f32,
                        [0.0, 0.0, 0.0, alpha],
                    )),
                    text_bg.clone(),
                ));
            }
        }
        draw_items.push((
            self.vertex_buffer(quad_vertices_px(
                text_x as f32,
                text_y as f32,
                text_cache.width as f32,
                text_cache.height as f32,
                width as f32,
                height as f32,
                [text_rgb[0], text_rgb[1], text_rgb[2], 1.0],
            )),
            text_bg,
        ));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("haze-cgen-overlay-encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("haze-cgen-overlay-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.view,
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
            });
            pass.set_pipeline(&self.pipeline);
            for (vertices, bind_group) in &draw_items {
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.draw(0..6, 0..1);
            }
        }
        encoder.copy_texture_to_buffer(
            target.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(target.bytes_per_row),
                    rows_per_image: Some(target.height),
                },
            },
            wgpu::Extent3d {
                width: target.width,
                height: target.height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = target.readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::Wait)?;
        rx.recv()
            .map_err(|err| anyhow::anyhow!("wgpu readback callback dropped: {err}"))??;
        let mapped = slice.get_mapped_range();
        let row_bytes = (target.width as usize) * 4;
        let mut pixels = vec![0; row_bytes * target.height as usize];
        for row in 0..target.height as usize {
            let src_start = row * target.bytes_per_row as usize;
            let dst_start = row * row_bytes;
            pixels[dst_start..dst_start + row_bytes]
                .copy_from_slice(&mapped[src_start..src_start + row_bytes]);
        }
        drop(mapped);
        target.readback.unmap();

        Ok(RgbaOverlay {
            width: target.width,
            height: target.height,
            pixels,
        })
    }

    fn ensure_target(&mut self, width: u32, height: u32) {
        let needs_new = self
            .target
            .as_ref()
            .is_none_or(|target| target.width != width || target.height != height);
        if !needs_new {
            return;
        }
        let bytes_per_row = align_to(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("haze-cgen-overlay-target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("haze-cgen-overlay-readback"),
            size: u64::from(bytes_per_row) * u64::from(height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.target = Some(WgpuTarget {
            width,
            height,
            bytes_per_row,
            texture,
            view,
            readback,
        });
    }

    fn ensure_text_texture(
        &mut self,
        text: &str,
        font_family: &str,
        font_size: u32,
        scale: i32,
    ) -> anyhow::Result<()> {
        let scale = scale.max(1);
        let needs_new = self.text_cache.as_ref().is_none_or(|cache| {
            cache.text != text
                || cache.font_family != font_family
                || cache.font_size != font_size
                || cache.scale != scale
                || cache.width == 0
                || cache.height == 0
        });
        if !needs_new {
            return Ok(());
        }
        let (width, height, pixels) = self
            .fonts
            .resolve(font_family)
            .and_then(|font| raster_text_rgba_system(&font, text, font_size))
            .unwrap_or_else(|| raster_text_rgba_bitmap(text, scale));
        if width == 0 || height == 0 {
            anyhow::bail!("cannot render empty cgen ticker text texture");
        }
        let max_dim = self.device.limits().max_texture_dimension_2d;
        if width > max_dim || height > max_dim {
            anyhow::bail!("cgen ticker text texture exceeds max GPU texture size");
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("haze-cgen-ticker-texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.text_cache = Some(WgpuTextCache {
            text: text.to_string(),
            font_family: font_family.to_string(),
            font_size,
            scale,
            width,
            height,
            _texture: texture,
            view,
        });
        Ok(())
    }

    fn bind_group(&self, view: &wgpu::TextureView) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("haze-cgen-overlay-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    fn vertex_buffer(&self, vertices: [GpuVertex; 6]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("haze-cgen-overlay-quad-vertices"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            })
    }
}

#[cfg(feature = "gpu-wgpu")]
struct WgpuTarget {
    width: u32,
    height: u32,
    bytes_per_row: u32,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
}

#[cfg(feature = "gpu-wgpu")]
struct WgpuTextCache {
    text: String,
    font_family: String,
    font_size: u32,
    scale: i32,
    width: u32,
    height: u32,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg(feature = "gpu-wgpu")]
struct SystemFontCache {
    db: fontdb::Database,
    fonts: HashMap<String, Option<Arc<Font>>>,
}

#[cfg(feature = "gpu-wgpu")]
impl SystemFontCache {
    fn new() -> Self {
        let mut db = fontdb::Database::new();
        db.load_system_fonts();
        Self {
            db,
            fonts: HashMap::new(),
        }
    }

    fn resolve(&mut self, family: &str) -> Option<Arc<Font>> {
        let key = family.trim().to_ascii_lowercase();
        if let Some(font) = self.fonts.get(&key) {
            return font.clone();
        }
        let font = self.load_font(family);
        self.fonts.insert(key, font.clone());
        font
    }

    fn load_font(&self, family: &str) -> Option<Arc<Font>> {
        let family = family.trim();
        let id = if family.is_empty() {
            None
        } else {
            self.db.query(&fontdb::Query {
                families: &[fontdb::Family::Name(family)],
                ..fontdb::Query::default()
            })
        }
        .or_else(|| {
            self.db.query(&fontdb::Query {
                families: &[fontdb::Family::SansSerif],
                ..fontdb::Query::default()
            })
        });
        id.and_then(|id| {
            self.db
                .with_face_data(id, |data, face_index| {
                    Font::from_bytes(
                        data.to_vec(),
                        FontSettings {
                            collection_index: face_index,
                            scale: 40.0,
                            load_substitutions: true,
                        },
                    )
                    .ok()
                })
                .flatten()
                .map(Arc::new)
        })
        .or_else(load_platform_arial)
    }
}

#[cfg(all(feature = "gpu-wgpu", target_os = "windows"))]
fn load_platform_arial() -> Option<Arc<Font>> {
    std::fs::read("C:/Windows/Fonts/arial.ttf")
        .ok()
        .and_then(|data| {
            Font::from_bytes(
                data,
                FontSettings {
                    scale: 40.0,
                    load_substitutions: true,
                    ..FontSettings::default()
                },
            )
            .ok()
        })
        .map(Arc::new)
}

#[cfg(all(feature = "gpu-wgpu", not(target_os = "windows")))]
fn load_platform_arial() -> Option<Arc<Font>> {
    None
}

#[cfg(feature = "gpu-wgpu")]
struct RgbaOverlay {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

#[cfg(feature = "gpu-wgpu")]
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
}

#[cfg(feature = "gpu-wgpu")]
impl GpuVertex {
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

#[cfg(feature = "gpu-wgpu")]
const TICKER_SHADER: &str = r#"
struct VertexIn {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(input.position, 0.0, 1.0);
    out.uv = input.uv;
    out.color = input.color;
    return out;
}

@group(0) @binding(0) var overlay_texture: texture_2d<f32>;
@group(0) @binding(1) var overlay_sampler: sampler;

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let sampled = textureSample(overlay_texture, overlay_sampler, input.uv);
    return vec4<f32>(input.color.rgb, input.color.a * sampled.a);
}
"#;

#[cfg(feature = "gpu-wgpu")]
fn align_to(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

#[cfg(feature = "gpu-wgpu")]
fn quad_vertices_px(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    surface_width: f32,
    surface_height: f32,
    color: [f32; 4],
) -> [GpuVertex; 6] {
    let x0 = px_to_ndc_x(x, surface_width);
    let x1 = px_to_ndc_x(x + width, surface_width);
    let y0 = px_to_ndc_y(y, surface_height);
    let y1 = px_to_ndc_y(y + height, surface_height);
    [
        GpuVertex {
            position: [x0, y0],
            uv: [0.0, 0.0],
            color,
        },
        GpuVertex {
            position: [x1, y0],
            uv: [1.0, 0.0],
            color,
        },
        GpuVertex {
            position: [x1, y1],
            uv: [1.0, 1.0],
            color,
        },
        GpuVertex {
            position: [x0, y0],
            uv: [0.0, 0.0],
            color,
        },
        GpuVertex {
            position: [x1, y1],
            uv: [1.0, 1.0],
            color,
        },
        GpuVertex {
            position: [x0, y1],
            uv: [0.0, 1.0],
            color,
        },
    ]
}

#[cfg(feature = "gpu-wgpu")]
fn quad_vertices_px_gradient(
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    surface_width: f32,
    surface_height: f32,
    top_color: [f32; 4],
    bottom_color: [f32; 4],
) -> [GpuVertex; 6] {
    let x0 = px_to_ndc_x(x, surface_width);
    let x1 = px_to_ndc_x(x + width, surface_width);
    let y0 = px_to_ndc_y(y, surface_height);
    let y1 = px_to_ndc_y(y + height, surface_height);
    [
        GpuVertex {
            position: [x0, y0],
            uv: [0.0, 0.0],
            color: top_color,
        },
        GpuVertex {
            position: [x1, y0],
            uv: [1.0, 0.0],
            color: top_color,
        },
        GpuVertex {
            position: [x1, y1],
            uv: [1.0, 1.0],
            color: bottom_color,
        },
        GpuVertex {
            position: [x0, y0],
            uv: [0.0, 0.0],
            color: top_color,
        },
        GpuVertex {
            position: [x1, y1],
            uv: [1.0, 1.0],
            color: bottom_color,
        },
        GpuVertex {
            position: [x0, y1],
            uv: [0.0, 1.0],
            color: bottom_color,
        },
    ]
}

#[cfg(feature = "gpu-wgpu")]
fn rgba(rgb: [f32; 3], alpha: f32) -> [f32; 4] {
    [rgb[0], rgb[1], rgb[2], alpha]
}

#[cfg(feature = "gpu-wgpu")]
fn px_to_ndc_x(value: f32, width: f32) -> f32 {
    (value / width.max(1.0)) * 2.0 - 1.0
}

#[cfg(feature = "gpu-wgpu")]
fn px_to_ndc_y(value: f32, height: f32) -> f32 {
    1.0 - (value / height.max(1.0)) * 2.0
}

#[cfg(feature = "gpu-wgpu")]
struct TextLayoutMetrics {
    width: u32,
    height: u32,
    offset_x: f32,
    offset_y: f32,
    glyphs: Vec<fontdue::layout::GlyphPosition>,
}

#[cfg(feature = "gpu-wgpu")]
fn layout_text(font: &Font, font_size: u32, text: &str) -> Option<TextLayoutMetrics> {
    let px = font_size.max(8) as f32;
    let mut layout = Layout::new(CoordinateSystem::PositiveYDown);
    layout.reset(&LayoutSettings {
        x: 0.0,
        y: 0.0,
        ..LayoutSettings::default()
    });
    layout.append(&[font], &TextStyle::new(text, px, 0));
    let glyphs = layout.glyphs();
    if glyphs.is_empty() {
        return None;
    }
    let min_x = glyphs
        .iter()
        .map(|glyph| glyph.x)
        .fold(f32::INFINITY, f32::min);
    let min_y = glyphs
        .iter()
        .map(|glyph| glyph.y)
        .fold(f32::INFINITY, f32::min);
    let max_x = glyphs
        .iter()
        .map(|glyph| glyph.x + glyph.width as f32)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = glyphs
        .iter()
        .map(|glyph| glyph.y + glyph.height as f32)
        .fold(f32::NEG_INFINITY, f32::max);
    let width = (max_x - min_x).ceil().max(1.0) as u32;
    let height = (max_y - min_y).ceil().max(1.0) as u32;
    Some(TextLayoutMetrics {
        width,
        height,
        offset_x: min_x,
        offset_y: min_y,
        glyphs: glyphs.clone(),
    })
}

#[cfg(feature = "gpu-wgpu")]
fn measure_text_width_with_cache(
    fonts: &mut SystemFontCache,
    family: &str,
    font_size: u32,
    text: &str,
) -> Option<i32> {
    let font = fonts.resolve(family)?;
    let metrics = layout_text(&font, font_size, text)?;
    Some(metrics.width.max(1) as i32)
}

#[cfg(feature = "gpu-wgpu")]
fn raster_text_rgba_system(font: &Font, text: &str, font_size: u32) -> Option<(u32, u32, Vec<u8>)> {
    let layout = layout_text(font, font_size, text)?;
    let mut pixels = vec![0; layout.width as usize * layout.height as usize * 4];
    for glyph in &layout.glyphs {
        let (_metrics, bitmap) = font.rasterize_config(glyph.key);
        let base_x = (glyph.x - layout.offset_x).round() as i32;
        let base_y = (glyph.y - layout.offset_y).round() as i32;
        for gy in 0..glyph.height as i32 {
            let y = base_y + gy;
            if y < 0 || y >= layout.height as i32 {
                continue;
            }
            for gx in 0..glyph.width as i32 {
                let x = base_x + gx;
                if x < 0 || x >= layout.width as i32 {
                    continue;
                }
                let alpha = bitmap
                    .get((gy as usize * glyph.width) + gx as usize)
                    .copied()
                    .unwrap_or_default();
                if alpha == 0 {
                    continue;
                }
                let offset = ((y as u32 * layout.width + x as u32) * 4) as usize;
                pixels[offset] = 255;
                pixels[offset + 1] = 255;
                pixels[offset + 2] = 255;
                pixels[offset + 3] = alpha;
            }
        }
    }
    Some((layout.width, layout.height, pixels))
}

#[cfg(feature = "gpu-wgpu")]
fn raster_text_rgba_bitmap(text: &str, scale: i32) -> (u32, u32, Vec<u8>) {
    let scale = scale.max(1);
    let width = text_width_px(text, scale).max(1) as u32;
    let height = glyph_height_px(scale).max(1) as u32;
    let mut pixels = vec![0; width as usize * height as usize * 4];
    let mut cursor = 0i32;
    for ch in text.chars() {
        if ch == ' ' {
            cursor = cursor.saturating_add(4 * scale);
            continue;
        }
        if let Some(glyph) = glyph_rows(ch) {
            for (row, bits) in glyph.iter().enumerate() {
                for col in 0..5 {
                    if bits & (1 << (4 - col)) == 0 {
                        continue;
                    }
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let x = cursor + col * scale + sx;
                            let y = row as i32 * scale + sy;
                            if x < 0 || y < 0 {
                                continue;
                            }
                            let x = x as u32;
                            let y = y as u32;
                            if x >= width || y >= height {
                                continue;
                            }
                            let offset = ((y * width + x) * 4) as usize;
                            pixels[offset] = 255;
                            pixels[offset + 1] = 255;
                            pixels[offset + 2] = 255;
                            pixels[offset + 3] = 255;
                        }
                    }
                }
            }
        }
        cursor = cursor.saturating_add(6 * scale);
    }
    (width, height, pixels)
}

#[cfg(feature = "gpu-wgpu")]
fn tint_rgba_overlay(pixels: &[u8], rgb: [u8; 3], alpha_scale: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len());
    for px in pixels.chunks_exact(4) {
        out.push(rgb[0]);
        out.push(rgb[1]);
        out.push(rgb[2]);
        let alpha = (u16::from(px[3]) * u16::from(alpha_scale)) / 255;
        out.push(alpha as u8);
    }
    out
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct TickerGradient {
    top: [f32; 3],
    middle: [f32; 3],
    bottom: [f32; 3],
}

fn ticker_font_family(feed: &FeedConfig) -> &str {
    non_empty_ref(&feed.banner.font)
        .or_else(|| non_empty_ref(&feed.graphics.font))
        .unwrap_or("Arial")
}

fn ticker_y(feed: &FeedConfig, frame_height: i32) -> i32 {
    if feed.banner.y != 0 {
        return feed.banner.y;
    }
    ((frame_height.max(1) as f32) * 0.08).round() as i32
}

fn ticker_text_x(frame_width: i32, _text_width: i32, frame_index: u64, scroll_speed: u32) -> i32 {
    let start_x = frame_width.saturating_add(1).max(1);
    let speed = scroll_speed.max(1) as i64;
    let scroll = (frame_index as i64).saturating_mul(speed);
    i64::from(start_x).saturating_sub(scroll) as i32
}

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

fn ticker_scroll_frames(frame_width: i32, text_width: i32, scroll_speed: u32) -> u64 {
    let start_x = frame_width.saturating_add(1).max(1);
    let travel = start_x.saturating_add(text_width.max(1)).saturating_add(1);
    let speed = scroll_speed.max(1);
    u64::from((travel as u32).div_ceil(speed))
}

fn ticker_state_key(banner: Option<&BannerPayload>, audio: Option<&PriorityAudio>) -> String {
    let queue_id = audio
        .map(|audio| audio.queue_id.as_str())
        .unwrap_or_default();
    let banner_signature = banner
        .map(|banner| banner.signature.as_str())
        .unwrap_or_default();
    if !queue_id.is_empty() {
        return queue_id.to_string();
    }
    banner_signature.to_string()
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
fn fill_rect_yuv_gradient(
    frame: &mut AVFrame,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    gradient: TickerGradient,
    fallback_yuv: (u8, u8, u8),
) {
    let x0 = x.clamp(0, frame.width);
    let y0 = y.clamp(0, frame.height);
    let x1 = x.saturating_add(w).clamp(0, frame.width);
    let y1 = y.saturating_add(h).clamp(0, frame.height);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let height = (y1 - y0).max(1);
    for row in 0..height {
        let t = if height <= 1 {
            0.0
        } else {
            row as f32 / (height - 1) as f32
        };
        let rgb = if t <= 0.5 {
            lerp_rgb(gradient.top, gradient.middle, t * 2.0)
        } else {
            lerp_rgb(gradient.middle, gradient.bottom, (t - 0.5) * 2.0)
        };
        let (yy, u, v) = yuv_from_rgb_units(rgb).unwrap_or(fallback_yuv);
        fill_rect_yuv(frame, x0, y0 + row, x1 - x0, 1, yy, u, v);
    }
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn lerp_rgb(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn yuv_from_rgb_units(rgb: [f32; 3]) -> Option<(u8, u8, u8)> {
    let r = (rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    Some(yuv_from_rgb(r, g, b))
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

#[cfg(feature = "ffmpeg-rsmpeg")]
fn draw_text_yuv_field(
    frame: &mut AVFrame,
    text: &str,
    x: i32,
    y: i32,
    scale: i32,
    yy: u8,
    field_parity: i32,
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
                        fill_luma_rect_field(
                            frame,
                            cursor + col * scale,
                            y + row as i32 * scale,
                            scale,
                            scale,
                            yy,
                            field_parity,
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

#[cfg(feature = "ffmpeg-rsmpeg")]
fn fill_luma_rect_field(
    frame: &mut AVFrame,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    value: u8,
    field_parity: i32,
) {
    let x0 = x.clamp(0, frame.width);
    let y0 = y.clamp(0, frame.height);
    let x1 = x.saturating_add(w).clamp(0, frame.width);
    let y1 = y.saturating_add(h).clamp(0, frame.height);
    if x1 <= x0 || y1 <= y0 || frame.data[0].is_null() {
        return;
    }
    let Ok(stride) = usize::try_from(frame.linesize[0]) else {
        return;
    };
    let Ok(start) = usize::try_from(x0) else {
        return;
    };
    let Ok(len) = usize::try_from(x1 - x0) else {
        return;
    };
    let parity = field_parity.rem_euclid(2);
    for row in y0..y1 {
        if row.rem_euclid(2) != parity {
            continue;
        }
        let Ok(row) = usize::try_from(row) else {
            continue;
        };
        // SAFETY: x/y bounds are clamped to the active luma plane and the
        // frame owns a positive stride for this writable AVFrame.
        unsafe {
            std::ptr::write_bytes(frame.data[0].add(row * stride + start), value, len);
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

#[cfg(all(feature = "ffmpeg-rsmpeg", feature = "gpu-wgpu"))]
fn blend_rgba_overlay(
    frame: &mut AVFrame,
    pixels: &[u8],
    overlay_width: u32,
    overlay_height: u32,
    dst_x: i32,
    dst_y: i32,
) {
    if pixels.is_empty() || overlay_width == 0 || overlay_height == 0 || frame.data[0].is_null() {
        return;
    }
    let Ok(y_stride) = usize::try_from(frame.linesize[0]) else {
        return;
    };
    let u_stride = usize::try_from(frame.linesize[1]).unwrap_or_default();
    let v_stride = usize::try_from(frame.linesize[2]).unwrap_or_default();
    let width = frame.width.max(0);
    let height = frame.height.max(0);
    for oy in 0..overlay_height as i32 {
        let fy = dst_y + oy;
        if fy < 0 || fy >= height {
            continue;
        }
        for ox in 0..overlay_width as i32 {
            let fx = dst_x + ox;
            if fx < 0 || fx >= width {
                continue;
            }
            let offset = ((oy as u32 * overlay_width + ox as u32) * 4) as usize;
            let alpha = pixels.get(offset + 3).copied().unwrap_or_default();
            if alpha == 0 {
                continue;
            }
            let r = pixels[offset];
            let g = pixels[offset + 1];
            let b = pixels[offset + 2];
            let (yy, u, v) = yuv_from_rgb(r, g, b);
            let Ok(fy_usize) = usize::try_from(fy) else {
                continue;
            };
            let Ok(fx_usize) = usize::try_from(fx) else {
                continue;
            };
            // SAFETY: `fx`/`fy` are clamped to the frame bounds, and the
            // destination row is within the positive stride validated above.
            unsafe {
                let y_ptr = frame.data[0].add(fy_usize * y_stride + fx_usize);
                *y_ptr = blend_u8(*y_ptr, yy, alpha);
            }
            match frame.format {
                ffi::AV_PIX_FMT_YUV420P => {
                    if frame.data[1].is_null() || frame.data[2].is_null() {
                        continue;
                    }
                    let cx = fx_usize / 2;
                    let cy = fy_usize / 2;
                    // SAFETY: chroma coordinates are subsampled from in-bounds
                    // luma coordinates and line sizes were converted above.
                    unsafe {
                        let u_ptr = frame.data[1].add(cy * u_stride + cx);
                        let v_ptr = frame.data[2].add(cy * v_stride + cx);
                        *u_ptr = blend_u8(*u_ptr, u, alpha);
                        *v_ptr = blend_u8(*v_ptr, v, alpha);
                    }
                }
                ffi::AV_PIX_FMT_NV12 => {
                    if frame.data[1].is_null() {
                        continue;
                    }
                    let cx = (fx_usize / 2) * 2;
                    let cy = fy_usize / 2;
                    // SAFETY: NV12 chroma coordinates are derived from
                    // in-bounds luma coordinates. `cx` is the UV pair offset.
                    unsafe {
                        let uv_ptr = frame.data[1].add(cy * u_stride + cx);
                        *uv_ptr = blend_u8(*uv_ptr, u, alpha);
                        *uv_ptr.add(1) = blend_u8(*uv_ptr.add(1), v, alpha);
                    }
                }
                ffi::AV_PIX_FMT_YUV422P => {
                    if frame.data[1].is_null() || frame.data[2].is_null() {
                        continue;
                    }
                    let cx = fx_usize / 2;
                    // SAFETY: 4:2:2 chroma rows match luma rows; x is
                    // subsampled from an in-bounds luma coordinate.
                    unsafe {
                        let u_ptr = frame.data[1].add(fy_usize * u_stride + cx);
                        let v_ptr = frame.data[2].add(fy_usize * v_stride + cx);
                        *u_ptr = blend_u8(*u_ptr, u, alpha);
                        *v_ptr = blend_u8(*v_ptr, v, alpha);
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(all(feature = "ffmpeg-rsmpeg", feature = "gpu-wgpu"))]
fn blend_rgba_luma_overlay_field(
    frame: &mut AVFrame,
    pixels: &[u8],
    overlay_width: u32,
    overlay_height: u32,
    dst_x: i32,
    dst_y: i32,
    luma: u8,
    field_parity: i32,
) {
    if pixels.is_empty() || overlay_width == 0 || overlay_height == 0 || frame.data[0].is_null() {
        return;
    }
    let Ok(y_stride) = usize::try_from(frame.linesize[0]) else {
        return;
    };
    let width = frame.width.max(0);
    let height = frame.height.max(0);
    let parity = field_parity.rem_euclid(2);
    for oy in 0..overlay_height as i32 {
        let fy = dst_y + oy;
        if fy < 0 || fy >= height || fy.rem_euclid(2) != parity {
            continue;
        }
        for ox in 0..overlay_width as i32 {
            let fx = dst_x + ox;
            if fx < 0 || fx >= width {
                continue;
            }
            let offset = ((oy as u32 * overlay_width + ox as u32) * 4) as usize;
            let alpha = pixels.get(offset + 3).copied().unwrap_or_default();
            if alpha == 0 {
                continue;
            }
            let Ok(fy_usize) = usize::try_from(fy) else {
                continue;
            };
            let Ok(fx_usize) = usize::try_from(fx) else {
                continue;
            };
            // SAFETY: luma coordinates are clamped to the active frame and
            // the destination row is within the validated positive stride.
            unsafe {
                let y_ptr = frame.data[0].add(fy_usize * y_stride + fx_usize);
                *y_ptr = blend_u8(*y_ptr, luma, alpha);
            }
        }
    }
}

#[cfg(all(feature = "ffmpeg-rsmpeg", feature = "gpu-wgpu"))]
fn blend_u8(dst: u8, src: u8, alpha: u8) -> u8 {
    let alpha = u16::from(alpha);
    let inv = 255u16.saturating_sub(alpha);
    ((u16::from(dst) * inv + u16::from(src) * alpha + 127) / 255) as u8
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

#[cfg(feature = "gpu-wgpu")]
fn rgb_from_hex(value: &str) -> [f32; 3] {
    let value = value.trim().trim_start_matches('#');
    if value.len() != 6 {
        return [0.5, 0.5, 0.5];
    }
    let r = u8::from_str_radix(&value[0..2], 16).unwrap_or(128) as f32 / 255.0;
    let g = u8::from_str_radix(&value[2..4], 16).unwrap_or(128) as f32 / 255.0;
    let b = u8::from_str_radix(&value[4..6], 16).unwrap_or(128) as f32 / 255.0;
    [r, g, b]
}

#[cfg(feature = "ffmpeg-rsmpeg")]
fn yuv_from_rgb(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = f32::from(r);
    let g = f32::from(g);
    let b = f32::from(b);
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
        AudioConfig, BannerConfig, ClockConfig, EndpointConfig, GraphicsConfig, LadderConfig,
        PriorityInputConfig, StateConfig, SyncConfig, TextConfig, VideoConfig,
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

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn raster_text_rgba_bitmap_emits_alpha_glyphs() {
        let (width, height, pixels) = raster_text_rgba_bitmap("A", 2);
        assert_eq!(height, 14);
        assert!(width >= 12);
        assert_eq!(pixels.len(), width as usize * height as usize * 4);
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] == 255));
    }

    #[test]
    fn ticker_scroll_starts_off_right_and_exits_left() {
        assert_eq!(ticker_text_x(1920, 320, 0, 4), 1921);
        let exit_frame = (1921 + 320) / 4;
        assert!(ticker_text_x(1920, 320, exit_frame as u64, 4) <= -319);
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
        let mut feed = test_feed();
        feed.banner.y = 0;
        assert_eq!(ticker_y(&feed, 1080), 86);
        feed.banner.y = 32;
        assert_eq!(ticker_y(&feed, 1080), 32);
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

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn quad_vertices_map_pixels_to_ndc() {
        let vertices = quad_vertices_px(0.0, 0.0, 100.0, 50.0, 200.0, 100.0, [1.0; 4]);
        assert_eq!(vertices[0].position, [-1.0, 1.0]);
        assert_eq!(vertices[2].position, [0.0, 0.0]);
        assert_eq!(vertices[2].uv, [1.0, 1.0]);
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
            ladder: LadderConfig::default(),
            banner: BannerConfig::default(),
            graphics: GraphicsConfig::default(),
            clock: ClockConfig::default(),
            text: TextConfig::default(),
            state: StateConfig::default(),
            sync: SyncConfig::default(),
        }
    }
}
