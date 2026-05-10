// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use bytemuck::{Pod, Zeroable};
use iced::widget::shader::{self, Storage, Viewport, wgpu};
use iced::{Rectangle, mouse};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

// ──────────────────────────── Constants ────────────────────────────

const MAX_SCOPE_SAMPLES: usize = 4096;
pub const NUM_SPECTRO_BINS: usize = 1024;
pub const DEFAULT_SPECTRO_COLS: usize = 2048;
pub const MIN_SPECTRO_COLS: usize = 128;
pub const MAX_SPECTRO_COLS: usize = 8192;
const FFT_SIZE: usize = 4096;
const DB_FLOOR: f32 = -80.0;
const RESIZE_SETTLE: Duration = Duration::from_millis(150);

// ──────────────────────────── WGSL: Oscilloscope ────────────────────────────

const SCOPE_WGSL: &str = r#"
struct Uniforms {
    width: f32,
    height: f32,
    num_samples: u32,
    gain: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> samples: array<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// sRGB → linear (see SPECTRO_WGSL for context).
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= cutoff);
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = select(-1.0, 3.0, idx == 1u);
    let y = select(-1.0, 3.0, idx == 2u);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = in.uv.x * u.width;
    let py = in.uv.y * u.height;
    let half_w = u.width * 0.5;
    let cy = u.height * 0.5;
    let amp = cy * 0.82;

    // Which half: left channel = left half, right channel = right half
    let is_right = px >= half_w;
    let ch_left = select(0.0, half_w + 1.0, is_right);
    let ch_right = select(half_w - 1.0, u.width, is_right);
    let ch_w = ch_right - ch_left;

    // Background: panel surface #2B2B40 — same as the rest of the panel chrome.
    var color = srgb_to_linear(vec3<f32>(0.16863, 0.16863, 0.25098));

    // Trace color: accent green #00E89C — unified across both channels;
    // the L/R split is communicated by the divider, not by hue.
    let wave_color = srgb_to_linear(vec3<f32>(0.0, 0.90980, 0.61176));

    // Subtle vertical divider between L/R: a half-step lighter than the surface.
    let divider = srgb_to_linear(vec3<f32>(0.25098, 0.25098, 0.41961)); // #40406B
    let dd = abs(px - half_w);
    if dd < 0.75 { color = mix(color, divider, 0.6 * (1.0 - dd / 0.75)); }

    // No samples: draw center line + flat waveform trace and return
    if u.num_samples < 2u {
        let cd = abs(py - cy);
        if cd < 0.5 { color = mix(color, divider, 0.5 * (1.0 - cd / 0.5)); }
        let flat_alpha = 1.0 - smoothstep(0.0, 1.5, cd);
        color = mix(color, wave_color * 0.5, flat_alpha);
        return vec4<f32>(color, 0.75);
    }

    // Sample position within this channel's half
    let t_x = clamp((px - ch_left) / ch_w, 0.0, 1.0);
    let sample_pos = t_x * f32(u.num_samples - 1u);
    let idx0 = u32(sample_pos);
    let idx1 = min(idx0 + 1u, u.num_samples - 1u);
    let frac = sample_pos - f32(idx0);

    // Sample the appropriate channel
    let ch_offset = select(0u, u.num_samples, is_right);
    let s0 = clamp(samples[ch_offset + idx0] * u.gain, -1.0, 1.0);
    let s1 = clamp(samples[ch_offset + idx1] * u.gain, -1.0, 1.0);
    let sv = mix(s0, s1, frac);
    let s_y = cy - sv * amp;
    let s_lo = min(s0, s1);
    let s_hi = max(s0, s1);
    let s_y_lo = cy - s_hi * amp;
    let s_y_hi = cy - s_lo * amp;
    let sd = select(abs(py - s_y), 0.0, py >= s_y_lo && py <= s_y_hi);
    let s_alpha = 1.0 - smoothstep(0.0, 1.5, sd);
    color = mix(color, wave_color, s_alpha);

    return vec4<f32>(color, 0.75);
}
"#;

// ──────────────────────────── WGSL: Spectrogram ────────────────────────────

const SPECTRO_WGSL: &str = r#"
struct Uniforms {
    width: f32,
    height: f32,
    num_bins: u32,
    num_columns: u32,
    write_index: u32,
    sample_rate: f32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> magnitudes: array<f32>;

// Max half-width of the per-axis Gaussian tap grid. Total taps per axis = 2*NMAX_HALF + 1 = 17;
// 2D worst case = 17*17 = 289 taps/fragment before stride widening kicks in.
const NMAX_HALF: i32 = 8;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = select(-1.0, 3.0, idx == 1u);
    let y = select(-1.0, 3.0, idx == 2u);
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// sRGB-encoded color (0..1) → linear-light, channel-wise. iced 0.13 selects
// an sRGB framebuffer; shader writes are encoded on store, so designer hex
// values must be converted before they're written. Without this, every
// palette stop displays much lighter than the spec.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= cutoff);
}

// Heatmap derived from the app palette so it reads native to the UI:
//   silence (panel bg #2B2B40) → mid-low (#40406B) → mid (#5C5C8A)
//   → loud (accent green #00E89C) → clipping/danger (amber #FFD33C).
// Four equal segments across t ∈ [0, 1].
fn heatmap(t: f32) -> vec3<f32> {
    let c0 = srgb_to_linear(vec3(0.16863, 0.16863, 0.25098)); // #2B2B40
    let c1 = srgb_to_linear(vec3(0.25098, 0.25098, 0.41961)); // #40406B
    let c2 = srgb_to_linear(vec3(0.36078, 0.36078, 0.54118)); // #5C5C8A
    let c3 = srgb_to_linear(vec3(0.00000, 0.90980, 0.61176)); // #00E89C
    let c4 = srgb_to_linear(vec3(1.00000, 0.82745, 0.23529)); // #FFD33C
    if t < 0.25 {
        return mix(c0, c1, t / 0.25);
    } else if t < 0.5 {
        return mix(c1, c2, (t - 0.25) / 0.25);
    } else if t < 0.75 {
        return mix(c2, c3, (t - 0.5) / 0.25);
    } else {
        return mix(c3, c4, (t - 0.75) / 0.25);
    }
}

// Lift quieter bins so the display shows more than only the loudest energy.
fn contrast_curve(t: f32) -> f32 {
    let x = clamp(t, 0.0, 1.0);
    let lifted = pow(x, 0.5);
    let shaped = smoothstep(0.0, 0.95, lifted);
    return mix(x, shaped, 0.9);
}

// Screen-pixel position → continuous base-texel coords. y-axis is log-frequency.
fn screen_to_texel(px: f32, py: f32) -> vec2<f32> {
    let fx = px / u.width * f32(u.num_columns) - 0.5;
    let min_freq = 50.0;
    let nyquist = u.sample_rate * 0.5;
    let t = 1.0 - py / u.height;
    let freq = min_freq * pow(nyquist / min_freq, t);
    let fy = freq * f32(u.num_bins) / nyquist - 0.5;
    return vec2<f32>(fx, fy);
}

// Nearest-tap base-buffer read: edge-clamp in (col, bin), circular wrap in col via write_index.
fn sample_base_nearest(col_tex: i32, bin_tex: i32) -> f32 {
    let cols_i = i32(u.num_columns);
    let bins_i = i32(u.num_bins);
    let col = clamp(col_tex, 0, cols_i - 1);
    let bin = clamp(bin_tex, 0, bins_i - 1);
    let c = (u.write_index + u32(col)) % u.num_columns;
    return magnitudes[c * u.num_bins + u32(bin)];
}

// Bilinear fast path for the magnification case (≤1 tap / pixel per axis).
fn sample_base_bilinear(fx_base: f32, fy_base: f32) -> f32 {
    let col0 = i32(floor(fx_base));
    let bin0 = i32(floor(fy_base));
    let fxf = clamp(fx_base - floor(fx_base), 0.0, 1.0);
    let fyf = clamp(fy_base - floor(fy_base), 0.0, 1.0);
    let m00 = sample_base_nearest(col0,     bin0);
    let m10 = sample_base_nearest(col0 + 1, bin0);
    let m01 = sample_base_nearest(col0,     bin0 + 1);
    let m11 = sample_base_nearest(col0 + 1, bin0 + 1);
    return mix(mix(m00, m10, fxf), mix(m01, m11, fxf), fyf);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if u.num_columns == 0u || u.num_bins == 0u {
        // Panel background #2B2B40 — silence reads as the surrounding chrome.
        let bg = srgb_to_linear(vec3(0.16863, 0.16863, 0.25098));
        return vec4<f32>(bg, 0.75);
    }

    let px = in.uv.x * u.width;
    let py = in.uv.y * u.height;
    let coords = screen_to_texel(px, py);

    // Footprint half-radius per axis in texel space.
    let dcx = dpdx(coords);
    let dcy = dpdy(coords);
    let fx = 0.5 * (abs(dcx.x) + abs(dcy.x));
    let fy = 0.5 * (abs(dcx.y) + abs(dcy.y));

    var mag: f32;
    if fx <= 0.5 && fy <= 0.5 {
        mag = sample_base_bilinear(coords.x, coords.y);
    } else {
        let sigma_x = max(fx, 0.5);
        let sigma_y = max(fy, 0.5);

        let want_nx = i32(ceil(3.0 * fx));
        let want_ny = i32(ceil(3.0 * fy));
        let nxh = min(want_nx, NMAX_HALF);
        let nyh = min(want_ny, NMAX_HALF);
        let stride_x = max(1, (want_nx + NMAX_HALF - 1) / NMAX_HALF);
        let stride_y = max(1, (want_ny + NMAX_HALF - 1) / NMAX_HALF);

        let cx = i32(floor(coords.x + 0.5));
        let cy = i32(floor(coords.y + 0.5));

        let inv_2sx2 = 1.0 / (2.0 * sigma_x * sigma_x);
        let inv_2sy2 = 1.0 / (2.0 * sigma_y * sigma_y);

        // Tabulate x-axis Gaussian weights once (j-invariant). Indexed as [i + NMAX_HALF].
        var wx_tab: array<f32, 17>;
        for (var k: i32 = -nxh; k <= nxh; k = k + 1) {
            let ox = f32(k * stride_x);
            wx_tab[k + NMAX_HALF] = exp(-ox * ox * inv_2sx2);
        }

        var accum: f32 = 0.0;
        var wsum: f32 = 0.0;
        for (var j: i32 = -nyh; j <= nyh; j = j + 1) {
            let oy = f32(j * stride_y);
            let wy = exp(-oy * oy * inv_2sy2);
            for (var i: i32 = -nxh; i <= nxh; i = i + 1) {
                let wx = wx_tab[i + NMAX_HALF];
                let w = wx * wy;
                let v = sample_base_nearest(cx + i * stride_x, cy + j * stride_y);
                accum = accum + v * w;
                wsum = wsum + w;
            }
        }
        mag = accum / max(wsum, 1e-20);
    }

    let color = heatmap(contrast_curve(mag));
    return vec4<f32>(color, 0.75);
}
"#;

// ──────────────────────────── Oscilloscope types ────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ScopeUniforms {
    width: f32,
    height: f32,
    num_samples: u32,
    gain: f32,
}

#[derive(Debug)]
struct ScopePipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct ScopePrimitive {
    uniforms: ScopeUniforms,
    samples: Vec<f32>,
}

impl shader::Primitive for ScopePrimitive {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        storage: &mut Storage,
        _bounds: &Rectangle,
        _viewport: &Viewport,
    ) {
        if !storage.has::<ScopePipeline>() {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scope_shader"),
                source: wgpu::ShaderSource::Wgsl(SCOPE_WGSL.into()),
            });

            let storage_buf_size = (MAX_SCOPE_SAMPLES * 2 * 4) as u64;
            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scope_uniforms"),
                size: std::mem::size_of::<ScopeUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("scope_samples"),
                size: storage_buf_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scope_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scope_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: storage_buffer.as_entire_binding(),
                    },
                ],
            });

            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scope_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("scope_pipeline"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            storage.store(ScopePipeline {
                pipeline,
                uniform_buffer,
                storage_buffer,
                bind_group,
            });
        }

        let pl = storage.get::<ScopePipeline>().unwrap();
        queue.write_buffer(&pl.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        if !self.samples.is_empty() {
            queue.write_buffer(&pl.storage_buffer, 0, bytemuck::cast_slice(&self.samples));
        }
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        storage: &Storage,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let pl = storage.get::<ScopePipeline>().unwrap();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scope_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_viewport(
            clip_bounds.x as f32,
            clip_bounds.y as f32,
            clip_bounds.width as f32,
            clip_bounds.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pass.set_pipeline(&pl.pipeline);
        pass.set_bind_group(0, &pl.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[derive(Debug)]
pub struct ScopeProgram {
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub gain: f32,
}

impl<Message> shader::Program<Message> for ScopeProgram {
    type State = ();
    type Primitive = ScopePrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        let n = self.left.len().min(MAX_SCOPE_SAMPLES);
        let mut samples = Vec::with_capacity(n * 2);
        samples.extend_from_slice(&self.left[..n]);
        samples.extend_from_slice(&self.right[..n.min(self.right.len())]);
        samples.resize(n * 2, 0.0);

        ScopePrimitive {
            uniforms: ScopeUniforms {
                width: bounds.width,
                height: bounds.height,
                num_samples: n as u32,
                gain: self.gain,
            },
            samples,
        }
    }
}

// ──────────────────────────── Spectrogram types ────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SpectroUniforms {
    width: f32,
    height: f32,
    num_bins: u32,
    num_columns: u32,
    write_index: u32,
    sample_rate: f32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Debug)]
struct SpectroPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    bgl: wgpu::BindGroupLayout,
    storage_capacity_cols: usize,
}

fn build_spectro_storage_and_bg(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    cols: usize,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let storage_buf_size = (cols * NUM_SPECTRO_BINS * 4) as u64;
    let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("spectro_magnitudes"),
        size: storage_buf_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("spectro_bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage_buffer.as_entire_binding(),
            },
        ],
    });
    (storage_buffer, bind_group)
}

#[derive(Debug, Clone)]
pub struct DirtySpan {
    pub start_col: usize,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub struct SpectroPrimitive {
    uniforms: SpectroUniforms,
    dirty: Vec<DirtySpan>,
    num_cols: usize,
    full_upload: Option<Vec<f32>>,
    observed_width_px: Arc<AtomicU32>,
}

impl shader::Primitive for SpectroPrimitive {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        storage: &mut Storage,
        bounds: &Rectangle,
        viewport: &Viewport,
    ) {
        let physical = (bounds.width * viewport.scale_factor() as f32)
            .round()
            .max(0.0) as u32;
        self.observed_width_px.store(physical, Ordering::Relaxed);

        if !storage.has::<SpectroPipeline>() {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("spectro_shader"),
                source: wgpu::ShaderSource::Wgsl(SPECTRO_WGSL.into()),
            });

            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("spectro_uniforms"),
                size: std::mem::size_of::<SpectroUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("spectro_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let (storage_buffer, bind_group) =
                build_spectro_storage_and_bg(device, &bgl, &uniform_buffer, self.num_cols);

            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("spectro_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("spectro_pipeline"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

            storage.store(SpectroPipeline {
                pipeline,
                uniform_buffer,
                storage_buffer,
                bind_group,
                bgl,
                storage_capacity_cols: self.num_cols,
            });
        } else {
            let pl = storage.get_mut::<SpectroPipeline>().unwrap();
            if pl.storage_capacity_cols != self.num_cols {
                let (new_storage, new_bg) = build_spectro_storage_and_bg(
                    device,
                    &pl.bgl,
                    &pl.uniform_buffer,
                    self.num_cols,
                );
                pl.storage_buffer = new_storage;
                pl.bind_group = new_bg;
                pl.storage_capacity_cols = self.num_cols;
            }
        }

        let pl = storage.get::<SpectroPipeline>().unwrap();
        queue.write_buffer(&pl.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        if let Some(full) = &self.full_upload {
            queue.write_buffer(&pl.storage_buffer, 0, bytemuck::cast_slice(full));
        } else {
            for span in &self.dirty {
                let offset =
                    (span.start_col * NUM_SPECTRO_BINS * std::mem::size_of::<f32>()) as u64;
                queue.write_buffer(&pl.storage_buffer, offset, bytemuck::cast_slice(&span.data));
            }
        }
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        storage: &Storage,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let pl = storage.get::<SpectroPipeline>().unwrap();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("spectro_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_viewport(
            clip_bounds.x as f32,
            clip_bounds.y as f32,
            clip_bounds.width as f32,
            clip_bounds.height as f32,
            0.0,
            1.0,
        );
        pass.set_scissor_rect(
            clip_bounds.x,
            clip_bounds.y,
            clip_bounds.width,
            clip_bounds.height,
        );
        pass.set_pipeline(&pl.pipeline);
        pass.set_bind_group(0, &pl.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[derive(Debug)]
pub struct SpectroProgram {
    pub dirty: Vec<DirtySpan>,
    pub write_index: usize,
    pub sample_rate: f32,
    pub full_upload: Option<Vec<f32>>,
    pub num_cols: usize,
    pub observed_width_px: Arc<AtomicU32>,
}

impl<Message> shader::Program<Message> for SpectroProgram {
    type State = ();
    type Primitive = SpectroPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        SpectroPrimitive {
            uniforms: SpectroUniforms {
                width: bounds.width,
                height: bounds.height,
                num_bins: NUM_SPECTRO_BINS as u32,
                num_columns: self.num_cols as u32,
                write_index: self.write_index as u32,
                sample_rate: self.sample_rate,
                _pad0: 0,
                _pad1: 0,
            },
            dirty: self.dirty.clone(),
            num_cols: self.num_cols,
            full_upload: self.full_upload.clone(),
            observed_width_px: Arc::clone(&self.observed_width_px),
        }
    }
}

// ──────────────────────────── Spectrogram State ────────────────────────────

pub struct SpectrogramState {
    magnitudes: Vec<f32>,
    write_index: usize,
    num_cols: usize,
    observed_width_px: Arc<AtomicU32>,
    pending_resize: Option<(usize, Instant)>,
    full_upload_needed: Cell<bool>,
    dirty_count: Cell<usize>,
    planner: FftPlanner<f64>,
    window: Vec<f64>,
}

impl SpectrogramState {
    pub fn new() -> Self {
        let window: Vec<f64> = (0..FFT_SIZE)
            .map(|i| {
                let phase = 2.0 * std::f64::consts::PI * i as f64 / (FFT_SIZE - 1) as f64;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        Self {
            magnitudes: vec![0.0; DEFAULT_SPECTRO_COLS * NUM_SPECTRO_BINS],
            write_index: 0,
            num_cols: DEFAULT_SPECTRO_COLS,
            observed_width_px: Arc::new(AtomicU32::new(0)),
            pending_resize: None,
            full_upload_needed: Cell::new(true),
            dirty_count: Cell::new(0),
            planner: FftPlanner::new(),
            window,
        }
    }

    pub fn clear(&mut self) {
        self.magnitudes.fill(0.0);
        self.write_index = 0;
        self.full_upload_needed.set(true);
        self.dirty_count.set(0);
    }

    pub fn push_fft(&mut self, waveform: &[f64]) {
        let frames = waveform.len() / 2;
        if frames < 4 {
            return;
        }

        let n = frames.min(FFT_SIZE);
        let fft = self.planner.plan_fft_forward(FFT_SIZE);
        let mut buf: Vec<Complex<f64>> = (0..FFT_SIZE)
            .map(|i| {
                let sample = if i < n {
                    let base = i * 2;
                    let l = waveform[base];
                    let r = waveform.get(base + 1).copied().unwrap_or(l);
                    (l + r) * 0.5
                } else {
                    0.0
                };
                Complex::new(sample * self.window[i], 0.0)
            })
            .collect();

        fft.process(&mut buf);

        let col_start = self.write_index * NUM_SPECTRO_BINS;
        let fft_bins = FFT_SIZE / 2;
        let downsample = fft_bins / NUM_SPECTRO_BINS; // 2048 / 1024 = 2
        for bin in 0..NUM_SPECTRO_BINS {
            // Take max magnitude across each group of `downsample` FFT bins
            let mut peak = 0.0f64;
            for j in 0..downsample {
                let mag = buf[bin * downsample + j].norm() * 2.0 / FFT_SIZE as f64;
                if mag > peak {
                    peak = mag;
                }
            }
            let db = if peak > 1e-12 {
                20.0 * peak.log10()
            } else {
                DB_FLOOR as f64
            };
            let normalized = ((db as f32 - DB_FLOOR) / -DB_FLOOR).clamp(0.0, 1.0);
            self.magnitudes[col_start + bin] = normalized;
        }

        self.write_index = (self.write_index + 1) % self.num_cols;
        self.dirty_count
            .set((self.dirty_count.get() + 1).min(self.num_cols));
    }

    pub fn magnitudes(&self) -> &[f32] {
        &self.magnitudes
    }

    pub fn write_index(&self) -> usize {
        self.write_index
    }

    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    pub fn observed_width_handle(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.observed_width_px)
    }

    /// Check the observed viewport width. If it's settled on a new value for
    /// `RESIZE_SETTLE`, return the new column count; otherwise return `None`.
    pub fn poll_desired_cols(&mut self, now: Instant) -> Option<usize> {
        let px = self.observed_width_px.load(Ordering::Relaxed) as usize;
        if px == 0 {
            return None;
        }
        let desired = px.clamp(MIN_SPECTRO_COLS, MAX_SPECTRO_COLS);
        if desired == self.num_cols {
            self.pending_resize = None;
            return None;
        }
        match self.pending_resize {
            Some((target, start)) if target == desired => {
                if now.duration_since(start) >= RESIZE_SETTLE {
                    self.pending_resize = None;
                    Some(desired)
                } else {
                    None
                }
            }
            _ => {
                self.pending_resize = Some((desired, now));
                None
            }
        }
    }

    /// Resize the ring buffer to `new_cols`, preserving the most recent history.
    pub fn resize(&mut self, new_cols: usize) {
        if new_cols == self.num_cols || new_cols == 0 {
            return;
        }
        let old_cols = self.num_cols;
        let keep = old_cols.min(new_cols);
        let mut new_mag = vec![0.0f32; new_cols * NUM_SPECTRO_BINS];
        // Place the most recent `keep` columns at [0..keep), oldest-first.
        for i in 0..keep {
            let src_col = (self.write_index + old_cols - keep + i) % old_cols;
            let src = src_col * NUM_SPECTRO_BINS;
            let dst = i * NUM_SPECTRO_BINS;
            new_mag[dst..dst + NUM_SPECTRO_BINS]
                .copy_from_slice(&self.magnitudes[src..src + NUM_SPECTRO_BINS]);
        }
        self.magnitudes = new_mag;
        self.num_cols = new_cols;
        self.write_index = keep % new_cols;
        self.full_upload_needed.set(true);
        self.dirty_count.set(0);
    }

    #[cfg(test)]
    fn latest_column_index(&self) -> usize {
        if self.write_index == 0 {
            self.num_cols - 1
        } else {
            self.write_index - 1
        }
    }

    #[cfg(test)]
    fn latest_column(&self) -> &[f32] {
        let col = self.latest_column_index();
        let start = col * NUM_SPECTRO_BINS;
        &self.magnitudes[start..start + NUM_SPECTRO_BINS]
    }

    /// Returns `true` (and clears the flag) if the full buffer needs uploading.
    pub fn take_full_upload_needed(&self) -> bool {
        self.full_upload_needed.replace(false)
    }

    /// Drain the pending-column counter and return up to two contiguous spans
    /// covering every column written since the last drain. If the backlog grew
    /// to or beyond `num_cols`, promotes to a full upload instead and returns
    /// empty.
    pub fn take_dirty_spans(&self) -> Vec<DirtySpan> {
        let count = self.dirty_count.replace(0);
        if count == 0 {
            return Vec::new();
        }
        if count >= self.num_cols {
            // Backlog outran the ring — can't identify which columns are stale.
            // Arm full_upload_needed; the next frame consumes it and uploads
            // everything. This frame uploads nothing, one-frame catchup lag.
            self.full_upload_needed.set(true);
            return Vec::new();
        }
        let num = self.num_cols;
        let bins = NUM_SPECTRO_BINS;
        let start = (self.write_index + num - count) % num;
        if start + count <= num {
            let off = start * bins;
            let data = self.magnitudes[off..off + count * bins].to_vec();
            vec![DirtySpan {
                start_col: start,
                data,
            }]
        } else {
            let first = num - start;
            let second = count - first;
            vec![
                DirtySpan {
                    start_col: start,
                    data: self.magnitudes[start * bins..num * bins].to_vec(),
                },
                DirtySpan {
                    start_col: 0,
                    data: self.magnitudes[0..second * bins].to_vec(),
                },
            ]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_waveform(freq_hz: f64, frames: usize) -> Vec<f64> {
        (0..frames)
            .flat_map(|i| {
                let s = (i as f64 * freq_hz * 2.0 * std::f64::consts::PI / 48_000.0).sin() * 0.8;
                [s, s]
            })
            .collect()
    }

    #[test]
    fn resize_preserves_latest_column() {
        let mut state = SpectrogramState::new();
        assert_eq!(state.num_cols(), DEFAULT_SPECTRO_COLS);

        // Fill enough columns that resize can never legitimately drop the
        // most-recent one. Use distinct frequencies so latest_column is unique.
        for i in 0..600 {
            let wave = test_waveform(220.0 + i as f64 * 5.0, 2048);
            state.push_fft(&wave);
        }
        assert_eq!(state.write_index(), 600);
        let latest_before = state.latest_column().to_vec();
        assert!(latest_before.iter().any(|&v| v > 0.0));

        // Shrink: most recent column must survive (512 < 600, so keep window
        // still ends at the most-recent push).
        state.resize(512);
        assert_eq!(state.num_cols(), 512);
        assert_eq!(state.latest_column(), latest_before.as_slice());

        // Grow: preserved data persists, capacity expands.
        state.resize(3000);
        assert_eq!(state.num_cols(), 3000);
        assert_eq!(state.latest_column(), latest_before.as_slice());
        assert_eq!(
            state.magnitudes().len(),
            3000 * NUM_SPECTRO_BINS,
            "backing Vec must match new capacity"
        );
    }

    #[test]
    fn resize_no_op_when_unchanged() {
        let mut state = SpectrogramState::new();
        let wave = test_waveform(440.0, 2048);
        state.push_fft(&wave);
        let before = state.latest_column().to_vec();
        state.resize(DEFAULT_SPECTRO_COLS);
        assert_eq!(state.num_cols(), DEFAULT_SPECTRO_COLS);
        assert_eq!(state.latest_column(), before.as_slice());
    }

    #[test]
    fn poll_desired_cols_debounces_and_clamps() {
        let mut state = SpectrogramState::new();
        let handle = state.observed_width_handle();
        let t0 = Instant::now();

        // No observation yet → None.
        assert_eq!(state.poll_desired_cols(t0), None);

        // Observe a new width; first poll arms the pending timer.
        handle.store(1920, Ordering::Relaxed);
        assert_eq!(state.poll_desired_cols(t0), None);

        // Before settle window, still None.
        assert_eq!(
            state.poll_desired_cols(t0 + Duration::from_millis(100)),
            None
        );

        // After settle window, resize fires.
        assert_eq!(
            state.poll_desired_cols(t0 + Duration::from_millis(200)),
            Some(1920)
        );

        // If the value changes before settle, timer resets.
        handle.store(3840, Ordering::Relaxed);
        let t1 = t0 + Duration::from_millis(300);
        assert_eq!(state.poll_desired_cols(t1), None);
        // Clamp to MAX_SPECTRO_COLS.
        handle.store(20_000, Ordering::Relaxed);
        let t2 = t1 + Duration::from_millis(50);
        assert_eq!(state.poll_desired_cols(t2), None);
        assert_eq!(
            state.poll_desired_cols(t2 + Duration::from_millis(200)),
            Some(MAX_SPECTRO_COLS)
        );

        // Clamp to MIN_SPECTRO_COLS.
        handle.store(1, Ordering::Relaxed);
        let t3 = t2 + Duration::from_millis(1000);
        assert_eq!(state.poll_desired_cols(t3), None);
        assert_eq!(
            state.poll_desired_cols(t3 + Duration::from_millis(200)),
            Some(MIN_SPECTRO_COLS)
        );
    }

    #[test]
    fn dirty_spans_single_contiguous_run() {
        let mut state = SpectrogramState::new();
        // Drain the new()-time full_upload flag so it doesn't conflate the test.
        assert!(state.take_full_upload_needed());
        assert!(state.take_dirty_spans().is_empty());

        for i in 0..5 {
            state.push_fft(&test_waveform(220.0 + i as f64 * 5.0, 2048));
        }
        let spans = state.take_dirty_spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].start_col, 0);
        assert_eq!(spans[0].data.len(), 5 * NUM_SPECTRO_BINS);

        // Span contents must match the underlying magnitudes at the reported offset.
        let expected = &state.magnitudes()[0..5 * NUM_SPECTRO_BINS];
        assert_eq!(spans[0].data.as_slice(), expected);

        // Second call without new pushes drains to empty.
        assert!(state.take_dirty_spans().is_empty());
    }

    #[test]
    fn dirty_spans_wrap_around() {
        let mut state = SpectrogramState::new();
        let _ = state.take_full_upload_needed();
        let num = state.num_cols();

        // Push until write_index is 2 short of num_cols, then drain.
        for _ in 0..(num - 2) {
            state.push_fft(&test_waveform(440.0, 2048));
        }
        let _ = state.take_dirty_spans();
        assert_eq!(state.write_index(), num - 2);

        // Push 5 more → should wrap, producing two spans (2 tail + 3 head).
        for _ in 0..5 {
            state.push_fft(&test_waveform(660.0, 2048));
        }
        assert_eq!(state.write_index(), 3);
        let spans = state.take_dirty_spans();
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start_col, num - 2);
        assert_eq!(spans[0].data.len(), 2 * NUM_SPECTRO_BINS);
        assert_eq!(spans[1].start_col, 0);
        assert_eq!(spans[1].data.len(), 3 * NUM_SPECTRO_BINS);
    }

    #[test]
    fn dirty_spans_overflow_triggers_full_upload() {
        let mut state = SpectrogramState::new();
        let _ = state.take_full_upload_needed();
        let num = state.num_cols();

        // Push more than num_cols without draining — saturates dirty_count.
        for _ in 0..(num + 3) {
            state.push_fft(&test_waveform(440.0, 2048));
        }
        let spans = state.take_dirty_spans();
        assert!(spans.is_empty(), "overflow path returns empty spans");
        assert!(
            state.take_full_upload_needed(),
            "overflow should arm full_upload_needed"
        );
    }
}
