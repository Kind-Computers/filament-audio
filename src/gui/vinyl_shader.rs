// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use bytemuck::{Pod, Zeroable};
use iced::widget::shader::{self, Storage, Viewport, wgpu};
use iced::{Rectangle, mouse};

pub const MAX_GROOVE_SAMPLES: usize = 131072;

const VINYL_WGSL: &str = r#"
struct Uniforms {
    width: f32,
    height: f32,
    angle: f32,
    num_groove_samples: u32,
    frame: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> groove_data: array<f32>;

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
// an sRGB framebuffer (GAMMA_CORRECTION=true) and re-encodes shader writes
// on store; designer hex values must be converted before they're written.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = vec3<f32>(0.04045);
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= cutoff);
}

// Full per-pixel record computation. Extracted so fs_main can supersample.
fn sample_vinyl(px: f32, py: f32) -> vec3<f32> {
    let cx = u.width * 0.5;
    let cy = u.height * 0.5;
    let dx = px - cx;
    let dy = py - cy;
    let dist = sqrt(dx * dx + dy * dy);

    let record_radius = min(u.height * 0.42, u.width * 0.35);
    let label_radius = record_radius * 0.35;
    let spindle_radius = record_radius * 0.03;

    // Outside record — dark background
    if dist > record_radius {
        return vec3<f32>(0.035, 0.035, 0.06);
    }

    // Spindle hole
    if dist < spindle_radius {
        return vec3<f32>(0.02, 0.02, 0.03);
    }

    // Label area — #5C5C8A with a subtle radial darkening toward the rim.
    // Iced uses an sRGB framebuffer, so shader writes are interpreted as
    // linear-light and re-encoded on store. We convert the sRGB hex value
    // into linear space here so the displayed color matches #5C5C8A.
    if dist < label_radius {
        let t = dist / label_radius;
        // #5C5C8A in sRGB → 0x5C/255, 0x5C/255, 0x8A/255 = (0.36078, 0.36078, 0.54118)
        let center = srgb_to_linear(vec3<f32>(0.36078, 0.36078, 0.54118));
        let edge = center * 0.85;
        return mix(edge, center, 1.0 - t);
    }

    // ── Groove zone ──
    let groove_span = record_radius - label_radius;
    let i = dist - label_radius;

    // Groove spiral mapping
    let groove_spacing = 1.5;
    let num_revolutions = groove_span / groove_spacing;
    let rev_from_outer = (record_radius - dist) / groove_spacing;

    // Angular position (rotates with record), normalized to 0-1
    let theta = atan2(dy, dx) - u.angle;
    let angle_01 = fract(theta / 6.2832 + 0.5);

    // Song progress through the spiral
    let song_t = (floor(rev_from_outer) + angle_01) / num_revolutions;

    // Look up groove amplitude from song data (or use fallback)
    var amplitude: f32;
    if u.num_groove_samples > 0u {
        let sample_idx = u32(clamp(song_t, 0.0, 1.0) * f32(u.num_groove_samples - 1u));
        amplitude = groove_data[sample_idx];
    } else {
        amplitude = 0.3;
    }

    // Per-pixel surface normal: groove depth modulated by song amplitude
    let groove_freq = 6.2832 / groove_spacing;
    let depth = 0.15 + amplitude * 0.6;
    let dh_dr = groove_freq * cos(dist * groove_freq) * depth;
    let inv_dist = select(1.0 / dist, 0.0, dist < 0.001);
    let radial = vec2<f32>(dx * inv_dist, dy * inv_dist);
    let normal = normalize(vec3<f32>(radial * -dh_dr, 1.0));

    // Groove tangent: perpendicular to radial (concentric circles)
    let tangent = vec3<f32>(-radial.y, radial.x, 0.0);

    // Light from upper-right, view from above (orthographic)
    let light_dir = normalize(vec3<f32>(1.0, -1.0, 1.0));
    let view_dir = vec3<f32>(0.0, 0.0, 1.0);
    let half_vec = normalize(light_dir + view_dir);

    // PVC/vinyl: eta = 1.54, F0 = ((1.54 - 1) / (1.54 + 1))^2 = 0.0452
    let F0 = 0.0452;

    // Fresnel (Schlick approximation)
    let NdotV = max(dot(normal, view_dir), 0.0);
    let fresnel = F0 + (1.0 - F0) * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);

    // Anisotropic specular (Kajiya-Kay) — the bright groove reflection band
    let TdotH = dot(tangent, half_vec);
    let sin_TH = sqrt(max(1.0 - TdotH * TdotH, 0.0));
    let aniso_spec = pow(sin_TH, 80.0);
    let aniso_sheen = pow(sin_TH, 10.0);

    // Fresnel-weighted specular: narrow highlight + broader sheen
    let specular = fresnel * (aniso_spec * 1.6 + aniso_sheen * 0.12);

    // Black vinyl diffuse: PVC absorbs nearly all light
    let NdotL = max(dot(normal, light_dir), 0.0);
    let albedo = vec3<f32>(0.012, 0.012, 0.015);
    let diffuse = albedo * (0.25 + NdotL * 0.75);

    // Subtle per-groove ridge texture (every 3rd groove slightly raised)
    let ridge = select(0.0, 0.003, i32(i) % 3 == 0);

    return max(diffuse + vec3<f32>(specular + ridge), vec3(0.005));
}

// Halton base-2 and base-3 sequences for temporally stable jitter.
// 8 frames of offsets that tile well and avoid clustering.
fn halton_jitter(frame: u32) -> vec2<f32> {
    let idx = frame % 8u;
    // Halton(2): 1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, 1/16
    // Halton(3): 1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, 8/9
    // Centered to [-0.5, 0.5)
    var jx: f32; var jy: f32;
    switch idx {
        case 0u: { jx =  0.0;    jy = -0.167; }
        case 1u: { jx = -0.25;   jy =  0.167; }
        case 2u: { jx =  0.25;   jy = -0.389; }
        case 3u: { jx = -0.375;  jy = -0.056; }
        case 4u: { jx =  0.125;  jy =  0.278; }
        case 5u: { jx = -0.125;  jy = -0.278; }
        case 6u: { jx =  0.375;  jy =  0.056; }
        default: { jx = -0.4375; jy =  0.389; }
    }
    return vec2(jx, jy);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let px = in.uv.x * u.width;
    let py = in.uv.y * u.height;

    // Temporal jitter: shift the RGSS pattern each frame using Halton sequence.
    // Over 8 frames at 120 Hz, the eye integrates 4*8 = 32 effective samples.
    let jitter = halton_jitter(u.frame);

    var color = (
        sample_vinyl(px - 0.125 + jitter.x, py - 0.375 + jitter.y) +
        sample_vinyl(px + 0.375 + jitter.x, py - 0.125 + jitter.y) +
        sample_vinyl(px - 0.375 + jitter.x, py + 0.125 + jitter.y) +
        sample_vinyl(px + 0.125 + jitter.x, py + 0.375 + jitter.y)
    ) * 0.25;

    // Vignette tinted to match the LP label (#5C5C8A), opacity fades from edges toward center
    let label_tint = srgb_to_linear(vec3<f32>(0.36078, 0.36078, 0.54118));
    let cx = u.width * 0.5;
    let cy = u.height * 0.5;
    let dx = (px - cx) / cx;
    let dy = (py - cy) / cy;
    let r = sqrt(dx * dx + dy * dy);
    let vignette = smoothstep(0.3, 1.3, r) * 0.1625;
    color = mix(color, label_tint, vignette);

    return vec4(color, 1.0);
}
"#;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct VinylUniforms {
    width: f32,
    height: f32,
    angle: f32,
    num_groove_samples: u32,
    frame: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[derive(Debug)]
struct VinylPipeline {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    storage_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Debug)]
pub struct VinylPrimitive {
    uniforms: VinylUniforms,
    groove_data: Vec<f32>,
}

impl shader::Primitive for VinylPrimitive {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        storage: &mut Storage,
        _bounds: &Rectangle,
        _viewport: &Viewport,
    ) {
        if !storage.has::<VinylPipeline>() {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vinyl_shader"),
                source: wgpu::ShaderSource::Wgsl(VINYL_WGSL.into()),
            });

            let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vinyl_uniforms"),
                size: std::mem::size_of::<VinylUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Storage buffer sized for max groove data (512 KB)
            let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vinyl_grooves"),
                size: (MAX_GROOVE_SAMPLES * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("vinyl_bgl"),
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
                label: Some("vinyl_bg"),
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
                label: Some("vinyl_pl"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("vinyl_pipeline"),
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
                        blend: None,
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

            storage.store(VinylPipeline {
                pipeline,
                uniform_buffer,
                storage_buffer,
                bind_group,
            });
        }

        let pl = storage.get::<VinylPipeline>().unwrap();
        queue.write_buffer(&pl.uniform_buffer, 0, bytemuck::bytes_of(&self.uniforms));
        if !self.groove_data.is_empty() {
            queue.write_buffer(
                &pl.storage_buffer,
                0,
                bytemuck::cast_slice(&self.groove_data),
            );
        }
    }

    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        storage: &Storage,
        target: &wgpu::TextureView,
        clip_bounds: &Rectangle<u32>,
    ) {
        let pl = storage.get::<VinylPipeline>().unwrap();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("vinyl_pass"),
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

#[derive(Debug, Clone)]
pub struct VinylProgram {
    pub rotation_angle: f32,
    pub frame: u32,
    pub groove_data: Vec<f32>,
}

impl<Message> shader::Program<Message> for VinylProgram {
    type State = ();
    type Primitive = VinylPrimitive;

    fn draw(
        &self,
        _state: &Self::State,
        _cursor: mouse::Cursor,
        bounds: Rectangle,
    ) -> Self::Primitive {
        VinylPrimitive {
            uniforms: VinylUniforms {
                width: bounds.width,
                height: bounds.height,
                angle: self.rotation_angle,
                num_groove_samples: self.groove_data.len() as u32,
                frame: self.frame,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            },
            groove_data: self.groove_data.clone(),
        }
    }
}

/// Render full module audio to an RMS-envelope for vinyl groove modulation.
/// Renders at 8363 Hz (classic Amiga Paula rate) — plenty of detail for a
/// ~200px radius disc, and ~6x faster than rendering at 48 kHz.
/// Returns up to MAX_GROOVE_SAMPLES f32 values in [0, 1].
pub fn render_groove_data(
    file_data: &[u8],
    stereo_separation: i32,
    interpolation_filter: i32,
) -> Result<Vec<f32>, String> {
    const GROOVE_RENDER_RATE: u32 = 8363; // Amiga Paula
    let samples = crate::render::render_module_to_samples_at_rate(
        file_data,
        stereo_separation,
        interpolation_filter,
        GROOVE_RENDER_RATE,
    )?;

    // Downmix stereo to mono
    let frames = samples.len() / 2;
    if frames == 0 {
        return Ok(Vec::new());
    }

    let target_len = frames.min(MAX_GROOVE_SAMPLES);
    let window_size = (frames / target_len).max(1);
    let mut envelope = Vec::with_capacity(target_len);

    for chunk_start in (0..frames).step_by(window_size) {
        let chunk_end = (chunk_start + window_size).min(frames);
        let n = chunk_end - chunk_start;
        if n == 0 {
            break;
        }
        // RMS of mono downmix
        let mut sum_sq = 0.0f64;
        for f in chunk_start..chunk_end {
            let l = samples[f * 2];
            let r = samples.get(f * 2 + 1).copied().unwrap_or(l);
            let mono = (l + r) * 0.5;
            sum_sq += mono * mono;
        }
        let rms = (sum_sq / n as f64).sqrt();
        envelope.push(rms as f32);
        if envelope.len() >= target_len {
            break;
        }
    }

    // Normalize to [0, 1]
    let peak = envelope.iter().copied().fold(0.0f32, f32::max);
    if peak > 1e-6 {
        for v in &mut envelope {
            *v /= peak;
        }
    }

    Ok(envelope)
}
