// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#[path = "../src/simd.rs"]
mod simd;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn mono_signal(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = i as f32;
            ((x * 0.017).sin() * 0.7) + ((x * 0.0031).cos() * 0.15)
        })
        .collect()
}

fn stereo_signal(frames: usize) -> Vec<f32> {
    (0..frames)
        .flat_map(|i| {
            let x = i as f32;
            [
                ((x * 0.013).sin() * 0.6) + 0.05,
                ((x * 0.021).cos() * 0.5) - 0.04,
            ]
        })
        .collect()
}

fn spectral_pair(len: usize) -> (Vec<f64>, Vec<f64>) {
    let a: Vec<f64> = (0..len)
        .map(|i| {
            let x = i as f64;
            ((x * 0.031).sin() * 0.8) + ((x * 0.0017).cos() * 0.07)
        })
        .collect();
    let b: Vec<f64> = a
        .iter()
        .enumerate()
        .map(|(i, &value)| value + ((i as f64 * 0.0071).sin() * 0.0004))
        .collect();
    (a, b)
}

fn bench_mix_to_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("mix_to_mono");
    for &frames in &[4_096usize, 32_768, 131_072] {
        let stereo = stereo_signal(frames);
        group.bench_with_input(
            BenchmarkId::new("stereo_frames", frames),
            &stereo,
            |b, input| {
                b.iter(|| simd::mix_to_mono(black_box(input), 2));
            },
        );
    }
    group.finish();
}

fn bench_pearson_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pearson_correlation");
    for &len in &[4_096usize, 32_768, 131_072] {
        let (a, b) = spectral_pair(len);
        group.bench_with_input(BenchmarkId::new("bins", len), &(a, b), |bench, pair| {
            bench.iter(|| simd::pearson_correlation(black_box(&pair.0), black_box(&pair.1)));
        });
    }
    group.finish();
}

fn bench_normalize_sample_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_sample_kernel");
    for &len in &[4_096usize, 32_768, 131_072] {
        let data = mono_signal(len);
        group.bench_with_input(BenchmarkId::new("samples", len), &data, |b, input| {
            b.iter(|| {
                let rms = simd::rms_f32(black_box(input));
                let mut scaled = input.clone();
                if rms > 1e-10 {
                    simd::scale_in_place(black_box(&mut scaled), 0.42 / rms);
                }
                scaled
            });
        });
    }
    group.finish();
}

fn bench_normalize_render_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_render_kernel");
    for &frames in &[4_096usize, 32_768, 131_072] {
        let data = stereo_signal(frames);
        group.bench_with_input(
            BenchmarkId::new("stereo_frames", frames),
            &data,
            |b, input| {
                b.iter(|| {
                    let peak = simd::peak_abs_f32(black_box(input));
                    let mut scaled = input.clone();
                    if peak > 0.0 {
                        simd::scale_and_clamp_in_place(
                            black_box(&mut scaled),
                            1.0 / peak,
                            -1.0,
                            1.0,
                        );
                    }
                    scaled
                });
            },
        );
    }
    group.finish();
}

fn bench_pcm_scale_pack(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcm_scale_pack");
    for &len in &[4_096usize, 32_768, 131_072] {
        let data = mono_signal(len);
        group.bench_with_input(BenchmarkId::new("i16", len), &data, |b, input| {
            b.iter(|| simd::scale_to_i16(black_box(input)));
        });
        group.bench_with_input(BenchmarkId::new("i24le", len), &data, |b, input| {
            b.iter(|| simd::scale_to_i24le_bytes(black_box(input)));
        });
    }
    group.finish();
}

fn bench_sum_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_f32");
    for &len in &[4_096usize, 32_768, 131_072] {
        let data = mono_signal(len);
        group.bench_with_input(BenchmarkId::new("samples", len), &data, |b, input| {
            b.iter(|| simd::sum_f32(black_box(input)));
        });
    }
    group.finish();
}

fn bench_subtract_in_place(c: &mut Criterion) {
    let mut group = c.benchmark_group("subtract_in_place");
    for &len in &[4_096usize, 32_768, 131_072] {
        let data = mono_signal(len);
        group.bench_with_input(BenchmarkId::new("samples", len), &data, |b, input| {
            b.iter(|| {
                let mut buf = input.clone();
                simd::subtract_in_place(black_box(&mut buf), 0.42);
                buf
            });
        });
    }
    group.finish();
}

fn bench_scale_from_i24le_bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale_from_i24le_bytes");
    for &len in &[4_096usize, 32_768, 131_072] {
        let data = mono_signal(len);
        let encoded = simd::scale_to_i24le_bytes(&data);
        group.bench_with_input(BenchmarkId::new("samples", len), &encoded, |b, input| {
            b.iter(|| simd::scale_from_i24le_bytes(black_box(input)));
        });
    }
    group.finish();
}

fn bench_deinterleave_interleave_stereo(c: &mut Criterion) {
    let mut group = c.benchmark_group("deinterleave_interleave_stereo");
    for &frames in &[4_096usize, 32_768, 131_072] {
        let stereo = stereo_signal(frames);
        group.bench_with_input(
            BenchmarkId::new("deinterleave_frames", frames),
            &stereo,
            |b, input| {
                b.iter(|| simd::deinterleave_stereo(black_box(input)));
            },
        );
        let (left, right) = simd::deinterleave_stereo(&stereo);
        group.bench_with_input(
            BenchmarkId::new("interleave_frames", frames),
            &(left, right),
            |b, pair| {
                b.iter(|| simd::interleave_stereo(black_box(&pair.0), black_box(&pair.1)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_mix_to_mono,
    bench_pearson_correlation,
    bench_normalize_sample_kernel,
    bench_normalize_render_kernel,
    bench_pcm_scale_pack,
    bench_sum_f32,
    bench_subtract_in_place,
    bench_scale_from_i24le_bytes,
    bench_deinterleave_interleave_stereo
);
criterion_main!(benches);
