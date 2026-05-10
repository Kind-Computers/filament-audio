// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(clippy::manual_is_multiple_of)]

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use super::CleanupMode;
use super::shared::{
    Region, RepairMode, RepairParams, ResidualDetectParams, apply_pre_declip_to_channels,
    deinterleave_channels, detect_residual_mask, dilate_mask, interleave_channels, kurtosis,
    mark_regions, mask_to_regions, merge_close_regions, odd_window, percentile_of_slice,
    repair_regions_for_all_channels, second_difference, total_region_len,
};

const QUIET_FLOOR_MARGIN_DB: f64 = 6.0;
const CLICK_CREST_DB_MAX: f64 = 7.0;
const CRACKLE_CREST_DB_MAX: f64 = 9.0;
const CLICK_FLUX_MIN: f64 = 0.20;
const CRACKLE_FLUX_MIN: f64 = 0.15;
const CLICK_KURTOSIS_MIN: f64 = 6.0;
const CRACKLE_KURTOSIS_MIN: f64 = 5.0;
const MIN_CONTEXT_SAMPLES: usize = 16;
const EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy)]
struct V21PassConfig {
    fft_size: usize,
    hop_size: usize,
    flatness_threshold: f64,
    energy_delta_db: f64,
    min_flag_frames: usize,
    aux_required: usize,
    crest_db_max: f64,
    flux_min: f64,
    residual_kurtosis_min: f64,
    candidate_dilation: usize,
    candidate_merge_gap: usize,
    max_candidate_span: usize,
    relocalizer: ResidualRelocalizer,
    final_merge_gap: usize,
    residual_fallback_sample_rate_limit: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct ResidualRelocalizer {
    mad_window: usize,
    z_hi: f64,
    z_lo: f64,
    edge_rms_mult: f64,
    max_span: usize,
    repair_pad: usize,
}

#[derive(Debug, Clone, Copy)]
struct FrameFeatures {
    energy_db: f64,
    flatness: f64,
    crest_db: f64,
    flux: f64,
    residual_kurtosis: f64,
    candidate: bool,
}

impl V21PassConfig {
    fn for_click(sample_rate: u32) -> Self {
        let fft_size = fft_size_for_sample_rate(sample_rate);
        let hop_size = hop_size_for_sample_rate(sample_rate);
        Self {
            fft_size,
            hop_size,
            flatness_threshold: 0.65,
            energy_delta_db: 6.0,
            min_flag_frames: 2,
            aux_required: 1,
            crest_db_max: CLICK_CREST_DB_MAX,
            flux_min: CLICK_FLUX_MIN,
            residual_kurtosis_min: CLICK_KURTOSIS_MIN,
            candidate_dilation: (hop_size / 2).max(1),
            candidate_merge_gap: hop_size,
            max_candidate_span: ((0.012 * sample_rate as f64) as usize).clamp(128, 1024),
            relocalizer: ResidualRelocalizer {
                mad_window: odd_window(((0.004 * sample_rate as f64) as usize).max(5), usize::MAX),
                z_hi: 8.0,
                z_lo: 3.5,
                edge_rms_mult: 2.5,
                max_span: ((0.003 * sample_rate as f64) as usize).clamp(32, 256),
                repair_pad: 1,
            },
            final_merge_gap: 2,
            residual_fallback_sample_rate_limit: None,
        }
    }

    fn for_crackle(sample_rate: u32) -> Self {
        let fft_size = fft_size_for_sample_rate(sample_rate);
        let hop_size = hop_size_for_sample_rate(sample_rate);
        Self {
            fft_size,
            hop_size,
            flatness_threshold: 0.55,
            energy_delta_db: 4.0,
            min_flag_frames: 1,
            aux_required: 2,
            crest_db_max: CRACKLE_CREST_DB_MAX,
            flux_min: CRACKLE_FLUX_MIN,
            residual_kurtosis_min: CRACKLE_KURTOSIS_MIN,
            candidate_dilation: (hop_size / 4).max(1),
            candidate_merge_gap: (hop_size / 2).max(1),
            max_candidate_span: ((0.004 * sample_rate as f64) as usize).clamp(32, 256),
            relocalizer: ResidualRelocalizer {
                mad_window: odd_window(((0.003 * sample_rate as f64) as usize).max(5), usize::MAX),
                z_hi: 6.0,
                z_lo: 2.8,
                edge_rms_mult: 1.6,
                max_span: ((0.001 * sample_rate as f64) as usize).clamp(8, 64),
                repair_pad: 0,
            },
            final_merge_gap: 1,
            residual_fallback_sample_rate_limit: Some(12_000),
        }
    }
}

pub(crate) fn apply_cleanup_v21(
    data: &[f64],
    sample_rate: u32,
    channels: usize,
    mode: CleanupMode,
) -> Result<Vec<f64>, String> {
    if channels == 0 || data.is_empty() {
        return Ok(Vec::new());
    }
    if mode == CleanupMode::Off {
        return Ok(data.to_vec());
    }

    let mut separated = deinterleave_channels(data, channels);
    apply_pre_declip_to_channels(&mut separated, sample_rate);

    match mode {
        CleanupMode::Off => {}
        CleanupMode::DeclickAr => {
            let config = V21PassConfig::for_click(sample_rate);
            let regions = detect_v21_regions(&separated, sample_rate, config);
            if !regions.is_empty() {
                repair_regions_for_all_channels(
                    &mut separated,
                    &regions,
                    RepairParams::for_click(),
                    RepairMode::Ar,
                );
            }
        }
        CleanupMode::DeclickMedian => {
            let config = V21PassConfig::for_click(sample_rate);
            let regions = detect_v21_regions(&separated, sample_rate, config);
            if !regions.is_empty() {
                repair_regions_for_all_channels(
                    &mut separated,
                    &regions,
                    RepairParams::for_click(),
                    RepairMode::Median,
                );
            }
        }
        CleanupMode::Decrackle => {
            let config = V21PassConfig::for_crackle(sample_rate);
            let mut first_pass_samples: Option<usize> = None;
            let mut previous_samples: Option<usize> = None;
            for _ in 0..3 {
                let regions = detect_v21_regions(&separated, sample_rate, config);
                let repaired_samples = total_region_len(&regions);
                if repaired_samples == 0 {
                    break;
                }

                if let Some(first_pass_samples) = first_pass_samples {
                    let min_activity = first_pass_samples.saturating_div(10).max(8);
                    let pass_grew_too_much = previous_samples
                        .map(|previous| repaired_samples * 4 > previous * 5)
                        .unwrap_or(false);
                    if repaired_samples < min_activity || pass_grew_too_much {
                        break;
                    }
                }

                repair_regions_for_all_channels(
                    &mut separated,
                    &regions,
                    RepairParams::for_crackle(),
                    RepairMode::Ar,
                );

                first_pass_samples.get_or_insert(repaired_samples);
                previous_samples = Some(repaired_samples);
            }
        }
    }

    Ok(interleave_channels(&separated))
}

fn detect_v21_regions(
    channels: &[Vec<f64>],
    sample_rate: u32,
    config: V21PassConfig,
) -> Vec<Region> {
    let Some(len) = channels.iter().map(Vec::len).min() else {
        return Vec::new();
    };
    if len < 5 {
        return Vec::new();
    }

    let analyzer = SpectralAnalyzer::new(config.fft_size, config.hop_size);
    let candidate_windows = detect_candidate_windows(channels, sample_rate, &analyzer, config, len);
    if candidate_windows.is_empty() {
        return Vec::new();
    }

    let mut relocalized = Vec::new();
    for (start, end) in candidate_windows {
        relocalized.extend(relocalize_candidate_window(channels, start, end, config));
    }

    let merged = merge_close_regions(&relocalized, config.final_merge_gap);
    merged
        .into_iter()
        .filter(|(start, end)| {
            let span = end.saturating_sub(*start);
            span > 0
                && span <= config.relocalizer.max_span
                && *start >= MIN_CONTEXT_SAMPLES
                && len.saturating_sub(*end) >= MIN_CONTEXT_SAMPLES
        })
        .collect()
}

fn detect_candidate_windows(
    channels: &[Vec<f64>],
    sample_rate: u32,
    analyzer: &SpectralAnalyzer,
    config: V21PassConfig,
    len: usize,
) -> Vec<Region> {
    let mut union_mask = vec![false; len];
    for channel in channels {
        let regions = candidate_windows_for_channel(&channel[..len], analyzer, config);
        mark_regions(&mut union_mask, &regions);
    }
    let merged = merge_close_regions(&mask_to_regions(&union_mask), config.candidate_merge_gap);
    let filtered: Vec<Region> = merged
        .into_iter()
        .filter(|(start, end)| end.saturating_sub(*start) <= config.max_candidate_span)
        .collect();

    let Some(sample_rate_limit) = config.residual_fallback_sample_rate_limit else {
        return filtered;
    };
    if sample_rate > sample_rate_limit {
        return filtered;
    }

    let fallback = fallback_crackle_candidate_windows(channels, sample_rate, config, len);
    supplement_candidate_windows(filtered, fallback, config.candidate_merge_gap)
}

fn candidate_windows_for_channel(
    samples: &[f64],
    analyzer: &SpectralAnalyzer,
    config: V21PassConfig,
) -> Vec<Region> {
    let (frame_starts, features) = analyzer.analyze(samples, config);
    if features.is_empty() {
        return Vec::new();
    }

    candidate_regions_from_features(&frame_starts, &features, samples.len(), config)
}

fn candidate_regions_from_features(
    frame_starts: &[usize],
    features: &[FrameFeatures],
    sample_len: usize,
    config: V21PassConfig,
) -> Vec<Region> {
    let mut regions = Vec::new();
    let mut i = 0usize;
    while i < features.len() {
        if features[i].candidate {
            let run_start = i;
            while i < features.len() && features[i].candidate {
                i += 1;
            }
            let run_len = i - run_start;
            if run_len >= config.min_flag_frames {
                let start_frame = run_start;
                let end_frame = i - 1;
                let mut start = frame_starts[start_frame];
                let mut end = frame_starts[end_frame]
                    .saturating_add(config.fft_size)
                    .min(sample_len);
                start = start.saturating_sub(config.candidate_dilation);
                end = (end + config.candidate_dilation).min(sample_len);
                if end > start && end - start <= config.max_candidate_span {
                    regions.push((start, end));
                }
            }
        } else {
            i += 1;
        }
    }

    merge_close_regions(&regions, config.candidate_merge_gap)
}

fn relocalize_candidate_window(
    channels: &[Vec<f64>],
    start: usize,
    end: usize,
    config: V21PassConfig,
) -> Vec<Region> {
    let candidate_len = end.saturating_sub(start);
    if candidate_len < 5 {
        return Vec::new();
    }

    let detect_params = relocalizer_params_for_window(config.relocalizer, candidate_len);
    let mut union_mask = vec![false; candidate_len];
    for channel in channels {
        let local_mask = detect_residual_mask(&channel[start..end], detect_params);
        for (dst, src) in union_mask.iter_mut().zip(local_mask.iter()) {
            *dst |= *src;
        }
    }

    dilate_mask(&mut union_mask, detect_params.pad_samples);
    let local_regions = merge_close_regions(&mask_to_regions(&union_mask), config.final_merge_gap);
    local_regions
        .into_iter()
        .map(|(local_start, local_end)| (start + local_start, start + local_end))
        .filter(|(region_start, region_end)| {
            region_end.saturating_sub(*region_start) <= config.relocalizer.max_span
        })
        .collect()
}

fn relocalizer_params_for_window(
    relocalizer: ResidualRelocalizer,
    candidate_len: usize,
) -> ResidualDetectParams {
    ResidualDetectParams {
        mad_window: odd_window(relocalizer.mad_window, candidate_len),
        z_hi: relocalizer.z_hi,
        z_lo: relocalizer.z_lo,
        edge_rms_mult: relocalizer.edge_rms_mult,
        max_click_samples: relocalizer.max_span.min(candidate_len),
        pad_samples: relocalizer.repair_pad,
    }
}

fn fallback_crackle_candidate_windows(
    channels: &[Vec<f64>],
    sample_rate: u32,
    config: V21PassConfig,
    len: usize,
) -> Vec<Region> {
    let sr = sample_rate as usize;
    let detect_params = ResidualDetectParams {
        mad_window: (((0.003 * sr as f64) as usize).max(31)) | 1,
        z_hi: 7.5,
        z_lo: 3.2,
        edge_rms_mult: 1.8,
        max_click_samples: ((0.00045 * sr as f64) as usize).clamp(2, 24),
        pad_samples: 0,
    };
    let mut regions = Vec::new();
    let mut union_mask = vec![false; len];
    for channel in channels {
        let mask = detect_residual_mask(&channel[..len], detect_params);
        for (dst, src) in union_mask.iter_mut().zip(mask.iter()) {
            *dst |= *src;
        }
    }
    for (start, end) in merge_close_regions(&mask_to_regions(&union_mask), config.final_merge_gap) {
        let start = start.saturating_sub(config.candidate_dilation);
        let end = (end + config.candidate_dilation).min(len);
        if end > start && end - start <= config.max_candidate_span {
            regions.push((start, end));
        }
    }
    regions
}

fn regions_overlap(lhs: Region, rhs: Region) -> bool {
    lhs.0 < rhs.1 && rhs.0 < lhs.1
}

fn supplement_candidate_windows(
    mut spectral: Vec<Region>,
    fallback: Vec<Region>,
    merge_gap: usize,
) -> Vec<Region> {
    let supplemental: Vec<Region> = fallback
        .into_iter()
        .filter(|candidate| {
            spectral
                .iter()
                .all(|existing| !regions_overlap(*candidate, *existing))
        })
        .collect();
    spectral.extend(supplemental);
    spectral.sort_unstable_by_key(|region| region.0);

    merge_close_regions(&spectral, merge_gap)
}

fn fft_size_for_sample_rate(sample_rate: u32) -> usize {
    let samples = ((0.006 * sample_rate as f64).round() as usize).max(32);
    samples.next_power_of_two().clamp(128, 512)
}

fn hop_size_for_sample_rate(sample_rate: u32) -> usize {
    ((0.0015 * sample_rate as f64).round() as usize).max(16)
}

struct SpectralAnalyzer {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f64>,
    fft: std::sync::Arc<dyn rustfft::Fft<f64>>,
}

impl SpectralAnalyzer {
    fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(fft_size);
        let window = (0..fft_size)
            .map(|i| {
                let phase =
                    2.0 * std::f64::consts::PI * i as f64 / (fft_size.saturating_sub(1)) as f64;
                0.5 - 0.5 * phase.cos()
            })
            .collect();
        Self {
            fft_size,
            hop_size,
            window,
            fft,
        }
    }

    fn analyze(&self, samples: &[f64], config: V21PassConfig) -> (Vec<usize>, Vec<FrameFeatures>) {
        let frame_starts = frame_starts(samples.len(), self.hop_size);
        if frame_starts.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut features = Vec::with_capacity(frame_starts.len());
        let mut previous_power: Option<Vec<f64>> = None;
        for &start in &frame_starts {
            let frame_samples = extract_frame(samples, start, self.fft_size);
            let residual = second_difference(&frame_samples);
            let mut spectrum: Vec<Complex<f64>> = residual
                .iter()
                .zip(self.window.iter())
                .map(|(sample, window)| Complex::new(sample * window, 0.0))
                .collect();
            self.fft.process(&mut spectrum);

            let power_bins: Vec<f64> = spectrum[1..(self.fft_size / 2)]
                .iter()
                .map(|bin| bin.norm_sqr())
                .collect();
            if power_bins.is_empty() {
                features.push(FrameFeatures {
                    energy_db: -120.0,
                    flatness: 0.0,
                    crest_db: 0.0,
                    flux: 0.0,
                    residual_kurtosis: 0.0,
                    candidate: false,
                });
                previous_power = Some(power_bins);
                continue;
            }

            let energy = power_bins.iter().copied().sum::<f64>();
            let mean_power = energy / power_bins.len() as f64;
            let max_power = power_bins.iter().copied().fold(0.0f64, f64::max);
            let log_mean = power_bins
                .iter()
                .map(|power| (power + EPSILON).ln())
                .sum::<f64>()
                / power_bins.len() as f64;
            let flatness = log_mean.exp() / (mean_power + EPSILON);
            let crest_db = 10.0 * ((max_power + EPSILON) / (mean_power + EPSILON)).log10();
            let flux = if let Some(previous_power) = previous_power.as_ref() {
                let positive_flux = power_bins
                    .iter()
                    .zip(previous_power.iter())
                    .map(|(current, previous)| (current - previous).max(0.0))
                    .sum::<f64>();
                positive_flux / (energy + EPSILON)
            } else {
                0.0
            };
            let residual_kurtosis = kurtosis(&residual);
            features.push(FrameFeatures {
                energy_db: 10.0 * (energy + EPSILON).log10(),
                flatness,
                crest_db,
                flux,
                residual_kurtosis,
                candidate: false,
            });
            previous_power = Some(power_bins);
        }

        let energies: Vec<f64> = features.iter().map(|feature| feature.energy_db).collect();
        let quiet_floor = percentile_of_slice(&energies, 0.20);
        for (index, feature) in features.iter_mut().enumerate() {
            let baseline = local_median(&energies, index, 8);
            feature.candidate =
                frame_qualifies_as_candidate(*feature, baseline, quiet_floor, config);
        }

        (frame_starts, features)
    }
}

fn frame_qualifies_as_candidate(
    feature: FrameFeatures,
    baseline: f64,
    quiet_floor: f64,
    config: V21PassConfig,
) -> bool {
    let energy_delta_db = feature.energy_db - baseline;
    let core_gate = feature.flatness >= config.flatness_threshold
        && energy_delta_db >= config.energy_delta_db
        && feature.energy_db >= quiet_floor + QUIET_FLOOR_MARGIN_DB;

    let mut aux_count = 0usize;
    if feature.crest_db <= config.crest_db_max {
        aux_count += 1;
    }
    if feature.flux >= config.flux_min {
        aux_count += 1;
    }
    if feature.residual_kurtosis >= config.residual_kurtosis_min {
        aux_count += 1;
    }

    core_gate && aux_count >= config.aux_required
}

fn frame_starts(len: usize, hop_size: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let mut starts = Vec::new();
    let mut start = 0usize;
    while start < len {
        starts.push(start);
        if hop_size == 0 {
            break;
        }
        if start + hop_size >= len {
            break;
        }
        start += hop_size;
    }
    starts
}

fn extract_frame(samples: &[f64], start: usize, fft_size: usize) -> Vec<f64> {
    let mut frame = vec![0.0f64; fft_size];
    let available = samples.len().saturating_sub(start).min(fft_size);
    frame[..available].copy_from_slice(&samples[start..start + available]);
    frame
}

fn local_median(values: &[f64], index: usize, radius: usize) -> f64 {
    let start = index.saturating_sub(radius);
    let end = (index + radius + 1).min(values.len());
    let mut local = values[start..end].to_vec();
    local.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    if local.is_empty() {
        0.0
    } else if local.len() % 2 == 0 {
        let mid = local.len() / 2;
        0.5 * (local[mid - 1] + local[mid])
    } else {
        local[local.len() / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn impulse_signal() -> Vec<f64> {
        let mut samples = vec![0.0f64; 4096];
        for offset in 0..12 {
            let burst = if offset % 2 == 0 { 1.0 } else { -1.0 };
            samples[1020 + offset] = burst;
        }
        samples
    }

    #[test]
    fn flatness_is_higher_for_impulsive_content_than_tonal_content() {
        let analyzer = SpectralAnalyzer::new(256, 64);
        let tonal: Vec<f64> = (0..4096)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                0.25 * (2.0 * std::f64::consts::PI * 220.0 * t).sin()
            })
            .collect();
        let (_, tonal_features) = analyzer.analyze(&tonal, V21PassConfig::for_click(16_000));
        let (_, impulse_features) =
            analyzer.analyze(&impulse_signal(), V21PassConfig::for_click(16_000));

        let tonal_peak = tonal_features
            .iter()
            .map(|feature| feature.flatness)
            .fold(0.0f64, f64::max);
        let impulse_peak = impulse_features
            .iter()
            .map(|feature| feature.flatness)
            .fold(0.0f64, f64::max);
        assert!(impulse_peak > tonal_peak);
    }

    #[test]
    fn quiet_noise_is_suppressed_by_candidate_gates() {
        let analyzer = SpectralAnalyzer::new(256, 64);
        let noise: Vec<f64> = (0..4096)
            .map(|i| ((i as f64 * 0.731).sin() * (i as f64 * 1.113).cos()) * 0.002)
            .collect();
        let (_, features) = analyzer.analyze(&noise, V21PassConfig::for_click(16_000));
        assert!(features.iter().all(|feature| !feature.candidate));
    }

    #[test]
    fn relocalizer_finds_narrow_spans_inside_wider_candidates() {
        let samples = impulse_signal();
        let channels = vec![samples.clone()];
        let config = V21PassConfig::for_click(16_000);
        let regions = relocalize_candidate_window(&channels, 980, 1080, config);
        assert!(!regions.is_empty());
        assert!(
            regions
                .iter()
                .all(|(start, end)| end - start <= config.relocalizer.max_span)
        );
        assert!(total_region_len(&regions) < 100);
    }

    #[test]
    fn click_candidate_windows_cover_fft_support_near_sample_boundaries() {
        let sample_rate = 16_000;
        let config = V21PassConfig::for_click(sample_rate);
        let sample_len = 512usize;
        let frame_starts = frame_starts(sample_len, config.hop_size);
        assert!(frame_starts.len() >= config.min_flag_frames);

        let mut features = vec![
            FrameFeatures {
                energy_db: 0.0,
                flatness: 0.0,
                crest_db: 0.0,
                flux: 0.0,
                residual_kurtosis: 0.0,
                candidate: false,
            };
            frame_starts.len()
        ];

        features[0].candidate = true;
        features[1].candidate = true;
        let start_regions =
            candidate_regions_from_features(&frame_starts, &features, sample_len, config);
        assert!(
            start_regions
                .iter()
                .any(|(start, end)| *start == 0 && *end >= config.fft_size),
            "first-frame candidates should cover the full FFT support at the start boundary",
        );

        for feature in &mut features {
            feature.candidate = false;
        }
        let last = features.len() - 1;
        features[last - 1].candidate = true;
        features[last].candidate = true;
        let end_regions =
            candidate_regions_from_features(&frame_starts, &features, sample_len, config);
        assert!(
            end_regions
                .iter()
                .any(|(start, end)| *start <= frame_starts[last - 1] && *end == sample_len),
            "last-frame candidates should extend through the end of the sample",
        );
    }

    #[test]
    fn bright_valid_attack_produces_few_or_no_final_spans() {
        let mut samples = vec![0.0f64; 4096];
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f64 / 16_000.0;
            let env = if i < 128 { i as f64 / 128.0 } else { 1.0 };
            *sample = env
                * (0.22 * (2.0 * std::f64::consts::PI * 1400.0 * t).sin()
                    + 0.14 * (2.0 * std::f64::consts::PI * 3200.0 * t).sin());
        }
        let channels = vec![samples];
        let regions = detect_v21_regions(&channels, 16_000, V21PassConfig::for_click(16_000));
        assert!(regions.len() <= 1);
        assert!(total_region_len(&regions) <= 16);
    }

    #[test]
    fn crackle_requires_two_auxiliary_features() {
        let config = V21PassConfig::for_crackle(8_000);
        let baseline = -12.0;
        let quiet_floor = -24.0;

        let one_aux_feature = FrameFeatures {
            energy_db: -4.0,
            flatness: 0.70,
            crest_db: 8.5,
            flux: 0.10,
            residual_kurtosis: 4.0,
            candidate: false,
        };
        assert!(
            !frame_qualifies_as_candidate(one_aux_feature, baseline, quiet_floor, config),
            "crackle should reject frames that only satisfy one auxiliary feature",
        );

        let two_aux_features = FrameFeatures {
            flux: 0.20,
            ..one_aux_feature
        };
        assert!(
            frame_qualifies_as_candidate(two_aux_features, baseline, quiet_floor, config),
            "crackle should accept frames that satisfy two auxiliary features",
        );
    }

    #[test]
    fn low_rate_crackle_fallback_supplements_non_overlapping_spectral_windows() {
        let config = V21PassConfig::for_crackle(8_000);
        let spectral = vec![(96usize, 132usize)];
        let fallback = vec![(100usize, 128usize), (220usize, 246usize)];
        let combined =
            supplement_candidate_windows(spectral.clone(), fallback, config.candidate_merge_gap);

        assert_eq!(combined, vec![(96, 132), (220, 246)]);
    }

    #[test]
    fn crackle_config_detects_dense_short_defects() {
        let samples: Vec<f64> = (0..8192)
            .map(|i| {
                let t = i as f64 / 8_000.0;
                let crackle = if i % 173 == 0 { 0.9 } else { 0.0 };
                0.35 * (2.0 * std::f64::consts::PI * 220.0 * t).sin() + crackle
            })
            .collect();
        let channels = vec![samples.clone()];
        let config = V21PassConfig::for_crackle(8_000);
        let analyzer = SpectralAnalyzer::new(config.fft_size, config.hop_size);
        let windows = detect_candidate_windows(&channels, 8_000, &analyzer, config, samples.len());
        assert!(!windows.is_empty());
        let regions = detect_v21_regions(&channels, 8_000, config);
        assert!(!regions.is_empty());
    }
}
