// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

#![allow(clippy::manual_is_multiple_of)]

use std::f64::consts::PI;

use crate::simd;

pub(crate) type Region = (usize, usize);

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidualDetectParams {
    pub(crate) mad_window: usize,
    pub(crate) z_hi: f64,
    pub(crate) z_lo: f64,
    pub(crate) edge_rms_mult: f64,
    pub(crate) max_click_samples: usize,
    pub(crate) pad_samples: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RepairParams {
    pub(crate) lpc_order: usize,
    pub(crate) min_context: usize,
    pub(crate) max_context: usize,
    pub(crate) merge_gap: usize,
}

impl RepairParams {
    pub(crate) fn for_click() -> Self {
        Self {
            lpc_order: 24,
            min_context: 128,
            max_context: 1024,
            merge_gap: 2,
        }
    }

    pub(crate) fn for_crackle() -> Self {
        Self {
            lpc_order: 18,
            min_context: 64,
            max_context: 384,
            merge_gap: 1,
        }
    }

    pub(crate) fn for_declip() -> Self {
        Self {
            lpc_order: 24,
            min_context: 96,
            max_context: 768,
            merge_gap: 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ClipDetectParams {
    peak_ratio: f64,
    min_region_samples: usize,
    max_region_samples: usize,
    flat_span_ratio: f64,
    edge_rms_mult: f64,
    pad_samples: usize,
}

impl ClipDetectParams {
    pub(crate) fn for_sample_rate(sample_rate: u32) -> Self {
        let sr = sample_rate as usize;
        Self {
            peak_ratio: 0.985,
            min_region_samples: 2,
            max_region_samples: ((0.00025 * sr as f64) as usize).clamp(2, 16),
            flat_span_ratio: 0.015,
            edge_rms_mult: 1.2,
            pad_samples: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RepairMode {
    Ar,
    Median,
}

pub(crate) fn apply_pre_declip_to_channels(channels: &mut [Vec<f64>], sample_rate: u32) {
    let clip_params = ClipDetectParams::for_sample_rate(sample_rate);
    let regions = detect_union_regions(
        channels,
        clip_params.pad_samples,
        RepairParams::for_declip().merge_gap,
        |samples| detect_clip_mask(samples, clip_params),
    );
    if regions.is_empty() {
        return;
    }
    repair_regions_for_all_channels(
        channels,
        &regions,
        RepairParams::for_declip(),
        RepairMode::Ar,
    );
}

pub(crate) fn repair_regions_for_all_channels(
    channels: &mut [Vec<f64>],
    regions: &[Region],
    repair_params: RepairParams,
    repair_mode: RepairMode,
) {
    for channel in channels {
        repair_regions(channel, regions, repair_params, repair_mode);
    }
}

pub(crate) fn detect_union_regions<F>(
    channels: &[Vec<f64>],
    pad_samples: usize,
    merge_gap: usize,
    detect_mask: F,
) -> Vec<Region>
where
    F: Fn(&[f64]) -> Vec<bool>,
{
    let Some(len) = channels.iter().map(Vec::len).min() else {
        return Vec::new();
    };
    if len == 0 {
        return Vec::new();
    }

    let mut union_mask = vec![false; len];
    for channel in channels {
        let mask = detect_mask(&channel[..len]);
        for (dst, src) in union_mask.iter_mut().zip(mask.iter()) {
            *dst |= *src;
        }
    }

    dilate_mask(&mut union_mask, pad_samples);
    merge_close_regions(&mask_to_regions(&union_mask), merge_gap)
}

pub(crate) fn deinterleave_channels(data: &[f64], channels: usize) -> Vec<Vec<f64>> {
    if channels == 0 {
        return Vec::new();
    }
    if channels == 1 {
        return vec![data.to_vec()];
    }
    if channels == 2 {
        let (left, right) = simd::deinterleave_stereo_f64(data);
        return vec![left, right];
    }

    let frames = data.len() / channels;
    let mut separated = vec![Vec::with_capacity(frames); channels];
    for frame in data.chunks_exact(channels) {
        for (channel, &sample) in separated.iter_mut().zip(frame.iter()) {
            channel.push(sample);
        }
    }
    separated
}

pub(crate) fn interleave_channels(channels: &[Vec<f64>]) -> Vec<f64> {
    if channels.is_empty() {
        return Vec::new();
    }
    if channels.len() == 1 {
        return channels[0].clone();
    }
    if channels.len() == 2 {
        return simd::interleave_stereo_f64(&channels[0], &channels[1]);
    }

    let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
    let mut interleaved = Vec::with_capacity(frames * channels.len());
    for frame in 0..frames {
        for channel in channels {
            interleaved.push(channel[frame]);
        }
    }
    interleaved
}

pub(crate) fn detect_residual_mask(samples: &[f64], params: ResidualDetectParams) -> Vec<bool> {
    let len = samples.len();
    let mut mask = vec![false; len];
    if len < 5 {
        return mask;
    }

    let residual = second_difference(samples);
    let half_window = params.mad_window / 2;
    let mut z = vec![0.0f64; len];
    for i in 1..len - 1 {
        let sigma = local_sigma_from_abs_median(&residual, i, half_window);
        z[i] = residual[i].abs() / sigma.max(1.0e-9);
    }

    let mut i = 2usize;
    while i < len.saturating_sub(2) {
        let rms = local_rms(samples, i, half_window);
        let edge = (samples[i] - samples[i - 1])
            .abs()
            .max((samples[i] - samples[i + 1]).abs());
        if z[i] > params.z_hi && edge > params.edge_rms_mult * rms.max(1.0e-9) {
            let mut start = i;
            let mut end = i + 1;

            while start > 1
                && z[start - 1] > params.z_lo
                && end - (start - 1) <= params.max_click_samples
            {
                start -= 1;
            }
            while end < len - 1
                && z[end] > params.z_lo
                && (end + 1) - start <= params.max_click_samples
            {
                end += 1;
            }

            if end - start <= params.max_click_samples {
                for slot in &mut mask[start..end] {
                    *slot = true;
                }
            }

            i = end + 1;
        } else {
            i += 1;
        }
    }

    mask
}

pub(crate) fn detect_clip_mask(samples: &[f64], params: ClipDetectParams) -> Vec<bool> {
    let len = samples.len();
    let mut mask = vec![false; len];
    if len < params.min_region_samples {
        return mask;
    }

    let peak_abs = samples
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f64, f64::max);
    if peak_abs < 0.98 {
        return mask;
    }

    let threshold = peak_abs * params.peak_ratio;
    let half_window = 32usize.min(len.saturating_sub(1)).max(1);
    let mut i = 0usize;
    while i < len {
        if samples[i].abs() >= threshold {
            let start = i;
            let sign = samples[i].signum();
            let mut min_sample = samples[i];
            let mut max_sample = samples[i];
            while i < len
                && samples[i].abs() >= threshold
                && (sign == 0.0 || samples[i].signum() == sign || samples[i] == 0.0)
                && i - start <= params.max_region_samples
            {
                min_sample = min_sample.min(samples[i]);
                max_sample = max_sample.max(samples[i]);
                i += 1;
            }
            let end = i;
            let region_len = end.saturating_sub(start);
            if region_len >= params.min_region_samples && region_len <= params.max_region_samples {
                let span = (max_sample - min_sample).abs();
                let edge_left = if start > 0 {
                    (samples[start] - samples[start - 1]).abs()
                } else {
                    0.0
                };
                let edge_right = if end < len {
                    (samples[end - 1] - samples[end]).abs()
                } else {
                    0.0
                };
                let edge = edge_left.max(edge_right);
                let rms = local_rms(samples, start.min(len - 1), half_window);
                if span <= peak_abs * params.flat_span_ratio
                    && edge > params.edge_rms_mult * rms.max(1.0e-4)
                {
                    for slot in &mut mask[start..end] {
                        *slot = true;
                    }
                }
            }
        } else {
            i += 1;
        }
    }

    mask
}

pub(crate) fn second_difference(samples: &[f64]) -> Vec<f64> {
    let len = samples.len();
    let mut residual = vec![0.0f64; len];
    if len < 3 {
        return residual;
    }
    for i in 1..len - 1 {
        residual[i] = samples[i + 1] - 2.0 * samples[i] + samples[i - 1];
    }
    residual
}

pub(crate) fn local_sigma_from_abs_median(
    samples: &[f64],
    index: usize,
    half_window: usize,
) -> f64 {
    let start = index.saturating_sub(half_window);
    let end = (index + half_window + 1).min(samples.len());
    let median_abs = median_of_abs(&samples[start..end]);
    (median_abs / 0.6745).max(1.0e-9)
}

pub(crate) fn local_rms(samples: &[f64], index: usize, half_window: usize) -> f64 {
    let start = index.saturating_sub(half_window);
    let end = (index + half_window + 1).min(samples.len());
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &sample in &samples[start..end] {
        sum += sample * sample;
        count += 1;
    }
    (sum / count.max(1) as f64).sqrt()
}

pub(crate) fn median_of_abs(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = samples.iter().map(|sample| sample.abs()).collect();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    median_of_sorted(&sorted)
}

pub(crate) fn median_of_sorted(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

pub(crate) fn median_of_slice(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    median_of_sorted(&sorted)
}

pub(crate) fn percentile_of_slice(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let idx = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).floor() as usize;
    sorted[idx]
}

pub(crate) fn dilate_mask(mask: &mut [bool], pad: usize) {
    if pad == 0 || mask.is_empty() {
        return;
    }

    let original = mask.to_vec();
    for (index, flagged) in original.iter().copied().enumerate() {
        if flagged {
            let start = index.saturating_sub(pad);
            let end = (index + pad + 1).min(mask.len());
            for slot in &mut mask[start..end] {
                *slot = true;
            }
        }
    }
}

pub(crate) fn mark_regions(mask: &mut [bool], regions: &[Region]) {
    for &(start, end) in regions {
        let start = start.min(mask.len());
        let end = end.min(mask.len());
        if start < end {
            for slot in &mut mask[start..end] {
                *slot = true;
            }
        }
    }
}

pub(crate) fn mask_to_regions(mask: &[bool]) -> Vec<Region> {
    let mut regions = Vec::new();
    let mut i = 0usize;
    while i < mask.len() {
        if mask[i] {
            let start = i;
            while i < mask.len() && mask[i] {
                i += 1;
            }
            regions.push((start, i));
        } else {
            i += 1;
        }
    }
    regions
}

pub(crate) fn merge_close_regions(regions: &[Region], max_gap: usize) -> Vec<Region> {
    if regions.is_empty() {
        return Vec::new();
    }

    let mut merged = Vec::new();
    let mut current = regions[0];
    for &(start, end) in &regions[1..] {
        if start <= current.1 + max_gap {
            current.1 = current.1.max(end);
        } else {
            merged.push(current);
            current = (start, end);
        }
    }
    merged.push(current);
    merged
}

pub(crate) fn total_region_len(regions: &[Region]) -> usize {
    regions
        .iter()
        .map(|(start, end)| end.saturating_sub(*start))
        .sum()
}

pub(crate) fn odd_window(window: usize, max_window: usize) -> usize {
    let mut result = window.max(3);
    if result % 2 == 0 {
        result += 1;
    }
    let max_odd = if max_window % 2 == 0 {
        max_window.saturating_sub(1)
    } else {
        max_window
    }
    .max(3);
    result.min(max_odd)
}

pub(crate) fn repair_regions(
    samples: &mut [f64],
    regions: &[Region],
    params: RepairParams,
    repair_mode: RepairMode,
) {
    for &(start, end) in regions {
        if start >= end || end > samples.len() {
            continue;
        }
        repair_gap(samples, start, end, params, repair_mode);
    }
}

pub(crate) fn repair_gap(
    samples: &mut [f64],
    start: usize,
    end: usize,
    params: RepairParams,
    repair_mode: RepairMode,
) {
    match repair_mode {
        RepairMode::Ar => repair_gap_ar(samples, start, end, params),
        RepairMode::Median => repair_gap_median(samples, start, end, params),
    }
}

pub(crate) fn repair_gap_ar(samples: &mut [f64], start: usize, end: usize, params: RepairParams) {
    let gap = end.saturating_sub(start);
    if gap == 0 {
        return;
    }

    let context = (gap * 8).clamp(params.min_context, params.max_context);
    let left_len = start.min(context);
    let right_len = (samples.len() - end).min(context);
    if left_len < 8 || right_len < 8 {
        linear_fill(samples, start, end);
        return;
    }

    let left = &samples[start - left_len..start];
    let right = &samples[end..end + right_len];
    let order = params
        .lpc_order
        .min(left.len() / 3)
        .min(right.len() / 3)
        .max(1);
    if order < 4 {
        linear_fill(samples, start, end);
        return;
    }

    let a_left = burg_lpc(left, order);
    let mut right_rev = right.to_vec();
    right_rev.reverse();
    let a_right = burg_lpc(&right_rev, order);

    let pred_fwd = predict_forward(left, &a_left, gap);
    let mut pred_bwd = predict_forward(&right_rev, &a_right, gap);
    pred_bwd.reverse();

    let left_bound = left
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f64, f64::max);
    let right_bound = right
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f64, f64::max);
    let clamp_bound = left_bound.max(right_bound).max(1.0);

    for i in 0..gap {
        let t = (i + 1) as f64 / (gap + 1) as f64;
        let w_right = 0.5 - 0.5 * (PI * t).cos();
        let w_left = 1.0 - w_right;
        let repaired = w_left * pred_fwd[i] + w_right * pred_bwd[i];
        samples[start + i] = repaired.clamp(-clamp_bound, clamp_bound);
    }
}

pub(crate) fn repair_gap_median(
    samples: &mut [f64],
    start: usize,
    end: usize,
    params: RepairParams,
) {
    let gap = end.saturating_sub(start);
    if gap == 0 {
        return;
    }

    let window = params.min_context.min(32);
    let left_start = start.saturating_sub(window);
    let right_end = (end + window).min(samples.len());
    let left = &samples[left_start..start];
    let right = &samples[end..right_end];
    if left.is_empty() || right.is_empty() {
        linear_fill(samples, start, end);
        return;
    }

    let left_median = median_of_slice(left);
    let right_median = median_of_slice(right);
    let bound = left
        .iter()
        .chain(right.iter())
        .map(|sample| sample.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);
    for i in 0..gap {
        let t = (i + 1) as f64 / (gap + 1) as f64;
        let bridged = left_median * (1.0 - t) + right_median * t;
        samples[start + i] = bridged.clamp(-bound, bound);
    }
}

pub(crate) fn linear_fill(samples: &mut [f64], start: usize, end: usize) {
    let gap = end.saturating_sub(start);
    if gap == 0 {
        return;
    }

    let left = if start > 0 { samples[start - 1] } else { 0.0 };
    let right = if end < samples.len() {
        samples[end]
    } else {
        left
    };
    for i in 0..gap {
        let t = (i + 1) as f64 / (gap + 1) as f64;
        samples[start + i] = left * (1.0 - t) + right * t;
    }
}

pub(crate) fn burg_lpc(samples: &[f64], order: usize) -> Vec<f64> {
    let len = samples.len();
    let order = order.min(len.saturating_sub(2));
    if order == 0 || len < 3 {
        return vec![1.0];
    }

    let epsilon = 1.0e-12f64;
    let mut ef = samples.to_vec();
    let mut eb = samples.to_vec();
    let mut a = vec![1.0f64];

    for m in 0..order {
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        for t in (m + 1)..len {
            numerator += ef[t] * eb[t - 1];
            denominator += ef[t] * ef[t] + eb[t - 1] * eb[t - 1];
        }

        let reflection = if denominator.abs() < epsilon {
            0.0
        } else {
            (-2.0 * numerator / (denominator + epsilon)).clamp(-0.9999, 0.9999)
        };

        let mut updated = vec![0.0f64; a.len() + 1];
        updated[0] = 1.0;
        for i in 1..a.len() {
            updated[i] = a[i] + reflection * a[a.len() - i];
        }
        updated[a.len()] = reflection;
        a = updated;

        for t in ((m + 1)..len).rev() {
            let ef_old = ef[t];
            let eb_old = eb[t - 1];
            ef[t] = ef_old + reflection * eb_old;
            eb[t - 1] = eb_old + reflection * ef_old;
        }
    }

    a
}

pub(crate) fn predict_forward(seed: &[f64], coefficients: &[f64], count: usize) -> Vec<f64> {
    let order = coefficients.len().saturating_sub(1);
    if order == 0 || seed.is_empty() || count == 0 {
        return vec![0.0; count];
    }

    let mut history = seed.to_vec();
    let mut output = Vec::with_capacity(count);
    for _ in 0..count {
        let hist_len = history.len();
        let used_order = order.min(hist_len);
        let mut predicted = 0.0f64;
        for i in 1..=used_order {
            predicted -= coefficients[i] * history[hist_len - i];
        }
        output.push(predicted);
        history.push(predicted);
    }
    output
}

pub(crate) fn kurtosis(samples: &[f64]) -> f64 {
    if samples.len() < 4 {
        return 0.0;
    }
    let mean = samples.iter().copied().sum::<f64>() / samples.len() as f64;
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for &sample in samples {
        let centered = sample - mean;
        let sq = centered * centered;
        m2 += sq;
        m4 += sq * sq;
    }
    let n = samples.len() as f64;
    let var = m2 / n;
    if var <= 1.0e-12 {
        return 0.0;
    }
    (m4 / n) / (var * var)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_near_clipped(samples: &[f64]) -> usize {
        samples.iter().filter(|sample| sample.abs() >= 0.99).count()
    }

    fn clipped_plateau_source() -> Vec<f64> {
        let mut original = vec![0.0f64; 4096];
        for (i, sample) in original.iter_mut().enumerate() {
            let t = i as f64 / 16_000.0;
            *sample = 0.22 * (2.0 * PI * 220.0 * t).sin();
        }
        for sample in &mut original[1200..1204] {
            *sample = 1.0;
        }
        for sample in &mut original[2400..2404] {
            *sample = -1.0;
        }
        original
    }

    #[test]
    fn clipped_plateaus_are_repaired_without_length_change() {
        let mut channels = vec![clipped_plateau_source()];
        let original = channels[0].clone();

        apply_pre_declip_to_channels(&mut channels, 16_000);

        assert_eq!(channels[0].len(), original.len());
        assert!(count_near_clipped(&channels[0]) < count_near_clipped(&original));
    }

    #[test]
    fn ar_repair_stays_finite_and_continuous() {
        let mut samples: Vec<f64> = (0..512)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                0.3 * (2.0 * PI * 330.0 * t).sin()
            })
            .collect();
        for sample in &mut samples[120..124] {
            *sample = 0.0;
        }

        repair_gap_ar(&mut samples, 120, 124, RepairParams::for_click());

        assert!(samples[120..124].iter().all(|sample| sample.is_finite()));
        assert!((samples[120] - samples[119]).abs() < 0.2);
        assert!((samples[123] - samples[124]).abs() < 0.2);
    }

    #[test]
    fn median_repair_only_changes_marked_region_and_stays_bounded() {
        let mut samples: Vec<f64> = (0..256)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                0.25 * (2.0 * PI * 220.0 * t).sin()
            })
            .collect();
        let original = samples.clone();
        for sample in &mut samples[80..84] {
            *sample = 1.0;
        }

        repair_gap_median(&mut samples, 80, 84, RepairParams::for_click());

        assert_eq!(&samples[..80], &original[..80]);
        assert_eq!(&samples[84..], &original[84..]);
        assert!(samples[80..84].iter().all(|sample| sample.is_finite()));
        let bound = original
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0f64, f64::max);
        assert!(
            samples[80..84]
                .iter()
                .all(|sample| sample.abs() <= bound + 1.0e-3)
        );
    }
}
