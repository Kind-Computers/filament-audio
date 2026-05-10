// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use super::shared::{
    RepairMode, RepairParams, ResidualDetectParams, apply_pre_declip_to_channels,
    deinterleave_channels, detect_residual_mask, detect_union_regions, interleave_channels,
    repair_regions_for_all_channels,
};
use super::{CleanupMode, RetiredCleanupPreset};

pub(crate) fn apply_cleanup_v1(
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
            apply_repair_pass(
                &mut separated,
                click_detect_params(sample_rate),
                RepairParams::for_click(),
                RepairMode::Ar,
            );
        }
        CleanupMode::DeclickMedian => {
            apply_repair_pass(
                &mut separated,
                click_detect_params(sample_rate),
                RepairParams::for_click(),
                RepairMode::Median,
            );
        }
        CleanupMode::Decrackle => {
            for _ in 0..2 {
                apply_repair_pass(
                    &mut separated,
                    crackle_detect_params(sample_rate),
                    RepairParams::for_crackle(),
                    RepairMode::Ar,
                );
            }
        }
    }
    Ok(interleave_channels(&separated))
}

pub(crate) fn apply_retired_cleanup_preset_v1(
    data: &[f64],
    sample_rate: u32,
    channels: usize,
    preset: RetiredCleanupPreset,
) -> Result<Vec<f64>, String> {
    if channels == 0 || data.is_empty() {
        return Ok(Vec::new());
    }

    let mut separated = deinterleave_channels(data, channels);
    apply_pre_declip_to_channels(&mut separated, sample_rate);
    match preset {
        RetiredCleanupPreset::Light => {
            apply_repair_pass(
                &mut separated,
                click_detect_params(sample_rate),
                RepairParams::for_click(),
                RepairMode::Ar,
            );
        }
        RetiredCleanupPreset::Archival => {
            apply_repair_pass(
                &mut separated,
                click_detect_params(sample_rate),
                RepairParams::for_click(),
                RepairMode::Ar,
            );
            for _ in 0..2 {
                apply_repair_pass(
                    &mut separated,
                    crackle_detect_params(sample_rate),
                    RepairParams::for_crackle(),
                    RepairMode::Ar,
                );
            }
        }
    }
    Ok(interleave_channels(&separated))
}

pub(super) fn click_detect_params(sample_rate: u32) -> ResidualDetectParams {
    let sr = sample_rate as usize;
    ResidualDetectParams {
        mad_window: (((0.004 * sr as f64) as usize).max(63)) | 1,
        z_hi: 10.0,
        z_lo: 4.5,
        edge_rms_mult: 3.0,
        max_click_samples: ((0.0010 * sr as f64) as usize).clamp(8, 64),
        pad_samples: 1,
    }
}

pub(super) fn crackle_detect_params(sample_rate: u32) -> ResidualDetectParams {
    let sr = sample_rate as usize;
    ResidualDetectParams {
        mad_window: (((0.003 * sr as f64) as usize).max(31)) | 1,
        z_hi: 7.5,
        z_lo: 3.2,
        edge_rms_mult: 1.8,
        max_click_samples: ((0.00045 * sr as f64) as usize).clamp(2, 24),
        pad_samples: 0,
    }
}

fn apply_repair_pass(
    channels: &mut [Vec<f64>],
    detect_params: ResidualDetectParams,
    repair_params: RepairParams,
    repair_mode: RepairMode,
) -> bool {
    let regions = detect_union_regions(
        channels,
        detect_params.pad_samples,
        repair_params.merge_gap,
        |samples| detect_residual_mask(samples, detect_params),
    );
    if regions.is_empty() {
        return false;
    }
    repair_regions_for_all_channels(channels, &regions, repair_params, repair_mode);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cleanup::shared::{RepairMode, RepairParams};

    #[test]
    fn stereo_union_detection_keeps_channels_aligned() {
        let mut left: Vec<f64> = (0..1024)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
            })
            .collect();
        let right = left.clone();
        left[320] = 1.0;
        let mut channels = vec![left, right.clone()];

        let changed = apply_repair_pass(
            &mut channels,
            click_detect_params(16_000),
            RepairParams::for_click(),
            RepairMode::Ar,
        );
        assert!(changed);

        let right_diffs: Vec<usize> = channels[1]
            .iter()
            .zip(right.iter())
            .enumerate()
            .filter_map(|(idx, (lhs, rhs))| ((lhs - rhs).abs() > 1.0e-6).then_some(idx))
            .collect();
        assert!(!right_diffs.is_empty());
        assert!(right_diffs.iter().all(|idx| (316..=324).contains(idx)));
    }
}
