// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Kind Computers, LLC.

use iced::window;
use std::path::Path;

pub fn create_icon() -> window::Icon {
    let rgba = create_icon_rgba(32);
    window::icon::from_rgba(rgba, 32, 32).expect("valid icon dimensions")
}

pub fn create_icon_rgba(size: u32) -> Vec<u8> {
    let center = (size as f32 - 1.0) / 2.0;
    let scale = size as f32 / 32.0; // normalize distances relative to 32px design
    let mut rgba = vec![0u8; (size * size * 4) as usize];

    for y in 0..size {
        for x in 0..size {
            let dx = (x as f32 - center) / scale;
            let dy = (y as f32 - center) / scale;
            let dist = (dx * dx + dy * dy).sqrt();
            let angle = dy.atan2(dx);

            // 8-pointed star ray intensity
            let ray = ray_intensity(angle, dist);

            // Radial glow layers
            let (r, g, b, a) = if dist < 3.0 {
                // Bright white core
                let t = dist / 3.0;
                let v = 255.0 - t * 30.0;
                (v as u8, v as u8, v as u8, 255)
            } else if dist < 7.0 {
                // Cyan glow ring
                let t = (dist - 3.0) / 4.0;
                let r = lerp(200.0, 0.0, t);
                let g = lerp(240.0, 180.0, t);
                let b = lerp(255.0, 240.0, t);
                let a = lerp(255.0, 200.0, t);
                (r as u8, g as u8, b as u8, a as u8)
            } else if dist < 14.0 {
                // Outer glow fade + rays
                let t = (dist - 7.0) / 7.0;
                let glow = (1.0 - t) * (1.0 - t);
                let intensity = (glow + ray * 1.5 * (1.0 - t * 0.5)).min(1.0);
                let r = (intensity * 0.0) as u8;
                let g = (intensity * 200.0) as u8;
                let b = (intensity * 255.0) as u8;
                let a = (intensity * 220.0) as u8;
                (r, g, b, a.max(1))
            } else {
                // Background with faint ray traces
                let faint = ray * (1.0 - ((dist - 14.0) / 4.0).min(1.0));
                if faint > 0.05 {
                    let g = (faint * 80.0) as u8;
                    let b = (faint * 120.0) as u8;
                    (0, g, b, (faint * 150.0) as u8)
                } else {
                    (0, 0, 0, 0)
                }
            };

            let idx = ((y * size + x) * 4) as usize;
            rgba[idx] = r;
            rgba[idx + 1] = g;
            rgba[idx + 2] = b;
            rgba[idx + 3] = a;
        }
    }

    rgba
}

pub fn save_icon_png(path: &Path) -> Result<(), String> {
    let size = 256u32;
    let rgba = create_icon_rgba(size);
    image::save_buffer(path, &rgba, size, size, image::ColorType::Rgba8)
        .map_err(|e| format!("Failed to save icon PNG: {e}"))
}

pub fn ray_intensity(angle: f32, dist: f32) -> f32 {
    // 8 rays (every 45 degrees) + 4 thinner intermediate rays
    let primary = (angle * 4.0).cos().abs(); // 8-fold symmetry
    let secondary = (angle * 4.0 + std::f32::consts::FRAC_PI_4).cos().abs();

    // Sharpen the rays with power function
    let primary = primary.powf(8.0);
    let secondary = secondary.powf(16.0) * 0.4;

    // Rays get thinner with distance
    let width_falloff = 1.0 / (1.0 + dist * 0.05);

    (primary + secondary) * width_falloff
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
