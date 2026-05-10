# Changelog

All notable changes to Filament Audio are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-10

Initial public release.

### Player
- Vendored libopenmpt fork rebuilt for end-to-end double-precision (`mixsample_t = double`)
- 64-bit mixer pipeline with Hermite smoothstep volume ramps and cascaded 4-pole
  channel filter (IT-style 2-pole resonant biquad + Butterworth post-filter)
- 64-tap polyphase sinc resampler with 8192 phases and an octave-spaced mipmap chain
- Full `double`-precision pitch tracking (`PitchT = double`, `FreqT = double`) for
  vibrato, portamento, and slide effects
- SIMD kernels compiled for SSE2, AVX, AVX2, and AVX-512 with runtime dispatch
- Module loading directly from archives: `.zip`, `.7z`, `.rar`, `.tar.*`, `.lha`,
  `.cab`, `.iso`

### Remastering
- AI sample upscaling via four optional external backends: AudioSR, LavaSR,
  FLowHigh, AP-BWE
- Live A/B between Original, 48 kHz reference (sinc-resampled), and any AI engine
  during playback — no song restart
- Pattern offset effects (`Oxx`, `SAx`) and portamento are auto-rescaled when sample
  rate changes
- Reference-only cleanup pipeline available without AI engines (declick, AR)
- Persistent on-disk sample cache at the platform cache directory

### Rendering
- Export to FLAC or AAC (512 kbps), defaulting to 96 kHz
- Batch CLI rendering for directories of modules with per-engine selection
- HRTF-based binaural rendering option

### Platform
- Linux `x86_64-unknown-linux-gnu` is the supported public target
- Linux desktop integration via `--install-icon` (XDG `.desktop` + 256×256 icon)
- GPU and hybrid CPU/GPU upscaling modes (NVIDIA CUDA, AMD ROCm, Intel XPU)

### Repository
- Public source release at <https://github.com/Kind-Computers/filament-audio>
- MIT licensed; AI backends remain external (not bundled)
