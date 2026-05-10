// Aniso64Kernel.h — Template SIMD kernels for 64-tap anisotropic sinc interpolation
//
// Demoscene spirit: one template, four ISA instantiations, fully unrolled.
// SSE2 → AVX → AVX2+FMA → AVX-512: same algorithm, wider pipes.
//
// Each SimdTraits<WIDTH> specialization maps ISA intrinsics to a uniform interface.
// The compiler unrolls all loops (ACC_COUNT and OUTER are compile-time constants).
// FMA vs mul+add is resolved by #ifdef __FMA__ at compile time — zero runtime cost.
//
// Register-optimal accumulator counts per ISA:
//   SSE2:    mono=8  stereo=4/ch  (fills 16 XMM regs)
//   AVX:     mono=4  stereo=2/ch  (16 YMM regs, no FMA → mul+add)
//   AVX2:    mono=4  stereo=2/ch  (16 YMM regs, FMA hides latency)
//   AVX-512: mono=2  stereo=1/ch  (32 ZMM regs, 8 doubles/vec)

#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

template<int WIDTH> struct SimdTraits;


// ==========================================================================
// SSE2 — baseline x86-64: 2 doubles per XMM register
// ==========================================================================
template<>
struct SimdTraits<2>
{
	using vec_t = __m128d;
	static vec_t zero() { return _mm_setzero_pd(); }
	static vec_t load(const double *p) { return _mm_loadu_pd(p); }
	static vec_t add(vec_t a, vec_t b) { return _mm_add_pd(a, b); }

	// SSE2 has no FMA — always emulate as mul + add
	static vec_t fmadd(vec_t a, vec_t b, vec_t c)
	{
		return _mm_add_pd(_mm_mul_pd(a, b), c);
	}

	// Deinterleave 2 stereo pairs: [L0 R0] [L1 R1] → [L0 L1] [R0 R1]
	static void deinterleave(vec_t s0, vec_t s1, vec_t &L, vec_t &R)
	{
		L = _mm_unpacklo_pd(s0, s1);  // [s0[0], s1[0]] = [L0, L1]
		R = _mm_unpackhi_pd(s0, s1);  // [s0[1], s1[1]] = [R0, R1]
	}

	// Horizontal sum: v[0] + v[1]
	static double reduce(vec_t v)
	{
		__m128d hi = _mm_unpackhi_pd(v, v);       // [v[1], v[1]]
		return _mm_cvtsd_f64(_mm_add_sd(v, hi));   // v[0] + v[1]
	}
};


// ==========================================================================
// AVX / AVX2+FMA — 4 doubles per YMM register
// Compiled with -mavx (AVX1) or -mavx2 -mfma (AVX2+FMA3).
// The #ifdef __FMA__ / __AVX2__ selects optimal instructions at compile time.
// ==========================================================================
#ifdef __AVX__
template<>
struct SimdTraits<4>
{
	using vec_t = __m256d;
	static vec_t zero() { return _mm256_setzero_pd(); }
	static vec_t load(const double *p) { return _mm256_loadu_pd(p); }
	static vec_t add(vec_t a, vec_t b) { return _mm256_add_pd(a, b); }

	static vec_t fmadd(vec_t a, vec_t b, vec_t c)
	{
#ifdef __FMA__
		return _mm256_fmadd_pd(a, b, c);              // AVX2+FMA3: single instruction
#else
		return _mm256_add_pd(_mm256_mul_pd(a, b), c);  // AVX1: mul then add
#endif
	}

	// Deinterleave 4 stereo pairs: [L0 R0 L1 R1] [L2 R2 L3 R3] → [L0..L3] [R0..R3]
	static void deinterleave(vec_t s0, vec_t s1, vec_t &L, vec_t &R)
	{
#ifdef __AVX2__
		// AVX2: cross-lane permute via VPERMPD
		vec_t lo = _mm256_unpacklo_pd(s0, s1);    // [L0 L2 | L1 L3] (within-lane unpack)
		vec_t hi = _mm256_unpackhi_pd(s0, s1);    // [R0 R2 | R1 R3]
		L = _mm256_permute4x64_pd(lo, 0xD8);      // [L0 L1 L2 L3] (0b11_01_10_00)
		R = _mm256_permute4x64_pd(hi, 0xD8);      // [R0 R1 R2 R3]
#else
		// AVX1: no cross-lane integer permute — extract to 128-bit halves
		__m128d s0_lo = _mm256_castpd256_pd128(s0);       // [L0 R0]
		__m128d s0_hi = _mm256_extractf128_pd(s0, 1);     // [L1 R1]
		__m128d s1_lo = _mm256_castpd256_pd128(s1);       // [L2 R2]
		__m128d s1_hi = _mm256_extractf128_pd(s1, 1);     // [L3 R3]
		__m128d l01 = _mm_unpacklo_pd(s0_lo, s0_hi);      // [L0 L1]
		__m128d r01 = _mm_unpackhi_pd(s0_lo, s0_hi);      // [R0 R1]
		__m128d l23 = _mm_unpacklo_pd(s1_lo, s1_hi);      // [L2 L3]
		__m128d r23 = _mm_unpackhi_pd(s1_lo, s1_hi);      // [R2 R3]
		L = _mm256_insertf128_pd(_mm256_castpd128_pd256(l01), l23, 1);  // [L0 L1 | L2 L3]
		R = _mm256_insertf128_pd(_mm256_castpd128_pd256(r01), r23, 1);  // [R0 R1 | R2 R3]
#endif
	}

	// Horizontal sum: v[0] + v[1] + v[2] + v[3]
	static double reduce(vec_t v)
	{
		__m128d lo  = _mm256_castpd256_pd128(v);      // [v[0], v[1]]
		__m128d hi  = _mm256_extractf128_pd(v, 1);    // [v[2], v[3]]
		__m128d sum = _mm_add_pd(lo, hi);              // [v[0]+v[2], v[1]+v[3]]
		__m128d shuf = _mm_unpackhi_pd(sum, sum);      // [v[1]+v[3], v[1]+v[3]]
		return _mm_cvtsd_f64(_mm_add_sd(sum, shuf));
	}
};
#endif // __AVX__


// ==========================================================================
// AVX-512 — 8 doubles per ZMM register
// ==========================================================================
#ifdef __AVX512F__
template<>
struct SimdTraits<8>
{
	using vec_t = __m512d;
	static vec_t zero() { return _mm512_setzero_pd(); }
	static vec_t load(const double *p) { return _mm512_loadu_pd(p); }
	static vec_t add(vec_t a, vec_t b) { return _mm512_add_pd(a, b); }

	static vec_t fmadd(vec_t a, vec_t b, vec_t c)
	{
		return _mm512_fmadd_pd(a, b, c);
	}

	// Deinterleave 8 stereo pairs via cross-register permute
	static void deinterleave(vec_t s0, vec_t s1, vec_t &L, vec_t &R)
	{
		const __m512i even_idx = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
		const __m512i odd_idx  = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
		L = _mm512_permutex2var_pd(s0, even_idx, s1);
		R = _mm512_permutex2var_pd(s0, odd_idx, s1);
	}

	// Single-instruction horizontal sum
	static double reduce(vec_t v)
	{
		return _mm512_reduce_add_pd(v);
	}
};
#endif // __AVX512F__


// ==========================================================================
// Kernel templates — fully unrolled at compile time
// ==========================================================================

// 64-tap mono dot product: Σ kernel[i] * samples[i], i=0..63
//
// WIDTH:     doubles per SIMD vector (2, 4, or 8)
// ACC_COUNT: independent accumulators to hide mul-add / FMA latency
//
// Both loops are fully unrolled: OUTER and ACC_COUNT are compile-time constants.
// Total SIMD ops: 64/WIDTH fmadd + (ACC_COUNT-1) add + 1 reduce.
template<int WIDTH, int ACC_COUNT>
struct Aniso64MonoKernel
{
	static_assert(64 % WIDTH == 0, "64 taps must divide evenly by SIMD width");
	static_assert((64 / WIDTH) % ACC_COUNT == 0, "vector count must divide evenly by accumulator count");

	static double compute(const double * __restrict__ kernel, const double * __restrict__ samples)
	{
		using T = SimdTraits<WIDTH>;
		using vec_t = typename T::vec_t;

		vec_t acc[ACC_COUNT];
		for(int a = 0; a < ACC_COUNT; a++)
			acc[a] = T::zero();

		constexpr int VECTORS = 64 / WIDTH;
		constexpr int OUTER = VECTORS / ACC_COUNT;

		for(int i = 0; i < OUTER; i++)
		{
			for(int a = 0; a < ACC_COUNT; a++)
			{
				const int offset = (i * ACC_COUNT + a) * WIDTH;
				acc[a] = T::fmadd(T::load(kernel + offset),
				                   T::load(samples + offset), acc[a]);
			}
		}

		vec_t total = acc[0];
		for(int a = 1; a < ACC_COUNT; a++)
			total = T::add(total, acc[a]);
		return T::reduce(total);
	}
};


// 64-tap stereo dot product: processes interleaved [L0,R0,L1,R1,...] samples.
// Writes separate L and R results.
//
// ACC_COUNT: independent accumulators PER CHANNEL (total regs = 2 * ACC_COUNT)
template<int WIDTH, int ACC_COUNT>
struct Aniso64StereoKernel
{
	static_assert(64 % WIDTH == 0, "64 taps must divide evenly by SIMD width");
	static_assert((64 / WIDTH) % ACC_COUNT == 0, "vector count must divide evenly by accumulator count");

	static void compute(
		const double * __restrict__ kernel,
		const double * __restrict__ samples,
		double * __restrict__ outL,
		double * __restrict__ outR)
	{
		using T = SimdTraits<WIDTH>;
		using vec_t = typename T::vec_t;

		vec_t accL[ACC_COUNT], accR[ACC_COUNT];
		for(int a = 0; a < ACC_COUNT; a++)
		{
			accL[a] = T::zero();
			accR[a] = T::zero();
		}

		constexpr int VECTORS = 64 / WIDTH;
		constexpr int OUTER = VECTORS / ACC_COUNT;

		for(int i = 0; i < OUTER; i++)
		{
			for(int a = 0; a < ACC_COUNT; a++)
			{
				const int kidx = (i * ACC_COUNT + a) * WIDTH;
				const int sidx = kidx * 2;  // stride 2: interleaved L/R

				vec_t k  = T::load(kernel + kidx);
				vec_t s0 = T::load(samples + sidx);
				vec_t s1 = T::load(samples + sidx + WIDTH);
				vec_t L, R;
				T::deinterleave(s0, s1, L, R);

				accL[a] = T::fmadd(k, L, accL[a]);
				accR[a] = T::fmadd(k, R, accR[a]);
			}
		}

		vec_t totalL = accL[0], totalR = accR[0];
		for(int a = 1; a < ACC_COUNT; a++)
		{
			totalL = T::add(totalL, accL[a]);
			totalR = T::add(totalR, accR[a]);
		}
		*outL = T::reduce(totalL);
		*outR = T::reduce(totalR);
	}
};

#endif // x86
