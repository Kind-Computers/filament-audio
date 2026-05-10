// Aniso64AVX512.cpp — AVX-512F+VL instantiation of 64-tap anisotropic sinc kernel
//
// This file MUST be compiled with -mavx512f -mavx512vl (separate from the rest of the build).
// 8 doubles per ZMM register — 64 taps in just 8 FMA instructions.
// _mm512_reduce_add_pd gives single-instruction horizontal sum.

#include "Aniso64Kernel.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

extern "C" {

double aniso64_dot_mono_avx512(
	const double * __restrict__ kernel,
	const double * __restrict__ samples)
{
	return Aniso64MonoKernel<8, 2>::compute(kernel, samples);
}

void aniso64_dot_stereo_avx512(
	const double * __restrict__ kernel,
	const double * __restrict__ samples,
	double * __restrict__ outL,
	double * __restrict__ outR)
{
	Aniso64StereoKernel<8, 1>::compute(kernel, samples, outL, outR);
}

} // extern "C"

#endif // x86
