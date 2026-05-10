// Aniso64AVX2.cpp — AVX2+FMA3 instantiation of 64-tap anisotropic sinc kernel
//
// This file MUST be compiled with -mavx2 -mfma (separate from the rest of the build).
// 4 doubles per YMM register, vfmadd (single-instruction fused multiply-add).
// Same template + accumulator count as AVX, but FMA halves the critical path.

#include "Aniso64Kernel.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

extern "C" {

double aniso64_dot_mono_avx2(
	const double * __restrict__ kernel,
	const double * __restrict__ samples)
{
	return Aniso64MonoKernel<4, 4>::compute(kernel, samples);
}

void aniso64_dot_stereo_avx2(
	const double * __restrict__ kernel,
	const double * __restrict__ samples,
	double * __restrict__ outL,
	double * __restrict__ outR)
{
	Aniso64StereoKernel<4, 2>::compute(kernel, samples, outL, outR);
}

} // extern "C"

#endif // x86
