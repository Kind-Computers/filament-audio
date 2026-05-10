// Aniso64SSE2.cpp — SSE2 instantiation of 64-tap anisotropic sinc kernel
//
// Baseline x86-64: no special compiler flags required.
// 2 doubles per XMM register, 8 accumulators (mono), 4/ch (stereo).
// 32 vector iterations fully unrolled = 64 multiply-adds + 7 reduction adds.

#include "Aniso64Kernel.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

extern "C" {

double aniso64_dot_mono_sse2(
	const double * __restrict__ kernel,
	const double * __restrict__ samples)
{
	return Aniso64MonoKernel<2, 8>::compute(kernel, samples);
}

void aniso64_dot_stereo_sse2(
	const double * __restrict__ kernel,
	const double * __restrict__ samples,
	double * __restrict__ outL,
	double * __restrict__ outR)
{
	Aniso64StereoKernel<2, 4>::compute(kernel, samples, outL, outR);
}

} // extern "C"

#endif // x86
