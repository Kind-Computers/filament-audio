/*
 * MixFuncTable.h
 * --------------
 * Purpose: Table containing all mixer functions.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once

#include "openmpt/all/BuildSettings.hpp"

#include "MixerInterface.h"

OPENMPT_NAMESPACE_BEGIN

namespace MixFuncTable
{
	// Table index bits:
	// [b2-b0] format (8-bit-mono, 16-bit-mono, 8-bit-stereo, 16-bit-stereo, float32-mono, float64-mono, float32-stereo, float64-stereo)
	// [b3]    ramp
	// [b4]    filter
	// [b7-b5] src type

	// Sample type / processing type index
	enum FunctionIndex
	{
		ndx16Bit  = 0x01,
		ndxStereo = 0x02,
		ndxFloat  = 0x04,
		ndxRamp   = 0x08,
		ndxFilter = 0x10,
	};

	// SRC index
	enum ResamplingIndex
	{
		ndxNoInterpolation = 0x00,
		ndxLinear          = 0x20,
		ndxFastSinc        = 0x40,
		ndxKaiser          = 0x60,
		ndxFIRFilter       = 0x80,
		ndxAmigaBlep       = 0xA0,
		ndxAniso64         = 0xC0,
		ndxCatmullRom      = 0xE0,
	};

	extern const MixFuncInterface Functions[8 * 32];

	ResamplingIndex ResamplingModeToMixFlags(ResamplingMode resamplingMode);
}

OPENMPT_NAMESPACE_END
