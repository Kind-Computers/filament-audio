/*
 * AGC.h
 * -----
 * Purpose: Automatic Gain Control
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */

#pragma once

#include "openmpt/all/BuildSettings.hpp"
#include "openmpt/base/Types.hpp"


OPENMPT_NAMESPACE_BEGIN


#ifndef NO_AGC

enum class AGCProfile
{
	Stock = 0,
	Gentle = 1,
};

class CAGC
{
private:
	uint32 m_nAGC;
	std::size_t m_nAGCRecoverCount;
	std::size_t m_nAGCAttackCount;
	uint32 m_Timeout;
	AGCProfile m_Profile;
public:
	CAGC();
	void Initialize(bool bReset, uint32 MixingFreq);
	AGCProfile GetProfile() const { return m_Profile; }
	void SetProfile(AGCProfile profile) { m_Profile = profile; }
public:
#ifdef MPT_INTMIXER
	void Process(int *MixSoundBuffer, int *RearSoundBuffer, std::size_t count, std::size_t nChannels);
#else
	void Process(float *MixSoundBuffer, float *RearSoundBuffer, std::size_t count, std::size_t nChannels);
	void Process(double *MixSoundBuffer, double *RearSoundBuffer, std::size_t count, std::size_t nChannels);
#endif
	void Adjust(uint32 oldVol, uint32 newVol);
};

#endif // NO_AGC


OPENMPT_NAMESPACE_END
