/*
 * AGC.cpp
 * -------
 * Purpose: Automatic Gain Control
 * Notes  : Ugh... This should really be removed at some point.
 * Authors: Olivier Lapicque
 *          OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#include "stdafx.h"
#include "../sounddsp/AGC.h"
#include "../soundlib/Mixer.h"


OPENMPT_NAMESPACE_BEGIN

	
//////////////////////////////////////////////////////////////////////////////////
// Automatic Gain Control

#ifndef NO_AGC

#define AGC_PRECISION		10
#define AGC_UNITY			(1 << AGC_PRECISION)

// Limiter
#define MIXING_LIMITMAX		(0x08100000)
#define MIXING_LIMITMIN		(-MIXING_LIMITMAX)

struct AGCProcessResult
{
	uint32 agc;
	std::size_t clipCount;
};

struct AGCProfileSettings
{
	std::size_t attackStepSamples;
	uint32 recoveryTimeoutMul;
};

static AGCProfileSettings GetAGCProfileSettings(AGCProfile profile)
{
	switch(profile)
	{
	case AGCProfile::Gentle:
		return {4u, 4u};
	case AGCProfile::Stock:
	default:
		return {1u, 1u};
	}
}


static AGCProcessResult ProcessAGC(int *pBuffer, int *pRearBuffer, std::size_t nSamples, std::size_t nChannels, int nAGC, std::size_t attackStepSamples, std::size_t &attackCount)
{
	std::size_t clipCount = 0;
	if(nChannels == 1)
	{
		while(nSamples--)
		{
			int val = (int)(((int64)*pBuffer * (int32)nAGC) >> AGC_PRECISION);
			if(val < MIXING_LIMITMIN || val > MIXING_LIMITMAX)
			{
				clipCount++;
				attackCount++;
				if(attackCount >= attackStepSamples)
				{
					nAGC--;
					attackCount -= attackStepSamples;
				}
			}
			*pBuffer = val;
			pBuffer++;
		}
	} else
	{
		if(nChannels == 2)
		{
			while(nSamples--)
				{
					int fl = (int)(((int64)pBuffer[0] * (int32)nAGC) >> AGC_PRECISION);
					int fr = (int)(((int64)pBuffer[1] * (int32)nAGC) >> AGC_PRECISION);
					bool dec = false;
					dec = dec || (fl < MIXING_LIMITMIN || fl > MIXING_LIMITMAX);
					dec = dec || (fr < MIXING_LIMITMIN || fr > MIXING_LIMITMAX);
					if(dec)
					{
						clipCount++;
						attackCount++;
						if(attackCount >= attackStepSamples)
						{
							nAGC--;
							attackCount -= attackStepSamples;
						}
					}
					pBuffer[0] = fl;
					pBuffer[1] = fr;
					pBuffer += 2;
				}
			} else if(nChannels == 4)
		{
			while(nSamples--)
			{
				int fl = (int)(((int64)pBuffer[0] * (int32)nAGC) >> AGC_PRECISION);
				int fr = (int)(((int64)pBuffer[1] * (int32)nAGC) >> AGC_PRECISION);
				int rl = (int)(((int64)pRearBuffer[0] * (int32)nAGC) >> AGC_PRECISION);
				int rr = (int)(((int64)pRearBuffer[1] * (int32)nAGC) >> AGC_PRECISION);
					bool dec = false;
					dec = dec || (fl < MIXING_LIMITMIN || fl > MIXING_LIMITMAX);
					dec = dec || (fr < MIXING_LIMITMIN || fr > MIXING_LIMITMAX);
					dec = dec || (rl < MIXING_LIMITMIN || rl > MIXING_LIMITMAX);
					dec = dec || (rr < MIXING_LIMITMIN || rr > MIXING_LIMITMAX);
					if(dec)
					{
						clipCount++;
						attackCount++;
						if(attackCount >= attackStepSamples)
						{
							nAGC--;
							attackCount -= attackStepSamples;
						}
					}
					pBuffer[0] = fl;
					pBuffer[1] = fr;
					pRearBuffer[0] = rl;
					pRearBuffer[1] = rr;
				pBuffer += 2;
				pRearBuffer += 2;
			}
		}
	}
	return {static_cast<uint32>(nAGC), clipCount};
}


CAGC::CAGC()
{
	m_Profile = AGCProfile::Stock;
	Initialize(true, 48000);
}


#ifdef MPT_INTMIXER
void CAGC::Process(int *MixSoundBuffer, int *RearSoundBuffer, std::size_t count, std::size_t nChannels)
{
	const AGCProfileSettings settings = GetAGCProfileSettings(m_Profile);
	AGCProcessResult result = ProcessAGC(MixSoundBuffer, RearSoundBuffer, count, nChannels, m_nAGC, settings.attackStepSamples, m_nAGCAttackCount);
	uint32 agc = result.agc;
	// Some kind custom law, so that the AGC stays quite stable, but slowly
	// goes back up if the sound level stays below a level inversely proportional
	// to the AGC level. (J'me comprends)
	if(result.clipCount == 0)
	{
		m_nAGCAttackCount = 0;
	}
	if((agc >= m_nAGC) && (m_nAGC < AGC_UNITY))
	{
		m_nAGCRecoverCount += count;
		if(m_nAGCRecoverCount >= m_Timeout)
		{
			m_nAGCRecoverCount = 0;
			m_nAGC++;
		}
	} else
	{
		m_nAGC = agc;
		m_nAGCRecoverCount = 0;
	}
}


#else
void CAGC::Process(float *MixSoundBuffer, float *RearSoundBuffer, std::size_t count, std::size_t nChannels)
{
	const AGCProfileSettings settings = GetAGCProfileSettings(m_Profile);
	// Convert fixed-point AGC gain to float: m_nAGC is in 10-bit fixed-point, unity = 1024
	const float agcScale = 1.0f / static_cast<float>(AGC_UNITY);
	// In float mixer, samples are roughly [-1, 1].  The int mixer's limit
	// (MIXING_LIMITMAX ≈ 2^27 * 1.008) maps to ~1.008 in float space.
	const float limitMax = static_cast<float>(MIXING_LIMITMAX) / MIXING_SCALEF;
	float gain = static_cast<float>(m_nAGC) * agcScale;
	std::size_t clipCount = 0;

	for(std::size_t i = 0; i < count; i++)
	{
		bool clipped = false;
		if(nChannels >= 2)
		{
			float fl = MixSoundBuffer[i * 2] * gain;
			float fr = MixSoundBuffer[i * 2 + 1] * gain;
			clipped = (std::abs(fl) > limitMax) || (std::abs(fr) > limitMax);
			MixSoundBuffer[i * 2] = fl;
			MixSoundBuffer[i * 2 + 1] = fr;
			if(nChannels >= 4 && RearSoundBuffer)
			{
				float rl = RearSoundBuffer[i * 2] * gain;
				float rr = RearSoundBuffer[i * 2 + 1] * gain;
				clipped = clipped || (std::abs(rl) > limitMax) || (std::abs(rr) > limitMax);
				RearSoundBuffer[i * 2] = rl;
				RearSoundBuffer[i * 2 + 1] = rr;
			}
		} else
		{
			float val = MixSoundBuffer[i] * gain;
			clipped = (std::abs(val) > limitMax);
			MixSoundBuffer[i] = val;
		}
		if(clipped)
		{
			clipCount++;
			m_nAGCAttackCount++;
			if(m_nAGCAttackCount >= settings.attackStepSamples)
			{
				m_nAGC--;
				gain = static_cast<float>(m_nAGC) * agcScale;
				m_nAGCAttackCount -= settings.attackStepSamples;
			}
		}
	}

	if(clipCount == 0)
	{
		m_nAGCAttackCount = 0;
	}
	uint32 agc = m_nAGC;
	if((agc >= m_nAGC) && (m_nAGC < AGC_UNITY))
	{
		m_nAGCRecoverCount += count;
		if(m_nAGCRecoverCount >= m_Timeout)
		{
			m_nAGCRecoverCount = 0;
			m_nAGC++;
		}
	} else
	{
		m_nAGC = agc;
		m_nAGCRecoverCount = 0;
	}
}


void CAGC::Process(double *MixSoundBuffer, double *RearSoundBuffer, std::size_t count, std::size_t nChannels)
{
	const AGCProfileSettings settings = GetAGCProfileSettings(m_Profile);
	const double agcScale = 1.0 / static_cast<double>(AGC_UNITY);
	const double limitMax = static_cast<double>(MIXING_LIMITMAX) / MIXING_SCALEF;
	double gain = static_cast<double>(m_nAGC) * agcScale;
	std::size_t clipCount = 0;

	for(std::size_t i = 0; i < count; i++)
	{
		bool clipped = false;
		if(nChannels >= 2)
		{
			double fl = MixSoundBuffer[i * 2] * gain;
			double fr = MixSoundBuffer[i * 2 + 1] * gain;
			clipped = (std::abs(fl) > limitMax) || (std::abs(fr) > limitMax);
			MixSoundBuffer[i * 2] = fl;
			MixSoundBuffer[i * 2 + 1] = fr;
			if(nChannels >= 4 && RearSoundBuffer)
			{
				double rl = RearSoundBuffer[i * 2] * gain;
				double rr = RearSoundBuffer[i * 2 + 1] * gain;
				clipped = clipped || (std::abs(rl) > limitMax) || (std::abs(rr) > limitMax);
				RearSoundBuffer[i * 2] = rl;
				RearSoundBuffer[i * 2 + 1] = rr;
			}
		} else
		{
			double val = MixSoundBuffer[i] * gain;
			clipped = (std::abs(val) > limitMax);
			MixSoundBuffer[i] = val;
		}
		if(clipped)
		{
			clipCount++;
			m_nAGCAttackCount++;
			if(m_nAGCAttackCount >= settings.attackStepSamples)
			{
				m_nAGC--;
				gain = static_cast<double>(m_nAGC) * agcScale;
				m_nAGCAttackCount -= settings.attackStepSamples;
			}
		}
	}

	if(clipCount == 0)
		m_nAGCAttackCount = 0;
	uint32 agc = m_nAGC;
	if((agc >= m_nAGC) && (m_nAGC < AGC_UNITY))
	{
		m_nAGCRecoverCount += count;
		if(m_nAGCRecoverCount >= m_Timeout)
		{
			m_nAGCRecoverCount = 0;
			m_nAGC++;
		}
	} else
	{
		m_nAGC = agc;
		m_nAGCRecoverCount = 0;
	}
}
#endif


void CAGC::Adjust(uint32 oldVol, uint32 newVol)
{
	m_nAGC = m_nAGC * oldVol / newVol;
	if (m_nAGC > AGC_UNITY) m_nAGC = AGC_UNITY;
}


void CAGC::Initialize(bool bReset, uint32 MixingFreq)
{
	if(bReset)
	{
		m_nAGC = AGC_UNITY;
		m_nAGCRecoverCount = 0;
		m_nAGCAttackCount = 0;
	}
	const AGCProfileSettings settings = GetAGCProfileSettings(m_Profile);
	m_Timeout = ((MixingFreq >> (AGC_PRECISION-8)) >> 1) * settings.recoveryTimeoutMul;
}


#else


MPT_MSVC_WORKAROUND_LNK4221(AGC)


#endif // NO_AGC


OPENMPT_NAMESPACE_END
