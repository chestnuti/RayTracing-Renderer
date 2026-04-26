#pragma once

#include "Core.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"
#include <vector>

#pragma warning( disable : 4244)

class SceneBounds
{
public:
	Vec3 sceneCentre;
	float sceneRadius;
};

class Light
{
public:
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec3 normal(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec3 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}

	Colour evaluate(const Vec3& wi)
	{
		if (Dot(wi, triangle->gNormal()) < 0)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 1.0f / triangle->area;
	}

	bool isArea()
	{
		return true;
	}

	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return triangle->gNormal();
	}

	float totalIntegratedPower()
	{
		return (triangle->area * emission.Lum());
	}

	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		return triangle->sample(sampler, pdf);
	}

	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = Vec3(0, 0, 1);
		pdf = 1.0f;
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
	}
};

class BackgroundColour : public Light
{
public:
	Colour emission;

	BackgroundColour(Colour _emission)
	{
		emission = _emission;
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}

	Colour evaluate(const Vec3& wi)
	{
		return emission;
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return SamplingDistributions::uniformSpherePDF(wi);
	}

	bool isArea()
	{
		return false;
	}

	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}

	float totalIntegratedPower()
	{
		return emission.Lum() * 4.0f * M_PI;
	}

	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}

	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light
{
public:
	Texture* env = NULL;
	std::vector<float> m_luminance;
	std::vector<float> m_importance;
	std::vector<float> m_marginalCDF;
	std::vector<float> m_conditionalCDF;
	float m_totalLuminance = 0.0f;
	float m_totalImportance = 0.0f;

	EnvironmentMap(Texture* _env)
	{
		load(_env);
	}

	void load(Texture* _env)
	{
		env = _env;
		m_luminance.clear();
		m_importance.clear();
		m_marginalCDF.clear();
		m_conditionalCDF.clear();
		m_totalLuminance = 0.0f;
		m_totalImportance = 0.0f;

		if (env == NULL || env->width <= 0 || env->height <= 0 || env->texels == NULL)
		{
			return;
		}

		const int width = env->width;
		const int height = env->height;
		const int pixelCount = width * height;

		m_luminance.resize(pixelCount, 0.0f);
		m_importance.resize(pixelCount, 0.0f);
		m_marginalCDF.resize(height, 0.0f);
		m_conditionalCDF.resize(pixelCount, 0.0f);

		std::vector<float> rowWeights(height, 0.0f);

		// LightTransport 1.pdf -> Environment Lighting -> Sampling.
		// Build a tabulated UV-domain PDF using luminance * sin(theta) so that the
		// solid-angle density is proportional to radiance after Jacobian conversion.
		for (int v = 0; v < height; ++v)
		{
			const float vv = ((float)v + 0.5f) / (float)height;
			const float theta = vv * (float)M_PI;
			const float sinTheta = std::max(sinf(theta), EPSILON);

			float rowWeight = 0.0f;
			for (int u = 0; u < width; ++u)
			{
				const int idx = (v * width) + u;
				const float luminance = env->texels[idx].Lum();
				const float importance = luminance * sinTheta;

				m_luminance[idx] = luminance;
				m_importance[idx] = importance;
				m_totalLuminance += luminance;
				rowWeight += importance;
			}

			rowWeights[v] = rowWeight;
			m_totalImportance += rowWeight;
		}

		if (m_totalImportance <= EPSILON)
		{
			for (int v = 0; v < height; ++v)
			{
				m_marginalCDF[v] = (float)(v + 1) / (float)height;
				for (int u = 0; u < width; ++u)
				{
					m_conditionalCDF[(v * width) + u] = (float)(u + 1) / (float)width;
				}
			}
			m_marginalCDF[height - 1] = 1.0f;
			return;
		}

		float marginalAccum = 0.0f;
		for (int v = 0; v < height; ++v)
		{
			marginalAccum += rowWeights[v] / m_totalImportance;
			m_marginalCDF[v] = marginalAccum;

			float conditionalAccum = 0.0f;
			if (rowWeights[v] > EPSILON)
			{
				for (int u = 0; u < width; ++u)
				{
					const int idx = (v * width) + u;
					conditionalAccum += m_importance[idx] / rowWeights[v];
					m_conditionalCDF[idx] = conditionalAccum;
				}
			}
			else
			{
				for (int u = 0; u < width; ++u)
				{
					m_conditionalCDF[(v * width) + u] = (float)(u + 1) / (float)width;
				}
			}

			m_conditionalCDF[(v * width) + (width - 1)] = 1.0f;
		}

		m_marginalCDF[height - 1] = 1.0f;
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		if (env == NULL || env->width <= 0 || env->height <= 0 || sampler == NULL)
		{
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// LightTransport 1.pdf -> Environment Lighting -> Sampling.
		// Sample marginal CDF in v/theta first, then conditional CDF in u/phi.
		const float xi1 = std::min(sampler->next(), 1.0f - EPSILON);
		const float xi2 = std::min(sampler->next(), 1.0f - EPSILON);

		const int vIdx = sampleMarginalIndex(xi1);
		const int uIdx = sampleConditionalIndex(vIdx, xi2);

		const float u = ((float)uIdx + 0.5f) / (float)env->width;
		const float v = ((float)vIdx + 0.5f) / (float)env->height;
		Vec3 wi = uvToDirection(u, v);

		pdf = PDF(shadingData, wi);
		if (pdf <= EPSILON)
		{
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		reflectedColour = evaluate(wi);
		return wi;
	}

	Colour evaluate(const Vec3& wi)
	{
		if (env == NULL || env->width <= 0 || env->height <= 0)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}

		// LightTransport 1.pdf -> Environment Lighting -> Lookup.
		float theta;
		float phi;
		float u;
		float v;
		if (directionToUV(wi, theta, phi, u, v) == false)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}

		return env->sample(u, v);
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		if (env == NULL || env->width <= 0 || env->height <= 0 || m_totalImportance <= EPSILON)
		{
			return 0.0f;
		}

		// LightTransport 1.pdf -> Environment Lighting -> Sampling.
		// Jacobian conversion: p_omega = p(u,v) / (2*pi^2*sin(theta)).
		float theta;
		float phi;
		float u;
		float v;
		if (directionToUV(wi, theta, phi, u, v) == false)
		{
			return 0.0f;
		}

		const float sinTheta = sinf(theta);
		if (sinTheta <= EPSILON)
		{
			return 0.0f;
		}

		const int uIdx = std::min((int)(u * (float)env->width), env->width - 1);
		const int vIdx = std::min((int)(v * (float)env->height), env->height - 1);
		if (uIdx < 0 || uIdx >= env->width || vIdx < 0 || vIdx >= env->height)
		{
			return 0.0f;
		}

		const int idx = (vIdx * env->width) + uIdx;
		const float pixelProbability = m_importance[idx] / m_totalImportance;
		const float pixelAreaUV = 1.0f / (float)(env->width * env->height);
		const float pdfUV = pixelProbability / pixelAreaUV;
		return pdfUV / (2.0f * (float)M_PI * (float)M_PI * sinTheta);
	}

	bool isArea()
	{
		return false;
	}

	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}

	float totalIntegratedPower()
	{
		if (env == NULL || env->width <= 0 || env->height <= 0)
		{
			return 0.0f;
		}

		// LightTransport 1.pdf -> Environment Lighting.
		// Approximate the environment integral on the sphere with the lat-long sin(theta) Jacobian.
		return m_totalImportance * (2.0f * (float)M_PI * (float)M_PI) / (float)(env->width * env->height);
	}

	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 1.0f / (4 * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}

	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Colour reflectedColour;
		ShadingData shadingData;
		return sample(shadingData, sampler, reflectedColour, pdf);
	}

private:
	Vec3 worldToEnvLocal(const Vec3& worldDir) const
	{
		// LightTransport 1.pdf uses a z-up spherical parameterization, while this
		// renderer's world/environment convention is y-up. Convert before applying
		// the course formulas so the environment remains upright in world space.
		return Vec3(worldDir.x, worldDir.z, worldDir.y);
	}

	Vec3 envLocalToWorld(const Vec3& envDir) const
	{
		return Vec3(envDir.x, envDir.z, envDir.y);
	}

	bool directionToUV(const Vec3& worldDir, float& theta, float& phi, float& u, float& v) const
	{
		const Vec3 envDirWorld = worldToEnvLocal(worldDir);
		const float lenSq = (envDirWorld.x * envDirWorld.x) + (envDirWorld.y * envDirWorld.y) + (envDirWorld.z * envDirWorld.z);
		if (lenSq <= (EPSILON * EPSILON))
		{
			return false;
		}

		const float invLen = 1.0f / sqrtf(lenSq);
		const Vec3 dir = envDirWorld * invLen;

		theta = acosf(std::max(-1.0f, std::min(1.0f, dir.z)));
		phi = atan2f(dir.y, dir.x);
		if (phi < 0.0f)
		{
			phi += 2.0f * (float)M_PI;
		}

		u = phi / (2.0f * (float)M_PI);
		v = theta / (float)M_PI;

		if (u < 0.0f)
		{
			u += 1.0f;
		}
		if (u >= 1.0f)
		{
			u = u - floorf(u);
		}
		v = std::min(std::max(v, 0.0f), 1.0f - EPSILON);
		return true;
	}

	Vec3 uvToDirection(float u, float v) const
	{
		const float phi = 2.0f * (float)M_PI * u;
		const float theta = (float)M_PI * v;
		const float sinTheta = sinf(theta);
		const Vec3 envDir(sinTheta * cosf(phi), sinTheta * sinf(phi), cosf(theta));
		return envLocalToWorld(envDir);
	}

	int sampleMarginalIndex(float xi) const
	{
		std::vector<float>::const_iterator it = std::lower_bound(m_marginalCDF.begin(), m_marginalCDF.end(), xi);
		if (it == m_marginalCDF.end())
		{
			return (int)m_marginalCDF.size() - 1;
		}
		return (int)(it - m_marginalCDF.begin());
	}

	int sampleConditionalIndex(int row, float xi) const
	{
		const int width = env->width;
		const float* rowStart = &m_conditionalCDF[row * width];
		const float* rowEnd = rowStart + width;
		const float* it = std::lower_bound(rowStart, rowEnd, xi);
		if (it == rowEnd)
		{
			return width - 1;
		}
		return (int)(it - rowStart);
	}
};
