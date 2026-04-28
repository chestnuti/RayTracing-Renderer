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
		//* Add code to sample a direction from the light
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wiLocal);
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
		// Samples a point on the bounding sphere of the scene. Feel free to improve this.
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}

	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		//* Replace this tabulated sampling of environment maps
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light
{
public:
	Texture* env = NULL;
	std::vector<float> luminance;
	std::vector<float> importance;
	std::vector<float> marginalCDF;
	std::vector<float> conditionalCDF;
	float totalLuminance = 0.0f;
	float totalImportance = 0.0f;
	bool enableEnvmap = true;

	EnvironmentMap(Texture* _env)
	{
		if (enableEnvmap)
			load(_env);
		else
			env = _env;
	}
	void load(Texture* _env)
	{
		env = _env;
		luminance.clear();
		importance.clear();
		marginalCDF.clear();
		conditionalCDF.clear();
		totalLuminance = 0.0f;
		totalImportance = 0.0f;

		if (env == NULL || env->width <= 0 || env->height <= 0 || env->texels == NULL)
		{
			return;
		}

		const int width = env->width;
		const int height = env->height;
		const int pixelCount = width * height;

		luminance.resize(pixelCount, 0.0f);
		importance.resize(pixelCount, 0.0f);
		marginalCDF.resize(height, 0.0f);
		conditionalCDF.resize(pixelCount, 0.0f);

		std::vector<float> rowWeights(height, 0.0f);

		// Build the tabulated importance distribution
		for (int v = 0; v < height; ++v)
		{
			const float vv = ((float)v + 0.5f) / (float)height;
			const float theta = vv * (float)M_PI;
			const float sinTheta = std::max(sinf(theta), EPSILON);

			float rowWeight = 0.0f;
			for (int u = 0; u < width; ++u)
			{
				const int idx = (v * width) + u;
				const float lum = env->texels[idx].Lum();
				const float imp = lum * sinTheta;

				luminance[idx] = lum;
				importance[idx] = imp;
				totalLuminance += lum;
				rowWeight += imp;
			}

			rowWeights[v] = rowWeight;
			totalImportance += rowWeight;
		}

		if (totalImportance <= EPSILON)
		{
			for (int v = 0; v < height; ++v)
			{
				marginalCDF[v] = (float)(v + 1) / (float)height;
				for (int u = 0; u < width; ++u)
				{
					conditionalCDF[(v * width) + u] = (float)(u + 1) / (float)width;
				}
			}
			marginalCDF[height - 1] = 1.0f;
			return;
		}

		float marginalAccum = 0.0f;
		for (int v = 0; v < height; ++v)
		{
			marginalAccum += rowWeights[v] / totalImportance;
			marginalCDF[v] = marginalAccum;

			float conditionalAccum = 0.0f;
			if (rowWeights[v] > EPSILON)
			{
				for (int u = 0; u < width; ++u)
				{
					const int idx = (v * width) + u;
					conditionalAccum += importance[idx] / rowWeights[v];
					conditionalCDF[idx] = conditionalAccum;
				}
			}
			else
			{
				for (int u = 0; u < width; ++u)
				{
					conditionalCDF[(v * width) + u] = (float)(u + 1) / (float)width;
				}
			}

			conditionalCDF[(v * width) + (width - 1)] = 1.0f;
		}

		marginalCDF[height - 1] = 1.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		if (env == NULL || env->width <= 0 || env->height <= 0 || sampler == NULL)
		{
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		if (enableEnvmap == false)
		{
			Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
			pdf = SamplingDistributions::uniformSpherePDF(wi);
			reflectedColour = evaluate(wi);
			return wi;
		}

		// Sample direction from the environment map
		const float x1 = std::min(sampler->next(), 1.0f - EPSILON);
		const float x2 = std::min(sampler->next(), 1.0f - EPSILON);

		const int vIdx = sampleMarginalIndex(x1);
		const int uIdx = sampleConditionalIndex(vIdx, x2);

		const float u = ((float)uIdx + 0.5f) / (float)env->width;
		const float v = ((float)vIdx + 0.5f) / (float)env->height;
		Vec3 wi = uvToDirection(u, v);

		// Convert tabulated probability to a solid-angle PDF for path-space integration
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

		if (enableEnvmap == false)
		{
			float u = atan2f(wi.z, wi.x);
			u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
			u = u / (2.0f * M_PI);
			float v = acosf(wi.y) / M_PI;
			return env->sample(u, v);
		}

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
		if (env == NULL || env->width <= 0 || env->height <= 0 || totalImportance <= EPSILON)
		{
			return 0.0f;
		}

		if (enableEnvmap == false)
		{
			return SamplingDistributions::uniformSpherePDF(wi);
		}

		// Return the solid-angle PDF
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

		// Convert uv to pixel indices
		const int uIdx = std::min((int)(u * (float)env->width), env->width - 1);
		const int vIdx = std::min((int)(v * (float)env->height), env->height - 1);
		if (uIdx < 0 || uIdx >= env->width || vIdx < 0 || vIdx >= env->height)
		{
			return 0.0f;
		}

		// Compute the PDF with respect to solid angle
		const int idx = (vIdx * env->width) + uIdx;
		const float pixelProbability = importance[idx] / totalImportance;
		const float pixelAreaUV = 1.0f / (float)(env->width * env->height);
		// Apply the lat-long Jacobian to get PDF over solid angle
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
		float total = 0;
		for (int i = 0; i < env->height; i++)
		{
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++)
			{
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		return total * 4.0f * M_PI;
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
		Vec3 wiToEnv = sample(shadingData, sampler, reflectedColour, pdf);
		return -wiToEnv;
	}

	Vec3 worldToEnvLocal(const Vec3& worldDir) const
	{
		// Convert to the environment local frame
		return Vec3(worldDir.x, worldDir.z, worldDir.y);
	}

	Vec3 envLocalToWorld(const Vec3& envDir) const
	{
		return Vec3(envDir.x, envDir.z, envDir.y);
	}

	bool directionToUV(const Vec3& worldDir, float& theta, float& phi, float& u, float& v) const
	{
		// Convert world-space direction to latitude-longitude coordinates
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
		// Convert uv to spherical coordinates
		const float phi = 2.0f * (float)M_PI * u;
		const float theta = (float)M_PI * v;
		const float sinTheta = sinf(theta);
		const Vec3 envDir(sinTheta * cosf(phi), sinTheta * sinf(phi), cosf(theta));
		// Convert to world space
		return envLocalToWorld(envDir);
	}

	int sampleMarginalIndex(float xi) const
	{
		// Select theta row from the marginalCDF
		std::vector<float>::const_iterator it = std::lower_bound(marginalCDF.begin(), marginalCDF.end(), xi);
		if (it == marginalCDF.end())
		{
			return (int)marginalCDF.size() - 1;
		}
		return (int)(it - marginalCDF.begin());
	}

	int sampleConditionalIndex(int row, float xi) const
	{
		// Select phi column from the conditionalCDF row
		const int width = env->width;
		const float* rowStart = &conditionalCDF[row * width];
		const float* rowEnd = rowStart + width;
		const float* it = std::lower_bound(rowStart, rowEnd, xi);
		if (it == rowEnd)
		{
			return width - 1;
		}
		return (int)(it - rowStart);
	}
};
