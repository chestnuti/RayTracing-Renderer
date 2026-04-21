#pragma once

#include "Core.h"
#include "Imaging.h"
#include "Sampling.h"

#pragma warning( disable : 4244)
#pragma warning( disable : 4305) // Double to float

class BSDF;

class ShadingData
{
public:
	Vec3 x;
	Vec3 wo;
	Vec3 sNormal;
	Vec3 gNormal;
	float tu;
	float tv;
	Frame frame;
	BSDF* bsdf;
	float t;
	ShadingData() {}
	ShadingData(Vec3 _x, Vec3 n)
	{
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = NULL;
	}
};

class ShadingHelper
{
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt)
	{
		//* Add code here
		cosTheta = fabsf(cosTheta);
		cosTheta = std::min(cosTheta, 1.0f);

		// Calculate sinTheta using trigonometric identity
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		float sinThetaT = (iorExt / iorInt) * sinTheta;

		// Total internal reflection
		if (sinThetaT >= 1.0f) return 1.0f;

		float cosThetaT = sqrtf(std::max(0.0f, 1.0f - sinThetaT * sinThetaT));

		// Calculate Rs and Rp
		float Rs = (iorExt * cosTheta - iorInt * cosThetaT) / (iorExt * cosTheta + iorInt * cosThetaT);
		float Rp = (iorInt * cosTheta - iorExt * cosThetaT) / (iorInt * cosTheta + iorExt * cosThetaT);

		return (Rs * Rs + Rp * Rp) * 0.5f;
	}
	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k)
	{
		//* Add code here
		cosTheta = fabsf(cosTheta);
		cosTheta = std::min(cosTheta, 1.0f);

		float cos2 = cosTheta * cosTheta;
		float sin2 = std::max(0.0f, 1.0f - cos2);

		return Colour(
			fresnelChannel(sin2, cos2, cosTheta, ior.r, k.r),
			fresnelChannel(sin2, cos2, cosTheta, ior.g, k.g),
			fresnelChannel(sin2, cos2, cosTheta, ior.b, k.b)
		);
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		//* Add code here
		float cosTheta = fabsf(wi.z);

		// no shadow
		if (cosTheta >= 1.0f) return 0.0f;

		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		float tanTheta = sinTheta / cosTheta;

		float alphaTan = alpha * tanTheta;

		return (-1.0f + sqrtf(1.0f + alphaTan * alphaTan)) * 0.5f;
	}
	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		//* Add code here
		return 1.0f / (1.0f + lambdaGGX(wi, alpha) + lambdaGGX(wo, alpha));
	}
	static float Dggx(Vec3 h, float alpha)
	{
		//* Add code here
		float cosTheta = h.z;

		// Only the upper hemisphere is physically meaningful
		if (cosTheta <= 0.0f) return 0.0f;

		float alpha2 = alpha * alpha;
		float cos2 = cosTheta * cosTheta;
		float denom = cos2 * (alpha2 - 1.0f) + 1.0f;   // (n¡¤h)^2*(alpha^2-1)+1

		return alpha2 / ((float)M_PI * denom * denom);
	}

	static float fresnelChannel(float sin2,float cos2, float cosTheta ,float eta, float kk)
	{
		float eta2 = eta * eta;
		float k2 = kk * kk;

		// t0 = eta^2 - k^2 - sin^2(theta)
		float t0 = eta2 - k2 - sin2;
		// a2b2 = sqrt( t0^2 + 4*eta^2*k^2 )  = (a^2 + b^2)
		float a2b2 = sqrtf(std::max(0.0f, t0 * t0 + 4.0f * eta2 * k2));
		// a = sqrt( (a^2 + b^2 + t0) / 2 )
		float a = sqrtf(std::max(0.0f, 0.5f * (a2b2 + t0)));

		// Rs
		float term1 = a2b2 + cos2;
		float term2 = 2.0f * a * cosTheta;
		float Rs = (term1 - term2) / std::max(term1 + term2, EPSILON);

		// Rp
		float term3 = cos2 * a2b2 + sin2 * sin2;      // cos^2*(a^2+b^2) + sin^4
		float term4 = term2 * sin2;                   // 2*a*cos*sin^2
		float Rp = Rs * (term3 - term4) / std::max(term3 + term4, EPSILON);

		float F = 0.5f * (Rp + Rs);

		return std::min(std::max(F, 0.0f), 1.0f);
	}
};

class BSDF
{
public:
	Colour emission;
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight()
	{
		return emission.Lum() > 0 ? true : false;
	}
	void addLight(Colour _emission)
	{
		emission = _emission;
	}
	Colour emit(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	virtual float mask(const ShadingData& shadingData) = 0;
};


class DiffuseBSDF : public BSDF
{
public:
	Texture* albedo;
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		//* Add correct sampling code here
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wi);

		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Add correct PDF code here
		const Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (wiLocal.z <= 0.0f) return 0.0f;
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class MirrorBSDF : public BSDF
{
public:
	Texture* albedo;
	MirrorBSDF() = default;
	MirrorBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		//* Replace this with Mirror sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 0.0f);
		}
		Vec3 wiLocal(-woLocal.x, -woLocal.y, woLocal.z);
		pdf = 1.0f;
		float cosTheta = std::max(std::abs(wiLocal.z), EPSILON);
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / cosTheta;
		return shadingData.frame.toWorld(wiLocal);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Mirror evaluation code
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Mirror PDF
		return 0.0f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};


class ConductorBSDF : public BSDF
{
public:
	Texture* albedo;
	Colour eta;
	Colour k;
	float alpha;
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
	{
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		//* Replace this with Conductor sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f) { 
			reflectedColour = Colour(0, 0, 0); 
			pdf = 1.0f; 
			return shadingData.frame.toWorld(Vec3(0, 0, 1)); 
		}

		// Sample half-vector
		float phi = 2.0f * (float)M_PI * sampler->next();
		float tanTheta2 = alpha * alpha * sampler->next() / std::max(1.0f - sampler->next(), 1e-6f);
		float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		Vec3 h = Vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
		float dot_wo_h = Dot(woLocal, h);
		if (dot_wo_h <= 0.0f) { 
			reflectedColour = Colour(0, 0, 0); 
			pdf = 1.0f; 
			return shadingData.frame.toWorld(Vec3(0, 0, 1)); 
		}

		// Reflect wo around h to get wi
		Vec3 wiLocal = Vec3(2.0f * dot_wo_h * h.x - woLocal.x, 2.0f * dot_wo_h * h.y - woLocal.y, 2.0f * dot_wo_h * h.z - woLocal.z);
		if (wiLocal.z <= 0.0f) { 
			reflectedColour = Colour(0, 0, 0); 
			pdf = 1.0f; 
			return shadingData.frame.toWorld(Vec3(0, 0, 1)); 
		}

		// Evaluate BSDF
		reflectedColour = evaluate(shadingData, shadingData.frame.toWorld(wiLocal));
		float  D = ShadingHelper::Dggx(h, alpha);
		pdf = PDF(shadingData, shadingData.frame.toWorld(wiLocal));

		return shadingData.frame.toWorld(wiLocal);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Conductor evaluation code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return Colour(0, 0, 0);

		Vec3  sum = Vec3(woLocal.x + wiLocal.x, woLocal.y + wiLocal.y, woLocal.z + wiLocal.z);
		float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
		if (len < 1e-6f) return Colour(0, 0, 0);
		Vec3 h = Vec3(sum.x / len, sum.y / len, sum.z / len);
		if (h.z <= 0.0f) return Colour(0, 0, 0);

		float dot_wo_h = Dot(woLocal, h);
		if (dot_wo_h <= 0.0f) return Colour(0, 0, 0);

		Colour F = ShadingHelper::fresnelConductor(dot_wo_h, eta, k);
		float  D = ShadingHelper::Dggx(h, alpha);
		float  G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float  denom = 4.0f * woLocal.z * wiLocal.z;
		if (denom < 1e-8f) return Colour(0, 0, 0);

		return albedo->sample(shadingData.tu, shadingData.tv) * F * (D * G / denom);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Conductor PDF
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;
		Vec3  sum = Vec3(woLocal.x + wiLocal.x, woLocal.y + wiLocal.y, woLocal.z + wiLocal.z);
		float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
		if (len < 1e-6f) return 0.0f;
		Vec3  h = Vec3(sum.x / len, sum.y / len, sum.z / len);
		if (h.z <= 0.0f) return 0.0f;
		float d = Dot(woLocal, h);
		if (d <= 0.0f) return 0.0f;
		return ShadingHelper::Dggx(h, alpha) * h.z / (4.0f * d);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class GlassBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	GlassBSDF() = default;
	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}

	static bool refractLocal(const Vec3& v, const Vec3& n, float etaI, float etaT, Vec3& outT)
	{
		// v points away from surface (same convention as woLocal/wiLocal)
		float cosI = Dot(v, n);
		float eta = etaI / etaT;

		float sin2I = std::max(0.0f, 1.0f - cosI * cosI);
		float sin2T = eta * eta * sin2I;

		// Total internal reflection
		if (sin2T >= 1.0f) return false;

		float cosT = sqrtf(std::max(0.0f, 1.0f - sin2T));

		// PBRT-style refraction
		outT = v * (-eta) + n * (eta * cosI - cosT);
		return true;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		//* Replace this with Glass sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);

		if (fabsf(woLocal.z) < EPSILON)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// Determine if the ray is entering or exiting the surface
		bool entering = woLocal.z > 0.0f;
		Vec3 n = entering ? Vec3(0.0f, 0.0f, 1.0f) : Vec3(0.0f, 0.0f, -1.0f);

		float etaI = entering ? extIOR : intIOR;
		float etaT = entering ? intIOR : extIOR;

		// Calculate Fresnel
		float cosThetaI = fabsf(Dot(woLocal, n));

		// Use the Fresnel equations to determine the reflection
		float Fr = ShadingHelper::fresnelDielectric(cosThetaI, etaT, etaI);
		Fr = std::min(std::max(Fr, 0.0f), 1.0f);

		Colour base = albedo->sample(shadingData.tu, shadingData.tv);

		float xi = sampler->next();

		// Try to refract
		Vec3 wtLocal;
		bool canRefract = refractLocal(woLocal, n, etaI, etaT, wtLocal);

		// reflect
		if (!canRefract)
		{
			Vec3 wiLocal = -woLocal - n * (2.0f * Dot(-woLocal, n));
			pdf = 1.0f;

			float cosO = std::max(fabsf(wiLocal.z), EPSILON);
			reflectedColour = base / cosO;
			return shadingData.frame.toWorld(wiLocal);
		}

		// Choose reflection or refraction
		if (xi < Fr)
		{
			Vec3 wiLocal = -woLocal - n * (2.0f * Dot(-woLocal, n));
			pdf = Fr;

			float cosO = std::max(fabsf(wiLocal.z), EPSILON);
			reflectedColour = base * (Fr / cosO);
			return shadingData.frame.toWorld(wiLocal);
		}
		else
		{
			// Transmit
			Vec3 wiLocal = wtLocal;
			pdf = 1.0f - Fr;

			// Radiance transport factor
			float eta = etaI / etaT;
			float cosT = std::max(fabsf(wiLocal.z), EPSILON);

			reflectedColour = base * ((1.0f - Fr) * (eta * eta) / cosT);
			return shadingData.frame.toWorld(wiLocal);
		}
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Glass evaluation code
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with GlassPDF
		return 0.0f;
	}
	bool isPureSpecular()
	{
		return true;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Dielectric sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo;
	float sigma;
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with OrenNayar sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	float alphaToPhongExponent()
	{
		return (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Plastic sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Plastic PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class LayeredBSDF : public BSDF
{
public:
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR;
	float extIOR;
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR)
	{
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}
	bool isPureSpecular()
	{
		return base->isPureSpecular();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return base->mask(shadingData);
	}
};
