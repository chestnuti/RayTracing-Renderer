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
		float eta = iorExt / iorInt;

		// Sin(theta) of angle of incidence
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));

		float sinThetaT = eta * sinTheta;

		// Total internal reflection
		if (sinThetaT >= 1.0f) return 1.0f;

		float cosThetaT = sqrtf(std::max(0.0f, 1.0f - sinThetaT * sinThetaT));

		// Fresnel reflectance
		float Rs = (eta * cosTheta - cosThetaT) / (eta * cosTheta + cosThetaT);
		float Rp = (cosTheta - eta * cosThetaT) / (cosTheta + eta * cosThetaT);

		// Average reflectance
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
			fresnelConductorChannel(sin2, cos2, cosTheta, ior.r, k.r),
			fresnelConductorChannel(sin2, cos2, cosTheta, ior.g, k.g),
			fresnelConductorChannel(sin2, cos2, cosTheta, ior.b, k.b)
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

		// Only the upper hemisphere
		if (cosTheta <= 0.0f) return 0.0f;

		float alpha2 = alpha * alpha;
		float cos2 = cosTheta * cosTheta;
		float denom = cos2 * (alpha2 - 1.0f) + 1.0f;

		return alpha2 / ((float)M_PI * denom * denom);
	}

	static float fresnelConductorChannel(float sin2,float cos2, float cosTheta ,float eta, float kk)
	{
		float eta2 = eta * eta;
		float k2 = kk * kk;


		float t0 = eta2 - k2 - sin2;
		float a2b2 = sqrtf(std::max(0.0f, t0 * t0 + 4.0f * eta2 * k2));
		float a = sqrtf(std::max(0.0f, 0.5f * (a2b2 + t0)));

		// Rs
		float term1 = a2b2 + cos2;
		float term2 = 2.0f * a * cosTheta;
		float Rs = (term1 - term2) / std::max(term1 + term2, EPSILON);

		// Rp
		float term3 = cos2 * a2b2 + sin2 * sin2;
		float term4 = term2 * sin2;
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

		reflectedColour = evaluate(shadingData, wi);
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Add correct evaluation code here
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
		// Reflect wo
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
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// Sample microfacet normal
		float u = sampler->next();
		float v = sampler->next();
		float phi = 2.0f * (float)M_PI * v;
		float tanThetaM2 = alpha * alpha * u / std::max(1.0f - u, EPSILON);
		float cosThetaM = 1.0f / sqrtf(1.0f + tanThetaM2);
		float sinThetaM = sqrtf(std::max(0.0f, 1.0f - cosThetaM * cosThetaM));
		Vec3 h = Vec3(sinThetaM * cosf(phi), sinThetaM * sinf(phi), cosThetaM);
		float dot_wo_h = Dot(woLocal, h);
		if (dot_wo_h <= 0.0f) {
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// Microfacet reflection
		Vec3 wiLocal = Vec3(2.0f * dot_wo_h * h.x - woLocal.x, 2.0f * dot_wo_h * h.y - woLocal.y, 2.0f * dot_wo_h * h.z - woLocal.z);
		if (wiLocal.z <= 0.0f) {
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// Evaluate BSDF
		reflectedColour = evaluate(shadingData, shadingData.frame.toWorld(wiLocal));
		pdf = PDF(shadingData, shadingData.frame.toWorld(wiLocal));

		return shadingData.frame.toWorld(wiLocal);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Conductor evaluation code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		// Microfacet half-vector
		Vec3 sum = Vec3(woLocal.x + wiLocal.x, woLocal.y + wiLocal.y, woLocal.z + wiLocal.z);
		float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
		if (len < EPSILON) return Colour(0.0f, 0.0f, 0.0f);
		Vec3 h = Vec3(sum.x / len, sum.y / len, sum.z / len);
		if (h.z <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		float dot_wo_h = Dot(woLocal, h);
		if (dot_wo_h <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		// Cook-Torrance microfacet model
		Colour F = ShadingHelper::fresnelConductor(dot_wo_h, eta, k);
		float D = ShadingHelper::Dggx(h, alpha);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float denom = 4.0f * woLocal.z * wiLocal.z;
		if (denom < EPSILON) return Colour(0.0f, 0.0f, 0.0f);

		return albedo->sample(shadingData.tu, shadingData.tv) * F * (D * G / denom);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Conductor PDF
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;
		Vec3 sum = Vec3(woLocal.x + wiLocal.x, woLocal.y + wiLocal.y, woLocal.z + wiLocal.z);
		float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
		if (len < EPSILON) return 0.0f;
		Vec3 h = Vec3(sum.x / len, sum.y / len, sum.z / len);
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

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		//* Replace this with Glass sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Colour alb = albedo->sample(shadingData.tu, shadingData.tv);

		// Determine if the ray is entering or exiting the surface
		bool entering = woLocal.z > 0.0f;
		float etaI = entering ? extIOR : intIOR;
		float etaT = entering ? intIOR : extIOR;
		float eta = etaI / etaT;

		float cosThetaO = fabsf(woLocal.z);

		// Fresnel reflectance.
		float F = ShadingHelper::fresnelDielectric(cosThetaO, etaT, etaI);

		float xi = sampler->next();
		Vec3 wiLocal;

		// Reflection
		if (xi < F)
		{
			wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
			pdf = F;
			reflectedColour = alb * (F / std::max(fabsf(wiLocal.z), (float)EPSILON));
		}
		// Refraction
		else
		{
			// Snell law
			float sin2ThetaT = eta * eta * std::max(0.0f, 1.0f - cosThetaO * cosThetaO);

			// Total internal reflection
			if (sin2ThetaT >= 1.0f)
			{
				wiLocal = Vec3(-woLocal.x, -woLocal.y, woLocal.z);
				pdf = 1.0f;
				reflectedColour = alb / std::max(fabsf(wiLocal.z), (float)EPSILON);
			}
			else
			{
				// Refraction direction
				float cosThetaT = sqrtf(1.0f - sin2ThetaT);
				if (entering) cosThetaT = -cosThetaT;

				wiLocal = Vec3(-eta * woLocal.x, -eta * woLocal.y, cosThetaT);
				pdf = 1.0f - F;

				reflectedColour = alb * ((1.0f - F) * eta * eta / std::max(fabsf(wiLocal.z), (float)EPSILON));
			}
		}

		return shadingData.frame.toWorld(wiLocal);
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
		//* Replace this with OrenNayar sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wi);

		reflectedColour = evaluate(shadingData, wi);
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with OrenNayar evaluation code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		float cosThetaO = woLocal.z, cosThetaI = wiLocal.z;
		// validate
        if (cosThetaO <= 0.0f || cosThetaI <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		// Oren-Nayar
		float sigma2 = sigma * sigma;
		float A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
		float B = 0.45f * sigma2 / (sigma2 + 0.09f);

		float sinThetaO = sqrtf(std::max(0.0f, 1.0f - cosThetaO * cosThetaO));
		float sinThetaI = sqrtf(std::max(0.0f, 1.0f - cosThetaI * cosThetaI));

		// Compute cos(phiI - phiO)
		float cosDeltaPhi = 0.0f;
        if (sinThetaO > EPSILON && sinThetaI > EPSILON)
		{
        float cosPhiO = woLocal.x / sinThetaO, sinPhiO = woLocal.y / sinThetaO;
			float cosPhiI = wiLocal.x / sinThetaI, sinPhiI = wiLocal.y / sinThetaI;
			cosDeltaPhi = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
		}

		// Compute alpha and beta
		float ThetaO = acosf(std::min(cosThetaO, 1.0f));
		float ThetaI = acosf(std::min(cosThetaI, 1.0f));
		float sinAlpha = sinf(std::max(ThetaO, ThetaI));
		float tanBeta = tanf(std::min(ThetaO, ThetaI));

		float oren = A + B * std::max(0.0f, cosDeltaPhi) * sinAlpha * tanBeta;
		return albedo->sample(shadingData.tu, shadingData.tv) * (oren / (float)M_PI);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with OrenNayar PDF
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
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		//* Replace this with Plastic sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		float F = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float e = (2.0f / SQ(std::max(alpha, EPSILON))) - 2.0f;

		Vec3 wiLocal;

		if (sampler->next() < F)
		{
			// Specular reflection direction
			Vec3 wr(-woLocal.x, -woLocal.y, woLocal.z);

			// Sample lobe
			float u1 = sampler->next();
			float u2 = sampler->next();
			float cosThetaLobe = powf(std::max(u1, EPSILON), 1.0f / (e + 1.0f));
			float sinThetaLobe = sqrtf(std::max(0.0f, 1.0f - cosThetaLobe * cosThetaLobe));
			float phiLobe = 2.0f * (float)M_PI * u2;
			Vec3 wLobe(sinThetaLobe * cosf(phiLobe), sinThetaLobe * sinf(phiLobe), cosThetaLobe);

			// Build an orthonormal basis with wr
			Vec3 tangent, bitangent;
			if (fabsf(wr.z) < 0.999f)
			{
				float invLen = 1.0f / sqrtf(wr.x * wr.x + wr.y * wr.y);
				tangent = Vec3(-wr.y * invLen, wr.x * invLen, 0.0f);
			}
			else
			{
				tangent = Vec3(1.0f, 0.0f, 0.0f);
			}
			// bitangent = wr x tangent
			bitangent = Vec3(wr.y * tangent.z - wr.z * tangent.y, wr.z * tangent.x - wr.x * tangent.z, wr.x * tangent.y - wr.y * tangent.x);

			wiLocal = Vec3(tangent.x * wLobe.x + bitangent.x * wLobe.y + wr.x * wLobe.z, tangent.y * wLobe.x + bitangent.y * wLobe.y + wr.y * wLobe.z, tangent.z * wLobe.x + bitangent.z * wLobe.y + wr.z * wLobe.z);

			// Discard samples below the hemisphere
			if (wiLocal.z <= 0.0f)
			{
				pdf = 0.0f;
				reflectedColour = Colour(0.0f, 0.0f, 0.0f);
				return Vec3(0.0f, 0.0f, 0.0f);
			}
		}
		else
		{
			// Lambert diffuse
			wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		}

		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		pdf = PDF(shadingData, wiWorld);
		reflectedColour = evaluate(shadingData, wiWorld);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Plastic evaluation code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		Colour rho = albedo->sample(shadingData.tu, shadingData.tv);

		float Fo = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float Fi = ShadingHelper::fresnelDielectric(fabsf(wiLocal.z), intIOR, extIOR);

		// Lambert diffuse
		Colour diffuse = rho * ((1.0f - Fo) * (1.0f - Fi) / (float)M_PI);

		// Phong glossy
		float e = (2.0f / SQ(std::max(alpha, EPSILON))) - 2.0f;
		Vec3 wr(-woLocal.x, -woLocal.y, woLocal.z);
		float cosAlpha = std::max(0.0f, Dot(wr, wiLocal));
		float phongLobe = ((e + 2.0f) / (2.0f * (float)M_PI)) * powf(cosAlpha, e);
		float specular = Fo * phongLobe;

		return diffuse + Colour(specular, specular, specular);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Plastic PDF
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;

		float F = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float e = (2.0f / SQ(std::max(alpha, EPSILON))) - 2.0f;

		// Phong lobe PDF
		Vec3 wr(-woLocal.x, -woLocal.y, woLocal.z);
		float cosAlpha = std::max(0.0f, Dot(wr, wiLocal));
		float pPhong = ((e + 1.0f) / (2.0f * (float)M_PI)) * powf(cosAlpha, e);

		// Cosine-weighted hemisphere PDF
		float pCos = wiLocal.z / (float)M_PI;

		return F * pPhong + (1.0f - F) * pCos;
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
