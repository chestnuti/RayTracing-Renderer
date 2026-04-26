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
		// F = (Rs^2 + Rp^2) / 2
		cosTheta = fabsf(cosTheta);
		cosTheta = std::min(cosTheta, 1.0f);

		// Sin(theta) of angle of incidence
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));

		float sinThetaT = (iorExt / iorInt) * sinTheta;

		// Total internal reflection
		if (sinThetaT >= 1.0f) return 1.0f;

		float cosThetaT = sqrtf(std::max(0.0f, 1.0f - sinThetaT * sinThetaT));

		// Fresnel reflectance
		float Rs = (iorExt * cosTheta - iorInt * cosThetaT) / (iorExt * cosTheta + iorInt * cosThetaT);
		float Rp = (iorInt * cosTheta - iorExt * cosThetaT) / (iorInt * cosTheta + iorExt * cosThetaT);

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

		// Only the upper hemisphere
		if (cosTheta <= 0.0f) return 0.0f;

		float alpha2 = alpha * alpha;
		float cos2 = cosTheta * cosTheta;
		float denom = cos2 * (alpha2 - 1.0f) + 1.0f;

		return alpha2 / ((float)M_PI * denom * denom);
	}

	static float fresnelChannel(float sin2,float cos2, float cosTheta ,float eta, float kk)
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
			reflectedColour = Colour(0, 0, 0);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// Sample microfacet normal
		float u = sampler->next();
		float v = sampler->next();
		float phi = 2.0f * (float)M_PI * v;
		float tanTheta2 = alpha * alpha * u / std::max(1.0f - u, 1e-6f);
		float cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		Vec3 h = Vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
		float dot_wo_h = Dot(woLocal, h);
		if (dot_wo_h <= 0.0f) {
			reflectedColour = Colour(0, 0, 0);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// Microfacet reflection
		Vec3 wiLocal = Vec3(2.0f * dot_wo_h * h.x - woLocal.x, 2.0f * dot_wo_h * h.y - woLocal.y, 2.0f * dot_wo_h * h.z - woLocal.z);
		if (wiLocal.z <= 0.0f) {
			reflectedColour = Colour(0, 0, 0);
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
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return Colour(0, 0, 0);

		// Microfacet half-vector
		Vec3  sum = Vec3(woLocal.x + wiLocal.x, woLocal.y + wiLocal.y, woLocal.z + wiLocal.z);
		float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
		if (len < 1e-6f) return Colour(0, 0, 0);
		Vec3 h = Vec3(sum.x / len, sum.y / len, sum.z / len);
		if (h.z <= 0.0f) return Colour(0, 0, 0);

		float dot_wo_h = Dot(woLocal, h);
		if (dot_wo_h <= 0.0f) return Colour(0, 0, 0);

		// Cook-Torrance microfacet model
		Colour F = ShadingHelper::fresnelConductor(dot_wo_h, eta, k);
		float  D = ShadingHelper::Dggx(h, alpha);
		float  G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);
		float  denom = 4.0f * woLocal.z * wiLocal.z;
		if (denom < 1e-6f) return Colour(0, 0, 0);

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
		Vec3  wiLocal;

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
		//* Replace this with Dielectric sampling code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (fabsf(woLocal.z) < 1e-6f)
		{
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		bool entering = (woLocal.z > 0.0f);
		float n_sign = entering ? 1.0f : -1.0f;

		// Sample microfacet normal and reflect or refract wo
		float u1 = sampler->next();
		float u2 = sampler->next();
		float phi = 2.0f * (float)M_PI * u1;

		// GGX sampling
		float tan2theta = alpha * alpha * u2 / std::max(1.0f - u2, 1e-6f);
		float cos_thetam = 1.0f / sqrtf(1.0f + tan2theta);
		float sin_thetam = sqrtf(std::max(0.0f, 1.0f - cos_thetam * cos_thetam));
		Vec3 wm = Vec3(sin_thetam * cosf(phi), sin_thetam * sinf(phi), n_sign * cos_thetam);

		float wo_dot_wm = Dot(woLocal, wm);
		if (wo_dot_wm <= 0.0f)
		{
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			pdf = 0.0f;
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		// Probability of reflection
		float F = ShadingHelper::fresnelDielectric(wo_dot_wm, intIOR, extIOR);

		Vec3 wiLocal;
		if (sampler->next() < F)
		{
			// Microfacet reflection
			wiLocal = Vec3(2.0f * wo_dot_wm * wm.x - woLocal.x, 2.0f * wo_dot_wm * wm.y - woLocal.y, 2.0f * wo_dot_wm * wm.z - woLocal.z);
			if (wiLocal.z * woLocal.z <= 0.0f)
			{
				reflectedColour = Colour(0.0f, 0.0f, 0.0f);
				pdf = 0.0f;
				return Vec3(0.0f, 0.0f, 0.0f);
			}
		}
		else
		{
			// Microfacet refraction
			float eta_o = entering ? extIOR : intIOR;  // medium on wo side
			float eta_t = entering ? intIOR : extIOR;  // medium on wi side
			float eta_ratio = eta_o / eta_t;	// = sin(theta_t)/sin(theta_o)

			// Snell law
			float sin2_t = eta_ratio * eta_ratio * (1.0f - wo_dot_wm * wo_dot_wm);
			if (sin2_t >= 1.0f)
			{
				// Total internal reflection
				reflectedColour = Colour(0.0f, 0.0f, 0.0f);
				pdf = 0.0f;
				return Vec3(0.0f, 0.0f, 0.0f);
			}
			float cos_t = sqrtf(1.0f - sin2_t);
			// Refraction direction
			float k = eta_ratio * wo_dot_wm - cos_t;
			wiLocal = Vec3(k * wm.x - eta_ratio * woLocal.x, k * wm.y - eta_ratio * woLocal.y, k * wm.z - eta_ratio * woLocal.z);

			if (wiLocal.z * woLocal.z >= 0.0f)
			{
				// Refraction on the same side is invalid
				reflectedColour = Colour(0.0f, 0.0f, 0.0f);
				pdf = 0.0f;
				return Vec3(0.0f, 0.0f, 0.0f);
			}
		}

		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		pdf = PDF(shadingData, wiWorld);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Dielectric evaluation code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (fabsf(woLocal.z) < 1e-6f || fabsf(wiLocal.z) < 1e-6f) return Colour(0.0f, 0.0f, 0.0f);

		Colour alb = albedo->sample(shadingData.tu, shadingData.tv);
		bool entering = (woLocal.z > 0.0f);
		bool sameSide = (woLocal.z * wiLocal.z > 0.0f);

		if (sameSide)
		{
			// Reflection if wi and wo are on the same side
			Vec3 sum(woLocal.x + wiLocal.x, woLocal.y + wiLocal.y, woLocal.z + wiLocal.z);
			float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
			if (len < 1e-6f) return Colour(0.0f, 0.0f, 0.0f);
			Vec3 wm(sum.x / len, sum.y / len, sum.z / len);
			// Align wm with wo side
			if (wm.z * woLocal.z < 0.0f) wm = Vec3(-wm.x, -wm.y, -wm.z);
			float wo_dot_wm = Dot(woLocal, wm);
			if (wo_dot_wm <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

			float F = ShadingHelper::fresnelDielectric(wo_dot_wm, intIOR, extIOR);
			Vec3 wmUp = (wm.z >= 0.0f) ? wm : Vec3(-wm.x, -wm.y, -wm.z);
			Vec3 woUp(woLocal.x, woLocal.y, fabsf(woLocal.z));
			Vec3 wiUp(wiLocal.x, wiLocal.y, fabsf(wiLocal.z));
			float D = ShadingHelper::Dggx(wmUp, alpha);
			float G = ShadingHelper::Gggx(wiUp, woUp, alpha);

			// Cook-Torrance reflection
			float denom = 4.0f * fabsf(woLocal.z) * fabsf(wiLocal.z);
			if (denom < 1e-6f) return Colour(0.0f, 0.0f, 0.0f);
			return alb * (F * D * G / denom);
		}
		else
		{
			// Refraction if wi and wo are on different sides
			float eta = entering ? (intIOR / extIOR) : (extIOR / intIOR);
			Vec3 sum(eta * wiLocal.x + woLocal.x, eta * wiLocal.y + woLocal.y, eta * wiLocal.z + woLocal.z);
			float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
			if (len < 1e-6f) return Colour(0.0f, 0.0f, 0.0f);
			Vec3 wm(sum.x / len, sum.y / len, sum.z / len);
			// Align wm with wo's side
			if (wm.z * woLocal.z < 0.0f) wm = Vec3(-wm.x, -wm.y, -wm.z);

			float wo_dot_wm = Dot(woLocal, wm);
			float wi_dot_wm = Dot(wiLocal, wm);
			if (wo_dot_wm <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

			float F = ShadingHelper::fresnelDielectric(wo_dot_wm, intIOR, extIOR);
			Vec3 wmUp = (wm.z >= 0.0f) ? wm : Vec3(-wm.x, -wm.y, -wm.z);
			Vec3 woUp(woLocal.x, woLocal.y, fabsf(woLocal.z));
			Vec3 wiUp(wiLocal.x, wiLocal.y, fabsf(wiLocal.z));
			float D = ShadingHelper::Dggx(wmUp, alpha);
			float G = ShadingHelper::Gggx(wiUp, woUp, alpha);

			// Cook-Torrance refraction
			float denomTerm = wi_dot_wm + wo_dot_wm / eta;
			float denomSq = denomTerm * denomTerm;
			if (denomSq < 1e-6f) return Colour(0.0f, 0.0f, 0.0f);

			float fr = (1.0f - F) * D * G * fabsf(wi_dot_wm) * fabsf(wo_dot_wm) / (denomSq * fabsf(wiLocal.z) * fabsf(woLocal.z));
			return alb * fr;
		}
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with Dielectric PDF
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (fabsf(woLocal.z) < 1e-6f || fabsf(wiLocal.z) < 1e-6f) return 0.0f;

		bool entering = (woLocal.z > 0.0f);
		bool sameSide = (woLocal.z * wiLocal.z > 0.0f);

		if (sameSide)
		{
			// Reflection branch
			Vec3 sum(woLocal.x + wiLocal.x, woLocal.y + wiLocal.y, woLocal.z + wiLocal.z);
			float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
			if (len < 1e-6f) return 0.0f;
			Vec3 wm(sum.x / len, sum.y / len, sum.z / len);
			if (wm.z * woLocal.z < 0.0f) wm = Vec3(-wm.x, -wm.y, -wm.z);
			float wo_dot_wm = Dot(woLocal, wm);
			if (wo_dot_wm <= 0.0f) return 0.0f;

			float F = ShadingHelper::fresnelDielectric(wo_dot_wm, intIOR, extIOR);
			Vec3 wmUp = (wm.z >= 0.0f) ? wm : Vec3(-wm.x, -wm.y, -wm.z);
			float D = ShadingHelper::Dggx(wmUp, alpha);
			float pReflect = D * fabsf(wm.z) / (4.0f * wo_dot_wm);
			return F * pReflect;
		}
		else
		{
			// Refraction branch
			float eta = entering ? (intIOR / extIOR) : (extIOR / intIOR);
			Vec3 sum(eta * wiLocal.x + woLocal.x,
				eta * wiLocal.y + woLocal.y,
				eta * wiLocal.z + woLocal.z);
			float len = sqrtf(sum.x * sum.x + sum.y * sum.y + sum.z * sum.z);
			if (len < 1e-6f) return 0.0f;
			Vec3 wm(sum.x / len, sum.y / len, sum.z / len);
			if (wm.z * woLocal.z < 0.0f) wm = Vec3(-wm.x, -wm.y, -wm.z);

			float wo_dot_wm = Dot(woLocal, wm);
			float wi_dot_wm = Dot(wiLocal, wm);
			if (wo_dot_wm <= 0.0f) return 0.0f;

			float F = ShadingHelper::fresnelDielectric(wo_dot_wm, intIOR, extIOR);
			Vec3 wmUp = (wm.z >= 0.0f) ? wm : Vec3(-wm.x, -wm.y, -wm.z);
			float D = ShadingHelper::Dggx(wmUp, alpha);

			float denomTerm = wi_dot_wm + wo_dot_wm / eta;
			float denomSq = denomTerm * denomTerm;
			if (denomSq < 1e-6f) return 0.0f;

			float pRefract = D * fabsf(wm.z) * fabsf(wo_dot_wm) / denomSq;
			return (1.0f - F) * pRefract;
		}
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
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 0.0f);
		}
		// Sample wi in local space
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		reflectedColour = evaluate(shadingData, wiWorld);
		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with OrenNayar evaluation code
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		float cos_o = woLocal.z, cos_i = wiLocal.z;
		// validate
		if (cos_o <= 0.0f || cos_i <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		// Oren-Nayar
		float sigma2 = sigma * sigma;
		float A = 1.0f - sigma2 / (2.0f * (sigma2 + 0.33f));
		float B = 0.45f * sigma2 / (sigma2 + 0.09f);

		float sin_o = sqrtf(std::max(0.0f, 1.0f - cos_o * cos_o));
		float sin_i = sqrtf(std::max(0.0f, 1.0f - cos_i * cos_i));

		// Compute cos(phi_i - phi_o)
		float cosDeltaPhi = 0.0f;
		if (sin_o > 1e-6f && sin_i > 1e-6f)
		{
			float cos_phi_o = woLocal.x / sin_o, sin_phi_o = woLocal.y / sin_o;
			float cos_phi_i = wiLocal.x / sin_i, sin_phi_i = wiLocal.y / sin_i;
			cosDeltaPhi = cos_phi_i * cos_phi_o + sin_phi_i * sin_phi_o;
		}

		// Compute alpha and beta
		float theta_o = acosf(std::min(cos_o, 1.0f));
		float theta_i = acosf(std::min(cos_i, 1.0f));
		float sinAlpha = sinf(std::max(theta_o, theta_i));
		float tanBeta = tanf(std::min(theta_o, theta_i));

		float oren = A + B * std::max(0.0f, cosDeltaPhi) * sinAlpha * tanBeta;
		return albedo->sample(shadingData.tu, shadingData.tv) * (oren / (float)M_PI);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Replace this with OrenNayar PDF
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;
		return wiLocal.z / (float)M_PI;
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
		float e = (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;

		Vec3 wiLocal;

		if (sampler->next() < F)
		{
			// Specular reflection direction
			Vec3 wr(-woLocal.x, -woLocal.y, woLocal.z);

			// Sample z-up lobe
			float u1 = sampler->next();
			float u2 = sampler->next();
			float cosThetaLobe = powf(std::max(u1, 1e-6f), 1.0f / (e + 1.0f));
			float sinThetaLobe = sqrtf(std::max(0.0f, 1.0f - cosThetaLobe * cosThetaLobe));
			float phiLobe = 2.0f * (float)M_PI * u2;
			Vec3 wLobe(sinThetaLobe * cosf(phiLobe),
				sinThetaLobe * sinf(phiLobe),
				cosThetaLobe);

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
		float e = (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
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
		float e = (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;

		// Phong lobe PDF around wr
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
		//* Add code to include layered sampling
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		if (woLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		float Fo = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);

		if (sampler->next() < Fo)
		{
			// Specular reflection
			Vec3 wiLocal(-woLocal.x, -woLocal.y, woLocal.z);
			Vec3 wi = shadingData.frame.toWorld(wiLocal);
			float invCos = 1.0f / std::max(wiLocal.z, 1e-6f);
			float c = Fo * invCos;  // F(wo)/cos(theta)
			reflectedColour = Colour(c, c, c);
			pdf = Fo;	// discrete branch probability for the delta
			return wi;
		}

		// Refraction
		float eta_in = extIOR / intIOR;
		float sin2_o = std::max(0.0f, 1.0f - woLocal.z * woLocal.z);
		float cos_oRef = sqrtf(std::max(1e-6f, 1.0f - eta_in * eta_in * sin2_o));
		Vec3 woRef(eta_in * woLocal.x, eta_in * woLocal.y, cos_oRef);

		// Sample the base BSDF
		ShadingData sdRef = shadingData;
		sdRef.wo = shadingData.frame.toWorld(woRef);

		Colour baseVal;
		float basePdf;
		Vec3 wiRefWorld = base->sample(sdRef, sampler, baseVal, basePdf);
		if (basePdf <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return wiRefWorld;
		}

		Vec3 wiRefLocal = shadingData.frame.toLocal(wiRefWorld);
		if (wiRefLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return wiRefWorld;
		}

		// Reflection
		float eta_out = intIOR / extIOR;
		float sin2_iRef = std::max(0.0f, 1.0f - wiRefLocal.z * wiRefLocal.z);
		float sin2_i = eta_out * eta_out * sin2_iRef;
		if (sin2_i >= 1.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return wiRefWorld;
		}
		float cos_i = sqrtf(1.0f - sin2_i);
		Vec3 wiLocal(eta_out * wiRefLocal.x, eta_out * wiRefLocal.y, cos_i);
		Vec3 wi = shadingData.frame.toWorld(wiLocal);

		float Fi = ShadingHelper::fresnelDielectric(cos_i, intIOR, extIOR);

		// Beer-Lambert absorption in the layer
		float opticalDepth = thickness * (1.0f / cos_oRef + 1.0f / wiRefLocal.z);
		Colour Tr(expf(-sigmaa.r * opticalDepth), expf(-sigmaa.g * opticalDepth), expf(-sigmaa.b * opticalDepth));

		float etaSq = eta_in * eta_in;
		float invCosFactor = 1.0f / (cos_i * wiRefLocal.z);

		reflectedColour = baseVal * Tr * (etaSq * (1.0f - Fo) * (1.0f - Fi) * invCosFactor);

		// PDF Jacobian
		float jacobian = etaSq * cos_i / wiRefLocal.z;
		pdf = (1.0f - Fo) * basePdf * jacobian;
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Add code for evaluation of layer
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return Colour(0.0f, 0.0f, 0.0f);

		// Refraction
		float eta_in = extIOR / intIOR;
		float sin2_o = std::max(0.0f, 1.0f - woLocal.z * woLocal.z);
		float cos_oRef = sqrtf(std::max(1e-6f, 1.0f - eta_in * eta_in * sin2_o));
		Vec3 woRef(eta_in * woLocal.x, eta_in * woLocal.y, cos_oRef);

		float sin2_i = std::max(0.0f, 1.0f - wiLocal.z * wiLocal.z);
		float cos_iRef = sqrtf(std::max(1e-6f, 1.0f - eta_in * eta_in * sin2_i));
		Vec3 wiRef(eta_in * wiLocal.x, eta_in * wiLocal.y, cos_iRef);

		// Evaluate base BSDF
		ShadingData sdRef = shadingData;
		sdRef.wo = shadingData.frame.toWorld(woRef);
		Vec3 wiRefWorld = shadingData.frame.toWorld(wiRef);
		Colour baseEval = base->evaluate(sdRef, wiRefWorld);

		// Fresnel
		float Fo = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);
		float Fi = ShadingHelper::fresnelDielectric(fabsf(wiLocal.z), intIOR, extIOR);

		// Beer-Lambert
		float opticalDepth = thickness * (1.0f / cos_oRef + 1.0f / cos_iRef);
		Colour Tr(expf(-sigmaa.r * opticalDepth), expf(-sigmaa.g * opticalDepth), expf(-sigmaa.b * opticalDepth));

		float etaSq = eta_in * eta_in;
		float invCosFactor = 1.0f / (fabsf(wiLocal.z) * cos_iRef);

		return baseEval * Tr * (etaSq * (1.0f - Fo) * (1.0f - Fi) * invCosFactor);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		//* Add code to include PDF for sampling layered BSDF
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f) return 0.0f;

		float Fo = ShadingHelper::fresnelDielectric(fabsf(woLocal.z), intIOR, extIOR);

		// Refraction
		float eta_in = extIOR / intIOR;
		float sin2_o = std::max(0.0f, 1.0f - woLocal.z * woLocal.z);
		float cos_oRef = sqrtf(std::max(1e-6f, 1.0f - eta_in * eta_in * sin2_o));
		Vec3 woRef(eta_in * woLocal.x, eta_in * woLocal.y, cos_oRef);

		float sin2_i = std::max(0.0f, 1.0f - wiLocal.z * wiLocal.z);
		float cos_iRef = sqrtf(std::max(1e-6f, 1.0f - eta_in * eta_in * sin2_i));
		Vec3 wiRef(eta_in * wiLocal.x, eta_in * wiLocal.y, cos_iRef);

		ShadingData sdRef = shadingData;
		sdRef.wo = shadingData.frame.toWorld(woRef);
		Vec3 wiRefWorld = shadingData.frame.toWorld(wiRef);

		float basePdf = base->PDF(sdRef, wiRefWorld);

		// Jacobian for refraction
		float jacobian = (eta_in * eta_in) * fabsf(wiLocal.z) / cos_iRef;
		return (1.0f - Fo) * basePdf * jacobian;
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
