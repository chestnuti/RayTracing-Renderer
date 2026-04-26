#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <iostream>
#if defined(USE_OIDN) && __has_include("OpenImageDenoise/oidn.hpp")
#include "OpenImageDenoise/oidn.hpp"
#else
#define RTBASE_HAS_OIDN 0
#endif

struct Tile {
	unsigned int x, y, w, h;
};

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom* samplers;
	std::thread** threads;
	int numProcs;

	std::vector<Tile> tiles;	// List of tiles to render
	std::atomic<int> tileIndex;	// Atomic index to keep track of the next tile to render
	std::mutex filmMutex;	// protect film->splat()
	std::mutex canvasMutex;	// protect canvas->draw()

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread * [numProcs];
		samplers = new MTRandom[numProcs];
		clear();
	}

	void clear()
	{
		film->clear();
	}

	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		// Skip direct lighting
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}

		// Sample a light source from the scene
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		if (!light)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}

		// Estimate direct lighting from the sampled light source
		float lightPdf;
		Colour emission;
		Vec3 lightSamplePoint = light->sample(shadingData, sampler, emission, lightPdf);

		if (light->isArea())
		{
			// Evaluate the direct-light contribution from the area light-sampled direction
			Vec3 wi = (lightSamplePoint - shadingData.x).normalize();
			float dist = (lightSamplePoint - shadingData.x).length();
			float cosAtSurface = std::max(0.0f, Dot(shadingData.sNormal, wi));
			float cosAtLight = std::max(0.0f, Dot(light->normal(shadingData, wi), -wi));
			float G = cosAtSurface * cosAtLight / (dist * dist);

			if (!scene->visible(shadingData.x, lightSamplePoint))
			{
				return Colour(0.0f, 0.0f, 0.0f);
			}

			Colour bsdf = shadingData.bsdf->evaluate(shadingData, wi);
			return bsdf * emission * G / (lightPdf * pmf);
		}

		// Accumulate direct lighting from the environment with MIS
		Colour directLighting(0.0f, 0.0f, 0.0f);

		// Treat non-area lights as environment lights and combine two estimators with MIS
		Vec3 wiEnv = lightSamplePoint;
		// Strategy PDF = select-this-light PMF * environment directional PDF
		const float envStrategyPdf = lightPdf * pmf;
		if (envStrategyPdf > EPSILON && emission.Lum() > 0.0f)
		{
			// Evaluate the direct-light contribution from the environment-sampled direction
			Ray shadowRay;
			shadowRay.init(shadingData.x + wiEnv * EPSILON, wiEnv);
			if (scene->bvh->traverseVisible(shadowRay, scene->triangles, FLT_MAX))
			{
				// If visible, evaluate the BSDF and MIS balance weight for this strategy
				Colour bsdfValue = shadingData.bsdf->evaluate(shadingData, wiEnv);
				const float bsdfPdf = shadingData.bsdf->PDF(shadingData, wiEnv);
				const float cosTheta = fabsf(Dot(shadingData.sNormal, wiEnv));
				const float misWeight = envStrategyPdf / (envStrategyPdf + bsdfPdf + EPSILON);

				if (bsdfValue.Lum() > 0.0f && cosTheta > 0.0f)
				{
					directLighting = directLighting + (emission * bsdfValue * (cosTheta * misWeight / envStrategyPdf));
				}
			}
		}

		float bsdfPdf = 0.0f;
		Colour bsdfValue;
		// Second MIS strategy: sample a direction from the BSDF
		Vec3 wiBsdf = shadingData.bsdf->sample(shadingData, sampler, bsdfValue, bsdfPdf);
		if (bsdfPdf > EPSILON && bsdfValue.Lum() > 0.0f)
		{
			// Evaluate the direct-light contribution from the BSDF-sampled direction
			Ray shadowRay;
			shadowRay.init(shadingData.x + wiBsdf * EPSILON, wiBsdf);
			if (scene->bvh->traverseVisible(shadowRay, scene->triangles, FLT_MAX))
			{
				// If visible, evaluate the environment radiance from that direction
				Colour envRadiance = light->evaluate(wiBsdf);
				if (envRadiance.Lum() > 0.0f)
				{
					// Compute the MIS balance weight against the environment-sampling strategy
					const float envPdf = light->PDF(shadingData, wiBsdf) * pmf;
					const float cosTheta = fabsf(Dot(shadingData.sNormal, wiBsdf));
					const float misWeight = bsdfPdf / (envPdf + bsdfPdf + EPSILON);
					directLighting = directLighting + (envRadiance * bsdfValue * (cosTheta * misWeight / bsdfPdf));
				}
			}
		}

		return directLighting;
	}

	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler)
	{
		constexpr int kMaxDepth = 8;
		if (depth >= kMaxDepth)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}

		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		if (shadingData.t >= FLT_MAX)
		{
			return pathThroughput * scene->background->evaluate(r.dir);
		}

		if (shadingData.bsdf->isLight())
		{
			if (depth == 0)
			{
				return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return Colour(0.0f, 0.0f, 0.0f);
		}

		Colour L(0.0f, 0.0f, 0.0f);
		L = L + pathThroughput * computeDirect(shadingData, sampler);

		// End low-throughput paths with Russian roulette
		float q = std::min(pathThroughput.Lum(), 1.0f);
		if (q <= EPSILON || sampler->next() > q)
		{
			return L;
		}
		pathThroughput = pathThroughput / q;

		float pdf = 0.0f;
		Colour bsdfVal;
		Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, bsdfVal, pdf);
		if (pdf <= EPSILON || bsdfVal.Lum() <= 0.0f)
		{
			return L;
		}

		float cosTheta = std::abs(Dot(shadingData.sNormal, wi));
		pathThroughput = pathThroughput * bsdfVal * cosTheta / pdf;

		// Continue the path with the sampled BSDF direction
		Ray nextRay;
		nextRay.init(shadingData.x + wi * EPSILON, wi);
		IntersectionData nextIntersection = scene->traverse(nextRay);
		if (nextIntersection.t >= FLT_MAX)
		{
			// Only specular chains allowed to pick up background radiance
			if (shadingData.bsdf->isPureSpecular() == false)
			{
				return L;
			}

			Colour envRadiance = scene->background->evaluate(wi);
			return L + (pathThroughput * envRadiance);
		}

		ShadingData nextShadingData = scene->calculateShadingData(nextIntersection, nextRay);
		if (nextShadingData.bsdf->isLight())
		{
			// Only specular chains allowed to pick up emissive surfaces
			if (shadingData.bsdf->isPureSpecular() == false)
			{
				return L;
			}
			return L + (pathThroughput * nextShadingData.bsdf->emit(nextShadingData, nextShadingData.wo));
		}

		return L + pathTrace(nextRay, pathThroughput, depth + 1, sampler);
	}

	Colour direct(Ray& r, Sampler* sampler)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return computeDirect(shadingData, sampler);
		}
		return scene->background->evaluate(r.dir);
	}

	Colour albedo(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate(r.dir);
	}

	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	PrimaryAOV computePrimaryAOV(const Ray& r)
	{
		PrimaryAOV aov;
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t >= FLT_MAX)
		{
			return aov;
		}

		Ray ray = r;
		ShadingData shadingData = scene->calculateShadingData(intersection, ray);
		aov.hit = true;
		aov.albedo = albedo(ray);
		aov.normal = Colour(shadingData.sNormal.x, shadingData.sNormal.y, shadingData.sNormal.z);
		return aov;
	}

	void finalizeAOVs()
	{
		film->finalizeAOVs();
	}

	bool denoiseOIDN()
	{
		const unsigned int pixelCount = film->width * film->height;
		if (pixelCount == 0)
		{
			return false;
		}

		std::vector<Colour> beauty = film->getAveragedBeauty();
		std::vector<Colour> out(pixelCount, Colour(0, 0, 0));

		oidn::DeviceRef device = oidn::newDevice();
		device.commit();
		oidn::FilterRef filter = device.newFilter("RT");

		const size_t imageBytes = static_cast<size_t>(pixelCount) * sizeof(Colour);
		oidn::BufferRef colorBuf = device.newBuffer(imageBytes);
		oidn::BufferRef outputBuf = device.newBuffer(imageBytes);
		std::memcpy(colorBuf.getData(), beauty.data(), imageBytes);

		filter.setImage("color", colorBuf, oidn::Format::Float3, film->width, film->height);
		filter.setImage("output", outputBuf, oidn::Format::Float3, film->width, film->height);

		oidn::BufferRef albedoBuf;
		if (film->aovAlbedo.size() == pixelCount)
		{
			albedoBuf = device.newBuffer(imageBytes);
			std::memcpy(albedoBuf.getData(), film->aovAlbedo.data(), imageBytes);
			filter.setImage("albedo", albedoBuf, oidn::Format::Float3, film->width, film->height);
		}

		oidn::BufferRef normalBuf;
		if (film->aovNormal.size() == pixelCount)
		{
			normalBuf = device.newBuffer(imageBytes);
			std::memcpy(normalBuf.getData(), film->aovNormal.data(), imageBytes);
			filter.setImage("normal", normalBuf, oidn::Format::Float3, film->width, film->height);
		}
		filter.set("hdr", true);
		filter.commit();
		filter.execute();

		const char* errorMessage;
		if (device.getError(errorMessage) != oidn::Error::None)
		{
			std::cerr << "OIDN error: " << errorMessage << std::endl;
			return false;
		}

		std::memcpy(out.data(), outputBuf.getData(), imageBytes);

		film->denoisedBeauty = std::move(out);
		film->hasDenoisedBeauty = true;
		return true;
	}

	void render()
	{
		film->incrementSPP();
		// Tile-based rendering loop
		unsigned int tileX = std::max(1u, film->width / (unsigned int)numProcs);
		unsigned int tileY = std::max(1u, film->height / (unsigned int)numProcs);
		tiles.clear();
		for (unsigned int y = 0; y < film->height; y += tileY)
		{
			for (unsigned int x = 0; x < film->width; x += tileX)
			{
				Tile tile;
				tile.x = x;
				tile.y = y;
				tile.w = std::min(tileX, film->width - x);
				tile.h = std::min(tileY, film->height - y);
				tiles.push_back(tile);
			}
		}

		auto renderTile = [&](int threadId) {
			while (true) {
				int index = tileIndex.fetch_add(1);
				if (index >= tiles.size()) {
					break;
				}
				Tile tile = tiles[index];
				for (unsigned int y = tile.y; y < tile.y + tile.h; y++)
				{
					for (unsigned int x = tile.x; x < tile.x + tile.w; x++)
					{
						float px = x + 0.5f;
						float py = y + 0.5f;
						Ray ray = scene->camera.generateRay(px, py);
						Colour startingThroughput(1, 1, 1);
						Colour col = pathTrace(ray, startingThroughput, 0, &samplers[threadId]);
						film->splat(px, py, col);
						PrimaryAOV aov = computePrimaryAOV(ray);
						film->splatAOV(x, y, aov);
						unsigned char r = (unsigned char)(col.r * 255);
						unsigned char g = (unsigned char)(col.g * 255);
						unsigned char b = (unsigned char)(col.b * 255);
						film->tonemap(x, y, r, g, b);
						canvas->draw(x, y, r, g, b);
					}
				}
			}
		};

		// Create threads to render tiles
		tileIndex = 0;
		for (int i = 0; i < numProcs; i++) {
			threads[i] = new std::thread(renderTile, i);
		}

		// Wait for all threads to finish
		for (int i = 0; i < numProcs; i++) {
			threads[i]->join();
			delete threads[i];
		}


		//for (unsigned int y = 0; y < film->height; y++)
		//{
		//	for (unsigned int x = 0; x < film->width; x++)
		//	{
		//		float px = x + 0.5f;
		//		float py = y + 0.5f;
		//		Ray ray = scene->camera.generateRay(px, py);
		//		//Colour col = viewNormals(ray);
		//		Colour col = albedo(ray);
		//		film->splat(px, py, col);
		//		unsigned char r = (unsigned char)(col.r * 255);
		//		unsigned char g = (unsigned char)(col.g * 255);
		//		unsigned char b = (unsigned char)(col.b * 255);
		//		film->tonemap(x, y, r, g, b);
		//		canvas->draw(x, y, r, g, b);
		//	}
		//}
	}

	int getSPP()
	{
		return film->SPP;
	}

	void saveHDR(std::string filename)
	{
		film->save(filename);
	}

	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}
};
