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
#ifdef USE_OIDN
#include "OpenImageDenoise/oidn.hpp"
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
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;

	std::vector<Tile> tiles;	// List of tiles to render
	std::atomic<int> tileIndex;	// Atomic index to keep track of the next tile to render
	std::mutex filmMutex;   // protect film->splat()
	std::mutex canvasMutex; // protect canvas->draw()

	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];
		clear();
	}
	void clear()
	{
		film->clear();
	}
	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		// Is surface is specular we cannot computing direct lighting
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		// Compute direct lighting here
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		if (!light) return Colour(0, 0, 0);

		float lightPdf;
		Colour emission;
		Vec3 lightSamplePoint = light->sample(shadingData, sampler, emission, lightPdf);

		// Distinguish area light
		if (light->isArea()) {
			Vec3 wi = (lightSamplePoint - shadingData.x).normalize();
			float dist = (lightSamplePoint - shadingData.x).length();
			// Geometric term
			float cosAtSurface = std::max(0.0f, Dot(shadingData.sNormal, wi));
			float cosAtLight = std::max(0.0f, Dot(light->normal(shadingData, wi), -wi));
			float G = cosAtSurface * cosAtLight / (dist * dist);
			// Visibility
			if (!scene->visible(shadingData.x, lightSamplePoint))
				return Colour(0, 0, 0);
			Colour bsdf = shadingData.bsdf->evaluate(shadingData, wi);
			return bsdf * emission * G / (lightPdf * pmf);
		}
		else {
			// Directional/Environment light
			Vec3 wi = lightSamplePoint;
			float cosTheta = std::max(0.0f, Dot(shadingData.sNormal, wi));
			// Visible if no intersection
			Ray shadowRay; shadowRay.init(shadingData.x + wi * EPSILON, wi);
			if (scene->bvh->traverseVisible(shadowRay, scene->triangles, FLT_MAX) == false)
				return Colour(0, 0, 0);
			Colour bsdf = shadingData.bsdf->evaluate(shadingData, wi);
			return bsdf * emission * cosTheta / (lightPdf * pmf);
		}
	}
	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler)
	{
		//* Add pathtracer code here
		// Limit the maximum depth
		constexpr int kMaxDepth = 8;
		if (depth >= kMaxDepth) return Colour(0, 0, 0);
		// Compute ray-scene intersection
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		if (shadingData.t >= FLT_MAX)
			return pathThroughput * scene->background->evaluate(r.dir);

		// to light source directly
		if (shadingData.bsdf->isLight())
		{
			// Only count emission for primary ray
			if (depth == 0)
				return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
			else
				return Colour(0, 0, 0);
		}

		Colour L(0, 0, 0);

		// NEE direct lighting
		L = L + pathThroughput * computeDirect(shadingData, sampler);

		// Russian roulette termination
		float q = std::min(pathThroughput.Lum(), 1.0f);
		if (sampler->next() > q) return L;
		pathThroughput = pathThroughput / q;

		// sample BSDF
		float pdf;
		Colour bsdfVal;
		Vec3 wi = shadingData.bsdf->sample(shadingData, sampler, bsdfVal, pdf);
		float cosTheta = std::abs(Dot(shadingData.sNormal, wi));
		pathThroughput = pathThroughput * bsdfVal * cosTheta / pdf;

		// irradiance along the sampled direction
		Ray nextRay; nextRay.init(shadingData.x + wi * EPSILON, wi);
		return L + pathTrace(nextRay, pathThroughput, depth + 1, sampler);
	}
	Colour direct(Ray& r, Sampler* sampler)
	{
		// Compute direct lighting for an image sampler here
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);

		// Miss -> background
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

		// Lambda function for rendering a tile
		auto renderTile = [&](int threadId) {
			while (true) {
				int index = tileIndex.fetch_add(1);
				if (index >= tiles.size()) {
					break; // No more tiles to render
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
		tileIndex = 0; // Initialize atomic tile index
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
