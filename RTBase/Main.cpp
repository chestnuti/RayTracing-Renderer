// Using Z-up coordinate system, left-handed

#include "GEMLoader.h"
#include "Renderer.h"
#include "SceneLoader.h"
#define NOMINMAX
#include "GamesEngineeringBase.h"
#include <unordered_map>

void runTests()
{
	// add triangle intersection tests here if required
	Triangle tri;
	tri.vertices[0].p = Vec3(-1.0f, -1.0f, 0.0f);
	tri.vertices[1].p = Vec3(1.0f, -1.0f, 0.0f);
	tri.vertices[2].p = Vec3(0.0f, 1.0f, 0.0f);
	tri.init(tri.vertices[0], tri.vertices[1], tri.vertices[2], 0);
	Ray r;
	r.o = Vec3(0.0f, 0.0f, -1.0f);
	r.dir = Vec3(0.0f, 0.0f, 1.0f);
	float t, u, v;
	if (tri.rayIntersect(r, t, u, v))
	{
		std::cout << "Intersection at t = " << t << ", u = " << u << ", v = " << v << std::endl;
	}
	else
	{
		std::cout << "No intersection" << std::endl;
	}
}

int main(int argc, char *argv[])
{
	// Add call to tests if required
	// runTests();
	
	// Initialize default parameters
	std::string sceneName = "MaterialsScene";
	std::string filename = "GI.hdr";
	unsigned int SPP = 128;
	bool enableDenoise = true;
    bool enableMIS = true;
	bool enableEnvmap = true;
	int renderMode = 0; // 0 = tile-based path tracing, 1 = light tracing, 2 = instant radiosity, 3 = albedo

	if (argc > 1)
	{
		std::unordered_map<std::string, std::string> args;
		for (int i = 1; i < argc; ++i)
		{
			std::string arg = argv[i];
			if (!arg.empty() && arg[0] == '-')
			{
				std::string argName = arg;
				if (i + 1 < argc)
				{
					std::string argValue = argv[++i];
					args[argName] = argValue;
				} else
				{
					std::cerr << "Error: Missing value for argument '" << arg << "'\n";
				}
			} else
			{
				std::cerr << "Warning: Ignoring unexpected argument '" << arg << "'\n";
			}
		}
		for (const auto& pair : args)
		{
			if (pair.first == "-scene")
			{
				sceneName = pair.second;
			}
			if (pair.first == "-outputFilename")
			{
				filename = pair.second;
			}
			if (pair.first == "-SPP")
			{
				SPP = stoi(pair.second);
			}
			if (pair.first == "-denoise")
			{
				enableDenoise = stoi(pair.second) != 0;
			}
           if (pair.first == "-MIS")
			{
				enableMIS = stoi(pair.second) != 0;
			}
		}
	}
	Scene* scene = loadScene(sceneName);
	GamesEngineeringBase::Window canvas;
	canvas.create((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, "Tracer", false);
	EnvironmentMap* envmap = dynamic_cast<EnvironmentMap*>(scene->background);
	if (envmap)
	{
		envmap->enableEnvmap = enableEnvmap;
	}
	RayTracer rt;
	rt.init(scene, &canvas);
    rt.enableMIS = enableMIS;
	bool running = true;
	GamesEngineeringBase::Timer timer;
	while (running)
	{
		canvas.checkInput();
		canvas.clear();
		if (canvas.keyPressed(VK_ESCAPE))
		{
			break;
		}
		if (canvas.keyPressed('W'))
		{
			viewcamera.forward();
			rt.clear();
		}
		if (canvas.keyPressed('S'))
		{
			viewcamera.back();
			rt.clear();
		}
		if (canvas.keyPressed('A'))
		{
			viewcamera.left();
			rt.clear();
		}
		if (canvas.keyPressed('D'))
		{
			viewcamera.right();
			rt.clear();
		}
		if (canvas.keyPressed('E'))
		{
			viewcamera.flyUp();
			rt.clear();
		}
		if (canvas.keyPressed('Q'))
		{
			viewcamera.flyDown();
			rt.clear();
		}
		// Time how long a render call takes
		timer.reset();
		// Render
		rt.render(renderMode);
		float t = timer.dt();
		// Write
		std::cout << t << std::endl;
		if (canvas.keyPressed('P'))
		{
			rt.saveHDR(filename);
		}
		if (canvas.keyPressed('L'))
		{
			size_t pos = filename.find_last_of('.');
			std::string ldrFilename = filename.substr(0, pos) + ".png";
			rt.savePNG(ldrFilename);
		}
		if (SPP == rt.getSPP())
		{
			rt.finalizeAOVs();
			if (enableDenoise)
			{
				rt.denoiseOIDN();
			}
			rt.saveHDR(filename);
			break;
		}
		canvas.present();
	}
	return 0;
}
