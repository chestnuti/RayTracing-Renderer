#pragma once

#include "Core.h"
#include "Sampling.h"

#define EPSILON 0.001f

class AABB;

class Ray
{
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray()
	{
	}
	Ray(Vec3 _o, Vec3 _d)
	{
		init(_o, _d);
	}
	void init(Vec3 _o, Vec3 _d)
	{
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}
	Vec3 at(const float t) const
	{
		return (o + (dir * t));
	}
};

class Plane
{
public:
	Vec3 n;
	float d;
	void init(Vec3& _n, float _d)
	{
		n = _n;
		d = _d;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		float tempT;
		tempT = (d - Dot(n, r.o)) / Dot(n, r.dir);
		if (t > EPSILON)
		{
			if (tempT < t)
			{
				t = tempT;
			}
			return true;
		}
		return false;
	}
};


class Triangle
{
public:
	Vertex vertices[3];
	Vec3 e1; // Edge 1
	Vec3 e2; // Edge 2
	Vec3 n; // Geometric Normal
	float area; // Triangle area
	float d; // For ray triangle if needed
	unsigned int materialIndex;
	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex)
	{
		materialIndex = _materialIndex;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;

		// Moller-Trumbore algorithm standard approach precomputes edges relative to vertex v0
		e1 = vertices[1].p - vertices[0].p;
		e2 = vertices[2].p - vertices[0].p;

		n = e1.cross(e2).normalize();
		area = e1.cross(e2).length() * 0.5f;
		d = Dot(n, vertices[0].p);
	}
	Vec3 centre() const
	{
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}
	// Add code here
	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const
	{
		// Moller-Trumbore ray-triangle intersection algorithm
		Vec3 pvec = r.dir.cross(e2);
		float det = e1.dot(pvec);
		if (fabs(det) < EPSILON)
		{
			return false; // Ray is parallel to the triangle
		}
		float invDet = 1.0f / det;
		Vec3 tvec = r.o - vertices[0].p;
		u = tvec.dot(pvec) * invDet;
		if (u < 0.0f || u > 1.0f)
		{
			return false; // Intersection is outside the triangle
		}
		Vec3 qvec = tvec.cross(e1);
		v = r.dir.dot(qvec) * invDet;
		if (v < 0.0f || u + v > 1.0f)
		{
			return false; // Intersection is outside the triangle
		}
		t = e2.dot(qvec) * invDet;
		if (t < EPSILON)
		{
			return false; // Intersection is behind the ray origin
		}

		return true;
	}
	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec3& interpolatedNormal, float& interpolatedU, float& interpolatedV) const
	{
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}
	// Add code here
	Vec3 sample(Sampler* sampler, float& pdf)
	{
		return Vec3(0, 0, 0);
	}
	Vec3 gNormal()
	{
		return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f));
	}
};

class AABB
{
public:
	Vec3 max;
	Vec3 min;
	AABB()
	{
		reset();
	}
	void reset()
	{
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	void extend(const Vec3 p)
	{
		max = Max(max, p);
		min = Min(min, p);
	}
	// Add code here
	bool rayAABB(const Ray& r, float& t)
	{
		Vec3 Tmin = (min - r.o) * r.invDir;
		Vec3 Tmax = (max - r.o) * r.invDir;
		Vec3 s1 = Min(Tmin, Tmax);
		Vec3 l1 = Max(Tmin, Tmax);
		float ts = std::max(s1.x, std::max(s1.y, s1.y));
		float tl = std::min(l1.x, std::min(l1.y, l1.z));
		t = std::min(ts, tl);
		return (ts <= tl) && (tl >= 0.0f);
	}
	// Add code here
	bool rayAABB(const Ray& r)
	{
		Vec3 s = (min - r.o) * r.invDir;
		Vec3 l = (max - r.o) * r.invDir;
		Vec3 s1 = Min(s, l);
		Vec3 l1 = Max(s, l);
		float ts = std::max(s1.x, std::max(s1.y, s.z));
		float tl = std::min(l1.x, std::min(l1.y, l.z));
		return (ts <= tl) && (tl >= 0.0f);
	}
	// Add code here
	float area()
	{
		Vec3 size = max - min;
		return ((size.x * size.y) + (size.y * size.z) + (size.x * size.z)) * 2.0f;
	}
};

class Sphere
{
public:
	Vec3 centre;
	float radius;
	void init(Vec3& _centre, float _radius)
	{
		centre = _centre;
		radius = _radius;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		// Ray-sphere intersection
		Vec3 oc = r.o - centre;
		float a = Dot(r.dir, r.dir);
		float b = 2.0f * Dot(oc, r.dir);
		float c = Dot(oc, oc) - radius * radius;
		float discriminant = b * b - 4 * a * c;
		if (discriminant < 0)
		{
			return false; // No intersection
		}
		float sqrtDiscriminant = sqrtf(discriminant);
		float t0 = (-b - sqrtDiscriminant) / (2.0f * a);
		float t1 = (-b + sqrtDiscriminant) / (2.0f * a);
		if (t0 > EPSILON)
		{
			t = t0;
			return true; // Intersection at t0
		}
		if (t1 > EPSILON)
		{
			t = t1;
			return true; // Intersection at t1
		}
		return false; // Intersection is behind the ray origin
	}
};

struct IntersectionData
{
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

#define MAXNODE_TRIANGLES 8
#define TRAVERSE_COST 1.0f
#define TRIANGLE_COST 2.0f
#define BUILD_BINS 32

class BVHNode
{
public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;

	int start;
	int count;

	bool leaf()
	{
		return (r == NULL) && (l == NULL);
	}
	// This can store an offset and number of triangles in a global triangle list for example
	// But you can store this however you want!
	// unsigned int offset;
	// unsigned char num;
	BVHNode()
	{
		r = NULL;
		l = NULL;
	}
	// Note there are several options for how to implement the build method. Update this as required
	void build(std::vector<Triangle>& inputTriangles)
	{
		buildRecursive(inputTriangles, 0, inputTriangles.size());
	}
	// irratetion method for building the BVH
	BVHNode* buildRecursive(std::vector<Triangle>& tris, int start, int end) {
		BVHNode* node = new BVHNode();

		// cacluate bounds for this node
		for (int i = start; i < end; i++) {
			node->bounds.extend(tris[i].vertices[0].p);
			node->bounds.extend(tris[i].vertices[1].p);
			node->bounds.extend(tris[i].vertices[2].p);
		}

		int count = end - start;

		// leaf node
		if (count <= 2) {
			node->start = start;
			node->count = count;
			return node;
		}

		// caculate centroid bounds
		AABB centroidBounds;
		centroidBounds.min = centroidBounds.max = tris[start].centre();

		for (int i = start + 1; i < end; i++) {
			centroidBounds.extend(tris[i].centre());
		}

		// select axis with largest extent
		Vec3 extent = {
			centroidBounds.max.x - centroidBounds.min.x,
			centroidBounds.max.y - centroidBounds.min.y,
			centroidBounds.max.z - centroidBounds.min.z
		};

		int axis = 0;
		if (extent.y > extent.x) axis = 1;
		if (extent.z > extent.y && extent.z > extent.x) axis = 2;

		// sort triangles by centroid along selected axis
		std::sort(tris.begin() + start, tris.begin() + end,
			[axis](const Triangle& a, const Triangle& b) {
				return a.centre()[axis] < b.centre()[axis];
			});

		// segment triangles into two equal sets
		int mid = (start + end) / 2;

		node->l = buildRecursive(tris, start, mid);
		node->r = buildRecursive(tris, mid, end);

		return node;
	}
	void traverse(const Ray& ray, const std::vector<Triangle>& triangles, IntersectionData& intersection)
	{
		// Add code here
		// use stack-based traversal to avoid recursion
		std::vector<BVHNode*> stack;
		stack.push_back(this);

		while (!stack.empty())
		{
			BVHNode* node = stack.back();
			stack.pop_back();

			// check ray-box intersection
			float t = intersection.t;
			if (!node->bounds.rayAABB(ray, t))
			{
				continue;
			}

			// if the closest intersection found so far is closer than the intersection with the bounding box, skip
			if (t >= intersection.t)
			{
				continue;
			}

			// leaf node: check ray-triangle intersections
			if (node->leaf())
			{
				for (int i = node->start; i < node->start + node->count; i++)
				{
					float u, v, triT = intersection.t;
					if (triangles[i].rayIntersect(ray, triT, u, v))
					{
						// find the closest intersection
						if (triT < intersection.t)
						{
							intersection.t = triT;
							intersection.ID = i;
							intersection.alpha = 1.0f - u - v; // barycentric coord
							intersection.beta = u;
							intersection.gamma = v;
						}
					}
				}
			}
			else
			{
				// internal node: add child nodes to stack for traversal
				// traverse closer child first
				if (node->l != NULL)
					stack.push_back(node->l);
				if (node->r != NULL)
					stack.push_back(node->r);
			}
		}
	}
	IntersectionData traverse(const Ray& ray, const std::vector<Triangle>& triangles)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}
	bool traverseVisible(const Ray& ray, const std::vector<Triangle>& triangles, const float maxT)
	{
		// Add visibility code here
		// Stack-based BVH traversal for visibility check
		std::vector<BVHNode*> stack;
		stack.push_back(this);

		while (!stack.empty())
		{
			BVHNode* node = stack.back();
			stack.pop_back();

			// Check ray-box intersection
			float t = maxT;
			if (!node->bounds.rayAABB(ray, t))
			{
				continue;
			}

			// If intersection is beyond maxT, skip this branch
			if (t >= maxT)
			{
				continue;
			}

			// Leaf node: check ray-triangle intersections
			if (node->leaf())
			{
				for (int i = node->start; i < node->start + node->count; i++)
				{
					float u, v, triT = maxT;
					if (triangles[i].rayIntersect(ray, triT, u, v))
					{
						// If intersection is within maxT range, ray is blocked
						if (triT < maxT)
						{
							return true;
						}
					}
				}
			}
			else
			{
				// Internal node: add child nodes to stack for traversal
				if (node->l != NULL)
					stack.push_back(node->l);
				if (node->r != NULL)
					stack.push_back(node->r);
			}
		}

		// No intersection found within maxT, ray is visible
		return false;
	}
};
