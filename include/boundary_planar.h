#ifndef BOUNDARY_PLANAR_H_
#define BOUNDARY_PLANAR_H_

#include "cuda.h"
#include "vec3d.h"

class Particle;

class Boundary_planar {
	bool temporary = false; // TODO: rather derive temporary boundaries
	bool planar = true;

	Vec3d plane_point;
	Vec3d normal;
	Vec3d p1, p2; // for display purposes

public:

	Boundary_planar() = delete;
	Boundary_planar(Vec3d p1, Vec3d p2, Vec3d p3) :
			plane_point(p1),
			normal(norm(crossProduct(p3 - p1, p2 - p1))),
			p1(p1),
			p2(p2) {
		planar = true;
	}

	bool operator==(const Boundary_planar &other) const;

	float distance(const Vec3d &point) const;
	float distance(const Particle &particle) const;
	__device__
	float distanceDev(const Vec3d *point) const;
	__device__
	float distanceDev(const Particle *particle) const;
	float distanceSigned(const Vec3d &point) const;
	__device__
	float distanceSigned(const Vec3d *point) const;
	__host__
	float distanceSigned(const Particle &particle) const;
	__device__
	float distanceSigned(const Particle *particle) const;

	void draw2D() const;

	__host__ __device__
	Vec3d getNormal() const {
		return normal;
	}
	__host__
	Vec3d getNormal(const Particle &particle) const {
		return normal;
	} // argument is not used, only to conform "interface"
	__device__
	Vec3d getNormal(const Particle *particle) const {
		return normal;
	}

	void setTemporary() {
		temporary = true;
	}
	__host__ __device__
	bool isTemporary() const {
		return temporary;
	}
	__host__ __device__
	bool isPlanar() const {
		return planar;
	}
};

#endif /* BOUNDARY_PLANAR_H_ */
