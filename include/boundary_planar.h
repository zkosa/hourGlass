#ifndef BOUNDARY_PLANAR_H_
#define BOUNDARY_PLANAR_H_

#include "boundary.h"
#include "vec3d.h"

class Boundary_planar: public Boundary {
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

	bool operator==(const Boundary &other) const override;

	float distance(const Vec3d &point) const;
	float distance(const Particle &particle) const override;
	__device__
	float distanceDev(const Particle *particle) const override;
	float distanceSigned(const Vec3d &point) const;
	__host__ __device__
	float distanceSigned(const Particle &particle) const override;

	void draw2D() override;

	Vec3d getNormal() const {
		return normal;
	}
	__host__
	Vec3d getNormal(const Particle &particle) const override {
		return normal;
	} // argument is not used, only to conform virtual function
	__device__
	Vec3d getNormal(const Particle *particle) const override {
		return normal;
	}
};

#endif /* BOUNDARY_PLANAR_H_ */
