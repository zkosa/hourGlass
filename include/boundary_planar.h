#ifndef BOUNDARY_PLANAR_H_
#define BOUNDARY_PLANAR_H_

#include "boundary.h"
#include "vec3d.h"

class Boundary_axissymmetric;

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

	bool operator==(const Boundary &other) const final;
	bool operator==(const Boundary_axissymmetric &other) const;
	bool operator==(const Boundary_planar &other) const;

	float distance(const Vec3d &point) const;
	float distance(const Particle &particle) const final;
	float distanceSigned(const Vec3d &point) const;
	float distanceSigned(const Particle &particle) const final;

	void draw2D() final;

	Vec3d getNormal() const {
		return normal;
	}
	Vec3d getNormal(const Particle &particle) const final {
		return normal;
	} // argument is not used, only to conform virtual function

};

#endif /* BOUNDARY_PLANAR_H_ */
