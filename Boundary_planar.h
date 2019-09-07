#ifndef BOUNDARY_PLANAR_H_
#define BOUNDARY_PLANAR_H_

#include "Boundary.h"


class Boundary_planar : public Boundary {
	Vec3d point;
	Vec3d normal;
	Vec3d p1, p2; // temporary, for display

public:
	Boundary_planar(Vec3d p1, Vec3d p2, Vec3d p3) :
		point(p1),
		normal( norm(crossProduct(p3-p1, p2-p1)) ),
		p1(p1),
		p2(p2)
	{ };

	double distance(const Particle & particle) const;

	void draw2D();

	Vec3d getNormal() { return normal; }

};

#endif /* BOUNDARY_PLANAR_H_ */
