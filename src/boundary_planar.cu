#include "boundary_planar.h"
#include "particle.h"


__device__
float Boundary_planar::distanceDev(const Vec3d *point) const {
	return Boundary_planar::distanceDev(point);
}

__device__
float Boundary_planar::distanceDev(const Particle *particle) const {
	return Boundary_planar::distanceDev(particle->cGetPos());
}

__device__
float Boundary_planar::distanceSigned(const Vec3d *point) const {
	return (*point - plane_point) * normal;
}

__device__
float Boundary_planar::distanceSigned(const Particle *particle) const {
	return Boundary_planar::distanceSigned(particle->cGetPos());
}

