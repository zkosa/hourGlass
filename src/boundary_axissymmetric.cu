#include "boundary_axissymmetric.h"
#include "particle.h"
#include "minimumdistance.h"

__device__
float Boundary_axissymmetric::distanceDev(const Vec3d* point) const {
	MinimumDistance minimum_distance(this, point);

	return minimum_distance.getDistance();
}

__device__
float Boundary_axissymmetric::distanceDev(const Particle *particle) const {
	return distanceDev(particle->cGetPos());
}
