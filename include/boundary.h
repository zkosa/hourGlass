#ifndef BOUNDARY_H_
#define BOUNDARY_H_

#include "vec3d.h"
#include "particle.h"

class Boundary {

	bool temporary = false;
public:

	virtual void draw2D() = 0;
	virtual float distance(const Particle &particle) const = 0;
	virtual float distanceSigned(const Particle &particle) const = 0;
	virtual Vec3d getNormal(const Particle &particle) const = 0;

	bool isTemporary() const {
		return temporary;
	}
	void setTemporary() {
		temporary = true;
	}

};

#endif /* BOUNDARY_H_ */
