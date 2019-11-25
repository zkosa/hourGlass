#ifndef BOUNDARY_H_
#define BOUNDARY_H_

#include "Vec3d.h"
#include "Particle.h"
#include <GLFW/glfw3.h>  // version 3.3 is installed to Dependencies

class Boundary {

	bool temporary = false;
public:
	//virtual ~Boundary();

	virtual void draw2D() = 0;
	//virtual void draw3D() = 0;
	virtual double distance(const Particle &particle) const = 0;
	virtual Vec3d getNormal(const Particle &particle) const = 0;

	bool isTemporary() const {
		return temporary;
	}
	void setTemporary() {
		temporary = true;
	}

};

#endif /* BOUNDARY_H_ */
