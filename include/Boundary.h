#ifndef BOUNDARY_H_
#define BOUNDARY_H_

#include "Vec3d.h"
#include "Particle.h"
#include <GLFW/glfw3.h>  // version 3.3 is installed to Dependencies

class Boundary {

public:
	//virtual ~Boundary();

	virtual void draw2D()=0;
	//virtual void draw3D()=0;
	virtual double distance(const class Particle & particle) const = 0;

};

#endif /* BOUNDARY_H_ */
