#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include "Particle.h"
#include "Boundary.h"

class Scene {

	double time_step = 0.001;
	double time = 0;
	std::vector<Particle> particles;
	std::vector<Boundary_planar> boundaries;
	std::vector<Boundary_axis_symmetric> boundaries_ax;

public:
	void init();
	void draw();
	void advance();
	void collide_boundaries();
	void collide_particles();

	std::vector<Particle>& getParticles() { return particles; }
	std::vector<Boundary_planar>& getBoundaries() { return boundaries; }

};

#endif /* SCENE_H_ */
