#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include "Particle.h"
#include "Boundary.h"
#include "BoundingBox.h"
#include "Cell.h"

class Scene {

	double time_step = 0.0004; //0.001
	double time = 0;
	std::vector<Particle> particles;
	std::vector<Boundary_planar> boundaries_pl;
	std::vector<Boundary_axis_symmetric> boundaries_ax;
	std::vector<Cell> cells;


public:
	void init();
	void draw();
	void advance();
	void collide_boundaries();
	void collide_particles();

	std::vector<Particle>& getParticles() { return particles; }
	std::vector<Boundary_planar>& getBoundariesPlanar() { return boundaries_pl; }
	std::vector<Boundary_axis_symmetric>& getBoundariesAxiSym() { return boundaries_ax; }

	BoundingBox boundingBox = BoundingBox(*this);

	void createCells();
	void drawCells();

};

#endif /* SCENE_H_ */
