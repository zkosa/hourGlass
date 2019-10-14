#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include "Particle.h"
#include "Boundary.h"
#include "BoundingBox.h"
#include "Cell.h"

class Scene {

	double time_step = 0.001; //0.001
	double time = 0;
	std::vector<Particle> particles;
	std::vector<Boundary_planar> boundaries_pl;
	std::vector<Boundary_axis_symmetric> boundaries_ax;
	std::vector<Cell> cells;


public:
	void init(int number_of_particles=500, double radius=0.01);
	void draw();
	void advance();
	void collide_boundaries();
	void collide_particles();
	void collide_cells();

	std::vector<Particle>& getParticles() { return particles; }
	std::vector<Boundary_planar>& getBoundariesPlanar() { return boundaries_pl; }
	std::vector<Boundary_axis_symmetric>& getBoundariesAxiSym() { return boundaries_ax; }

	BoundingBox boundingBox = BoundingBox(*this);

	void createCells(const int Nx=10, const int Ny=10, const int Nz=1);
	void drawCells();
	void populateCells();
	void clearCells();

};

#endif /* SCENE_H_ */
