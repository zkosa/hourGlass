#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include "Particle.h"
#include "Boundary.h"
#include "BoundingBox.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
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
	void resolve_constraints_on_init(int sweeps=20);
	void resolve_constraints_on_init_cells(int sweeps=20); // running the sweeps cell-wise
	void draw();
	void advance();
	void collide_boundaries();
	void collide_particles();
	void collide_cells();

	std::vector<Particle>& getParticles() { return particles; }
	std::vector<Boundary_planar>& getBoundariesPlanar() { return boundaries_pl; }
	std::vector<Boundary_axis_symmetric>& getBoundariesAxiSym() { return boundaries_ax; }

	BoundingBox boundingBox = BoundingBox(*this);

	void createCells();
	void drawCells();
	void populateCells();
	void clearCells();

	double energy();
	Vec3d impulse();
	double impulse_magnitude() { return abs(impulse()); }

};

#endif /* SCENE_H_ */
