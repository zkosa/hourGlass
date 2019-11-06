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

    enum Geometry {hourglass=0, box=1};
    Geometry geometry = hourglass;
    std::string  geometry_names[2] = {"hourglass", "box"};

	bool running = false;

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
	Geometry& getGeometry() { return geometry; }
	std::string  getGeometryName() { return geometry_names[geometry]; }

	BoundingBox boundingBox = BoundingBox(*this);

	void createGeometry(int);
	void createGeometry(Geometry);
	void createCells();
	void drawCells();
	void populateCells();
	void clearCells();
	void deleteCells();

	void clearParticles();
	void addParticles(int N, double y=1.0, double r=Particle::getUniformRadius(), bool randomize_y=true);


	double energy();
	Vec3d impulse();
	double impulse_magnitude() { return abs(impulse()); }

	bool isRunning() { return running; }
	void setRunning();
	void setStopping();
	void setGeometry(int _geo) { geometry = static_cast<Geometry>(_geo); };
	void setGeometry(Geometry _geo) { geometry = _geo; };

};

#endif /* SCENE_H_ */
