#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include "Particle.h"
#include "Boundary.h"
#include "BoundingBox.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include "Cell.h"

class MainWindow;

class Scene {

	MainWindow* viewer = nullptr;
	double time_step = 0.001; // [s]
	double time = 0;
	std::vector<Particle> particles;
	std::vector<Boundary_planar> boundaries_pl;
	std::vector<Boundary_axis_symmetric> boundaries_ax;
	std::vector<Cell> cells;

    enum Geometry {hourglass=0, box=1};
    Geometry geometry = hourglass;
    std::string  geometry_names[2] = {"hourglass", "box"};

    bool started = false;
	bool running = false;
	bool finished = false;
	bool benchmark_mode = false;

	double benchmark_simulation_time = 1./10; // [s]

public:
	void connectViewer(MainWindow* _mainwindow) { viewer = _mainwindow; }
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
	bool benchmarkMode() { return benchmark_mode; }
	bool isStarted() { return started; }
	bool isRunning() { return running; }
	bool isFinished() { return finished; }

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

	void setRunning();
	void setStopping();
	void setFinished();
	void setGeometry(int _geo) { geometry = static_cast<Geometry>(_geo); };
	void setGeometry(Geometry _geo) { geometry = _geo; };
	void setBenchmarkMode(bool b) { benchmark_mode = b; }

	void reset();

};

#endif /* SCENE_H_ */
