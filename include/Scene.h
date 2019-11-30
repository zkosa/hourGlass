#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include "Particle.h"
#include "Boundary.h"
#include "BoundingBox.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include "Cell.h"
#include "Timer.h"

class MainWindow;

enum Geometry {
	hourglass = 0, hourglass_with_removable_orifice = 1, box = 2
};

class Scene {

	MainWindow *viewer = nullptr;
	double time_step = 0.001; // [s]
	double time = 0;
	std::vector<Particle> particles;
	std::vector<Boundary_planar> boundaries_pl;
	std::vector<Boundary_axis_symmetric> boundaries_ax;
	std::vector<Cell> cells;

	Geometry geometry = hourglass;
	std::string geometry_names[3] = { "hourglass",
			"hourglass_with_removable_orifice", "box" };

	bool started = false;
	bool running = false;
	bool finished = false;
	bool benchmark_mode = false;

	double benchmark_simulation_time = 1.; // [s]

	int loop_counter = 0;
	double duration = 0.;

	struct Defaults {
		double time_step = 0.001;
		int Nx = 10, Ny = Nx, Nz = 1;
		Geometry geometry = hourglass;
		int number_of_particles = 5000;
		double particle_diameter = 0.005; // [m]
		double Cd = 0.5;
	};

	Defaults defaults;

public:
	Timer timer_all, timer;

	void connectViewer(MainWindow *window) {
		viewer = window;
	}
	void init(int number_of_particles = 500, double radius = 0.01);
	void resolveConstraintsOnInit(int sweeps = 20);
	void resolveConstraintsOnInitCells(int sweeps = 20); // running the sweeps cell-wise
	void resolveConstraintsCells(int max_sweeps = 500); // do while there is collision
	void draw();
	void advance();
	void collideWithBoundaries();
	void collideWithBoundariesCells();
	void collideParticles();
	void collideParticlesCells();

	std::vector<Particle>& getParticles() {
		return particles;
	}
	std::vector<Boundary_planar> getBoundariesPlanar() {
		return boundaries_pl;
	}
	std::vector<Boundary_axis_symmetric> getBoundariesAxiSym() {
		return boundaries_ax;
	}
	Geometry& getGeometry() {
		return geometry;
	}
	std::string getGeometryName() {
		return geometry_names[geometry];
	}
	bool benchmarkMode() {
		return benchmark_mode;
	}
	bool isStarted() {
		return started;
	}
	bool isRunning() {
		return running;
	}
	bool isFinished() {
		return finished;
	}

	BoundingBox bounding_box = BoundingBox(*this);

	void createGeometry(int);
	void createGeometry(Geometry);
	void removeTemporaryGeo();
	bool hasTemporaryGeo() const;
	void createCells();
	void markBoundaryCells();
	bool pointIsExternal(const Boundary_axis_symmetric &b, const Vec3d &point);
	bool pointIsExternal(const Boundary_planar &b, const Vec3d &point);
	bool pointIsInternal(const Boundary_axis_symmetric &b, const Vec3d &point) {
		return !pointIsExternal(b, point);
	}
	bool pointIsInternal(const Boundary_planar &b, const Vec3d &point) {
		return !pointIsExternal(b, point);
	}
	void markExternal(Cell &c);
	void markExternalCells();
	void removeExternalCells();
	void drawCells();
	void populateCells();
	void clearCells();
	void deleteCells();

	void clearParticles();
	void addParticles(int N, double y = 1.0, double r =
			Particle::getUniformRadius(), bool randomize_y = true);

	double energy();
	Vec3d impulse();
	double impulseMagnitude() {
		return abs(impulse());
	}

	void setRunning();
	void setStopping();
	void setFinished();
	void setGeometry(int geo) {
		this->geometry = static_cast<Geometry>(geo);
	}
	void setGeometry(Geometry geometry) {
		this->geometry = geometry;
	}
	void setBenchmarkMode(bool benchmark_mode) {
		this->benchmark_mode = benchmark_mode;
	}
	void setTimestep(double time_step) {
		this->time_step = time_step;
	}

	double getDuration() {
		return duration;
	}
	void setDuration(double duration) {
		this->duration = duration;
	}
	void addToDuration(double duration) {
		this->duration += duration;
	}

	void reset();

	void advanceCounter() {
		loop_counter += 1;
	}
	void resetCounter() {
		loop_counter = 0;
	}
	int getCounter() {
		return loop_counter;
	}

	void applyDefaults();

};

#endif /* SCENE_H_ */
