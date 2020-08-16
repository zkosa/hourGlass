#ifndef SCENE_H_
#define SCENE_H_

#include <vector>
#include "particle.h"
#include "boundary.h"
#include "boundingbox.h"
#include "boundary_planar.h"
#include "boundary_axissymmetric.h"
#include "cell.h"
#include "timer.h"

class MainWindow;

enum class Geometry {
	hourglass = 0, hourglass_with_removable_orifice = 1, box = 2, test = 3
};

using particle_boundary_pair = std::pair<Particle&, Boundary&>;

class Scene {
	// TODO: allow only a single instant!

	MainWindow *viewer = nullptr;
	float time_step = 0.001; // [s]
	float simulation_time = 0;
	std::vector<Particle> particles;
	std::vector<Boundary_planar> boundaries_pl;
	std::vector<Boundary_axissymmetric> boundaries_ax;
	std::vector<Cell> cells;

	Geometry geometry = Geometry::hourglass;
	std::string geometry_names[4] = { "hourglass",
			"hourglass_with_removable_orifice", "box", "test" };
	int number_of_particles = 0; // it is not a "status", but a "request"

	bool started = false;
	bool running = false;
	bool finished = false;
	bool benchmark_mode = false;

	float benchmark_simulation_time = 1.; // [s]

	int loop_counter = 0;
	float duration = 0.;  // [ms]

	struct Defaults {
		float time_step = 0.001; // [s]
		int Nx = 10, Ny = Nx, Nz = 1;
		Geometry geometry = Geometry::hourglass;
		int number_of_particles = 5000;
		float particle_diameter = 0.005; // [m]
		float Cd = 0.5;
	};

	Defaults defaults;

public:
	Timer timer_all, timer;

	Scene(); // TODO: consider the rule of 0/3/5 after adding a constructor!

	void connectViewer(MainWindow *window) {
		viewer = window;
	}
	void init(int number_of_particles = 500, float radius = 0.01);
	void resolveConstraintsOnInit(int sweeps = 20);
	void resolveConstraintsOnInitCells(int sweeps = 20); // running the sweeps cell-wise
	void resolveConstraintsCells(int max_sweeps = 500); // do while there is collision
	void draw();
	void calculatePhysics();
	void advance();
	void collideWithBoundaries();
	void collideWithBoundariesCells();
	void collideParticles();
	void collideParticlesCells();

	float getSimulationTime() {
		return simulation_time;
	}
	std::vector<Particle>& getParticles() {
		return particles;
	}
	const std::vector<Boundary_planar>& getBoundariesPlanar() {
		return boundaries_pl;
	}
	const std::vector<Boundary_axissymmetric>& getBoundariesAxiSym() {
		return boundaries_ax;
	}
	const Geometry& getGeometry() {
		return geometry;
	}
	std::string getGeometryName() {
		return geometry_names[(int)geometry];
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
	void setVeloThreeParticlesTest();
	void removeTemporaryGeo();
	bool hasTemporaryGeo() const;
	void createCells();
	void markBoundaryCells();
	bool pointIsExternal(const Boundary_axissymmetric &b, const Vec3d &point);
	bool pointIsExternal(const Boundary_planar &b, const Vec3d &point);
	bool pointIsInternal(const Boundary_axissymmetric &b, const Vec3d &point) {
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
	void addParticle(Particle p);
	void addParticles(int N, float y = 1.0, float r =
			Particle::getUniformRadius(), bool randomize_y = true);

	float energy();
	Vec3d impulse();
	float impulseMagnitude() {
		return abs(impulse());
	}

	void veloCheck();

	void setRunning();
	void setStopping();
	void setFinished();
	void setGeometry(int geo) {
		this->geometry = static_cast<Geometry>(geo);
	}
	void setGeometry(Geometry geometry) {
		this->geometry = geometry;
	}
	void setNumberOfParticles(int number_of_particles) {
		this->number_of_particles = number_of_particles;
	}
	void setBenchmarkMode(bool benchmark_mode) {
		this->benchmark_mode = benchmark_mode;
	}
	void setTimestep(float time_step) {
		this->time_step = time_step;
	}

	float getDuration() {
		return duration;
	}
	void setDuration(float duration) {
		this->duration = duration;
	}
	void resetDuration() {
		setDuration(0);
	}
	void addToDuration(float duration) {
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
	const BoundingBox& getBoundingBox() {
		return bounding_box;
	}
	const Defaults& getDefaults() {
		return defaults;
	}
	int getNumberOfParticles() const {
		return number_of_particles;
	}
	void applyDefaults();

	template <class T>
	static void removeDuplicates(std::vector<T> &vector);

};

#endif /* SCENE_H_ */
