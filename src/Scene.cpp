#include "Scene.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include <random>
#include <omp.h>
#include "mainwindow.h"

void Scene::createGeometry(int geo) {

	createGeometry(static_cast<Geometry>(geo));

}

void Scene::createGeometry(Geometry geometry) {

	boundaries_pl.clear();
	boundaries_ax.clear();

	float corner = 0.999;

	if (geometry == hourglass) {
		Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, -corner, 0),
				Vec3d(-1, -corner, 1));
		Boundary_axis_symmetric glass;

		boundaries_pl.push_back(ground);
		boundaries_ax.push_back(glass);
	} else if (geometry == box) {
		Boundary_planar slope(Vec3d(-corner, -corner, 0), Vec3d(corner, 0, 0),
				Vec3d(-corner, -corner, 1));
		Boundary_planar side_wall_left(Vec3d(-corner, -corner, 0),
				Vec3d(-corner, corner, 0), Vec3d(-corner, -corner, -1));
		Boundary_planar side_wall_right(Vec3d(corner, 0, 0),
				Vec3d(corner, corner, 0), Vec3d(corner, 0, 1));

		boundaries_pl.push_back(slope);
		boundaries_pl.push_back(side_wall_left);
		boundaries_pl.push_back(side_wall_right);

	}
	createCells();
}

void Scene::resolve_constraints_on_init(int sweeps) {

	for (int sweep = 0; sweep < sweeps; ++sweep) {
		std::cout << sweep << " " << std::flush;
		for (auto &p1 : particles) {
			for (auto &p2 : particles) {
				if (p1.distance(p2) < p1.getR() + p2.getR()) {
					if (&p1 != &p2) { // do not collide with itself
						p1.collide_particle(p2);
					}
				}
				for (auto &b : boundaries_pl) {
					if (b.distance(p1) < p1.getR()) {
						p1.collide_wall(b);
					}
					if (b.distance(p2) < p2.getR()) {
						p2.collide_wall(b);
					}
				}
				for (auto &b : boundaries_ax) {
					if (b.distance(p1) < p1.getR()) {
						p1.collide_wall(b);
					}
					if (b.distance(p2) < p2.getR()) {
						p2.collide_wall(b);
					}
				}
			}
		}
	}
}

void Scene::resolve_constraints_on_init_cells(int sweeps) {
	populateCells();
	for (int sweep = 0; sweep < sweeps; ++sweep) {
		std::cout << sweep << " " << std::flush;

		for (auto &c : cells) {
			for (int p1ID : c.getParticleIDs()) {
				auto &p1 = particles[p1ID];
				for (int p2ID : c.getParticleIDs()) {
					auto &p2 = particles[p2ID];
					if (p1.distance(p2) < p1.getR() + p2.getR()) {
						if (p1ID != p2ID) { // do not collide with itself
							p1.collide_particle_check_boundary(p2);
						}
					}
				}
			}
		}
		populateCells();

		// redraw the scene after each sweeps:
		//this->draw(); // glfwSwapBuffers is not available here!
	}
	std::cout << std::endl;
}

void Scene::resolve_constraints_cells(int max_sweeps) {
	populateCells();
	int sweep = 0;
	int collision_counter;
	do {
		std::cout << sweep++ << " " << std::flush;
		collision_counter = 0;

		for (auto &c : cells) {
			for (int p1ID : c.getParticleIDs()) {
				auto &p1 = particles[p1ID];
				for (int p2ID : c.getParticleIDs()) {
					auto &p2 = particles[p2ID];
					if (p1.distance(p2) < p1.getR() + p2.getR()) {
						if (p1ID != p2ID) { // do not collide with itself
							p1.collide_particle_check_boundary(p2);
							collision_counter++;
						}
					}
				}
			}
		}
		std::cout << " (" << collision_counter << ") " << std::flush;
		if (sweep%10 ==0) {
			std::cout << "\n" << std::flush;
		}
		populateCells();

	} while (collision_counter > 0 && sweep < max_sweeps);
	std::cout << std::endl;
}

void Scene::draw() {
	for (auto &b : boundaries_pl) {
		b.draw2D();
	}
	for (auto &b : boundaries_ax) {
		b.draw2D();
	}
	for (auto &p : particles) {
		p.draw2D();
	}
}

void Scene::advance() {
	if (benchmark_mode && time >= benchmark_simulation_time) { // in benchmark mode the simulation time is fixed
	//setStopping();
		setFinished();
		std::cout << "The benchmark has been finished." << std::endl;
	} else {
		time += time_step;
		for (auto &p : particles) {
			p.advance(time_step);
		}
	}
	advanceCounter();
	//std::cout << "Time: " << time << " s" << std::endl << std::flush;
}

void Scene::collide_boundaries() {
//#pragma omp parallel for
	for (auto &p : particles) {
		for (auto &b : boundaries_pl) {
			if (b.distance(p) < p.getR()) {
				p.collide_wall(b);
			}
		}
		for (auto &b : boundaries_ax) {
			if (b.distance(p) < p.getR()) {
				p.collide_wall(b);
			}
		}
	}
}

void Scene::collide_boundaries_cells() {

	for (auto &c : cells) {
		if (c.hasBoundary()) {
			for (int pID : c.getParticleIDs()) {
				auto &p = particles[pID];
				for (auto &b : boundaries_pl) {
					if (b.distance(p) < p.getR()) {
						p.collide_wall(b);
					}
				}
				for (auto &b : boundaries_ax) {
					if (b.distance(p) < p.getR()) {
						p.collide_wall(b);
					}
				}
			}
		}
	}
}

void Scene::collide_particles() {
	for (auto &p1 : particles) {
		for (auto &p2 : particles) {
			if (p1.distance(p2) < p1.getR() + p2.getR()) {
				if (&p1 != &p2) { // do not collide with itself
					p1.collide_particle(p2);
				}
			}
		}
	}
}

void Scene::collide_cells() {

	for (auto &c : cells) { // when no omp (: loops are not supported)
		if (c.hasBoundary()) {
			for (int p1ID : c.getParticleIDs()) {
				auto &p1 = particles[p1ID];
				for (int p2ID : c.getParticleIDs()) {
					auto &p2 = particles[p2ID];
					if (p1.distance(p2) < p1.getR() + p2.getR()) {
						if (p1ID != p2ID) { // do not collide with itself
							p1.collide_particle_check_boundary(p2);
						}
					}
				}
			}
		} else {
			for (int p1ID : c.getParticleIDs()) {
				auto &p1 = particles[p1ID];
				for (int p2ID : c.getParticleIDs()) {
					auto &p2 = particles[p2ID];
					if (p1.distance(p2) < p1.getR() + p2.getR()) {
						if (p1ID != p2ID) { // do not collide with itself
							p1.collide_particle(p2);
						}
					}
				}
			}
		}
	}
}

void Scene::createCells() {

	deleteCells();

	int Nx = Cell::getNx();
	int Ny = Cell::getNy();
	int Nz = Cell::getNz();

	double dx = boundingBox.diagonal().x / Nx;
	double dy = boundingBox.diagonal().y / Ny;
	double dz = boundingBox.diagonal().z / Nz;

	// add extra cell layer on top for the particles which go beyond y=1
	// during e.g. the initial geometric constraint resolution
	int extra_layers_on_top = 1;

	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny + extra_layers_on_top; ++j) {
			for (int k = 0; k < Nz; ++k) {
				cells.emplace_back(
						Cell(
								(boundingBox.getCorner1() * Vec3d::i) * Vec3d::i
										+ dx * (i + 0.5) * Vec3d::i
										+ (boundingBox.getCorner1() * Vec3d::j)
												* Vec3d::j
										+ dy * (j + 0.5) * Vec3d::j
										+ (boundingBox.getCorner1() * Vec3d::k)
												* Vec3d::k
										+ dz * (k + 0.5) * Vec3d::k,
								dx * Vec3d::i + dy * Vec3d::j + dy * Vec3d::k));
			}
		}
	}

	markBoundaryCells();
	markExternalCells();
	removeExternalCells();
}

void Scene::markBoundaryCells() {
	std::cout << "Marking boundary cells..." << std::endl;

	for (auto &c : cells) {
		c.setCellWithoutBoundary(); // clear values before update
		for (auto &b : boundaries_pl) {
			if (c.contains(b)) {
				c.setCellWithBoundary();
			}
		}
		for (auto &b : boundaries_ax) {
			if (c.contains(b)) {
				c.setCellWithBoundary();
			}
		}
	}
}

bool Scene::pointIsExternal(const Boundary_axis_symmetric &b,
		const Vec3d &point) {
	// rough method!
	auto contour = b.getContourFun();
	double contour_radius = contour(point * norm(b.getAxis()));
	double point_radius = abs(
			point - (point * norm(b.getAxis())) * norm(b.getAxis()));
	// tuning factor for marking also those cells where all points are external,
	// but there is still interference with a boundary:
	point_radius *= 1.0;

	if (point_radius > contour_radius) {
		return true;
	} else {
		return false;
	}
}

bool Scene::pointIsExternal(const Boundary_planar &b, const Vec3d &point) {
	if (b.distance_signed(point) < 0) {
		return true;
	} else {
		return false;
	}
}

void Scene::markExternal(Cell &c) {

	// stores internal/external value for each cell in relation to each boundaries:
	std::vector<bool> boundariesCellInternalStatus;
	bool internalStatus;

	for (auto &b : boundaries_ax) {
		internalStatus = false;
		for (const auto &point : c.getAllPoints()) {
			if (pointIsInternal(b, point)) {
				internalStatus = true;
				boundariesCellInternalStatus.push_back(internalStatus);
				break; // if at least one point is external, the cell is internal
			}
		}
		// executes if none of the points was internal:
		boundariesCellInternalStatus.push_back(internalStatus);
	}

	for (auto &b : boundaries_pl) {
		internalStatus = false;
		for (const auto &point : c.getAllPoints()) {
			if (pointIsInternal(b, point)) {
				internalStatus = true;
				boundariesCellInternalStatus.push_back(internalStatus);
				break; // if at least one point is external, the cell is internal
			}
		}
		// executes if none of the points was internal:
		boundariesCellInternalStatus.push_back(internalStatus);
	}

	// if the cell is on the outer side of any boundary, it is external:
	c.setInternal();
	for (const auto &internal : boundariesCellInternalStatus) {
		if (!internal) {
			c.setExternal();
		}
	}
}

void Scene::markExternalCells() {
	std::cout << "Marking external cells..." << std::endl;

	for (auto &c : cells) {
		markExternal(c);
	}

	std::cout << "Number of cells after marking:" << cells.size() << std::endl;
}

void Scene::removeExternalCells() {
	std::cout << "Removing external cells..." << std::endl;

	cells.erase(std::remove_if(cells.begin(), cells.end(), [](Cell &c) {
		return c.isExternal();
	}), cells.end());

	std::cout << "Number of cells after removal:" << cells.size() << std::endl;
}

void Scene::drawCells() {
	for (auto &c : cells) {
		c.draw2D();
	}
}

void Scene::populateCells() {
	this->clearCells();
//#pragma omp parallel for
	for (auto &c : cells) {
		c.populate(particles);
	}
}

void Scene::clearCells() {
	for (auto &c : cells) {
		c.clear();
	}
}

void Scene::deleteCells() {
	std::cout << "Deleting all cells..." << std::endl;
	cells.clear();
}

void Scene::clearParticles() {
	particles.clear();
}

void Scene::addParticles(int N, double y, double r, bool randomize_y) {

	int number_of_distinct_random = 500;
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator
	std::uniform_int_distribution<> distr(-number_of_distinct_random,
			number_of_distinct_random); // define the range

	float corner = 0.999;
	double x;
	double random1, random2;
	double radius;
	for (int i = 0; i < N; i++) {
		x = -corner * 0.99 + i * (2 * corner * 0.99) / N;
		random1 = 0; //double(distr(eng))  / number_of_distinct_random;
		random2 = double(distr(eng)) / number_of_distinct_random;
		radius = r * (1 + random1 / 2.);

		particles.emplace_back(Vec3d(x, y * (1 + random2 / 200.), 0),
				Vec3d(0, 0, 0), i, radius); // no need to type the constructor!!!
	}
}

double Scene::energy() {
	double energy = 0;
	for (auto &p : particles) {
		energy += p.energy();
	}
	return energy;
}

Vec3d Scene::impulse() {
	Vec3d impulse { 0, 0, 0 };
	for (auto &p : particles) {
		impulse = impulse + p.impulse();
	}
	return impulse;
}

void Scene::setRunning() {
	running = true;
	started = true;
	std::cout << "Starting..." << std::endl;
}

void Scene::setStopping() {
	running = false;
	std::cout << "Stopping..." << std::endl;
}

void Scene::setFinished() {
	running = false;
	finished = true;
	std::cout << "Finishing..." << std::endl;
	viewer->sendFinishedSignal();

	std::cout << "Execution time of physics loop: " << duration << std::endl;
	std::cout << "Execution time of physics loop / loop: "
			<< duration / getCounter() << std::endl;
}

void Scene::reset() {
	started = false;
	running = false;
	finished = false;
	std::cout << "Resetting..." << std::endl;

	time = 0;

	boundaries_ax.clear();
	boundaries_pl.clear();
	particles.clear();
	cells.clear();

	applyDefaults();

	createGeometry(geometry);
	addParticles(viewer->getNumberOfParticles());
	createCells();
	populateCells();
}

void Scene::applyDefaults() {
	setTimestep(defaults.time_step);
	setGeometry(defaults.geometry);
	Cell::setNx(defaults.Nx);
	Cell::setNy(defaults.Ny);
	Cell::setNz(defaults.Nz);
	viewer->setNumberOfParticles(defaults.number_of_particles);
	Particle::setUniformRadius(0.5 * defaults.particle_diameter);
	Particle::setCd(defaults.Cd);
}
