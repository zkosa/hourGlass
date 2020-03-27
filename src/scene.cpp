#include "scene.h"
#include "boundary_planar.h"
#include "boundary_axissymmetric.h"
#include <random>
//#include <omp.h>
#include "mainwindow.h"

void Scene::createGeometry(int geo) {

	createGeometry(static_cast<Geometry>(geo));

}

void Scene::createGeometry(Geometry geometry) {

	boundaries_pl.clear();
	boundaries_ax.clear();

	float corner = 0.999;

	if (geometry == hourglass || geometry == hourglass_with_removable_orifice) {
		Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, -corner, 0),
				Vec3d(-1, -corner, 1));
		Boundary_axissymmetric glass;

		boundaries_pl.push_back(ground);
		boundaries_ax.push_back(glass);
		if (geometry == hourglass_with_removable_orifice) {
			Boundary_planar temporary_orifice(Vec3d(-0.07, 0, 0),
					Vec3d(0.07, 0, 0), Vec3d(-0.07, 0, 1));
			temporary_orifice.setTemporary();
			boundaries_pl.push_back(temporary_orifice);
		}

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

	} else if (geometry == test) {
		Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, -corner, 0),
						Vec3d(-1, -corner, 1));
		Boundary_planar side_wall_left(Vec3d(-corner, -corner, 0),
				Vec3d(-corner, corner, 0), Vec3d(-corner, -corner, -1));
		Boundary_planar side_wall_right(Vec3d(corner, -corner, 0),
				Vec3d(corner, corner, 0), Vec3d(corner, -corner, 1));

		boundaries_pl.push_back(ground);
		boundaries_pl.push_back(side_wall_left);
		boundaries_pl.push_back(side_wall_right);

		clearParticles();
		setNumberOfParticles(3);
		// addParticles(getNumberOfParticles()); // it will be called implicitly via the
		//initializeTestThreeParticles(); // do not call it here, because it will be overridden by the add particles in MainWindow::on_Particle_number_slider_valueChanged

	}
	createCells();
}

void Scene::setVeloThreeParticlesTest() {
	// check, whether the context is appropriate
	if( geometry == test && getNumberOfParticles() == 3 ) {
		int particle_diameter_mm = 50;
		float r = particle_diameter_mm / 1000. / 2.; // int [mm] --> float [m], diameter --> radius
		Particle::setUniformRadius(r);
		float vx = 10.0f;
		// left particle:
		particles[0].setV(Vec3d(vx, 0.0f, 0.0f));
		// right particle:
		particles[2].setV(Vec3d(-vx, 0.0f, 0.0f));
	} else {
		std::string message = "wrong call to initializeTestThreeParticles()";
		std::cout << message << std::endl;
	}
}

void Scene::removeTemporaryGeo() {
	std::cout << "Removing temporary geometries..." << std::endl;
	boundaries_pl.erase(
			std::remove_if(boundaries_pl.begin(), boundaries_pl.end(),
					[](auto &b) {
						return b.isTemporary();
					}), boundaries_pl.end());
	boundaries_ax.erase(
			std::remove_if(boundaries_ax.begin(), boundaries_ax.end(),
					[](auto &b) {
						return b.isTemporary();
					}), boundaries_ax.end());
	createCells();
}

bool Scene::hasTemporaryGeo() const {
	for (const auto &b : boundaries_pl) {
		if (b.isTemporary()) {
			return true;
		}
	}
	for (const auto &b : boundaries_ax) {
		if (b.isTemporary()) {
			return true;
		}
	}
	return false;
}

void Scene::resolveConstraintsOnInit(int sweeps) {

	for (int sweep = 0; sweep < sweeps; ++sweep) {
		std::cout << sweep << " " << std::flush;
		for (auto &p1 : particles) {
			for (auto &p2 : particles) {
				if (p1.distance(p2) < p1.getR() + p2.getR()) {
					if (&p1 != &p2) { // do not collide with itself
						p1.collideToParticle(p2);
					}
				}
				for (auto &b : boundaries_pl) {
					if (b.distance(p1) < p1.getR()) {
						p1.collideToWall(b);
					}
					if (b.distance(p2) < p2.getR()) {
						p2.collideToWall(b);
					}
				}
				for (auto &b : boundaries_ax) {
					if (b.distance(p1) < p1.getR()) {
						p1.collideToWall(b);
					}
					if (b.distance(p2) < p2.getR()) {
						p2.collideToWall(b);
					}
				}
			}
		}
	}
}

void Scene::resolveConstraintsOnInitCells(int sweeps) {
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
							p1.collideToParticle_checkBoundary(p2);
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

void Scene::resolveConstraintsCells(int max_sweeps) {
	populateCells();
	int sweep = 0;
	int collision_counter;
	do {
		std::cout << sweep++ << " " << std::flush;
		collision_counter = 0;

		for (auto &c : cells) {
			if (c.hasBoundary()) { // TODO: check if the compiler can optimize it when it is moved into the loop!
				for (int p1ID : c.getParticleIDs()) {
					auto &p1 = particles[p1ID];
					for (auto &b : boundaries_pl) {
						if (b.distance(p1) < p1.getR()) {
							p1.collideToWall(b);
						}
					}
					for (auto &b : boundaries_ax) {
						if (b.distance(p1) < p1.getR()) {
							p1.collideToWall(b);
						}
					}
					for (int p2ID : c.getParticleIDs()) {
						auto &p2 = particles[p2ID];
						if (p1.distance(p2) < p1.getR() + p2.getR()) {
							if (p1ID != p2ID) { // do not collide with itself
								p1.collideToParticle_checkBoundary(p2);
								collision_counter++;
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
								p1.collideToParticle(p2);
								collision_counter++;
							}
						}
					}
				}
			}
		}
		std::cout << " (" << collision_counter << ") " << std::flush;
		if (sweep % 10 == 0) {
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

void Scene::calculatePhysics() {
	timer.start();
	populateCells();
	advance();
	populateCells();
	//std::cout << "before collision..." << std::endl;
	//veloCheck();
	//resolveConstraintsCells(5);
	//collide_boundaries();
	collideWithBoundariesCells();
	populateCells();
	collideParticlesCells();
	//std::cout << "after collision..." << std::endl;
	//veloCheck();
	timer.stop();
	addToDuration(timer.milliSeconds());
	std::cout << timer.milliSeconds() << "ms" << std::endl << std::flush;
}

void Scene::advance() {
	if (benchmark_mode && time >= benchmark_simulation_time) { // in benchmark mode the simulation time is fixed
		viewer->wrapStopButtonClicked();
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

void Scene::collideWithBoundaries() {
//#pragma omp parallel for
	for (auto &p : particles) {
		for (auto &b : boundaries_pl) {
			if (b.distance(p) < p.getR()) {
				p.collideToWall(b);
			}
		}
		for (auto &b : boundaries_ax) {
			if (b.distance(p) < p.getR()) {
				p.collideToWall(b);
			}
		}
	}
}

void Scene::collideWithBoundariesCells() {

	for (auto &c : cells) {
		if (c.hasBoundary()) {
			for (int pID : c.getParticleIDs()) {
				auto &p = particles[pID];
				for (auto &b : boundaries_pl) {
					if (b.distance(p) < p.getR()) {
						p.collideToWall(b);
					}
				}
				for (auto &b : boundaries_ax) {
					if (b.distance(p) < p.getR()) {
						p.collideToWall(b);
					}
				}
			}
		}
	}
}

void Scene::collideParticles() {
	for (auto &p1 : particles) {
		for (auto &p2 : particles) {
			if (p1.distance(p2) < p1.getR() + p2.getR()) {
				if (&p1 != &p2) { // do not collide with itself
					p1.collideToParticle(p2);
				}
			}
		}
	}
}

void Scene::collideParticlesCells() {

	for (auto &c : cells) { // when no omp (: loops are not supported)
		if (c.hasBoundary()) {
			for (int p1ID : c.getParticleIDs()) {
				auto &p1 = particles[p1ID];
				for (int p2ID : c.getParticleIDs()) {
					auto &p2 = particles[p2ID];
					if (p1.distance(p2) < p1.getR() + p2.getR()) {
						if (p1ID != p2ID) { // do not collide with itself
							p1.collideToParticle_checkBoundary(p2);
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
							p1.collideToParticle(p2);
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

	float dx = bounding_box.diagonal().x / Nx;
	float dy = bounding_box.diagonal().y / Ny;
	float dz = 0.0f; //dy; // bounding_box.diagonal().z / Nz;// 2D: keep the third dimension small !!!

	Cell::setDX(Vec3d(dx, dy, dz));

	// add extra cell layer on top for the particles which go beyond y=1
	// during e.g. the initial geometric constraint resolution
	int extra_layers_on_top = 1;

	std::cout << "Creating cells..." << std::endl;
	Vec3d corner1 = bounding_box.getCorner1();
	Vec3d cell_center;
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny + extra_layers_on_top; ++j) {
			for (int k = 0; k < Nz; ++k) {
				cell_center.x = corner1.x + dx * (i + 0.5);
				cell_center.y = corner1.y + dy * (j + 0.5);
				cell_center.z = 0.0; //corner1.z + dz * (k + 0.5);  // 2D workaround

				cells.emplace_back(cell_center);
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

bool Scene::pointIsExternal(const Boundary_axissymmetric &b,
		const Vec3d &point) {
	// rough method!
	auto contour = b.getContourFun();
	float contour_radius = contour(point * norm(b.getAxis()));
	float point_radius = abs(
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
	if (b.distanceSigned(point) < 0) {
		return true;
	} else {
		return false;
	}
}

void Scene::markExternal(Cell &c) {

	// stores internal/external value for each cell in relation to each boundaries:
	std::vector<bool> internal_status_per_boundary;
	bool internal;

	for (auto &b : boundaries_ax) {
		internal = false;
		for (const auto &point : c.getAllPoints()) {
			if (pointIsInternal(b, point)) {
				internal = true;
				internal_status_per_boundary.push_back(internal);
				break; // if at least one point is external, the cell is internal
			}
		}
		// executes if none of the points was internal:
		internal_status_per_boundary.push_back(internal);
	}

	for (auto &b : boundaries_pl) {
		/*if (b.isTemporary()) {
		 continue; // do not consider temporary boundaries in cell removal
		 }*/
		internal = false;
		for (const auto &point : c.getAllPoints()) {
			if (pointIsInternal(b, point)) {
				internal = true;
				internal_status_per_boundary.push_back(internal);
				break; // if at least one point is external, the cell is internal
			}
		}
		// executes if none of the points was internal:
		internal_status_per_boundary.push_back(internal);
	}

	// if the cell is on the outer side of any boundary, it is external:
	c.setInternal();
	for (const auto &internal : internal_status_per_boundary) {
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

	std::cout << "Number of cells after marking: " << cells.size() << std::endl;
}

void Scene::removeExternalCells() {
	std::cout << "Removing external cells..." << std::endl;

	cells.erase(std::remove_if(cells.begin(), cells.end(), [](Cell &c) {
		return c.isExternal();
	}), cells.end());

	std::cout << "Number of cells after removal: " << cells.size() << std::endl;
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
	//cells[0].size();
	for (auto &c : cells) {
		c.clear();
	}
}

void Scene::deleteCells() {
	if (cells.size() > 0) {
		std::cout << "Deleting all cells..." << std::endl;
		cells.clear();
	} else {
		std::cout << "No cells to delete." << std::endl;
	}
}

void Scene::clearParticles() {
	//particles[0].size();
	particles.clear();
}

void Scene::addParticle(const Particle &p) {
	particles.push_back(p);
}

void Scene::addParticles(int N, float y, float r, bool randomize_y) {

	if (geometry == test) {
		std::cout << "adding particles in test mode..." << std::endl;
	}

	int number_of_distinct_random = 500;
	std::random_device rd; // obtain a random number from hardware
	std::mt19937 eng(rd()); // seed the generator
	std::uniform_int_distribution<> distr(-number_of_distinct_random,
			number_of_distinct_random); // define the range

	float corner = 0.999;
	float x;
	float random_y;
	for (int i = 0; i < N; i++) {
		x = -corner * 0.99 + i * (2 * corner * 0.99) / N;

		if (randomize_y) {
			random_y = float(distr(eng)) / number_of_distinct_random;
		} else {
			random_y = 0;
		}

		Vec3d pos(x, y * (1 + random_y / 200.), 0);
		Vec3d vel(0,0,0);
		Particle p(pos, vel, i, r);

		addParticle(p);
	}

}

float Scene::energy() {
	float energy = 0;
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

void Scene::veloCheck() {

	std::vector<float> vels(sizeof(particles));
	std::vector<float> max_vels(sizeof(particles));
	for (const auto &p : particles) {
		vels.emplace_back(abs(p.getV()));
		max_vels.emplace_back(p.maxVelocity());
	}
	float domainActualMaxVel = *std::max_element(vels.begin(), vels.end());
	float domainTheoreticalMaxVel = *std::max_element(max_vels.begin(), max_vels.end());

	std::cout << "timeStepLimit: " << particles[0].timeStepLimit() << "\t"
			<< "domMaxVel: " << domainActualMaxVel << "\t"
			<< "maxV: " << particles[0].maxVelocity() << "\t"
			<< "terminalV: " << particles[0].terminalVelocity() << "\t"
			<< "maxFreeFallV: " << particles[0].maxFreeFallVelocity() << "\t"
			<< "v: " << abs(particles[0].getV())
			<< std::endl << std::flush;

	if ( domainActualMaxVel > domainTheoreticalMaxVel ) {
		std::cout << "Largest particle velocity: " << domainActualMaxVel
				<< "is higher than theoretical: " << domainTheoreticalMaxVel << std::endl;
	}

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

	std::cout << "Execution time of physics loop: " << duration << std::endl;
	std::cout << "Execution time of physics loop / loop: "
			<< duration / getCounter() << std::endl;
	resetCounter();
	resetDuration();
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
	addParticles(getNumberOfParticles());
	createCells();
	populateCells();
}

void Scene::applyDefaults() {
	setTimestep(defaults.time_step);
	setGeometry(defaults.geometry);
	Cell::setNx(defaults.Nx);
	Cell::setNy(defaults.Ny);
	Cell::setNz(defaults.Nz);
	setNumberOfParticles(defaults.number_of_particles);
	Particle::setUniformRadius(0.5 * defaults.particle_diameter);
	Particle::setCd(defaults.Cd);
}