#include "Scene.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include <random>
#include <algorithm>
#include <execution>

void Scene::init(int number_of_particles, double radius) {

    float corner = 0.999;
    Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, -corner, 0), Vec3d(-1, -corner, 1));
    /*
    Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, 0, 0), Vec3d(-1, -corner, 1));
    Boundary_planar side_wall(Vec3d(1, -corner, 0), Vec3d(1, corner, 0), Vec3d(1, 0, 1));
    Boundary_planar side_wall2(Vec3d(-corner, -corner, 0), Vec3d(-corner, corner, 0), Vec3d(-corner, 0, 1));
     */

    Boundary_axis_symmetric glass;

    boundaries_pl.push_back(ground);
    //boundaries.push_back(side_wall);
    //boundaries.push_back(side_wall2);
    boundaries_ax.push_back(glass);

    //int number_of_particles = 500; //500

    int number_of_distinct_random = 500;
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(-number_of_distinct_random, number_of_distinct_random); // define the range


    double x;
    double y = 1; //0.95;
    double r = radius; //0.01
    double random1, random2;
    for (int i=0; i < number_of_particles; i++) {
		//particle[i].setWindow(window);
		//x = double(distr(eng))  / number_of_distinct_random;
		x = -corner*0.99 + i*(2*corner*0.99)/number_of_particles;
		random1 = 0; //double(distr(eng))  / number_of_distinct_random;
		random2 = double(distr(eng))  / number_of_distinct_random;
		r *= (1 + random1/2.);

		particles.emplace_back(  Vec3d(x, y*(1+random2/200.), 0), Vec3d(0, 0, 0), i , r );  // no need to type the constructor!!!
    }
}

void Scene::draw() {
	for (auto& b : boundaries_pl) {
		b.draw2D();
	}
	for (auto& b : boundaries_ax) {
		b.draw2D();
	}

	for (auto& p : particles) {
		p.draw2D();
	}
}

void Scene::advance() {
	time += time_step;
	for (auto& p : particles) {
		p.advance(time_step);
	}
	//std::cout << "Time: " << time << " s" << std::endl << std::flush;
}

void Scene::collide_boundaries() {
	for (auto& p : particles) {
		for (auto& b : boundaries_pl) {
			if ( b.distance(p) < p.getR() ) {
				p.collide_wall(b);
			}
		}
		for (auto& b : boundaries_ax) {
			if ( b.distance(p) < p.getR() ) {
				p.collide_wall(b);
			}
		}
	}
}

void Scene::collide_particles() {
	for (auto& p1 : particles) {
		for (auto& p2 : particles) {
			if ( p1.distance(p2) < p1.getR() + p2.getR() ) {
				if ( &p1 != &p2 ) { // do not collide with itself
					p1.collide_particle(p2);
				}
			}
		}
	}
}

void Scene::collide_cells() {
	//for (auto& c : cells) {
	std::for_each( std::execution::par,
			cells.begin(),
			cells.end(),
			[&](auto&& c) {

		for (int p1ID : c.getParticleIDs()) {
			auto& p1 = particles[p1ID];
			for (int p2ID : c.getParticleIDs()) {
				auto& p2 = particles[p2ID];
				if ( p1.distance(p2) < p1.getR() + p2.getR() ) {
					if ( p1ID != p2ID ) { // do not collide with itself
						p1.collide_particle(p2);
					}
				}
			}
		}
	}
	);

}

void Scene::createCells(const int Nx, const int Ny, const int Nz) {

	double dx = boundingBox.diagonal().x/Nx;
	double dy = boundingBox.diagonal().y/Ny;
	double dz = boundingBox.diagonal().z/Nz;

	for (int i=0; i < Nx; ++i) {
		for (int j=0; j < Ny; ++j) {
			for (int k=0; k < Nz; ++k) {
				cells.emplace_back(
					Cell(( boundingBox.getCorner1()*Vec3d::i)*Vec3d::i + dx*(i+0.5)*Vec3d::i +
							(boundingBox.getCorner1()*Vec3d::j)*Vec3d::j + dy*(j+0.5)*Vec3d::j +
							(boundingBox.getCorner1()*Vec3d::k)*Vec3d::k + dz*(k+0.5)*Vec3d::k,
							dx*Vec3d::i + dy*Vec3d::j + dy*Vec3d::k)
				);
			}
		}
	}

}

void Scene::drawCells() {
	for ( auto& c : cells) {
		c.draw2D();
	}
}

void Scene::populateCells() {
	this->clearCells();
	for (auto& c : cells) {
		c.populate(particles);
	}
}

void Scene::clearCells() {
	for (auto& c : cells) {
		c.clear();
	}
}
