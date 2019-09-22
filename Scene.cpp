#include "Scene.h"
#include <random>

void Scene::init() {

    float corner = 0.999;
    Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, 0, 0), Vec3d(-1, -corner, 1));
    Boundary_planar side_wall(Vec3d(1, -corner, 0), Vec3d(1, corner, 0), Vec3d(1, 0, 1));
    Boundary_planar side_wall2(Vec3d(-corner, -corner, 0), Vec3d(-corner, corner, 0), Vec3d(-corner, 0, 1));

    boundaries.push_back(ground);
    boundaries.push_back(side_wall);
    boundaries.push_back(side_wall2);

    int number_of_particles = 500;

    int number_of_distinct_random = 500;
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(-number_of_distinct_random, number_of_distinct_random); // define the range


    double x;
    double y = 1; //0.95;
    double r = 0.01;
    double random1, random2;
    for (int i=0; i < number_of_particles; i++) {
		//particle[i].setWindow(window);
		//x = double(distr(eng))  / number_of_distinct_random;
    	x = -corner + i*(2*corner)/number_of_particles;
		random1 = 0; //double(distr(eng))  / number_of_distinct_random;
		random2 = double(distr(eng))  / number_of_distinct_random;
		r *= (1 + random1/2.);

		particles.emplace_back(  Vec3d(x, y*(1+random2/200.), 0), Vec3d(0, 0, 0), r  );  // no need to type the constructor!!!
    }
}

void Scene::draw() {
    for (auto& b : boundaries) {
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
		for (auto& b : boundaries) {
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
