#include "vec3d.h"
#include "particle.h"
#include "cell.h"
#include "minimumdistance.h"
#include "scene.h"
#include <iostream>

int main() {

	std::cout << "Object sizes:" << std::endl;
	std::cout << "-------------" << std::endl;

	Vec3d v;
	std::cout << "Vec3d:                  " << sizeof(v) << " B" << std::endl;

	Particle p(v, 0.005f);
	std::cout << "Particle:               " << sizeof(p) << " B" << std::endl;

	Cell c(v);
	std::cout << "Cell:                   " << sizeof(c) << " B" << std::endl;

	Boundary_axissymmetric hourglass;
	std::cout << "Boundary_axissymmetric: " << sizeof(hourglass) << " B" << std::endl;

	MinimumDistance md(hourglass, p);
	std::cout << "MinimumDistance:        " << sizeof(md) << " B" << std::endl;

	Scene s;
	std::cout << "Scene:                  " << sizeof(s) << " B" << std::endl;

	s.applyDefaults();
//	s.createGeometry(Geometry::hourglass);
//	s.createCells();
//	s.addParticles(s.getNumberOfParticles(), 0.0f, 0.005f, false);
//	std::cout << "Scene:                  " << sizeof(s) << " B" << std::endl;
//	std::cout << "Scene::Particles        " << sizeof(s.getParticles()) << " B" << std::endl;

	auto d = s.getDefaults();
	std::cout << "Scene::Defaults:        " << sizeof(d) << " B" << std::endl;


	std::cout << std::endl << "-------------" << std::endl;
	std::cout << "Accumulated object sizes in the benchmark run:" << std::endl;
	std::cout << "-------------" << std::endl;
	std::cout << "Particles: " << s.getNumberOfParticles() << " x " << sizeof(p) << "B = "
			  << s.getNumberOfParticles()*sizeof(p)/1000. << " kB" << std::endl;
	std::cout << "Cells: " << Cell::getNx() << " x (" << Cell::getNy() << " + 1) x " << Cell::getNz() << " x " << sizeof(c) << "B = "
			  << Cell::getNx() * (Cell::getNy() + 1) * Cell::getNz() * sizeof(c)/1000. << " kB" << std::endl;
}
