#include "scene.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE scene-TEST
#include <boost/test/unit_test.hpp>

#define FLOAT_TEST_PRECISION 1e-6f

void print(const std::vector<Particle>& particles) {
	int counter = 0;
	for (const auto& p : particles) {
		std::cout << counter++ << ": ";
		p.getV() .print();
	}
}


BOOST_AUTO_TEST_CASE( scene_cell_external_test )
{
	Scene scene;

	Boundary_axissymmetric glass;

	Vec3d center(0,0,0);
	Vec3d external(1,0,0);

	BOOST_REQUIRE_EQUAL( scene.pointIsInternal(glass, center), true );
	BOOST_REQUIRE_EQUAL( scene.pointIsExternal(glass, center), false );

	BOOST_REQUIRE_EQUAL( scene.pointIsInternal(glass, external), false );
	BOOST_REQUIRE_EQUAL( scene.pointIsExternal(glass, external), true );


	Vec3d dX_small(0.01,0.01,0.01);
	Cell::setDX(dX_small);
	Cell c_internal(center);

	for (const auto&p : c_internal.getAllPoints()) {
		BOOST_REQUIRE_EQUAL( scene.pointIsInternal(glass, p), true );
		BOOST_REQUIRE_EQUAL( scene.pointIsExternal(glass, p), false );
	}

	Cell c_external(external);

	for (const auto&p : c_external.getAllPoints()) {
		BOOST_REQUIRE_EQUAL( scene.pointIsInternal(glass, p), false );
		BOOST_REQUIRE_EQUAL( scene.pointIsExternal(glass, p), true );
	}
}

BOOST_AUTO_TEST_CASE( scene_add_particles_test )
{
	Scene scene;

	Scene *scene_ptr = &scene; // TODO: check
	Particle::connectScene(scene_ptr);
	Cell::connectScene(scene_ptr);

	scene.applyDefaults(); // crashes
	scene.setGeometry(test);
	scene.createGeometry(test);
	scene.setNumberOfParticles(3);
	scene.addParticles(scene.getNumberOfParticles());
	scene.setVeloThreeParticlesTest();
	scene.populateCells();

	std::cout << "after creation:" << std::endl;
	print(scene.getParticles());

	//MainWindow::run()
	scene.setRunning();
	scene.resolveConstraintsOnInitCells(5);
	scene.populateCells();

	std::cout << "after start:" << std::endl;
	print(scene.getParticles());

	scene.calculatePhysics();
	std::cout << "after one loop:" << std::endl;
	print(scene.getParticles());

	scene.calculatePhysics();
	std::cout << "after one more loop:" << std::endl;
	print(scene.getParticles());

	for (auto p : scene.getParticles()) {
		std::cout << scene.getParticles()[0].getID() << "\t";
	}
	std::cout << std::endl;
}
