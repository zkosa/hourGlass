#include "scene.h"
#include "devtools.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER // it deactivates the tolerance!!!
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE scene-TEST
#include <boost/test/unit_test.hpp>

#define FLOAT_TEST_PRECISION 1e-6f

void print(const std::vector<Particle>& particles) {
	int counter = 0;
	for (const auto& p : particles) {
		std::cout << counter++ << ": ";
		p.getV().print();
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

BOOST_AUTO_TEST_CASE( scene_drag_test )
{
	// The middle particle drifted away when the static force field
	// was overwritten by the acceleration of the drag of the
	// left particle

	Scene scene;

	Scene *scene_ptr = &scene; // TODO: check
	Particle::connectScene(scene_ptr);
	Cell::connectScene(scene_ptr);

	scene.applyDefaults();
	//Particle::setCd(0.0f);

	scene.setGeometry(Geometry::test);
	scene.createGeometry(Geometry::test);
	scene.setNumberOfParticles(3);
	scene.addParticles(scene.getNumberOfParticles());

	auto& p_left = scene.getParticles()[0];
	auto& p_middle = scene.getParticles()[1];
	auto& p_right = scene.getParticles()[2];

	scene.setVeloThreeParticlesTest();
	p_left.setV(Vec3d(10, 0, 0));
	p_right.setV(Vec3d(-10, 0, 0));
	BOOST_REQUIRE_EQUAL( p_middle.getV().x , 0 );

	scene.populateCells();
	scene.advance();
	BOOST_REQUIRE_EQUAL( p_middle.getV().x , 0 );

	scene.calculatePhysics();
	BOOST_REQUIRE_EQUAL( p_middle.getV().x , 0 );
}

BOOST_AUTO_TEST_CASE( scene_three_particles_test )
{
	// The middle particle drifted away when the static force field
	// was overwritten by the acceleration of the drag of the
	// left particle

	Scene scene;

	Scene *scene_ptr = &scene; // TODO: check
	Particle::connectScene(scene_ptr);
	Cell::connectScene(scene_ptr);

	scene.applyDefaults();
	float Cd = 0.5f;
	Particle::setCd(Cd);
	scene.setGeometry(Geometry::test);
	scene.createGeometry(Geometry::test);
	scene.setNumberOfParticles(3);
	scene.addParticles(scene.getNumberOfParticles());
	scene.setVeloThreeParticlesTest();
	scene.populateCells();

	auto& p_left = scene.getParticles()[0];
	auto& p_middle = scene.getParticles()[1];
	auto& p_right = scene.getParticles()[2];

	print(scene.getParticles());

	//MainWindow::run()
	scene.setRunning();
	scene.resolveConstraintsOnInitCells(5);
	scene.populateCells();

	BOOST_REQUIRE_EQUAL( p_middle.getV().x , 0 );
	if (Cd < SMALL) {
		BOOST_REQUIRE_EQUAL( p_left.getV().x , 10 );
		BOOST_REQUIRE_EQUAL( p_right.getV().x , -10 );
	}

	scene.calculatePhysics();
	BOOST_REQUIRE_EQUAL( p_middle.getV().x , 0 );
	if (Cd < SMALL) {
		BOOST_REQUIRE_EQUAL( p_left.getV().x , 10 );
		BOOST_REQUIRE_EQUAL( p_right.getV().x , -10 );
	}

	for (auto p : scene.getParticles()) {
		std::cout << p.getID() << "\t";
	}
	std::cout << std::endl;
}

BOOST_AUTO_TEST_CASE( scene_collide_to_wall_test )
{
	Scene scene;

	Scene *scene_ptr = &scene; // TODO: check
	Particle::connectScene(scene_ptr);
	Cell::connectScene(scene_ptr);

	scene.applyDefaults();

	scene.setGeometry(Geometry::test); // TODO: combine setGeometry and createGeometry?
	scene.createGeometry(Geometry::test);

	float ground_level = -0.999;
	float r = 0.1;
	float overlap = 0.5; // in the ratio of the radius
	float height = ground_level + r*(1 - overlap);

	Vec3d pos(0, height, 0);
	Vec3d vel(0, 0, 0);

	Particle p(pos, vel, r);

	scene.addParticle(p);

	//Particle::setCd(0);
	Boundary_planar ground = scene.getBoundariesPlanar()[0];
	p.collideToWall(ground);

	// the particle touches the surface after collision:
	BOOST_REQUIRE_EQUAL( p.getY(), ground_level + r );

	p.collideToWall(ground);
	p.info();
	BOOST_REQUIRE_EQUAL( p.getY(), ground_level + r );

	p.setV(Vec3d(0,-1,0));
	p.collideToWall(ground);
	p.info();
	BOOST_REQUIRE_EQUAL( p.getY(), ground_level + r );

	p.setV(Vec3d(0,1,0));
	p.collideToWall(ground);
	p.info();
	BOOST_REQUIRE_EQUAL( p.getY(), ground_level + r );
}

BOOST_AUTO_TEST_CASE( scene_collideParticles_test ) {

	Scene scene;

	Scene *scene_ptr = &scene; // TODO: check
	Particle::connectScene(scene_ptr);
	Cell::connectScene(scene_ptr);

	scene.applyDefaults();

	scene.setGeometry(Geometry::test); // TODO: combine setGeometry and createGeometry?
	scene.createGeometry(Geometry::test);

	float r = Particle::getUniformRadius();
	Particle p1(Vec3d::null, Vec3d::null, r);
	Particle p2(p1);

	Vec3d half_dist{r/2, 0, 0};

	p1.move(-half_dist);
	p2.move(half_dist);

	BOOST_TEST_REQUIRE( p1.distance(p2) == abs(2.0f * half_dist) );

	scene.addParticle(p1);
	scene.addParticle(p2);

	scene.collideParticles();

	auto P1 = scene.getParticles()[0];
	auto P2 = scene.getParticles()[1];
	BOOST_TEST_REQUIRE( P1.distance(P2) == 2.0f *r, boost::test_tools::tolerance(1e-5f) );
}

BOOST_AUTO_TEST_CASE( scene_collideParticlesCells_test ) {

	Scene scene;

	Scene *scene_ptr = &scene;
	Particle::connectScene(scene_ptr);
	Cell::connectScene(scene_ptr);

	scene.applyDefaults();

	scene.setGeometry(Geometry::test);
	scene.createGeometry(Geometry::test);

	float r = Particle::getUniformRadius();
	Particle p1(Vec3d::null, Vec3d::null, r);
	Particle p2(p1);

	Vec3d half_dist{r/2, 0, 0};

	p1.move(-half_dist);
	p2.move(half_dist);

	BOOST_TEST_REQUIRE( p1.distance(p2) == abs(2.0f * half_dist) );

	scene.addParticle(p1);
	scene.addParticle(p2);

	// cellwise collision requires cells
	scene.createCells();
	scene.populateCells();

	scene.collideParticlesCells();

	auto P1 = scene.getParticles()[0];
	auto P2 = scene.getParticles()[1];
	BOOST_TEST_REQUIRE( P1.distance(P2) == 2.0f *r, boost::test_tools::tolerance(1e-5f) );
}

BOOST_AUTO_TEST_CASE( scene_createCells_performance_test )
{
	Scene scene;

	Timer timer;
	double duration;

	timer.start();
	Cell::setNx(100);
	Cell::setNy(100);
	Cell::setNz(1);
	scene.createCells();
	duration = timer.milliSeconds();
	watch(duration);
}

BOOST_AUTO_TEST_CASE( scene_addParticles_performance_test )
{
	Scene scene;

	Timer timer;
	double duration;

	timer.start();
	scene.addParticles(1e5);
	duration = timer.milliSeconds();
	watch(duration);
}
