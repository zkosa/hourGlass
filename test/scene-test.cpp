#include "scene.h"
#include "devtools.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER // it deactivates the tolerance!!!
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE scene-TEST
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/array.hpp>

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
	scene.applyDefaults();
	//Particle::setCd(0.0f);
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
	scene.applyDefaults();

	float Cd = 0.5f;
	Particle::setCd(Cd);
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
	scene.applyDefaults();
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
	scene.applyDefaults();
	scene.createGeometry(Geometry::test);

	float r = Particle::getUniformRadius();
	Particle p1(Vec3d::null, Vec3d::null, r);
	Particle p2(p1);

	Vec3d half_dist{r/2, 0, 0};

	p1.move(-half_dist);
	p2.move(half_dist);

	BOOST_TEST_REQUIRE( (p1.distance(p2) == abs(2.0f * half_dist)) );

	scene.addParticle(p1);
	scene.addParticle(p2);

	scene.collideParticles();

	auto P1 = scene.getParticles()[0];
	auto P2 = scene.getParticles()[1];
	BOOST_TEST_REQUIRE( P1.distance(P2) == 2.0f *r, boost::test_tools::tolerance(1e-5f) );
}

BOOST_AUTO_TEST_CASE( scene_collideParticlesCells_test ) {

	Scene scene;
	scene.applyDefaults();
	scene.createGeometry(Geometry::test);

	float r = Particle::getUniformRadius();
	Particle p1(Vec3d::null, Vec3d::null, r);
	Particle p2(p1);

	Vec3d half_dist{r/2, 0, 0};

	p1.move(-half_dist);
	p2.move(half_dist);

	BOOST_TEST_REQUIRE( (p1.distance(p2) == abs(2.0f * half_dist)) );

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

BOOST_AUTO_TEST_CASE( scene_collideWithBoundariesCells_planar_test ) {

	Scene scene;
	scene.applyDefaults();
	scene.createGeometry(Geometry::test);

	// taking the appropriate (nearest) wall, knowing the order of walls from createGeometry
	auto side_wall_left = scene.getBoundariesPlanar()[1];

	float r = Particle::getUniformRadius();
	float y = Cell::getDX().y * 0.5; // at cell mid, when the number of cells is even // TODO: add variation
	float vx = -10.0f;
	Particle p1(Vec3d{-0.999f + 0.7f*r, y, 0.0f}, Vec3d{vx, 0.0f, 0.0f}, r);

	Particle::setRestitutionCoefficient(1.0f);

	scene.addParticle(p1);

	// cellwise collision requires cells
	scene.createCells();
	scene.populateCells();

	scene.collideWithBoundariesCells();

	auto P1 = scene.getParticles()[0];
	BOOST_TEST_REQUIRE( side_wall_left.distance(P1) == r, boost::test_tools::tolerance(1e-5f) );
	BOOST_TEST_REQUIRE( P1.getV().x == -vx, boost::test_tools::tolerance(1e-5f) );
}

static const boost::array< int, 8 > cell_Ns{1, 2, 5, 10, 11, 30, 32, 99};

BOOST_DATA_TEST_CASE( scene_collideWithBoundariesCells_axisymm_test, cell_Ns, N ) {

	Scene scene;
	scene.applyDefaults();
	scene.createGeometry(Geometry::hourglass);

	auto hourglass = scene.getBoundariesAxiSym()[0];

	float r = Particle::getUniformRadius();
	float y = Cell::getDX().y * 0.0; // TODO: add variation
	float vx = -10.0f;
	Particle p1(Vec3d{-0.07f + 0.7f*r, y, 0.0f}, Vec3d{vx, 0.0f, 0.0f}, r);

	Particle::setRestitutionCoefficient(1.0f);

	scene.addParticle(p1);

	// specify the number of the cells explicitly,
	// because the issue occured earlier when a particle
	// was part of odd number of cells:
	// (colliding odd number of times has undone the velocity reversion)
	Cell::setNx(N);
	Cell::setNy(Cell::getNx());

	// cellwise collision requires cells
	scene.createCells();
	scene.populateCells();

	auto& P1 = scene.getParticles()[0];

	scene.collideWithBoundariesCells();

	BOOST_TEST_REQUIRE( hourglass.distance(P1) == r, boost::test_tools::tolerance(1e-5f) );
	BOOST_TEST_REQUIRE( P1.getV().x == -vx, boost::test_tools::tolerance(1e-5f) );
}

BOOST_AUTO_TEST_CASE ( scene_removeDuplicates_int_test )
{
	// test with ints because it is easier
	std::vector<int> vecti {
								1,
								2,
								1,  // this is a duplicate
								3,
								3,  // this is a duplicate
								3,  // this is a duplicate
								4
	};

	std::vector<int> res {
								1,
								2,
								3,
								4
	};

	BOOST_TEST_REQUIRE( vecti.size() == 7);
	Scene::removeDuplicates(vecti);
	BOOST_TEST_REQUIRE( vecti.size() == 4);

	BOOST_TEST_REQUIRE( (vecti[0] == res[0]) );
	BOOST_TEST_REQUIRE( (vecti[1] == res[1]) );
	BOOST_TEST_REQUIRE( (vecti[2] == res[2]) );
	BOOST_TEST_REQUIRE( (vecti[3] == res[3]) );
}

BOOST_AUTO_TEST_CASE ( scene_removeDuplicates_test )
{
	Particle p1(Vec3d::null, 0.005f);
	Particle p2 = p1;
	Particle p3 = p1;
	p3.move(Vec3d::i);

	Boundary_axissymmetric b_ax;

	std::vector<particle_boundary_pair> pb_pairs;
	//std::vector<std::pair<Particle&, Boundary&>> pb_pairs;
	pb_pairs.emplace_back(p1, b_ax);
	pb_pairs.emplace_back(p2, b_ax);  // this is a duplicate
	pb_pairs.emplace_back(p3, b_ax);

	BOOST_TEST_REQUIRE( pb_pairs.size() == 3);
	Scene::removeDuplicates(pb_pairs);
	BOOST_TEST_REQUIRE( pb_pairs.size() == 2);
	BOOST_TEST_REQUIRE( (pb_pairs[0] == particle_boundary_pair{p1, b_ax}) );
	BOOST_TEST_REQUIRE( (pb_pairs[1] == particle_boundary_pair{p3, b_ax}) ); // check whether the last was simply skipped
}

BOOST_AUTO_TEST_CASE( scene_createCells_performance_test )
{
	Scene scene;

	// create a geometry, because createCells marks and delete cells based on the geometry
	scene.createGeometry(Geometry::hourglass);

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
