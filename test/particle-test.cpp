#include "devtools.h"
#include "particle.h"
#include "boundary_planar.h"
#include "boundary_axissymmetric.h"
#include "scene.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER // it deactivates the tolerance!!!
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE particle-TEST
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/array.hpp>
#include <boost/math/special_functions/fpclassify.hpp>


BOOST_AUTO_TEST_CASE( construction_test )
{
	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);

	Particle p1(pos - r*Vec3d::i, vel, r);
	Particle p2(p1);
	p2.size();

	BOOST_TEST_REQUIRE( p1.getPos() == p2.getPos() );
	BOOST_TEST_REQUIRE( p1.getV() == p2.getV() );
	BOOST_TEST_REQUIRE( p1.getAcceleration() == p2.getAcceleration() );
	BOOST_TEST_REQUIRE( p1.getCd() == p2.getCd() );
	BOOST_TEST_REQUIRE( p1.getID() == p2.getID() );

	p1.setV(2*vel);
	BOOST_TEST_REQUIRE( p1.getV() == 2*vel );
}

BOOST_AUTO_TEST_CASE( ID_test )
{
	Particle::resetLastID();

	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);
/*
	Particle p0();
	// it has the default ID:
	BOOST_TEST_REQUIRE( p0.getID() == -1 );
*/
	Particle p1{pos, vel, 1, r};
	BOOST_TEST_REQUIRE( p1.getID() == 1 );
	p1.setID(2);
	BOOST_TEST_REQUIRE( p1.getID() == 2 );

	Scene scene;
	scene.addParticles(2, 0.0f, r);
	auto ps = scene.getParticles();
	for (std::size_t i = 0; i < ps.size(); i++) {
		BOOST_TEST_REQUIRE( ps[i].getID() == int(i) );
	}

	Particle p0{pos, vel, r};
	scene.addParticle(p0);
	watch(ps.size());
	int i = 0;
	for (const auto& p : scene.getParticles()) {
		watch(i++);
		watch(p.getID());
	}
	ps = scene.getParticles();
	BOOST_TEST_REQUIRE( ps[2].getID() == 2 );
}

BOOST_AUTO_TEST_CASE( move_test )
{
	Vec3d pos(0,0,0);
	Vec3d movement(-1,0,0);

	Particle p1(pos);
	Particle p2(p1);
	p2.move(movement);

	BOOST_TEST_REQUIRE( p2.getPos() - p1.getPos() == movement );
}

BOOST_AUTO_TEST_CASE( self_collision_test )
{
	// Nothing should change when the particle collides to itself
	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);

	Particle p1(pos, vel, r);

	p1.collideToParticle(p1);

	BOOST_TEST_REQUIRE( p1.getPos() == pos );
	BOOST_TEST_REQUIRE( p1.getV() == vel );
}

BOOST_AUTO_TEST_CASE( collision_touching_test )
{
	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);

	Particle p1(pos - r*Vec3d::i, vel, r);
	Particle p2(pos + r*Vec3d::i, -vel, r);

	p1.collideToParticle(p2);

	BOOST_TEST_REQUIRE( p1.getPos() == -p2.getPos() );
	BOOST_TEST_REQUIRE( p1.getV() == -p2.getV() );
}

BOOST_AUTO_TEST_CASE( collision_overlapping_test )
{
	float r = 0.005;
	float scale = 0.5;
	Vec3d pos1(-scale*r,0,0);
	Vec3d vel1(1,0,0);
	Vec3d pos2 = -pos1;
	Vec3d vel2 = -vel1;

	Particle::setCd(0);
	Particle p1(pos1, vel1, r);
	Particle p2(pos2, vel2, r);

	p1.collideToParticle(p2);

	BOOST_TEST_REQUIRE( p1.getPos() == -p2.getPos() );
	BOOST_TEST_REQUIRE( p1.getV() == -p2.getV() );
}

BOOST_AUTO_TEST_CASE( collision_distant_test )
{
	// Nothing should change when the particles are distant
	// it crashed before distance check in the collision
	float r = 0.005;
	float scale = 1.5;
	Vec3d pos1(-scale*r,0,0);
	Vec3d pos2(-pos1);
	Vec3d vel1(-1,0,0);
	Vec3d vel2(-vel1);

	Particle p1(pos1, vel1, r);
	Particle p2(pos2, vel2, r);

	p1.collideToParticle(p2);

	BOOST_TEST_REQUIRE( p1.getPos() == pos1 );
	BOOST_TEST_REQUIRE( p1.getV() == vel1 );
	BOOST_TEST_REQUIRE( p2.getPos() == pos2 );
	BOOST_TEST_REQUIRE( p2.getV() == vel2 );
}

BOOST_AUTO_TEST_CASE( collision_touching_parallel_test )
{
	// no friction --> no change in velo/pos
	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(0,1,0);

	Vec3d pos1 = pos - r*Vec3d::i;
	Vec3d pos2 = pos + r*Vec3d::i;

	Particle p1(pos1, vel, r);
	Particle p2(pos2, vel, r);

	p1.collideToParticle(p2);

	BOOST_TEST_REQUIRE( p1.getPos() == pos1 );
	BOOST_TEST_REQUIRE( p2.getPos() == pos2 );
	BOOST_TEST_REQUIRE( p1.getV() == vel );
	BOOST_TEST_REQUIRE( p2.getV() == vel );
}

BOOST_AUTO_TEST_CASE( overlap_with_wall_test )
{

	Scene scene;
	scene.createGeometry(Geometry::hourglass);

	Boundary_axissymmetric glass = scene.getBoundariesAxiSym()[0]; // hardcoded shape, orifice diameter 0.014 [m]
	Boundary_planar ground = scene.getBoundariesPlanar()[0];

	float radius = 0.005;
	float offset = 0.9*radius;
	Vec3d point(0, -0.999f + offset, 0); // point close to the ground plane
	Particle p(point, radius); // particle overlapping with the wall

	BOOST_TEST_REQUIRE( p.overlapWithWall(ground) == true );
	BOOST_TEST_REQUIRE( p.overlapWithWall(glass) == false );

	// tolerance does not work for user defined types
	//BOOST_TEST_REQUIRE( p.overlapVectorWithWall(ground) == Vec3d(0.0f, -1*(radius - offset), 0.0f), boost::test_tools::tolerance(Vec3d(1e-4f, 1e-4f, 1e-4f)) );
	BOOST_TEST_REQUIRE( p.overlapVectorWithWall(ground).x == Vec3d(0.0f, -1*(radius - offset), 0.0f).x, boost::test_tools::tolerance(1e-4f) );
	BOOST_TEST_REQUIRE( p.overlapVectorWithWall(ground).y == Vec3d(0.0f, -1*(radius - offset), 0.0f).y, boost::test_tools::tolerance(1e-4f) );
	BOOST_TEST_REQUIRE( p.overlapVectorWithWall(ground).z == Vec3d(0.0f, -1*(radius - offset), 0.0f).z, boost::test_tools::tolerance(1e-4f) );

	// BOOST_TEST_REQUIRE( p.overlapWithWalls() == true ); // fails, because scene is not known by p

	Scene *scene_ptr = &scene;
	Particle::connectScene(scene_ptr);

	BOOST_TEST_REQUIRE( p.overlapWithWalls() == true );
	BOOST_TEST_REQUIRE( p.overlapVectorWithWalls() == p.overlapVectorWithWall(ground) );

	 // make the particle huge, to overlap with all walls
	p.setR(1.5f);

	BOOST_TEST_REQUIRE( p.overlapWithWalls() == true );
	// it finds the ground, because overlap check is started with the planars
	BOOST_TEST_REQUIRE( p.overlapVectorWithWalls() == p.overlapVectorWithWall(ground) );

	// make the particle small, to overlap with none of the walls
	p.setR(1e-5f);

	BOOST_TEST_REQUIRE( p.overlapWithWalls() == false );
	BOOST_TEST_REQUIRE( p.overlapVectorWithWalls() == Vec3d::null );
}

BOOST_AUTO_TEST_CASE( collideToParticle_checkBoundary_noneTouches_test )
{
	// Two overlapping particles collide,
	// while NONE of them overlaps with a wall BEFORE collision.
	// We expect them to be touching each other after collision, and
	//
	Scene scene;
	Scene *scene_ptr = &scene;
	Particle::connectScene(scene_ptr); // TODO add to constructor, or at least a getPointer
	scene.createGeometry(Geometry::test); // box
	Boundary_planar ground = scene.getBoundariesPlanar()[0];

	float r = Particle::getUniformRadius();
	Particle p1( Vec3d(0.0f, -0.999f + r*1.6f, 0.0f), Vec3d(0.0f, 0.0f, 0.0f) );
	Particle p2(p1);
	p2.move(Vec3d(0.0f, 1.5f*r, 0.0f));


	watch(ground.distance(p1));
	watch(ground.distance(p2));
	p1.collideToParticle_checkBoundary(p2);

	auto tol = boost::test_tools::tolerance(float(2e-5f));

	BOOST_TEST_REQUIRE( p1.distance(p2) == 2.0f*r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p1) > r, tol ); // TODO: expect touch?
	BOOST_TEST_REQUIRE( ground.distance(p2) > r, tol );
	watch(ground.distance(p1));
	watch(ground.distance(p2));
}

BOOST_AUTO_TEST_CASE( collideToParticle_checkBoundary_noneTouches_butWouldAfterColl_test )
{
	// Two overlapping particles collide,
	// while NONE of them overlaps with a wall BEFORE collision.
	// We expect them to be touching each other after collision, and
	//
	Scene scene;
	Scene *scene_ptr = &scene;
	Particle::connectScene(scene_ptr); // TODO add to constructor, or at least a getPointer
	scene.createGeometry(Geometry::test); // box
	Boundary_planar ground = scene.getBoundariesPlanar()[0];

	float r = Particle::getUniformRadius();
	Particle p1( Vec3d(0.0f, -0.999f + r*1.1, 0.0f), Vec3d(0.0f, 0.0f, 0.0f) );
	Particle p2(p1);
	p2.move(Vec3d(0.0f, 1.5f*r, 0.0f));


	watch(ground.distance(p1));
	watch(ground.distance(p2));
	p1.collideToParticle_checkBoundary(p2);

	auto tol = boost::test_tools::tolerance(float(1e-6f));

	BOOST_TEST_REQUIRE( p1.distance(p2) == 2.0f*r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p1) > r, tol ); // TODO: expect touch?
	BOOST_TEST_REQUIRE( ground.distance(p2) > r, tol );
	watch(ground.distance(p1));
	watch(ground.distance(p2));
}

BOOST_AUTO_TEST_CASE( collideToParticle_checkBoundary_oneTouches_test )
{
	// Two overlapping particles collide,
	// while ONE of them overlaps with a wall too.
	// We expect them to be touching each other after collision
	Scene scene;
	Scene *scene_ptr = &scene;
	Particle::connectScene(scene_ptr); // TODO add to constructor, or at least a getPointer
	scene.createGeometry(Geometry::test); // box
	Boundary_planar ground = scene.getBoundariesPlanar()[0];

	float r = Particle::getUniformRadius();
	Particle p1( Vec3d(0.0f, -0.999f + r*0.9, 0.0f), Vec3d(0.0f, 0.0f, 0.0f) );
	Particle p2(p1);
	p2.move(Vec3d(0.0f, 1.5f*r, 0.0f));

	p1.collideToParticle_checkBoundary(p2);

	auto tol = boost::test_tools::tolerance(float(2e-5f));

	BOOST_TEST_REQUIRE( p1.distance(p2) == 2.0f*r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p1) == r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p2) == 3.0f*r, tol );
}

BOOST_AUTO_TEST_CASE( collideToParticle_checkBoundary_otherTouches_test )
{
	// Two overlapping particles collide,
	// while ONE of them overlaps with a wall too.
	// We expect them to be touching each other after collision
	Scene scene;
	Scene *scene_ptr = &scene;
	Particle::connectScene(scene_ptr); // TODO add to constructor, or at least a getPointer
	scene.createGeometry(Geometry::test); // box
	Boundary_planar ground = scene.getBoundariesPlanar()[0];

	float r = Particle::getUniformRadius();
	Particle p1( Vec3d(0.0f, -0.999f + r*0.9, 0.0f), Vec3d(0.0f, 0.0f, 0.0f) );
	Particle p2(p1);
	p2.move(Vec3d(0.0f, 1.5f*r, 0.0f));

	// now p2 becomes the "other" particle
	p2.collideToParticle_checkBoundary(p1);

	auto tol = boost::test_tools::tolerance(float(2e-5f));

	BOOST_TEST_REQUIRE( p1.distance(p2) == 2.0f*r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p1) == r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p2) == 3.0f*r, tol );
}

BOOST_AUTO_TEST_CASE( collideToParticle_checkBoundary_twoTouch_test )
{
	// Two overlapping particles collide,
	// while BOTH of them overlap with a wall too.
	// We expect them to be touching each other and the walls after collision
	Scene scene;
	Scene *scene_ptr = &scene;
	Particle::connectScene(scene_ptr); // TODO add to constructor, or at least a getPointer
	scene.createGeometry(Geometry::test); // box
	Boundary_planar ground = scene.getBoundariesPlanar()[0];

	float r = Particle::getUniformRadius();
	Particle p1( Vec3d(0.0f, -0.999f + r*0.9, 0.0f), Vec3d(0.0f, 0.0f, 0.0f) );
	Particle p2(p1);
	p2.move(Vec3d(1.5f*r, 0.0f, 0.0f));

	p1.collideToParticle_checkBoundary(p2);

	auto tol = boost::test_tools::tolerance(float(1e-6f));

	BOOST_TEST_REQUIRE( p1.distance(p2) == 2.0f*r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p1) == r, tol );
	BOOST_TEST_REQUIRE( ground.distance(p2) == r, tol );
}

BOOST_AUTO_TEST_CASE( no_drag_first_step_test )
{
	float time_step = 0.001; // [s]
	float r = 0.005;
	float height = 1;
	Vec3d pos(0,height,0);
	Vec3d vel(0,0,0);

	Particle p(pos, vel, r);
	Particle::setCd(0.0);

	p.advance(time_step);

	Vec3d calculated_travel = 0.5 * gravity * time_step * time_step;
	Vec3d calculated_speed = gravity * time_step;
	Vec3d calculated_acceleration = gravity;

	Vec3d simulated_travel = p.getPos() - pos;
	Vec3d simulated_speed = p.getV();
	Vec3d simulated_acceleration = p.getAcceleration();

	float tolerance = 1e-7; // [%]
	BOOST_REQUIRE_SMALL( abs(calculated_travel - simulated_travel), tolerance );
	BOOST_REQUIRE_SMALL( abs(calculated_speed - simulated_speed), tolerance );
	BOOST_REQUIRE_SMALL( abs(calculated_acceleration - simulated_acceleration), tolerance );
}

BOOST_AUTO_TEST_CASE( no_drag_fall_test )
{
	float time_step = 0.001; // [s]
	float r = 0.005;
	float height = 1.999;
	Vec3d pos(0,height,0);
	Vec3d vel(0,0,0);

	Particle p(pos, vel, r);
	Particle::setCd(0.0);

	float elapsed_time = 0;
	do {
		p.advance(time_step);
		elapsed_time += time_step;
	}
	while (p.getPos().y > 0); // particle reaching the ground, traveling 2 [m]

	float simulated_height = height - p.getPos().y;
	float calculated_time = std::sqrt(2*simulated_height/g);

	auto tol = boost::test_tools::tolerance(float(1e-5));
	BOOST_TEST_REQUIRE( calculated_time == elapsed_time, tol );

	// connect scene, otherwise maxFreeFallVelocity gives nan
	Scene scene;
	auto* scene_ptr = &scene;
	Particle::connectScene(scene_ptr);
	scene.createGeometry(Geometry::test); // box with level ground

	tol = boost::test_tools::tolerance(float(1e-3));
	BOOST_TEST_REQUIRE( std::abs(p.getV().y) == p.maxFreeFallVelocity(), tol );


	BOOST_TEST_REQUIRE( !boost::math::isfinite(p.terminalVelocity()) );

	// when there is no drag, the limit is the domain max free fall velocity
	BOOST_TEST_REQUIRE( p.maxVelocity() == p.maxFreeFallVelocity() );

	watch(p.terminalVelocity());
	watch(p.maxFreeFallVelocity());
	watch(p.maxVelocity());

}

static const boost::array< float, 6 > Cd_data{0.01, 0.1, 0.5, 5.0, 25.0, 100.0}; // it would fail with 0, because there the terminal velocity is infinite
//static const boost::array< float, 5 > r_data{0.001, 0.005, 0.025, 0.1, 0.25};

BOOST_DATA_TEST_CASE( terminal_velocity_test, Cd_data, Cd )
//BOOST_DATA_TEST_CASE( terminal_velocity_test, r_data, R )
{
	float time_step = 0.001; // [s]
	float r = 0.005;
	float height = 1;
	Vec3d pos(0,height,0);
	Vec3d vel(0,0,0);

	Particle p(pos, vel, r);
	Particle::setCd(Cd); // although it is the default, but needed, because in the previous test it was set to 0

	Vec3d velo_old;
	float calculated_terminal_velocity = p.terminalVelocity();
	do {
		velo_old = p.getV();
		p.advance(time_step);
		//p.getV().print();
	}
	while ( abs(p.getV() - velo_old) > SMALL ); // reaching equilibrium

	float simulated_terminal_velocity = abs(p.getV());

	//std::cout << calculated_terminal_velocity << std::endl;
	float tolerance = 0.05; // [%]
	BOOST_REQUIRE_CLOSE( calculated_terminal_velocity, simulated_terminal_velocity, tolerance );
}
