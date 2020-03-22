#include "particle.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE particle-TEST
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/array.hpp>


BOOST_AUTO_TEST_CASE( construction_test )
{
	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);

	Particle p1(pos - r*Vec3d::i, vel, r);
	Particle p2(p1);

	BOOST_REQUIRE_EQUAL( p1.getPos(), p2.getPos() );
	BOOST_REQUIRE_EQUAL( p1.getV(), p2.getV() );
	BOOST_REQUIRE_EQUAL( p1.getCd(), p2.getCd() );

	BOOST_REQUIRE_EQUAL( p1.getID(), p2.getID() );

	p1.setV(2*vel);
	BOOST_REQUIRE_EQUAL( p1.getV(), 2*vel );
}

BOOST_AUTO_TEST_CASE( collision_touching_test )
{
	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);

	Particle p1(pos - r*Vec3d::i, vel, r);
	Particle p2(pos + r*Vec3d::i, -1*vel, r);

	p1.collideToParticle(p2);
/*
	p1.getV().print();
	p2.getV().print();
	p1.getPos().print();
	p2.getPos().print();
*/
	BOOST_REQUIRE_EQUAL( p1.getPos(), -1*p2.getPos() );
	BOOST_REQUIRE_EQUAL( p1.getV(), -1*p2.getV() );
}

BOOST_AUTO_TEST_CASE( collision_overlapping_test )
{
	float r = 0.005;
	float scale = 0.5;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);

	Particle p1(pos - scale*r*Vec3d::i, vel, r);
	Particle p2(pos + scale*r*Vec3d::i, -1*vel, r);

	p1.collideToParticle(p2);

	BOOST_REQUIRE_EQUAL( p1.getPos(), -1*p2.getPos() );
	BOOST_REQUIRE_EQUAL( p1.getV(), -1*p2.getV() );
}

BOOST_AUTO_TEST_CASE( collision_distant_test )
{
	float r = 0.005;
	float scale = 1.5;
	Vec3d pos(0,0,0);
	Vec3d vel(-1,0,0);

	Particle p1(pos - scale*r*Vec3d::i, vel, r);
	Particle p2(pos + scale*r*Vec3d::i, -1*vel, r);

	p1.collideToParticle(p2);

	BOOST_REQUIRE_EQUAL( p1.getPos(), -1*p2.getPos() );
	BOOST_REQUIRE_EQUAL( p1.getV(), -1*p2.getV() );
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

	BOOST_REQUIRE_EQUAL( p1.getPos(), pos1 );
	BOOST_REQUIRE_EQUAL( p2.getPos(), pos2 );
	BOOST_REQUIRE_EQUAL( p1.getV(), vel );
	BOOST_REQUIRE_EQUAL( p2.getV(), vel );
}

BOOST_AUTO_TEST_CASE( no_drag_fall_test )
{
	float time_step = 0.001; // [s]
	float r = 0.005;
	float height = 1;
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

	float tolerance = 0.001; // [%]
	BOOST_REQUIRE_CLOSE( calculated_time, elapsed_time, tolerance );
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
