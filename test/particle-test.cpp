#include "particle.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE particle-TEST
#include <boost/test/unit_test.hpp>


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
	float r = 0.005;
	Vec3d pos(0,0,0);
	Vec3d vel(0,1,0);

	Particle p1(pos - r*Vec3d::i, vel, r);
	Particle p2(pos + r*Vec3d::i, vel, r);

	p1.collideToParticle(p2);
/*
	p1.getV().print();
	p2.getV().print();
	p1.getPos().print();
	p2.getPos().print();
*/
	BOOST_REQUIRE_EQUAL( p1.getPos(), -1*p2.getPos() );
	//BOOST_REQUIRE_EQUAL( p1.getV(), -1*p2.getV() ); // fails

}

BOOST_AUTO_TEST_CASE( advance_test )
{

	float dt = 0.001; // [s]
	float r = 0.005;
	Vec3d pos(0.5,1,0);
	Vec3d vel(0,0,0);

	Particle p(pos, vel, r);

	p.advance(dt);
	p.getPos().print();
	p.getV().print();

	p.advance(dt);
	p.getPos().print();
	p.getV().print();

}
