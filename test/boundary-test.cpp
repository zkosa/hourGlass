#include "boundary.h"
#include "boundary_axissymmetric.h"
#include "boundary_planar.h"
#include "particle.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE boundary-TEST
#include <boost/test/unit_test.hpp>

// TODO: add tests in planes other than (x,y)

BOOST_AUTO_TEST_CASE( distance_origin_test )
{
	float corner = 1;
	Boundary_planar ground(
			Vec3d(-1, -corner, 0),
			Vec3d(1, -corner, 0),
			Vec3d(-1, -corner, 1)
			);

	float radius = 0.005;
	Vec3d point(0,0,0);
	Particle p(point, radius);

	BOOST_REQUIRE_EQUAL( ground.distance(point), 1.0f );
	BOOST_REQUIRE_EQUAL( ground.distance(p), 1.0f );


	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.014 [m]

	BOOST_REQUIRE_EQUAL( glass.distance(point), 0.07f );
	BOOST_REQUIRE_EQUAL( glass.distance(p), 0.07f );
}

BOOST_AUTO_TEST_CASE( distance_on_the_axis_hourglass_test )
{
	// chek a point on the axis, check the normals too

	Boundary_axissymmetric glass; // hardcoded shape, the orifice diameter is 0.14 [m]

	Vec3d point2(0, 0.1, 0);

	// axial coordinate of closest point on the contour
	// https://www.wolframalpha.com/input/?i=minimize+%28x-0.1%29%5E2+%2B+%28x%5E2%2B0.07-0%29%5E2+
	float root = 0.1 * pow(25 + 2*sqrt(1871.0), 1.0/3.0) - 19.0/(10.0*pow(25.0 + 2*sqrt(1871.0), 1.0/3.0));
	// closes point on the contour
	Vec3d point_root(glass.getContourFun()(root), root, 0); // axis is the y axis!

	// hmmm, unexpected implicit particle to point conversion is happening TODO: investigate it
	BOOST_REQUIRE_EQUAL( glass.distance(point2), abs(point2 - point_root) );
	// not necessary:
	//BOOST_TEST_REQUIRE( abs(glass.getNormal(point2) - (point2 - point_root)) < 1e-5f );
}

BOOST_AUTO_TEST_CASE( distance_in_the_orifice_hourglass_test )
{
	// check a point in the bottleneck, halfway between the axis and the vertex

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.14 [m]

	Vec3d point2(0.035, 0, 0);

	Vec3d point_root(0.07, 0, 0); // axis is the y axis!

	float res = glass.distance(point2);
	BOOST_REQUIRE_EQUAL( res, abs(point2 - point_root) );
}
