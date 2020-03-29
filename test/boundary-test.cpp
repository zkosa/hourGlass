#include "boundary.h"
#include "boundary_axissymmetric.h"
#include "boundary_planar.h"
#include "particle.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE boundary-TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( distance_test )
{
	float corner = 1;
	Boundary_planar ground(
			Vec3d(-1, -corner, 0),
			Vec3d(1, -corner, 0),
			Vec3d(-1, -corner, 1)
			);

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.014 [m]

	float radius = 0.005;
	Vec3d point(0,0,0);
	Particle p(point, radius);

	BOOST_REQUIRE_EQUAL( ground.distance(point), 1.0f );
	BOOST_REQUIRE_EQUAL( ground.distance(p), 1.0f );

	BOOST_REQUIRE_EQUAL( glass.distance(point), 0.07f );
	BOOST_REQUIRE_EQUAL( glass.distance(p), 0.07f );

	Vec3d point1(0, 0.005, 0);
	point1.print();
	std::cout << "dist:" <<	glass.distance(point1) << std::endl;
	glass.getNormal(Particle(point1, radius)).print();

	Vec3d point2(0, 0.1, 0);
	point2.print();


	float a = 4;
	float b = 2.28;
	float c = -0.2;
	float root = 1/3.0f * (pow(-27*a*a*c + 3*sqrt(3)*sqrt(27 * a*a*a*a*c*c + 4 * a*a*b*b*b*c) - 2*b*b*b, 1/3.0f)/(pow(2, 1/3.0f) * a) + (pow(2, 1/3.f) *b*b)/(a * pow(-27 * a*a*c + 3 * sqrt(3) * sqrt(27 * a*a*a*a*c*c + 4 * a*a*b*b*b*c) - 2 *b*b*b, 1/3.0f) ) - b/a);

	std::cout << "root: " << root << std::endl;
	std::cout << "normal: " << glass.getNormal(Particle(point2, radius)) << std::endl;

	Vec3d point_root(glass.getContourFun()(root), root, 0); // axis is the y axis!

	point_root.print();

	BOOST_REQUIRE_EQUAL( glass.distance(point2), abs(point2 - point_root) );
}
