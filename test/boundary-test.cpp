#include "devtools.h"
#include "boundary.h"
#include "boundary_axissymmetric.h"
#include "boundary_planar.h"
#include "particle.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE boundary-TEST
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/array.hpp>
#include <boost/math/special_functions/fpclassify.hpp>


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

BOOST_AUTO_TEST_CASE( distance_external_planar_test )
{
	// check that the signed distance is the inverse of the signed
	// for external particles
	float corner = 1;
	Boundary_planar ground(
			Vec3d(-1, -corner, 0),
			Vec3d(1, -corner, 0),
			Vec3d(-1, -corner, 1)
			);

	Vec3d point(0,-1.5,0);
	Particle p(point);

	BOOST_REQUIRE_EQUAL( ground.distance(point), -ground.distanceSigned(point) );
	BOOST_REQUIRE_EQUAL( ground.distance(p), -ground.distanceSigned(p) );
}

BOOST_AUTO_TEST_CASE( distance_external_glass_test )
{
	// check that the signed distance is the inverse of the signed
	// for external particles

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.014 [m]

	Vec3d point(0.071,0,0);
	Particle p(point);

	BOOST_REQUIRE_EQUAL( glass.distance(point), -glass.distanceSigned(point) );
	BOOST_REQUIRE_EQUAL( glass.distance(p), -glass.distanceSigned(p) );
}

BOOST_AUTO_TEST_CASE( distance_on_the_axis_hourglass_test )
{
	// check a point on the axis, check the normals too
	// challenge is the zero radial component

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

float hourglass_contour_inverse (float rad) {
	float ax = std::sqrt(rad - 0.07);
	return ax;
}

// analytic derivative of hourglass contour:
float derivative_exact(float ax0) {
	return 2*ax0;
}

float derivative_numeric(std::function<float(const float)> f, float ax0, float dax=1e-5) {
	return ( f(ax0 + dax) - f(ax0 - dax) ) / (2*dax);
}

// inputs to be tested:
static const boost::array< float, 2 > rad_data{
	0.07f,
	0.32f // 0.5*0.5 + 0.07
};
// expected results:
static const boost::array< Vec3d, 2 > normal_target_data{
	norm(Vec3d{-1,0,0}),
	norm(Vec3d{-1,1,0})
};

BOOST_DATA_TEST_CASE( hourglass_numdiff_normal_test,
		rad_data, rad_//,
		//normal_target_data, normal_target_ // user types can not be used unfortunately
		)
{
	watch(rad_);
	// Check the numerical normal calculation

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.14 [m]

	float tol = 1e-6f;
	float rad = rad_;
	float ax = std::sqrt(rad - 0.07); // inverse of the contour function (one of the two solutions)
	// check the inversion (y is the rotation axis!)
	BOOST_REQUIRE( abs(glass.getContourFun()(ax) - rad) < tol );

	Vec3d point(rad, ax, 0);

	// check that the point and the particle are really on the curve before collision:
	// the tests may loose their sense when distance check is implemented
	BOOST_REQUIRE( glass.distance(point) < tol );

	glass.getNormalNumDiff(point).print();

	BOOST_REQUIRE( boost::math::isfinite( glass.getNormalNumDiff(point)) );

	// smaller tolerance for normal vectors:
	auto tol_normal = boost::test_tools::tolerance(1e-3f);
	// find the index of the current input, and use it to obtain the desired result:
	auto data_index = std::find(rad_data.begin(), rad_data.end(), rad_) - rad_data.begin();
	//BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point) == normal_target_data[data_index], boost::test_tools::tolerance(1e-3f) ); // type is not recognized, fails
	//BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point) == normal_target_data[data_index], boost::test_tools::tolerance(Vec3d(1e-3f,1e-3f,1e-3f)) ); // compile error
	BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).x == normal_target_data[data_index].x, tol_normal );
	BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).y == normal_target_data[data_index].y, tol_normal );
	BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).z == normal_target_data[data_index].z, tol_normal );
}

static const boost::array< float, 3 > rad_data2{0.07f, 0.071f, 0.45f};

BOOST_DATA_TEST_CASE( collide_hourglass_test, rad_data2, rad_ )
{
	watch(rad_);
	// Check if a particle on the contour is properly moved back to the domain
	// Motivation: particles cross the walls ( e.g. at x=0.45)

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.14 [m]

	float tolerance = 1e-6f;
	float rad = rad_;
	float ax = std::sqrt(rad - 0.07); // inverse of the contour function (one of the two solutions)
	// check the inversion (y is the rotation axis!)
	BOOST_REQUIRE( abs(glass.getContourFun()(ax) - rad) < tolerance );

	Vec3d point(rad, ax, 0);
	Vec3d vel(0, -3, 0);

	Particle p(point, vel);

	// check that the point and the particle are really on the curve before collision:
	// the tests may loose their sense when distance check is implemented
	BOOST_REQUIRE( glass.distance(point) < tolerance );
	BOOST_REQUIRE( glass.distance(p) < tolerance );

	glass.getNormal(p).print();
	 // only here, where the point is on the curve!
	glass.getNormalNumDiff(point).print();

	// check if no NaNs or Infs occur
	BOOST_REQUIRE( boost::math::isfinite( glass.getNormal(p) ) );
	BOOST_REQUIRE( boost::math::isfinite( glass.getNormalNumDiff(point) ) );


	std::cout << glass.distance(p) << std::endl;
	p.collideToWall(glass);
	std::cout << glass.distance(p) << std::endl;
	glass.getNormal(p).print();
	// check that the point is at touching distance (radius) after collision:
	BOOST_TEST_REQUIRE( glass.distance(p) == p.getR(), boost::test_tools::tolerance(1e-3f) );

}
