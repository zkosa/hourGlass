#include "devtools.h"
#include "boundary.h"
#include "boundary_axissymmetric.h"
#include "boundary_planar.h"
#include "particle.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER // it deactivates the tolerance!!!
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

	BOOST_TEST_REQUIRE( ground.distance(point) == 1.0f );
	BOOST_TEST_REQUIRE( ground.distance(p) == 1.0f );


	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.014 [m]

	BOOST_TEST_REQUIRE( glass.distance(point) == 0.07f );
	BOOST_TEST_REQUIRE( glass.distance(p) == 0.07f );
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

	BOOST_TEST_REQUIRE( ground.distance(point) == -ground.distanceSigned(point) );
	BOOST_TEST_REQUIRE( ground.distance(p) == -ground.distanceSigned(p) );
}

BOOST_AUTO_TEST_CASE( distance_external_glass_test )
{
	// check that the signed distance is the inverse of the signed
	// for external particles

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.014 [m]

	Vec3d point(0.071,0,0);
	Particle p(point);

	BOOST_TEST_REQUIRE( glass.distance(point) == -glass.distanceSigned(point) );
	BOOST_TEST_REQUIRE( glass.distance(p) == -glass.distanceSigned(p) );
}

BOOST_AUTO_TEST_CASE( distance_on_the_axis_hourglass_test )
{
	// check a point on the axis, check the normals too
	// challenge is the zero radial component

	Boundary_axissymmetric glass; // hardcoded shape, the orifice diameter is 0.14 [m]

	Vec3d point2(0, 0.1, 0);

	// axial coordinate of closest point on the contour
	// https://www.wolframalpha.com/input/?i=minimize+%28x-0.1%29%5E2+%2B+%28x%5E2%2B0.07-0%29%5E2+
	float root = 0.1f * pow(25.0f + 2.0f*sqrt(1871.0f), 1.0f/3.0f) - 19.0f/(10.0f*pow(25.0f + 2.0f*sqrt(1871.0f), 1.0f/3.0f));
	// closes point on the contour
	Vec3d point_root(glass.getContourFun()(root), root, 0); // axis is the y axis!

	BOOST_TEST_REQUIRE( glass.distance(point2) == abs(point2 - point_root) );
	BOOST_TEST_REQUIRE( glass.distance(point2) == glass.distanceSigned(point2), boost::test_tools::tolerance(1e-7f) );
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

// expected results:
static const boost::array< Vec3d, 2 > normal_target_data_negative_y{
	norm(Vec3d{-1,0,0}),
	norm(Vec3d{-1,-1,0})
};

BOOST_DATA_TEST_CASE( hourglass_numdiff_normal_test,
		rad_data, rad_//,
		//normal_target_data, normal_target_ // user types can not be used unfortunately
		)
{
	// Check the numerical normal calculation

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.14 [m]

	float rad = rad_;
	float ax = std::sqrt(rad - 0.07f); // inverse of the contour function (one of the two solutions)
	// check the inversion (y is the rotation axis!)
	BOOST_TEST_REQUIRE( glass.getContourFun()(ax) == rad );

	Vec3d point(rad, ax, 0);

	// check that the point and the particle are really on the curve before collision:
	// the tests may loose their sense when distance check is implemented
	BOOST_TEST_REQUIRE( glass.distance(point) == 0.0f,  boost::test_tools::tolerance(1e-6f) );

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

BOOST_DATA_TEST_CASE( hourglass_numdiff_normal_negative_y_test,
		rad_data, rad_//,
		//normal_target_data, normal_target_ // user types can not be used unfortunately
		)
{
	// Check the numerical normal calculation

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.14 [m]

	float rad = rad_;
	float ax = -std::sqrt(rad - 0.07f); // inverse of the contour function (one of the two solutions)
	// check the inversion (y is the rotation axis!)
	BOOST_TEST_REQUIRE( glass.getContourFun()(ax) == rad );

	Vec3d point(rad, ax, 0);

	// check that the point and the particle are really on the curve before collision:
	// the tests may loose their sense when distance check is implemented
	BOOST_TEST_REQUIRE( glass.distance(point) == 0.0f,  boost::test_tools::tolerance(1e-6f) );

	BOOST_REQUIRE( boost::math::isfinite( glass.getNormalNumDiff(point)) );

	// smaller tolerance for normal vectors:
	auto tol_normal = boost::test_tools::tolerance(1e-3f);
	// find the index of the current input, and use it to obtain the desired result:
	auto data_index = std::find(rad_data.begin(), rad_data.end(), rad_) - rad_data.begin();
	//BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point) == normal_target_data_negative_y[data_index], boost::test_tools::tolerance(1e-3f) ); // type is not recognized, fails
	//BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point) == normal_target_data_negative_y[data_index], boost::test_tools::tolerance(Vec3d(1e-3f,1e-3f,1e-3f)) ); // compile error
	BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).x == normal_target_data_negative_y[data_index].x, tol_normal );
	BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).y == normal_target_data_negative_y[data_index].y, tol_normal );
	BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).z == normal_target_data_negative_y[data_index].z, tol_normal );
}

BOOST_AUTO_TEST_CASE( hourglass_normal_in_all_quadrants_test )
{
	// Check the numerical normal calculation in all quadrants

	Boundary_axissymmetric glass;

	float ax = 0.5f;
	float rad = glass.getContourFun()(ax);

	std::vector<Vec3d> curve_points;
	curve_points.emplace_back(rad, ax, 0);
	curve_points.emplace_back(rad, -ax, 0);
	curve_points.emplace_back(-rad, ax, 0);
	curve_points.emplace_back(-rad, -ax, 0);

	std::vector<Vec3d> normal_targets;
	normal_targets.push_back( norm(Vec3d{-1,  1, 0}) );
	normal_targets.push_back( norm(Vec3d{-1, -1, 0}) );
	normal_targets.push_back( norm(Vec3d{ 1,  1, 0}) );
	normal_targets.push_back( norm(Vec3d{ 1, -1, 0}) );

	// smaller tolerance for normal vectors:
	auto tol_normal = boost::test_tools::tolerance(1e-3f);

	Vec3d point;
	Vec3d normal_target;
	for (std::size_t i = 0; i < curve_points.size(); i++) {

		point = curve_points[i];
		normal_target = normal_targets[i];

		// check that the point is really on the curve:
		BOOST_TEST_REQUIRE( glass.distance(point) == 0.0f,  boost::test_tools::tolerance(1e-6f) );

		BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).x == normal_target.x, tol_normal );
		BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).y == normal_target.y, tol_normal );
		BOOST_TEST_REQUIRE( glass.getNormalNumDiff(point).z == normal_target.z, tol_normal );
	}
}

static const boost::array< float, 6 > rad_data2{0.07f, 0.071f, 0.32f, 0.45f, 0.571f, 0.9f};

BOOST_DATA_TEST_CASE( collide_hourglass_zerovel_test, rad_data2, rad_ )
{
	// Check if a particle on the contour is properly moved back to the domain
	// Motivation: particles cross the walls ( e.g. at x=0.45) 0.571
	// This test is performed with steady particles

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.14 [m]

	auto tolerance = boost::test_tools::tolerance(1e-6f);
	float rad = rad_;
	float ax = std::sqrt(rad - 0.07f); // inverse of the contour function (one of the two solutions)
	// check the inversion
	BOOST_TEST_REQUIRE( glass.getContourFun()(ax) == rad, tolerance );

	Vec3d point(rad, ax, 0.0f);
	Vec3d vel(0.0f, 0.0f, 0.0f);

	Particle p(point, vel);
	glass.distance(point);
	// check that the point and the particle are really on the curve before collision:
	// the tests may loose their sense when distance check is implemented
	BOOST_TEST_REQUIRE( glass.distance(point) == 0.0f, tolerance );
	BOOST_TEST_REQUIRE( glass.distance(p) == 0.0f, tolerance );

	//watch(glass.getNormal(p));
	 // only here, where the point is on the curve!
	//watch(glass.getNormalNumDiff(point));

	// check if no NaNs or Infs occur
	BOOST_TEST_REQUIRE( boost::math::isfinite( glass.getNormal(p) ) );
	BOOST_TEST_REQUIRE( boost::math::isfinite( glass.getNormalNumDiff(point) ) );

	p.collideToWall(glass);

	// check that the point is at touching distance (radius) after collision:
	BOOST_TEST_REQUIRE( glass.distance(p) == p.getR(), boost::test_tools::tolerance(1e-5f) );

}

// when particle is moved backwards along the old velocity vector,
// it can still interfere with the (curved) wall --> test with planar too!
BOOST_DATA_TEST_CASE( collide_hourglass_downvel_test, rad_data2, rad_ )
{
	// Check if a particle on the contour is properly moved back to the domain
	// Motivation: particles cross the walls ( e.g. at x=0.45)

	Boundary_axissymmetric glass; // hardcoded shape, orifice diameter 0.14 [m]

	auto tolerance = boost::test_tools::tolerance(1e-6f);
	float rad = rad_;
	float ax = std::sqrt(rad - 0.07f); // inverse of the contour function (one of the two solutions)
	// check the inversion
	BOOST_TEST_REQUIRE( glass.getContourFun()(ax) == rad, tolerance );

	Vec3d point(rad, ax, 0.0f);
	Vec3d vel(0.0f, -3.0f, 0.0f);

	Particle p(point, vel);

	// check that the point and the particle are really on the curve before collision:
	// the tests may loose their sense when distance check is implemented
	BOOST_TEST_REQUIRE( glass.distance(point) == 0.0f, tolerance );
	BOOST_TEST_REQUIRE( glass.distance(p) == 0.0f, tolerance );

	//watch(glass.getNormal(p));
	 // only here, where the point is on the curve!
	//watch(glass.getNormalNumDiff(point));

	// check whether NaNs or Infs occur
	BOOST_TEST_REQUIRE( boost::math::isfinite( glass.getNormal(p) ) );
	BOOST_TEST_REQUIRE( boost::math::isfinite( glass.getNormalNumDiff(point) ) );

	p.collideToWall(glass);

	// check that the point is at touching distance (radius) after collision:
	BOOST_TEST_REQUIRE( glass.distance(p) == p.getR(), boost::test_tools::tolerance(1e-5f) );
}

static const boost::array< float, 3 > horizontal_offset_data{-1e-3f, 0.0f, 1e-3f};

BOOST_DATA_TEST_CASE( collide_planar_arbivel_test, horizontal_offset_data, offset_ )
{
	// Check if a particle on the wall is properly moved back to the domain
	// Arbitrary velocity vector

	float midpoint_offset = offset_;
	float corner = 0.999f;
	Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, -corner, 0),
					Vec3d(-1, -corner, 1));

	auto tolerance = boost::test_tools::tolerance(1e-6f);
	auto tolerance_large = boost::test_tools::tolerance(2e-5f);

	// it failed before adding negative signed
	Vec3d point(0.0f, -corner + midpoint_offset, 0.0f);
	Vec3d vel(2.0f, -3.0f, 0.0f);

	Particle p(point, vel);

	// check that the point and the particle are really at the expected distance from the surface before collision:
	BOOST_TEST_REQUIRE( ground.distance(point) == std::abs(midpoint_offset), tolerance_large );
	BOOST_TEST_REQUIRE( ground.distance(p) == std::abs(midpoint_offset), tolerance_large );
	BOOST_TEST_REQUIRE( ground.distanceSigned(point) == midpoint_offset, tolerance_large );
	BOOST_TEST_REQUIRE( ground.distanceSigned(p) == midpoint_offset, tolerance_large );

	p.collideToWall(ground);

	// check that the point is at touching distance (radius) after collision:
	BOOST_TEST_REQUIRE( ground.distance(p) == p.getR(), tolerance );
}

// Check that the default constructor has been deleted.
/*
BOOST_AUTO_TEST_CASE( planar_constructor_test )
{
	Boundary_planar ground; // it should produce a compile error
}
*/

BOOST_AUTO_TEST_CASE( planarity_test )
{
	float corner = 0.999f;
	Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, -corner, 0),
					Vec3d(-1, -corner, 1));

	Boundary_axissymmetric glass;

	BOOST_TEST_REQUIRE( ground.isPlanar() == true );
	BOOST_TEST_REQUIRE( glass.isPlanar() == false );
}
