#include <iostream>
#include "minimumdistance.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE minimumdistance-TEST
#include <boost/test/unit_test.hpp>

float vertex_height = 0.2;
float X_offset = 0.0;

// contour to be rotated around the axis
float parabola(float X) {
	return (X - X_offset)*(X - X_offset) + vertex_height;
}

BOOST_AUTO_TEST_CASE( parabola_minimum_distance_test )
{
	std::function<float(float)> contour = parabola;
	Vec3d point(0.1,0,0); // point at the bottleneck, halfway between axis and contour

	//MinimumDistance minimum_distance(parabola, point);
	MinimumDistance minimum_distance(contour, point);

	float tolerance = Minimum::getTolerance();
	float min_dist = minimum_distance.getDistance();
	float min_dist_from_vector = abs(minimum_distance.getDistanceVectorFromClosestPointOfContour());
	float truth = 0.1;

	BOOST_REQUIRE( std::abs(min_dist - truth) < tolerance );
	// check whether the two different calculation gives the same result:
	BOOST_TEST_REQUIRE( min_dist == min_dist_from_vector, boost::test_tools::tolerance(tolerance) );
}

float X0 = -0.999;
float a = 0.1;
float b = 0.5;

// contour to be rotated around the axis
float line(float X) {
	return a*(X - X0) + b;
}

BOOST_AUTO_TEST_CASE( line_minimum_distance_test )
{
	// test case with X!=0, still easy to calculate the exact distance
	std::function<float(float)> contour = line;
	Vec3d point(0,0,0); // origin

	MinimumDistance minimum_distance(contour, point);

	float tolerance = Minimum::getTolerance();
	float min_dist = minimum_distance.getDistance();
	float min_dist_from_vector = abs(minimum_distance.getDistanceVectorFromClosestPointOfContour());
	// calculate the exact point by intersecting:
	// - the original "line" y = a*(X - X0) + b
	// - a perpendicular line through the origin: y = -x/a
	float X_curve = (a*X0 - b) / (a + 1/a);
	float Y_curve = line(X_curve);
	VecAxiSym point_curve(X_curve, Y_curve);
	float truth = abs( point_curve - point.toYAxial() );

	BOOST_TEST_REQUIRE( min_dist == truth, boost::test_tools::tolerance(tolerance) );
	// check whether the two different calculation gives the same result:
	BOOST_TEST_REQUIRE( min_dist == min_dist_from_vector, boost::test_tools::tolerance(5e-3f) ); // TODO: check the reason for the large difference!
}

BOOST_AUTO_TEST_CASE( wrapped_hourglass_test ) {

	Boundary_axissymmetric glass; // (hourglass geometry is hardcoded)
	Vec3d point(0.035,0,0); // point at the bottleneck, halfway between axis and contour
	float r = 0.005;
	Particle particle(point, r);

	MinimumDistance minimum_distance(glass, particle);

	float tolerance = Minimum::getTolerance();
	float min_dist = minimum_distance.getDistance();
	float min_dist_from_vector = abs(minimum_distance.getDistanceVectorFromClosestPointOfContour());
	float truth = 0.035;

	BOOST_TEST_REQUIRE( min_dist == truth, boost::test_tools::tolerance(tolerance) );
	// check whether the two different calculation gives the same result:
	BOOST_TEST_REQUIRE( min_dist == min_dist_from_vector, boost::test_tools::tolerance(tolerance) );
}
