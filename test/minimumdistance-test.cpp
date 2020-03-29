#include <iostream>
#include "minimumdistance.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE minimumdistance-TEST
#include <boost/test/unit_test.hpp>

float vertex_height = 0.2;
float X_offset = 0.0;

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
	float mindist = minimum_distance.getDistance();
	float truth = 0.1;

	BOOST_REQUIRE( std::abs(mindist - truth) < tolerance );
}

float X0 = -0.999;
float a = 0.1;
float b = 0.5;

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
	float mindist = minimum_distance.getDistance();
	// calculate the exact point by intersecting:
	// - the original "line" y = a*(X - X0) + b
	// - a perpendicular line through the origin: y = -x/a
	float X_curve = (a*X0 - b) / (a + 1/a);
	float Y_curve = line(X_curve);
	VecAxiSym point_curve(X_curve, Y_curve);
	float truth = abs( point_curve - point.toYAxial() );

	BOOST_TEST_REQUIRE( mindist == truth, boost::test_tools::tolerance(tolerance) );
}
