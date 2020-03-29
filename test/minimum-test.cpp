#include <iostream>
#include "minimum.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE minimum-TEST
#include <boost/test/unit_test.hpp>

float vertex_height = 0.2;
float X_offset = 0.1;

float parabola(float X) {
	return (X - X_offset)*(X - X_offset) + vertex_height;
}

float distance2(float X, float X0, float R0) {
	// distance squared between the parabola and point (X0,R0)
	return (X - X0) * (X - X0) + (parabola(X) - R0) * (parabola(X) - R0);
}

BOOST_AUTO_TEST_CASE( parabola_minimum_test )
{
	// find the vertex of the parabola

	std::function<float(float)> parabola_contour = parabola;
	std::function<float(float, float, float)> distance2_to_parabola = distance2;

	std::cout << parabola(0.0f) << std::endl;

	Minimum curve_minimum(parabola);

	float tolerance = Minimum::getTolerance();
	float guess = 0.005;
	float root = curve_minimum.findRoot(guess);
	float truth = X_offset;

	std::cout << "truth: " << truth << ", root: " << root << std::endl;
	BOOST_REQUIRE( std::abs(root - truth) < tolerance );

	// check repeated execution:
	root = curve_minimum.findRoot(guess);
	BOOST_REQUIRE( std::abs(root - truth) < tolerance );
}
