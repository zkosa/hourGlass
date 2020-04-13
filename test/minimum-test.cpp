#include "devtools.h"
#include "constants.h"
#include "minimum.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER // it can deactivate the tolerance!!!
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE minimum-TEST
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/array.hpp>

float vertex_height = 0.2;
float X_offset = 0.1;

float parabola(float X) {
	return (X - X_offset)*(X - X_offset) + vertex_height;
}

// same result is expected with various starting values
static const boost::array< float, 9 > guess_data{
	-1.0f, -0.1f, 0.0f, 0.005f, X_offset - 1e-5f, X_offset, X_offset + 1e-6f, 0.5f, 1.0f };

BOOST_DATA_TEST_CASE( parabola_minimum_test, guess_data, guess_ )
{
	// find the vertex of the parabola

	Minimum curve_minimum(parabola);

	//auto tolerance = boost::test_tools::tolerance( float(3.0f * Minimum::getTolerance()) );
	auto tolerance = boost::test_tools::tolerance( float(3 * 1e-05f) );
	// changing the tolerance does not effect the result, only the number of iterations. hmmm...
	//curve_minimum.setTolerance(1e-9f);
	float guess = guess_;
	float root = curve_minimum.findRoot(guess);
	float truth = X_offset;

	//watch( (root - truth) / truth );
	//watch( curve_minimum.getPerformedIterations() );

	BOOST_TEST_REQUIRE( root == truth, tolerance );

	// check repeated execution:
	root = curve_minimum.findRoot(guess);
	BOOST_TEST_REQUIRE( root == truth, tolerance );
}

float negCosine(float X) {
	return -cos(X);
}

// same result is expected with various starting values
static const boost::array< float, 7 > guess_data_cos{
	-0.5f, -0.1f, -1e-5f, 0.0f, 1e-6f, 0.5f, 0.9f*pi/4.0f }; // 0.9f*pi/2.0f would find other minimum

BOOST_DATA_TEST_CASE( cos_minimum_test, guess_data_cos, guess_ )
{
	// find the minimum of the -cosine function

	Minimum curve_minimum(negCosine);

	auto tolerance = boost::test_tools::tolerance( float(3 * 1e-05f) );
	// changing the tolerance does not effect the result, only the number of iterations. hmmm...
	//curve_minimum.setTolerance(1e-9f);
	float guess = guess_;
	float root = curve_minimum.findRoot(guess);
	float truth = 0.0f;

	//std::cout << guess_ << "  "; watch(root);
	//watch( curve_minimum.getPerformedIterations() );

	BOOST_TEST_REQUIRE( root == truth, tolerance );

	// check repeated execution:
	root = curve_minimum.findRoot(guess);
	BOOST_TEST_REQUIRE( root == truth, tolerance );
}
