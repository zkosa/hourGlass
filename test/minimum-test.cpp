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
	auto tolerance = boost::test_tools::tolerance( float(3e-05f) );
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

	auto tolerance = boost::test_tools::tolerance( float(3e-05f) );
	// changing the tolerance does not effect the result, only the number of iterations. hmmm...
	//curve_minimum.setTolerance(1e-9f);
	float guess = guess_;
	float root = curve_minimum.findRoot(guess);
	float truth = 0.0f;

	BOOST_TEST_REQUIRE( root == truth, tolerance );

	// check repeated execution:
	root = curve_minimum.findRoot(guess);
	BOOST_TEST_REQUIRE( root == truth, tolerance );
}

// checking for the minimum in next period
BOOST_DATA_TEST_CASE( cos_minimum_offset_test, guess_data_cos, guess_ )
{
	// find the minimum of the -cosine function

	Minimum curve_minimum(negCosine);

	float offset = 2.0f * pi;
	auto tolerance = boost::test_tools::tolerance( float(3e-05f) );
	float guess = guess_ + offset;
	float root = curve_minimum.findRoot(guess);
	float truth = 0.0f + offset;

	BOOST_TEST_REQUIRE( root == truth, tolerance );
}

// checking for the minimum in next period, with specifying the guess via method
BOOST_DATA_TEST_CASE( cos_minimum_offset_setGuess_test, guess_data_cos, guess_ )
{
	// find the minimum of the -cosine function

	Minimum curve_minimum(negCosine);

	float offset = 2.0f * pi;
	auto tolerance = boost::test_tools::tolerance( float(3e-05f) );
	float guess = guess_ + offset;
	curve_minimum.setInitialGuess(guess);
	curve_minimum.setTolerance(1e-5f);
	float root = curve_minimum.findRoot(); // TODO: find the reason for Eclipse ambiguous error
	float truth = 0.0f + offset;

	BOOST_TEST_REQUIRE( root == truth, tolerance );
}


// check if the specified tolerance can is reached
BOOST_DATA_TEST_CASE( cos_minimum_setTolerance_test, guess_data_cos, guess_ )
{

	std::cout << "-------------------------" << std::endl;
	Minimum curve_minimum(negCosine);

	curve_minimum.setInitialGuess(guess_);
	float truth = 0.0f;
	float root;

	curve_minimum.setTolerance(1e-3f);
	root = curve_minimum.findRoot();
	watch( curve_minimum.getPerformedIterations() );
	BOOST_TEST_REQUIRE( root == truth, boost::test_tools::tolerance( float(curve_minimum.getTolerance()) ) );

	curve_minimum.setTolerance(1e-5f);
	root = curve_minimum.findRoot();
	watch( curve_minimum.getPerformedIterations() );
	BOOST_TEST_REQUIRE( root == truth, boost::test_tools::tolerance( float(curve_minimum.getTolerance()) ) );
/*
	// precision limit is reached, the new X_new is the same as the old one
	curve_minimum.setTolerance(1e-7f);
	root = curve_minimum.findRoot();
	watch( curve_minimum.getPerformedIterations() );
	BOOST_TEST_REQUIRE( root == truth, boost::test_tools::tolerance( float(curve_minimum.getTolerance()) ) );
*/
}
