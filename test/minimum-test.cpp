#include "minimum.h"

#define BOOST_TEST_TOOLS_UNDER_DEBUGGER
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE minimum-TEST
#include <boost/test/unit_test.hpp>

float height = 0.2;

float parabola(float X) {
	return X*X + height;
}

float distance2(float X, float X0, float R0) {
	// distance squared between the parabola and point (X0,R0)
	return (X - X0) * (X - X0)
			+ (parabola(X) - R0) * (parabola(X) - R0);
}

BOOST_AUTO_TEST_CASE( parabola_minimum_test )
{

	std::function<float(float)> parabola_contour = parabola;
	std::function<float(float, float, float)> distance2_to_parabola = distance2;

	std::cout << parabola(0.0f) << std::endl;

	Vec3d point(0,0,0);
	float X0 = point.toYAxial().X;
	float R0 = point.toYAxial().R;

	Minimum minimum(distance2_to_parabola, X0, R0); // problem: contour is uninitialized

	minimum.search(); // contour would be needed
	BOOST_REQUIRE_EQUAL( minimum.getDistance(), height );
}
