#ifndef INCLUDE_MINIMUMDISTANCE_H_
#define INCLUDE_MINIMUMDISTANCE_H_

#include "minimum.h"
#include "vec3d.h"

class MinimumDistance {

	float point_X0;
	float point_R0;

	std::function<float(float)> contour;

	float distance2(float X) const {
		return (X - point_X0) * (X - point_X0)
				+ (contour(X) - point_R0) * (contour(X) - point_R0);
	};
	// function pointer to the function to be minimized
	std::function<float(float)> function;

	Minimum minimum;

public:
	MinimumDistance(std::function<float(float)> contour_, Vec3d point) :
		point_X0(point.toYAxial().axial),
		point_R0(point.toYAxial().radial),
		contour(contour_),
		function( std::bind(&MinimumDistance::distance2, this, std::placeholders::_1) ),
		minimum(function)
	{

	};

	float getDistance2() {
		float curve_X = minimum.findRoot(); // location of minimum distance on the curve
		float curve_R = contour(curve_X); // radius of axisymmetric shape at curve_X

		float distance_squared = (curve_X - point_X0)*(curve_X - point_X0) +
								 (curve_R - point_R0)*(curve_R - point_R0);

		return distance_squared;
	}

	inline float getDistance() {
		return std::sqrt( getDistance2() );
	}

};



#endif /* INCLUDE_MINIMUMDISTANCE_H_ */
