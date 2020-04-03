#ifndef INCLUDE_MINIMUMDISTANCE_H_
#define INCLUDE_MINIMUMDISTANCE_H_

#include "minimum.h"
#include "vec3d.h"
#include "boundary_axissymmetric.h"
#include "particle.h"

class MinimumDistance {

	// member order is important because of dependency!

	// point where the distance is measured to the curve from:
	Vec3d point;
	float point_X0 = point.toYAxial().axial;
	float point_R0 = point.toYAxial().radial;

	Vec3d axis;

	std::function<float(float)> contour;

	float distance2(float X) const {
		return (X - point_X0) * (X - point_X0)
				+ (contour(X) - point_R0) * (contour(X) - point_R0);
	};
	// function pointer to the function (square of the distance) to be minimized
	std::function<float(float)> function;

	Minimum minimum;

public:
	MinimumDistance(std::function<float(float)> contour, const Vec3d& point) :
		point(point),
		contour(contour),
		function( std::bind(&MinimumDistance::distance2, this, std::placeholders::_1) ),
		minimum(function)
	{
// axis?
	};

	MinimumDistance(const Boundary_axissymmetric& boundary, const Particle& particle) :
		point(particle.getPos()),
		axis(boundary.getAxis()),
		contour(boundary.getContourFun()),
		function( std::bind(&MinimumDistance::distance2, this, std::placeholders::_1) ),
		minimum(function)
	{
		// execute minimum.findRoot() here, and share the results with getDistance and getNormal?
		// how to pass the guess then?
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

	Vec3d getDistanceVectorFromClosestPointOfContour() {
		float curve_X = minimum.findRoot(); // location of minimum distance point on the curve
		float curve_R = contour(curve_X); // radius of axisymmetric shape at curve_X

		VecAxiSym closestPointInRadialCoord(curve_X, curve_R);

		Vec3d radial = point - (point * axis) * axis; // radial vector. it becomes zero, when the point is on the axis!

		// pick a "random" unit vector, when it would be a null vector:
		if ( radial.isSmall() ) {
			radial = Vec3d::i;
		}

		Vec3d contactPoint = axis * closestPointInRadialCoord.axial
					+ norm(radial) * closestPointInRadialCoord.radial;

		Vec3d n = point - contactPoint;
		return n;
	}

	Vec3d getNormalizedDistanceVectorFromClosestPointOfContour() {
		return norm( getDistanceVectorFromClosestPointOfContour() );
	}

	Vec3d getNormal() {
		// preserving legacy implementation
		// TODO: implement it based on curve geometry at the point
		return getNormalizedDistanceVectorFromClosestPointOfContour();
	}

	void setInitualGuess(float guess) {
		minimum.setInitialGuess(guess);
	}
};

#endif /* INCLUDE_MINIMUMDISTANCE_H_ */
