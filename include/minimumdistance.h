#ifndef INCLUDE_MINIMUMDISTANCE_H_
#define INCLUDE_MINIMUMDISTANCE_H_

#include "minimum.h"
#include "vec3d.h"
#include "boundary_axissymmetric.h"
#include "particle.h"
#include "cuda.h"
//#include <functional>
#include "functionhandler.h"

class MinimumDistance {

	// member order is important because of dependency!

	// point where the distance is measured to the curve from:
	const Vec3d point;
	const float point_X0 = point.toYAxial().axial;
	const float point_R0 = point.toYAxial().radial;

	const Vec3d axis = Vec3d::j; // default axis

	const Boundary_axissymmetric* function_owner; // needed for storing the address of the owner of the member function held by the function handler
	constFunctionHandler<Boundary_axissymmetric> functionHandler_contour;

	__host__ __device__
	float distance2(float X) const {
		return (X - point_X0) * (X - point_X0)
				+ ((function_owner->*functionHandler_contour)(X) - point_R0) * ((function_owner->*functionHandler_contour)(X) - point_R0);
	};

	Minimum minimum;

	Vec3d closest_point_on_the_contour;

	// private, because it would not take effect, when called externally, after construction
	__host__ __device__
	void setInitialGuess(float guess) {
		minimum.setInitialGuess(guess);
	}

public:
	__host__ __device__
	MinimumDistance(const Boundary_axissymmetric& bax, const Vec3d& point) : // make bax const
		point(point),
		axis(bax.getAxis()),
		function_owner(&bax),
		functionHandler_contour(bax.functionHandler_contour),
		minimum(this, functionHandler_distance2)
		//function( std::bind(&MinimumDistance::distance2, this, std::placeholders::_1) ),
		//minimum(function),
	{
		// use the axial coordinate of the point as starting value for the Newton iteration
		setInitialGuess(point * axis);
		findClosestPointOnContour();
	};

	__host__ __device__
	MinimumDistance(const Boundary_axissymmetric* bax, const Vec3d& point) :
		point(point),
		axis(bax->getAxis()),
		function_owner(bax),
		functionHandler_contour(bax->functionHandler_contour),
		minimum(this, functionHandler_distance2)
	{
		// use the axial coordinate of the point as starting value for the Newton iteration
		setInitialGuess(point * axis);
		findClosestPointOnContour();
	};

	__host__ __device__
	MinimumDistance(const Boundary_axissymmetric* bax, const Vec3d* point) :
		point(*point),
		axis(bax->getAxis()),
		function_owner(bax),
		functionHandler_contour(bax->functionHandler_contour),
		minimum(this, functionHandler_distance2)
	{
		// use the axial coordinate of the point as starting value for the Newton iteration
		setInitialGuess(*point * axis);
		findClosestPointOnContour();
	};

	__host__ __device__
	MinimumDistance(const Boundary_axissymmetric* bax, const Particle* particle) :
		point(particle->getPos()),
		axis(bax->getAxis()),
		function_owner(bax),
		functionHandler_contour(bax->functionHandler_contour),
		minimum(this, functionHandler_distance2)
	{
		// use the axial coordinate of the point as starting value for the Newton iteration
		setInitialGuess(point * axis);
		findClosestPointOnContour();
	};

	// function pointer to the function (square of the distance) to be minimized:
	constFunctionHandler<MinimumDistance> functionHandler_distance2 = &MinimumDistance::distance2;

	// call in all constructors!
	__host__ __device__
	void findClosestPointOnContour() {
		const float curve_X = minimum.findRoot(); // location of minimum distance point on the curve
		constFunctionHandler<Boundary_axissymmetric> handler = this->functionHandler_contour;
		const float curve_R = (function_owner->*handler)(curve_X); // radius of axisymmetric shape at curve_X

		const VecAxiSym closestPointInRadialCoord(curve_X, curve_R);

		Vec3d radial = point - (point * axis) * axis; // radial vector. it becomes zero, when the point is on the axis!

		// pick a "random" unit vector, when it would be a null vector:
		if ( radial.isSmall() ) {
			radial = Vec3d(1, 0, 0); //Vec3d::i; // TODO: resolve unavailable static Vec3d::i
		}

		// convert to Cartesian coordinate system:
		closest_point_on_the_contour = axis * closestPointInRadialCoord.axial
					+ norm(radial) * closestPointInRadialCoord.radial;
	}

	__host__ __device__
	Vec3d getClosestPointOnTheContour() const {
		return closest_point_on_the_contour;
	}

	__host__ __device__
	float getDistance2() {
		const float curve_X = closest_point_on_the_contour.toYAxial().axial;
		const float curve_R = closest_point_on_the_contour.toYAxial().radial;

		const float distance_squared = (curve_X - point_X0)*(curve_X - point_X0) +
								 (curve_R - point_R0)*(curve_R - point_R0);

		return distance_squared;
	}

	__host__ __device__
	inline float getDistance() {
		return std::sqrt( getDistance2() );
	}

	__host__ __device__
	Vec3d getDistanceVectorFromClosestPointOfContour() {
		return point - closest_point_on_the_contour;
	}

	__host__ __device__
	Vec3d getNormalizedDistanceVectorFromClosestPointOfContour() {
		return norm( getDistanceVectorFromClosestPointOfContour() );
	}

	__host__ __device__
	Vec3d getNormal() {
		// preserving legacy implementation
		// TODO: implement it based on curve geometry at the point
		return getNormalizedDistanceVectorFromClosestPointOfContour();
	}

	__host__ __device__
	Vec3d getNormal2() {
		// legacy implementation for testing
		// TODO: clarify the misleading name!
		const float curve_X = minimum.findRoot(); // location of minimum distance point on the curve
		const float curve_R = (function_owner->*functionHandler_contour)(curve_X); // radius of axisymmetric shape at curve_X

		const VecAxiSym closestPointInRadialCoord(curve_X, curve_R);

		Vec3d radial = point - (point * axis) * axis; // radial vector. it becomes zero, when the point is on the axis!

		// pick an arbitrary unit vector, when it would be a null vector:
		if ( radial.isSmall() ) {
			radial = Vec3d::i;
		}

		const Vec3d contactPoint = axis * closestPointInRadialCoord.axial
					+ norm(radial) * closestPointInRadialCoord.radial;

		return norm(point - contactPoint);
	}

};

#endif /* INCLUDE_MINIMUMDISTANCE_H_ */
