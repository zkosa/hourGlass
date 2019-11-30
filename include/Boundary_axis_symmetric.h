#ifndef BOUNDARY_AXIS_SYMMETRIC_H_
#define BOUNDARY_AXIS_SYMMETRIC_H_

#include "Boundary.h"
#include <functional>
#include <unordered_map>

/*  // discards const qualifier!
 // inlined to avoid problems from multiple definitions
 inline double hour_glass_shape(double X) {
 double min_height = 0.07;
 return X*X + min_height;
 }
 */

class Boundary_axis_symmetric: public Boundary {
	Vec3d p1_axis { 0, -1, 0 };
	Vec3d p2_axis { 0, 1, 0 };
	Vec3d axis = norm(p2_axis - p1_axis);
	std::unordered_map<int, Vec3d> normals_to_particles;

	double hourGlassShape(double X) const {
		double min_height = 0.07;
		return X * X + min_height;
	}

	std::function<double(double)> contour = std::bind(
			&Boundary_axis_symmetric::hourGlassShape, this,
			std::placeholders::_1);
	/*
	 double distance2(double X, double X0, double R0) const {
	 return (X - X0)*(X - X0) + (contour(X) - R0)*(contour(X) - R0); // crashes
	 }
	 */
	double distance2(double X, double X0, double R0) const {
		return (X - X0) * (X - X0)
				+ (hourGlassShape(X) - R0) * (hourGlassShape(X) - R0); // fine
	}
	//std::function<double(double, double, double)> distance2_fun = distance2;
	std::function<double(double, double, double)> distance2_fun = std::bind(
			&Boundary_axis_symmetric::distance2, this, std::placeholders::_1,
			std::placeholders::_2, std::placeholders::_3);

	//std::function<double(double)> shape = hour_glass_shape(1.);
public:
	double distance(const Particle &particle) const override;

	void draw2D() override;

	Vec3d getNormal(const Particle &particle) const override;

	const Vec3d& getAxis() const {
		return axis;
	}

	std::function<double(const double)> getContourFun() const {
		return contour;
	}
	std::function<double(double, double, double)> getDistance2Fun() const {
		return distance2_fun;
	}

};

#endif /* BOUNDARY_AXIS_SYMMETRIC_H_ */
