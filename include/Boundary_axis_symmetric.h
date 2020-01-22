#ifndef BOUNDARY_AXIS_SYMMETRIC_H_
#define BOUNDARY_AXIS_SYMMETRIC_H_

#include "Boundary.h"
#include <functional>
#include <unordered_map>

class Boundary_axis_symmetric: public Boundary {
	Vec3d p1_axis { 0, -1, 0 };
	Vec3d p2_axis { 0, 1, 0 };
	Vec3d axis = norm(p2_axis - p1_axis);
	std::unordered_map<int, Vec3d> normals_to_particles;

	float hourGlassShape(float X) const {
		float min_height = 0.07;
		return X * X + min_height;
	}

	std::function<float(float)> contour = std::bind(
			&Boundary_axis_symmetric::hourGlassShape, this,
			std::placeholders::_1);

	float distance2(float X, float X0, float R0) const {
		return (X - X0) * (X - X0)
				+ (hourGlassShape(X) - R0) * (hourGlassShape(X) - R0);
	}

	std::function<float(float, float, float)> distance2_fun = std::bind(
			&Boundary_axis_symmetric::distance2, this, std::placeholders::_1,
			std::placeholders::_2, std::placeholders::_3);

public:

	float distance(const Particle &particle) const override;

	void draw2D() override;

	Vec3d getNormal(const Particle &particle) const override;

	const Vec3d& getAxis() const {
		return axis;
	}

	std::function<float(const float)> getContourFun() const {
		return contour;
	}
	std::function<float(float, float, float)> getDistance2Fun() const {
		return distance2_fun;
	}

};

#endif /* BOUNDARY_AXIS_SYMMETRIC_H_ */
