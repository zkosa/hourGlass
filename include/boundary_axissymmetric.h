#ifndef BOUNDARY_AXISSYMMETRIC_H_
#define BOUNDARY_AXISSYMMETRIC_H_

#include "boundary.h"
#include <functional>
#include <unordered_map>

class Boundary_axissymmetric: public Boundary {
	Vec3d p1_axis { 0, -1, 0 };
	Vec3d p2_axis { 0, 1, 0 };
	Vec3d axis = norm(p2_axis - p1_axis);
	std::unordered_map<int, Vec3d> normals_to_particles;

	float hourGlassShape(float X) const {
		float min_height = 0.07;
		return X * X + min_height;
	}

	std::function<float(float)> contour = std::bind(
			&Boundary_axissymmetric::hourGlassShape, this,
			std::placeholders::_1);

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

};

#endif /* BOUNDARY_AXISSYMMETRIC_H_ */
