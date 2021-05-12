#ifndef BOUNDARY_AXISSYMMETRIC_H_
#define BOUNDARY_AXISSYMMETRIC_H_

#include "boundary.h"
#include "vec3d.h"
#include <functional>

class Boundary_axissymmetric: public Boundary {
	Vec3d p1_axis { 0.0f, -1.0f, 0.0f };
	Vec3d p2_axis { 0.0f,  1.0f, 0.0f };
	Vec3d axis = norm(p2_axis - p1_axis);

	float hourGlassShape(float X) const {
		float min_height = 0.07f;
		return X * X + min_height;
	}

	std::function<float(float)> contour = std::bind(
			&Boundary_axissymmetric::hourGlassShape, this,
			std::placeholders::_1);

public:

	bool operator==(const Boundary &other) const override final;

	float distance(const Vec3d &point) const;
	float distance(const Particle &particle) const override final;
	float distanceSigned(const Vec3d &point) const;
	float distanceSigned(const Particle &particle) const override final;

	void draw2D() override final;

	Vec3d getNormal(const Particle &particle) const override final;
	Vec3d getNormalNumDiff(const Vec3d &curve_point) const;

	const Vec3d& getAxis() const {
		return axis;
	}

	std::function<float(const float)> getContourFun() const {
		return contour;
	}

};

#endif /* BOUNDARY_AXISSYMMETRIC_H_ */
