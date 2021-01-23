#ifndef BOUNDARY_AXISSYMMETRIC_H_
#define BOUNDARY_AXISSYMMETRIC_H_

#include "boundary.h"
#include "vec3d.h"
//#include <functional>
#include "functionhandler.h"
#include "minimum.h"

class Boundary_axissymmetric: public Boundary {
	Vec3d p1_axis { 0.0f, -1.0f, 0.0f };
	Vec3d p2_axis { 0.0f,  1.0f, 0.0f };
	Vec3d axis = norm(p2_axis - p1_axis);

	__host__ __device__
	float hourGlassShape(float X) const { // would be nice to have as static
		float min_height = 0.07f;
		return X * X + min_height;
	}

//	std::function<float(float)> contour = std::bind(
//			&Boundary_axissymmetric::hourGlassShape, this,
//			std::placeholders::_1);

	// function and bind are not available on the device:
	// let us use a function pointer on the device
	//function_t contour_ptr = hourGlassShape;


public:

	__host__ __device__
	Boundary_axissymmetric() {
		initializer();
	}

	constFunctionHandler<Boundary_axissymmetric> functionHandler_contour;

	__host__ __device__
	void initializer() {
		this->functionHandler_contour = &Boundary_axissymmetric::hourGlassShape;
	}

	bool operator==(const Boundary &other) const override;

	float distance(const Vec3d &point) const;
	__device__
	float distanceDev(const Vec3d *point) const;
	float distance(const Particle &particle) const override;
	__device__
	float distanceDev(const Particle *particle) const override;
	__host__ __device__
	float distanceSigned(const Vec3d &point) const;
	__host__ __device__
	float distanceSigned(const Particle &particle) const override;

	void draw2D() override;

	__host__ __device__
	Vec3d getNormal(const Particle &particle) const override;
	Vec3d getNormalNumDiff(const Vec3d &curve_point) const;

	__host__ __device__
	const Vec3d& getAxis() const {
		return axis;
	}

};

__global__
inline void initializeFunctionHandle(Boundary_axissymmetric* bax) {
	// it needs to be called after the object the object has been copied to the device!
	// takes a device pointer to the object to be initialized
	// unfortunately direct __host__ call of dev_bax->initializer() does not work as expected
	bax->initializer();
}

#endif /* BOUNDARY_AXISSYMMETRIC_H_ */
