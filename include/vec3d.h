#ifndef VEC3D_H_
#define VEC3D_H_

#include <math.h>
#include <iosfwd>
#include <stdexcept>
#include "constants.h"
#include "vecaxisym.h"
#include "cuda.h"

class Vec3d {

public:
	float x;
	float y;
	float z;

	CUDA_HOSTDEV
	Vec3d() : x(0.0f), y(0.0f), z(0.0f) {}

	CUDA_HOSTDEV
	Vec3d(float x, float y, float z) :
			x(x), y(y), z(z) {}

	CUDA_HOSTDEV
	Vec3d& operator+=(const Vec3d &other);
	CUDA_HOSTDEV
	Vec3d& operator-=(const Vec3d &other);
	CUDA_HOSTDEV
	Vec3d& operator*=(const float m);
	CUDA_HOSTDEV
	Vec3d& operator/=(const float div);

	std::ostream& print() const;
	CUDA_HOSTDEV
	bool isLarge() const;
	CUDA_HOSTDEV
	bool isSmall() const;

	CUDA_HOSTDEV
	VecAxiSym toYAxial() const {
		return VecAxiSym(y, sqrt(x*x + z*z));
	}

	const static Vec3d null;
	const static Vec3d i, j, k;

};


CUDA_HOSTDEV
inline Vec3d Add(const Vec3d &a, const Vec3d &b) {
	return Vec3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

CUDA_HOSTDEV
inline Vec3d Substract(const Vec3d &a, const Vec3d &b) {
	return Vec3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

CUDA_HOSTDEV
inline Vec3d Multiply(const Vec3d &a, const float m) {
	return Vec3d(a.x * m, a.y * m, a.z * m);
}

CUDA_HOSTDEV
inline Vec3d Divide(const Vec3d &a, const float m) {
	// device code does not support exception handling
	return Vec3d(a.x / m, a.y / m, a.z / m);
}

CUDA_HOSTDEV
inline bool operator==(const Vec3d &a, const Vec3d &b) {
	return ( a.x == b.x && a.y == b.y && a.z == b.z );
}

std::ostream& operator<<(std::ostream &out, const Vec3d &a);

CUDA_HOSTDEV
inline Vec3d operator+(const Vec3d &a, const Vec3d &b) {
	return Add(a, b);
}

CUDA_HOSTDEV
inline Vec3d& Vec3d::operator+=(const Vec3d &other)
{
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

CUDA_HOSTDEV
inline Vec3d operator-(const Vec3d &a, const Vec3d &b) {
	return Substract(a, b);
}

CUDA_HOSTDEV
inline Vec3d& Vec3d::operator-=(const Vec3d &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

CUDA_HOSTDEV
inline Vec3d operator*(const float m, const Vec3d &a) {
	return Multiply(a, m);
}

CUDA_HOSTDEV
inline Vec3d operator*(const Vec3d &a, const float m) {
	return Multiply(a, m);
}

CUDA_HOSTDEV
inline Vec3d& Vec3d::operator*=(const float m) {
	x *= m;
	y *= m;
	z *= m;
	return *this;
}

CUDA_HOSTDEV
inline Vec3d operator-(const Vec3d &a) {
	return Multiply(a, -1);
}

CUDA_HOSTDEV
inline float operator*(const Vec3d &a, const Vec3d &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

CUDA_HOSTDEV
inline Vec3d crossProduct(const Vec3d &a, const Vec3d &b) {
	return Vec3d(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

CUDA_HOSTDEV
inline Vec3d operator/(const Vec3d &a, const float m) {
	return Divide(a, m);
}

CUDA_HOSTDEV
inline Vec3d& Vec3d::operator/=(const float div) {
	// device code does not support exception handling
	x /= div;
	y /= div;
	z /= div;
	return *this;
}

CUDA_HOSTDEV
inline float abs(const Vec3d &a) {
	return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

CUDA_HOSTDEV
inline Vec3d norm(const Vec3d &a) {
	const float length = abs(a);
	if (length > SMALL) {
		return a / length;
	} else {
		return Vec3d(0, 0, 0);
	}
}

#endif /* VEC3D_H_ */
