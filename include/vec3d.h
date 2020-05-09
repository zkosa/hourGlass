#ifndef VEC3D_H_
#define VEC3D_H_

#include <math.h>
#include <iostream>
#include <stdexcept>
#include "constants.h"
#include "vecaxisym.h"

class Vec3d {

public:
	float x;
	float y;
	float z;

	Vec3d() : x(0.0f), y(0.0f), z(0.0f) {}

	Vec3d(float x, float y, float z) :
			x(x), y(y), z(z) {}

	Vec3d& operator+=(const Vec3d &other);
	Vec3d& operator-=(const Vec3d &other);
	Vec3d& operator*=(const float m);
	Vec3d& operator/=(const float div);

	std::ostream& print() const;
	bool isLarge() const;
	bool isSmall() const;

	VecAxiSym toYAxial() const {
		return VecAxiSym(y, sqrt(x*x + z*z));
	}

	const static Vec3d i, j, k;

};

const Vec3d gravity { 0, -g, 0 };

inline Vec3d Add(const Vec3d &a, const Vec3d &b) {
	return Vec3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Vec3d Substract(const Vec3d &a, const Vec3d &b) {
	return Vec3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline Vec3d Multiply(const Vec3d &a, const float m) {
	return Vec3d(a.x * m, a.y * m, a.z * m);
}

inline Vec3d Divide(const Vec3d &a, const float m) {
	if (std::abs(m) < VSMALL) {
		throw std::invalid_argument("Division by very small number.");
	}
	return Vec3d(a.x / m, a.y / m, a.z / m);
}

inline bool operator==(const Vec3d &a, const Vec3d &b) {
	return ( a.x == b.x && a.y == b.y && a.z == b.z );
}

inline std::ostream& operator<<(std::ostream &out, const Vec3d &a) {
	out << "(" << a.x << ", " << a.y << ", " << a.z << ")";
	return out;
}

inline Vec3d operator+(const Vec3d &a, const Vec3d &b) {
	return Add(a, b);
}

inline Vec3d& Vec3d::operator+=(const Vec3d &other)
{
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

inline Vec3d operator-(const Vec3d &a, const Vec3d &b) {
	return Substract(a, b);
}

inline Vec3d& Vec3d::operator-=(const Vec3d &other) {
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

inline Vec3d operator*(const float m, const Vec3d &a) {
	return Multiply(a, m);
}

inline Vec3d operator*(const Vec3d &a, const float m) {
	return Multiply(a, m);
}

inline Vec3d& Vec3d::operator*=(const float m) {
	x *= m;
	y *= m;
	z *= m;
	return *this;
}

inline Vec3d operator-(const Vec3d &a) {
	return Multiply(a, -1);
}

inline float operator*(const Vec3d &a, const Vec3d &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3d crossProduct(const Vec3d &a, const Vec3d &b) {
	return Vec3d(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

inline Vec3d operator/(const Vec3d &a, const float m) {
	return Divide(a, m);
}

inline Vec3d& Vec3d::operator/=(const float div) {
	if (std::abs(div) < VSMALL) {
		throw std::invalid_argument("Division by very small number.");
	}
	x /= div;
	y /= div;
	z /= div;
	return *this;
}

inline float abs(const Vec3d &a) {
	return std::sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}

inline Vec3d norm(const Vec3d &a) {
	float length = abs(a);
	if (length > SMALL) {
		return a / length;
	} else {
		return Vec3d(0, 0, 0);
	}
}

#endif /* VEC3D_H_ */
