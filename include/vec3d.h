#ifndef VEC3D_H_
#define VEC3D_H_

#include <math.h>
#include <iostream>
#include <stdexcept>
#include "constants.h"

struct Vec3d {

	float x;
	float y;
	float z;

	Vec3d() : x(0.0f), y(0.0f), z(0.0f) {}

	Vec3d(float x, float y, float z) :
			x(x), y(y), z(z) {}

	std::ostream& print() const;
	bool large() const;

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
	if (abs(m) < VSMALL) {
		throw std::invalid_argument("Division by very small number.");
	}
	return Vec3d(a.x / m, a.y / m, a.z / m);

}

inline bool operator==(const Vec3d &a, const Vec3d &b) {
	return ( a.x == b.x && a.y == b.y && a.z == b.z );
}

inline std::ostream& operator<<(std::ostream &out, const Vec3d &a) {
	out << "(" << a.x << ", " << a.y << ", " << a.z << ")" << std::endl;
	return out;
}

inline Vec3d operator+(const Vec3d &a, const Vec3d &b) {
	return Add(a, b);
}

inline Vec3d operator-(const Vec3d &a, const Vec3d &b) {
	return Substract(a, b);
}

inline Vec3d operator*(const float m, const Vec3d &a) {
	return Multiply(a, m);
}

inline Vec3d operator*(const Vec3d &a, const float m) {
	return Multiply(a, m);
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