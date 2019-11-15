#include "Vec3d.h"

std::ostream& Vec3d::print() const {
	return std::cout << x << ", " << y << ", " << z << std::endl;
}

bool Vec3d::large() const {
	return (abs(*this) > LARGE || abs(*this) < -LARGE || std::isnan(abs(*this)));
}

Vec3d operator+(const Vec3d &a, const Vec3d &b) {
	return Add(a, b);
}

Vec3d operator-(const Vec3d &a, const Vec3d &b) {
	return Substract(a, b);
}

Vec3d operator*(const double m, const Vec3d &a) {
	return Multiply(a, m);
}

Vec3d operator*(const Vec3d &a, const double m) {
	return Multiply(a, m);
}

double operator*(const Vec3d &a, const Vec3d &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3d crossProduct(const Vec3d &a, const Vec3d &b) {
	return Vec3d(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
}

Vec3d operator/(const Vec3d &a, const double m) {
	return Divide(a, m);
}

const Vec3d Vec3d::i = Vec3d(1, 0, 0);

const Vec3d Vec3d::j = Vec3d(0, 1, 0);

const Vec3d Vec3d::k = Vec3d(0, 0, 1);

Vec3d norm(const Vec3d &a) {
	double length = abs(a);
	if (length > SMALL) {
		return a / length;
	} else {
		return Vec3d(0, 0, 0);
	}
}
