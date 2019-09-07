#include "Vec3d.h"

std::ostream& Vec3d::print() {
	return std::cout << x << ", " << y << ", " << z;
}

Vec3d Add(const Vec3d& a, const Vec3d& b) {
	return Vec3d(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3d Substract(const Vec3d& a, const Vec3d& b) {
	return Vec3d(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3d Multiply(const Vec3d& a, const double m) {
	return Vec3d(a.x*m, a.y*m, a.z*m);
}

Vec3d Divide(const Vec3d& a, const double m) {
	// TODO: treat division by zero
	return Vec3d(a.x/m, a.y/m, a.z/m);
}

Vec3d operator+(const Vec3d& a, const Vec3d& b) {
	return Add(a, b);
}

Vec3d operator-(const Vec3d& a, const Vec3d& b) {
	return Substract(a, b);
}

Vec3d operator*(const double m, const Vec3d& a) {
	return Multiply(a, m);
}

Vec3d operator*(const Vec3d& a, const double m) {
	return Multiply(a, m);
}

double operator*(const Vec3d& a, const Vec3d& b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

Vec3d crossProduct(const Vec3d& a, const Vec3d& b) {
	return Vec3d(
			a.y* b.z - a.z*b.y,
			a.z* b.x - a.x*b.z,
			a.x*b.y - a.y*b.x
			);
}

Vec3d operator/(const Vec3d& a, const double m) {
	return Divide(a, m);
}

double abs(const Vec3d& a) {
	return std::sqrt( a.x*a.x + a.y*a.y + a.z*a.z);
}

Vec3d norm(const Vec3d& a){
	double length = abs(a);
	if (length > 1e-10) {
		return a/length;
	}
	else {
		return Vec3d(0,0,0);
	}
}
