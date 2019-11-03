#ifndef VEC3D_H_
#define VEC3D_H_

#include <math.h>
#include <iostream>
#include "Constants.h"

struct Vec3d;

struct Vec3d {

	double x;
	double y;
	double z;

	Vec3d(double x, double y, double z): x(x), y(y), z(z) {};
	std::ostream& print() const;
	bool large() const;

	const static Vec3d i, j, k;

};

const Vec3d gravity {0, -g, 0};

Vec3d Add(const Vec3d& a, const Vec3d& b);

Vec3d Substract(const Vec3d& a, const Vec3d& b);

Vec3d Multiply(const Vec3d& a, const double m);

Vec3d Divide(const Vec3d& a, const double m);

Vec3d operator+(const Vec3d& a, const Vec3d& b);

Vec3d operator-(const Vec3d& a, const Vec3d& b);

Vec3d operator*(const double m, const Vec3d& a);

Vec3d operator*(const Vec3d& a, const double m);

double operator*(const Vec3d& a, const Vec3d& b);

Vec3d crossProduct(const Vec3d& a, const Vec3d& b);

Vec3d operator/(const Vec3d& a, const double m);

double abs(const Vec3d& a);

Vec3d norm(const Vec3d& a);

#endif /* VEC3D_H_ */