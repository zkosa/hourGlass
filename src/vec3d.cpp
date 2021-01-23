#include "vec3d.h"
#include <iostream>

std::ostream& operator<<(std::ostream &out, const Vec3d &a) {
	out << "(" << a.x << ", " << a.y << ", " << a.z << ")";
	return out;
}

std::ostream& Vec3d::print() const {
	return std::cout << *this << std::endl;
}

// static members:
CUDA_HOSTDEV
const Vec3d Vec3d::null = Vec3d(0, 0, 0);

CUDA_HOSTDEV
const Vec3d Vec3d::i = Vec3d(1, 0, 0);

CUDA_HOSTDEV
const Vec3d Vec3d::j = Vec3d(0, 1, 0);

CUDA_HOSTDEV
const Vec3d Vec3d::k = Vec3d(0, 0, 1);
