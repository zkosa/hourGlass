#include "vec3d.h"

std::ostream& Vec3d::print() const {
	return std::cout << *this << std::endl;
}

bool Vec3d::isLarge() const {
	return ( abs(*this) > LARGE || std::isnan(abs(*this)) );
}

bool Vec3d::isSmall() const {
	return ( abs(*this) < SMALL || std::isnan(abs(*this)) );
}

// static members:
const Vec3d Vec3d::null = Vec3d(0, 0, 0);

const Vec3d Vec3d::i = Vec3d(1, 0, 0);

const Vec3d Vec3d::j = Vec3d(0, 1, 0);

const Vec3d Vec3d::k = Vec3d(0, 0, 1);
