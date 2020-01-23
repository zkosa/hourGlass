#include "Vec3d.h"

std::ostream& Vec3d::print() const {
	return std::cout << x << ", " << y << ", " << z << std::endl;
}

bool Vec3d::large() const {
	return (abs(*this) > LARGE || abs(*this) < -LARGE || std::isnan(abs(*this)));
}

const Vec3d Vec3d::i = Vec3d(1, 0, 0);

const Vec3d Vec3d::j = Vec3d(0, 1, 0);

const Vec3d Vec3d::k = Vec3d(0, 0, 1);

