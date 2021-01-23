#include "vec3d.h"

CUDA_HOSTDEV
bool Vec3d::isLarge() const {
	return ( abs(*this) > LARGE || std::isnan(abs(*this)) );
}
CUDA_HOSTDEV
bool Vec3d::isSmall() const {
	return ( abs(*this) < SMALL || std::isnan(abs(*this)) );
}
