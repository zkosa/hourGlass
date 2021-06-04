#include "vec3d.h"
#include "boundingbox.h"

BoundingBox::BoundingBox(Scene &scene) :
		corner1( { 0.0f, 0.0f, 0.0f }), corner2( { 0.0f, 0.0f, 0.0f }) {

	float x1, y1, z1, x2, y2, z2;

	// hardcoded:
	x1 = y1 = z1 = -1.0f;
	x2 = y2 = z2 = 1.0f;

	corner1 = { x1, y1, z1 };
	corner2 = { x2, y2, z2 };

}

Vec3d BoundingBox::center() const {
	return 0.5f * (corner2 + corner1);
}

Vec3d BoundingBox::diagonal() const {
	return corner2 - corner1;
}

float BoundingBox::volume() const {
	return diagonal().x * diagonal().y * diagonal().z;
}

