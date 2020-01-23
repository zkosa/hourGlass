#include "Vec3d.h"
#include "Scene.h"
#include "BoundingBox.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"

BoundingBox::BoundingBox(Scene &scene) :
		corner1( { 0, 0, 0 }), corner2( { 0, 0, 0 }) {

	float x1, y1, z1, x2, y2, z2;

	// hardcoded:
	x1 = y1 = z1 = -1;
	x2 = y2 = z2 = 1;

	corner1 = { x1, y1, z1 };
	corner2 = { x2, y2, z2 };

}

Vec3d BoundingBox::center() const {
	return (corner2 + corner1) / 2.;
}

Vec3d BoundingBox::diagonal() const {
	return corner2 - corner1;
}

float BoundingBox::volume() const {
	return diagonal().x * diagonal().y * diagonal().z;
}
