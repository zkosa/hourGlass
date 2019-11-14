#include "Boundary_planar.h"

double Boundary_planar::distance(const Vec3d &point) const {
	return abs((point - plane_point) * normal);
}

double Boundary_planar::distance(const Particle &particle) const {
	return abs((particle.getPos() - plane_point) * normal);
}

void Boundary_planar::draw2D() {
	if (std::abs(normal * Vec3d(0, 0, 1)) < 1e-3) { // we are parallel to the display plane
		glBegin(GL_LINE_LOOP);
		glVertex2f(float(p1.x), float(p1.y));
		glVertex2f(float(p2.x), float(p2.y));
		glEnd();
	} else {
		std::cout << "3D drawing of planes has not been implemented yet!"
				<< std::endl;
	}
}
