#include "Boundary_planar.h"

double Boundary_planar::distance(const Vec3d &point) const {
	return abs(Boundary_planar::distance_signed(point));
}

double Boundary_planar::distance(const Particle &particle) const {
	return Boundary_planar::distance(particle.getPos());
}

double Boundary_planar::distance_signed(const Vec3d &point) const {
	return (point - plane_point) * normal;
}

double Boundary_planar::distance_signed(const Particle &particle) const {
	return Boundary_planar::distance_signed(particle.getPos());
}

void Boundary_planar::draw2D() {
	if (std::abs(normal * Vec3d::k) < 1e-3) { // we are parallel to the display plane
		glBegin(GL_LINE_LOOP);
		glVertex2f(float(p1.x), float(p1.y));
		glVertex2f(float(p2.x), float(p2.y));
		glEnd();
	} else {
		std::cout << "3D drawing of planes has not been implemented yet!"
				<< std::endl;
	}
}
