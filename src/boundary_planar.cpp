#include "boundary_planar.h"
#include <QOpenGLWidget>

float Boundary_planar::distance(const Vec3d &point) const {
	return std::abs(Boundary_planar::distanceSigned(point));
}

float Boundary_planar::distance(const Particle &particle) const {
	return Boundary_planar::distance(particle.getPos());
}

float Boundary_planar::distanceSigned(const Vec3d &point) const {
	return (point - plane_point) * normal;
}

float Boundary_planar::distanceSigned(const Particle &particle) const {
	return Boundary_planar::distanceSigned(particle.getPos());
}

void Boundary_planar::draw2D() {
	if (std::abs(normal * Vec3d::k) < 1e-3f) { // we are parallel to the display plane
		glBegin(GL_LINE_LOOP);
		glVertex2f(float(p1.x), float(p1.y));
		glVertex2f(float(p2.x), float(p2.y));
		glEnd();
	} else {
		std::cout << "3D drawing of planes has not been implemented yet!"
				<< std::endl;
	}
}
