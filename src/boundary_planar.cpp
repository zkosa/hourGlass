#include "boundary_planar.h"
#include "particle.h"
#include <iostream>
#include <QOpenGLWidget>

bool Boundary_planar::operator==(const Boundary &other) const {
	const Boundary_planar* other_boundary_casted_to_this = dynamic_cast< const Boundary_planar* >( &other );
	if ( other_boundary_casted_to_this == nullptr ) {
		return false; // they have different derived type
	} else {
		if (plane_point == other_boundary_casted_to_this->plane_point &&
			normal == other_boundary_casted_to_this->normal) {
			return true;
		} else {
			return false;
		}
	}
}

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
		glVertex2f(p1.x, p1.y);
		glVertex2f(p2.x, p2.y);
		glEnd();
	} else {
		std::cout << "3D drawing of planes has not been implemented yet!"
				<< std::endl;
	}
}
