#include "boundary_axissymmetric.h"
#include "minimumdistance.h"
#include <QOpenGLWidget>

float Boundary_axissymmetric::distance(const Particle &particle) const {

	float start_X = particle.getPos() * axis; // use the axial coordinate as guess

	MinimumDistance minimum_distance(*this, particle);

	minimum_distance.setInitualGuess(start_X);

	return minimum_distance.getDistance();
}

Vec3d Boundary_axissymmetric::getNormal(const Particle &particle) const {
	// provides a normalized direction vector from the closest surface point to the particle
	// TODO: check, if this is what we really want and what we really do

	float start_X = particle.getPos() * axis; // use the axial coordinate as guess

	MinimumDistance minimum_distance(*this, particle);

	minimum_distance.setInitualGuess(start_X);

	return minimum_distance.getNormal();
}

void Boundary_axissymmetric::draw2D() {
	// hardcoded for x= 0 axis
	// TODO: generalize

	float X;
	int resolution = 20;
	glBegin(GL_LINE_LOOP);
	for (int i = 0; i <= resolution; ++i) {
		X = Vec3d(p1_axis + (p2_axis - p1_axis) * (i / float(resolution))).y;
		glVertex2f(float(contour(X)), X);
	}
	glEnd();

	glBegin(GL_LINE_LOOP);
	for (int i = 0; i <= resolution; ++i) {
		X = Vec3d(p1_axis + (p2_axis - p1_axis) * (i / float(resolution))).y;
		glVertex2f(-float(contour(X)), X);
	}
	glEnd();

}
