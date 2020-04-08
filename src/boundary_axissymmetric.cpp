#include "boundary_axissymmetric.h"
#include "minimumdistance.h"
#include <QOpenGLWidget>

float Boundary_axissymmetric::distance(const Particle &particle) const {

	MinimumDistance minimum_distance(*this, particle);

	return minimum_distance.getDistance();
}

Vec3d Boundary_axissymmetric::getNormal(const Particle &particle) const {
	// provides a normalized direction vector from the closest surface point to the particle
	// TODO: check, if this is what we really want and what we really do

	MinimumDistance minimum_distance(*this, particle);

	return minimum_distance.getNormal();
}

Vec3d Boundary_axissymmetric::getNormalNumDiff(const Vec3d &curve_point) const {
	// TODO: add check if is it really on the point?
	// alternatively: create a curve point type, and allow only that as argument?

	// step size for numerical derivative
	float dax = 1e-5f;
	// axial coordinate of the point where the normal is needed
	float ax = curve_point.toYAxial().axial;
	// slope of the tangent at the point (with respect to the axis)
	float slope_tangent = (contour(ax + dax) - contour(ax - dax)) / (2*dax);
	// slope of the normal at the point (with respect to the axis)
	// result of 90 deg rotation
	// must point inside, conforming to other definition (from surface to particle),
	// assuming that the particle is still inside
	//  TODO: try to use it as a check for particle loss!
	float slope_normal;

	Vec3d normal;
	if ( abs(slope_tangent) < SMALL ) {
		normal = Vec3d(-1,0,0);
	} else {
		slope_normal = 1 / slope_tangent;
		 // no need to normalize, and 2D only!
		// TODO: check in all quadrants, consider atan2 as well!
		normal = Vec3d(sin(atan(slope_normal)), cos(atan(slope_normal)), 0);
	}
// TODO: figure out, how to rotate in the right azimuthal direction
	return normal; // 2D only! TODO: generalize for 3D, check for negative y
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
