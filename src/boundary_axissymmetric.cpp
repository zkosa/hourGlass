#include "boundary_axissymmetric.h"
#include "minimumdistance.h"
#include <QOpenGLWidget>

float Boundary_axissymmetric::distance(const Particle &particle) const {
	return distance(particle.getPos());
}

float Boundary_axissymmetric::distance(const Vec3d &point) const {

	MinimumDistance minimum_distance(*this, point);

	return minimum_distance.getDistance();
}

float Boundary_axissymmetric::distanceSigned(const Particle &particle) const {
	return 	distanceSigned(particle.getPos());
}

float Boundary_axissymmetric::distanceSigned(const Vec3d &point) const {
	// Returns a signed distance between the contour and the particle.
	// Negative value indicates that the particle is on the outer side.
	// TODO: add tests
	// TODO: optimize
	MinimumDistance minimum_distance(*this, point);

	Vec3d contact_point = minimum_distance.getClosestPointOnTheContour();

	return (point - contact_point) * getNormal(point);
}

Vec3d Boundary_axissymmetric::getNormal(const Particle &particle) const {
	// provides a normalized direction vector from the closest surface point to the particle

	MinimumDistance minimum_distance(*this, particle);

	return getNormalNumDiff(minimum_distance.getClosestPointOnTheContour());
}

Vec3d Boundary_axissymmetric::getNormalNumDiff(const Vec3d &curve_point) const {
	// TODO: add check if is it really on the point?
	// alternatively: create a curve point type, and allow only that as argument?

	// step size for numerical derivative
	float dax = 1e-5f;
	// axial coordinate of the point where the normal is needed
	float ax = curve_point.toYAxial().axial;
	// must point inside, conforming to other definition (from surface to particle),
	// assuming that the particle is still inside!!!
	// TODO: try to use it as a check for particle loss!
	// TODO: check in all quadrants
	// rotate the tangent by 90 deg, to get normal:
	// swap x and y, and change sign of x
	Vec3d normal = norm(Vec3d(
			-dax,
			contour(ax + dax/2) - contour(ax - dax/2),
			0));

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
