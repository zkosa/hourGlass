#include "boundary_axissymmetric.h"
#include "minimum.h"
#include <QOpenGLWidget>

float Boundary_axissymmetric::distance(const Particle &particle) const {

	const Vec3d &pos = particle.getPos();
	float X0 = pos * axis; // axial coordinate
	Vec3d Radial = pos - (pos * axis) * axis; // radial vector
	float R0 = abs(Radial); // radial coordinate

	float start_X = X0;

	//const std::function<float(float, float, float)> f = std::bind( getDistance2Fun(), 0, X0, R0) ;
	const std::function<float(float, float, float)> f = getDistance2Fun();
	//std::cout << "calling f..." << std::endl;
	//f(1,2,3);
	//std::cout << "f was called" << std::endl;
	//std::cout << "distance2(1, X0, R0): " << distance2(1, X0, R0) << std::endl;  // OK
	//std::cout << "contour(0): " << contour(0) << std::endl; // OK
	//distance2_fun(1, X0, R0); // crashes

	// Newton iteration for minimum
	//Minimum minimum( f, X0, R0 );  // (0, X0, R0)
	Minimum minimum(*this, X0, R0);
	minimum.search(start_X);

	return minimum.getDistance();
}

Vec3d Boundary_axissymmetric::getNormal(const Particle &particle) const {
	// provides a normalized direction vector from the closest surface point to the particle
	// TODO: check, if this is what we really want and what we really do

	const Vec3d &pos = particle.getPos();
	float X0 = pos * axis; // axial coordinate
	Vec3d Radial = pos - (pos * axis) * axis; // radial vector
	float R0 = abs(Radial); // radial coordinate

	const std::function<float(float, float, float)> f = getDistance2Fun();

	// Newton iteration for minimum
	//Minimum minimum( f, X0, R0 );  // (0, X0, R0)
	Minimum minimum(*this, X0, R0);
	VecAxiSym contactPointInRadialCoord =
			minimum.getContactPointInRadialCoord();

	Vec3d contactPoint = axis * contactPointInRadialCoord.X
			+ norm(Radial) * contactPointInRadialCoord.R;

	Vec3d n = norm(particle.getPos() - contactPoint);

	return n;
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
