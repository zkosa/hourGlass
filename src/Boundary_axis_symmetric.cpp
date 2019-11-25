#include "Boundary_axis_symmetric.h"
#include "Minimum.h"
#include <QOpenGLWidget>

double Boundary_axis_symmetric::distance(const Particle &particle) const {

	const Vec3d &pos = particle.getPos();
	double X0 = pos * axis; // axial coordinate
	Vec3d Radial = pos - (pos * axis) * axis; // radial vector
	double R0 = abs(Radial); // radial coordinate

	double start_X = X0;

	//const std::function<double(double, double, double)> f = std::bind( getDistance2Fun(), 0, X0, R0) ;
	const std::function<double(double, double, double)> f = getDistance2Fun();
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

Vec3d Boundary_axis_symmetric::getNormal(const Particle &particle) const {

	const Vec3d &pos = particle.getPos();
	double X0 = pos * axis; // axial coordinate
	Vec3d Radial = pos - (pos * axis) * axis; // radial vector
	double R0 = abs(Radial); // radial coordinate

	const std::function<double(double, double, double)> f = getDistance2Fun();

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

void Boundary_axis_symmetric::draw2D() {
	// hardcoded for x= 0 axis
	// TODO: generalize

	float X;
	int resolution = 20;
	glBegin(GL_LINE_LOOP);
	for (int i = 0; i <= resolution; ++i) {
		X = Vec3d(p1_axis + (p2_axis - p1_axis) * (i / double(resolution))).y;
		glVertex2f(float(contour(X)), X);
	}
	glEnd();

	glBegin(GL_LINE_LOOP);
	for (int i = 0; i <= resolution; ++i) {
		X = Vec3d(p1_axis + (p2_axis - p1_axis) * (i / double(resolution))).y;
		glVertex2f(-float(contour(X)), X);
	}
	glEnd();

}
