#include "boundary_axissymmetric.h"
#include "minimumdistance.h"
#include <QOpenGLWidget>

bool Boundary_axissymmetric::operator==(const Boundary &other) const {
	const Boundary_axissymmetric* other_boundary_casted_to_this = dynamic_cast< const Boundary_axissymmetric* >( &other );
	if ( other_boundary_casted_to_this == nullptr ) {
		return false; // they have different derived type
	} else {
		return *this == other;
	}
}

bool Boundary_axissymmetric::operator==(const Boundary_planar &other) const {
	return false;
}

bool Boundary_axissymmetric::operator==(const Boundary_axissymmetric &other) const {
	if (axis == other.axis
			// && contour == other.contour  // TODO: std::function equality?
			) {
		return true;
	} else {
		return false;
	}
}

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
	const MinimumDistance minimum_distance(*this, point);

	const Vec3d contact_point = minimum_distance.getClosestPointOnTheContour();

	return (point - contact_point) * getNormal(point);
}

Vec3d Boundary_axissymmetric::getNormal(const Particle &particle) const {
	// provides a normalized direction vector from the closest surface point to the particle

	const MinimumDistance minimum_distance(*this, particle);

	return getNormalNumDiff(minimum_distance.getClosestPointOnTheContour());
}

Vec3d Boundary_axissymmetric::getNormalNumDiff(const Vec3d &curve_point) const {
	// Returns a normal vector at the specified point on the axial symmetric surface.
	// The normal points inside.
	 // 2D only! TODO: generalize for 3D!

	// TODO: add check if curve_point is really on the curve?
	// alternatively: create a curve point type, and allow only that as argument?

	// TODO: try to use the relation of the normal vector and the distance vector
	// from the curve_point to the particle to check if the particle is inside.

	// x* notation indicates, that it is the axial coordinate,
	// which is y (the vertical coordinate) in our case.
	// The f'(x*) = ( f(x* + dx/2) - f(x* - dx/2) ) / dx approximation is used.
	// In this case the tangent vector (in 2D) can be written as:
	// tangent = normalize( dx, f(x* + dx/2) - f(x* - dx/2) )
	// while the normal vector can be obtained by 90 deg rotation:
	// normal = normalize( f(x* + dx/2) - f(x* - dx/2), -dx )
	// The two coordinates have to be swapped at the end, because
	// the first coordinate is the radial one.

	// step size for numerical derivative (axial direction)
	constexpr float dax = 1e-5f;
	// axial coordinate of the point where the normal is needed
	const float ax = curve_point.toYAxial().axial;
	float dax_signed;
	if (curve_point.x >= 0.0f) {
		dax_signed = -dax;
	} else {
		dax_signed = dax;
	}
	const Vec3d normal = norm(Vec3d(
			dax_signed,
			contour(ax + dax/2.0f) - contour(ax - dax/2.0f),
			0));

	return normal;
}

void Boundary_axissymmetric::draw2D() {
	// hardcoded for x= 0 axis
	// TODO: generalize
	glLineWidth(2.0f);
	glColor4f(0.6f, 0.8f, 0.8f, 1.0f);

	float X;
	constexpr int resolution = 20;

	// the right hand side section in the (x,y) plane:
	glBegin(GL_LINE_LOOP);
	for (int i = 0; i <= resolution; ++i) {
		X = Vec3d(p1_axis + (p2_axis - p1_axis) * (static_cast<float>(i) / static_cast<float>(resolution))).y;
		glVertex2f(contour(X), X);
	}
	glEnd();

	// the left hand side section in the (x,y) plane:
	glBegin(GL_LINE_LOOP);
	for (int i = 0; i <= resolution; ++i) {
		X = Vec3d(p1_axis + (p2_axis - p1_axis) * (static_cast<float>(i) / static_cast<float>(resolution))).y;
		glVertex2f(-contour(X), X);
	}
	glEnd();
	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	glLineWidth(1.0f);
}
