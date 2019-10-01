#include "Particle.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include "Cell.h"
#include "Scene.h"

Cell::Cell(const Vec3d& center, const Vec3d& dX)
{
	bounds.x = (center - dX/2)*Vec3d::i;
	bounds.y = (center - dX/2)*Vec3d::j;
	bounds.z = (center - dX/2)*Vec3d::k;

	bounds.X = (center + dX/2)*Vec3d::i;
	bounds.Y = (center + dX/2)*Vec3d::j;
	bounds.Z = (center + dX/2)*Vec3d::k;
}

void Cell::init(const Scene& scene) {
}

void Cell::shrink() {
}

void Cell::update() {
}

bool Cell::contains(const Particle& p) {
	Vec3d pos = p.getPos(); // TODO implement getX, ...

	return ( pos.x > bounds.x && pos.x < bounds.X ) &&
			( pos.y > bounds.y && pos.y < bounds.Y ) &&
			( pos.z > bounds.z && pos.z < bounds.Z );
}

void Cell::addParticle(const Particle& p) {
	particle_IDs.emplace_back(p.getID());
}

void Cell::addBoundaryPlanar(const Boundary_planar& b) {
}

void Cell::addBoundaryAxiSym(const Boundary_axis_symmetric& b) {
}

void Cell::draw2D() {
	glBegin(GL_LINE_LOOP);
/*	glEnable( GL_BLEND );
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(.23,.78,.32,0.95);
*/	glVertex2f(float(bounds.x), float(bounds.y));
	glVertex2f(float(bounds.X), float(bounds.y));
	glVertex2f(float(bounds.X), float(bounds.Y));
	glVertex2f(float(bounds.x), float(bounds.Y));
	glEnd();
}
