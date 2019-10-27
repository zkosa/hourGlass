#include "Particle.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include "Cell.h"
#include "Scene.h"
#include <omp.h>

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

void Cell::clear() {
	particle_IDs.clear();
}

void Cell::populate(std::vector<Particle>& particles) {
//#pragma omp parallel for
	for (auto& p : particles) {
		if (this->contains(p)) {
			this->addParticle(p);
		}
	}
}

bool Cell::contains(const Particle& p) {
	Vec3d pos = p.getPos(); // TODO implement getX, ...
	double r = p.getR();

	return ( pos.x + r > bounds.x && pos.x - r < bounds.X ) &&
			( pos.y + r > bounds.y && pos.y - r < bounds.Y ) &&
			( pos.z + r > bounds.z && pos.z - r < bounds.Z );
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
