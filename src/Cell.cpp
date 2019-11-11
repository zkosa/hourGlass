#include "Particle.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include "Cell.h"
#include "Scene.h"
#include <omp.h>

int Cell::Nx = 10;

int Cell::Ny = Cell::Nx;

int Cell::Nz = 1; // 2D

Cell::Cell(const Vec3d& center, const Vec3d& dX)
{
	bounds.x = (center - dX/2)*Vec3d::i;
	bounds.y = (center - dX/2)*Vec3d::j;
	bounds.z = (center - dX/2)*Vec3d::k;

	bounds.X = (center + dX/2)*Vec3d::i;
	bounds.Y = (center + dX/2)*Vec3d::j;
	bounds.Z = (center + dX/2)*Vec3d::k;

	double factor = 0.93;
	bounds_display.x = (center - factor*dX/2)*Vec3d::i;
	bounds_display.y = (center - factor*dX/2)*Vec3d::j;
	bounds_display.z = (center - factor*dX/2)*Vec3d::k;

	bounds_display.X = (center + factor*dX/2)*Vec3d::i;
	bounds_display.Y = (center + factor*dX/2)*Vec3d::j;
	bounds_display.Z = (center + factor*dX/2)*Vec3d::k;

	r = abs(0.5*dX);

	center_ = center;
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

bool Cell::contains(const Boundary& b){
	if (b.distance(center_) <= r) { // + Particle::getUniformRadius()
		return true;
	}
	else {
		return false;
	}
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
//	glEnable( GL_BLEND ); // alpha seems to have no effect
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	if (hasBoundary()) {
		glColor4f(1,0,0, 1);
	}
	else {
		glColor4f(0,1,0, 0.1);
	}
	glVertex2f(float(bounds_display.x), float(bounds_display.y));
	glVertex2f(float(bounds_display.X), float(bounds_display.y));
	glVertex2f(float(bounds_display.X), float(bounds_display.Y));
	glVertex2f(float(bounds_display.x), float(bounds_display.Y));
	glColor4f(1, 1, 1, 1);
	glEnd();
}
