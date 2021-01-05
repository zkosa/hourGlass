#include "particle.h"
#include "boundary_planar.h"
#include "boundary_axissymmetric.h"
#include "cell.h"
#include "scene.h"
//#include <omp.h>
#include <QOpenGLWidget>
#include <iostream>
#include <algorithm>

// static members:
int Cell::Nx = 10;

int Cell::Ny = Cell::Nx;

int Cell::Nz = 1; // 2D

Scene *Cell::scene = nullptr;

Vec3d Cell::dX{0,0,0}; // 2D

// constructor
Cell::Cell(const Vec3d &center) {

	const Vec3d dX = Cell::dX;

	bounds.x1 = (center - dX / 2.) * Vec3d::i;
	bounds.y1 = (center - dX / 2.) * Vec3d::j;
	bounds.z1 = (center - dX / 2.) * Vec3d::k;

	bounds.x2 = (center + dX / 2.) * Vec3d::i;
	bounds.y2 = (center + dX / 2.) * Vec3d::j;
	bounds.z2 = (center + dX / 2.) * Vec3d::k;

	constexpr float factor = 0.94f;
	bounds_for_display.x1 = (center - factor * dX / 2.) * Vec3d::i;
	bounds_for_display.y1 = (center - factor * dX / 2.) * Vec3d::j;
	bounds_for_display.z1 = (center - factor * dX / 2.) * Vec3d::k;

	bounds_for_display.x2 = (center + factor * dX / 2.) * Vec3d::i;
	bounds_for_display.y2 = (center + factor * dX / 2.) * Vec3d::j;
	bounds_for_display.z2 = (center + factor * dX / 2.) * Vec3d::k;

	this->center = center;

}

pointData Cell::getCorners() const {
	pointData corners;
	corners.reserve(8);

	corners.emplace_back( bounds.x1, bounds.y1, bounds.z1 );
	corners.emplace_back( bounds.x1, bounds.y1, bounds.z2 );
	corners.emplace_back( bounds.x1, bounds.y2, bounds.z1 );
	corners.emplace_back( bounds.x1, bounds.y2, bounds.z2 );
	corners.emplace_back( bounds.x2, bounds.y1, bounds.z1 );
	corners.emplace_back( bounds.x2, bounds.y1, bounds.z2 );
	corners.emplace_back( bounds.x2, bounds.y2, bounds.z1 );
	corners.emplace_back( bounds.x2, bounds.y2, bounds.z2 );

	return corners;
}

pointData Cell::getFaceCenters() const {
	pointData face_centers;
	face_centers.reserve(6);

	face_centers.push_back(center + dX / 2. * Vec3d::i * Vec3d::i);
	face_centers.push_back(center + dX / 2. * Vec3d::j * Vec3d::j);
	face_centers.push_back(center + dX / 2. * Vec3d::k * Vec3d::k);
	face_centers.push_back(center - dX / 2. * Vec3d::i * Vec3d::i);
	face_centers.push_back(center - dX / 2. * Vec3d::j * Vec3d::j);
	face_centers.push_back(center - dX / 2. * Vec3d::k * Vec3d::k);

	return face_centers;
}

pointData Cell::getEdgeCenters() const {
	pointData edge_centers;
	edge_centers.reserve(12);

	edge_centers.push_back(
			center + 0.5 * dX * Vec3d::i * Vec3d::i
					+ 0.5 * dX * Vec3d::j * Vec3d::j);
	edge_centers.push_back(
			center + 0.5 * dX * Vec3d::i * Vec3d::i
					- 0.5 * dX * Vec3d::j * Vec3d::j);
	edge_centers.push_back(
			center - 0.5 * dX * Vec3d::i * Vec3d::i
					+ 0.5 * dX * Vec3d::j * Vec3d::j);
	edge_centers.push_back(
			center - 0.5 * dX * Vec3d::i * Vec3d::i
					- 0.5 * dX * Vec3d::j * Vec3d::j);
	edge_centers.push_back(
			center + 0.5 * dX * Vec3d::i * Vec3d::i
					+ 0.5 * dX * Vec3d::k * Vec3d::k);
	edge_centers.push_back(
			center + 0.5 * dX * Vec3d::i * Vec3d::i
					- 0.5 * dX * Vec3d::k * Vec3d::k);
	edge_centers.push_back(
			center - 0.5 * dX * Vec3d::i * Vec3d::i
					+ 0.5 * dX * Vec3d::k * Vec3d::k);
	edge_centers.push_back(
			center - 0.5 * dX * Vec3d::i * Vec3d::i
					- 0.5 * dX * Vec3d::k * Vec3d::k);
	edge_centers.push_back(
			center + 0.5 * dX * Vec3d::j * Vec3d::j
					+ 0.5 * dX * Vec3d::k * Vec3d::k);
	edge_centers.push_back(
			center + 0.5 * dX * Vec3d::j * Vec3d::j
					- 0.5 * dX * Vec3d::k * Vec3d::k);
	edge_centers.push_back(
			center - 0.5 * dX * Vec3d::j * Vec3d::j
					+ 0.5 * dX * Vec3d::k * Vec3d::k);
	edge_centers.push_back(
			center - 0.5 * dX * Vec3d::j * Vec3d::j
					- 0.5 * dX * Vec3d::k * Vec3d::k);

	return edge_centers;
}

pointData Cell::getAllPoints() const {
	pointData all;
	all.reserve(27);
	all.push_back(center);

	const pointData corners = getCorners();
	const pointData faceCenters = getFaceCenters();
	const pointData edgeCenters = getEdgeCenters();
	all.insert(all.end(), corners.begin(), corners.end());
	all.insert(all.end(), faceCenters.begin(), faceCenters.end());
	all.insert(all.end(), edgeCenters.begin(), edgeCenters.end());
	return all;
}

void Cell::clear() {
	particle_IDs.clear();
}

void Cell::populate(std::vector<Particle> &particles) {
//#pragma omp parallel for
	for (auto &p : particles) {
		if (contains(p)) {
			addParticle(p);
		}
	}
}




bool Cell::contains(const Particle &p) const {
	const float r = p.getR();

	return (p.getX() + r > bounds.x1 && p.getX() - r < bounds.x2)
			&& (p.getY() + r > bounds.y1 && p.getY() - r < bounds.y2)
			&& (p.getZ() + r > bounds.z1 && p.getZ() - r < bounds.z2);
}

bool Cell::contains(const Boundary &b) const {
	if (b.distance(center) <= getHalfDiagonal()) {
		return true;
	} else {
		return false;
	}
}

void Cell::addParticle(const Particle &p) {
	particle_IDs.emplace_back(p.getID());
}

void Cell::size() const {
	std::cout << "Size of cell object: " << sizeof(*this) << std::endl;
}

void Cell::draw2D() const {
	glBegin(GL_LINE_LOOP);
	if (hasBoundary()) {
		glColor4f(1, 0, 0, 0.7);
	} else {
		glColor4f(0, 1, 0, 0.5);
	}
//	if (isExternal()) {
//		glColor4f(0, 0, 1, 1);
//	} else {
//		glColor4f(1, 1, 0, 1);
//	}
	glVertex2f(float(bounds_for_display.x1), float(bounds_for_display.y1));
	glVertex2f(float(bounds_for_display.x2), float(bounds_for_display.y1));
	glVertex2f(float(bounds_for_display.x2), float(bounds_for_display.y2));
	glVertex2f(float(bounds_for_display.x1), float(bounds_for_display.y2));
	glColor4f(1, 1, 1, 1);
	glEnd();
}

Vec3d Cell::average(const pointData& pd) {
	return std::accumulate(pd.begin(), pd.end(), Vec3d::null) / pd.size();
}
