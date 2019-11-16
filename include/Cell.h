#ifndef CELL_H_
#define CELL_H_

#include <vector>
#include "Vec3d.h"

class Scene;
class Particle;
class Boundary_planar;
class Boundary_axis_symmetric;

typedef std::vector<Vec3d> pointData;

struct Bounds {
	// lower bound coordinates:
	double x1 = 0;
	double y1 = 0;
	double z1 = 0;
	// upper bound coordinates:
	double x2 = 0;
	double y2 = 0;
	double z2 = 0;
};

class Cell {

	static Scene *scene; // TODO make constant

	static int Nx, Ny, Nz;

	Bounds bounds;
	Bounds bounds_display; // scaled for avoiding overlap of edges during display

	Vec3d center = { 0, 0, 0 };
	pointData corners;
	pointData faceCenters;
	pointData edgeCenters; // TODO: check if it improves performance when they are not stored

	std::vector<int> particle_IDs; // TODO reserve the expected size
	std::vector<int> boundary_IDs_planar;
	std::vector<int> boundary_IDs_axis_symmetric;

	double half_diagonal; // center to corner distance

	bool cell_with_boundary = false;
	bool cell_is_external = false;

public:
	//Cell() {};
	Cell(const Vec3d &center, const Vec3d &dX);

	static void connectScene(Scene *scene) {
		Cell::scene = scene;
	}

	void clear();
	void populate(std::vector<Particle> &particles);
	bool contains(const Particle&);
	bool contains(const Boundary&);
	void addParticle(const Particle&);

	void draw2D();

	const std::vector<int>& getBoundaryIDsAxisSymmetric() const {
		return boundary_IDs_axis_symmetric;
	}
	const std::vector<int>& getBoundaryIDsPlanar() const {
		return boundary_IDs_planar;
	}
	const Bounds& getBounds() const {
		return bounds;
	}
	const Vec3d& getCenter() const {
		return center;
	}
	const pointData getCorners() const {
		return corners;
	}
	const pointData getFaceCenters() const {
		return faceCenters;
	}
	const pointData getEdgeCenters() const {
		return edgeCenters;
	}
	const pointData getAllPoints() const {
		pointData all;
		all.push_back(center);
		all.insert(all.end(), corners.begin(), corners.end());
		all.insert(all.end(), faceCenters.begin(), faceCenters.end());
		all.insert(all.end(), edgeCenters.begin(), edgeCenters.end());
		return all;
	}
	const double getHalfDiagonal() const {
		return half_diagonal;
	}
	const std::vector<int>& getParticleIDs() const {
		return particle_IDs;
	}

	bool hasBoundary() {
		return cell_with_boundary;
	}

	inline bool isExternal() {
		return cell_is_external;
	}

	void setCellWithBoundary() {
		cell_with_boundary = true;
	}
	void setCellWithoutBoundary() {
		cell_with_boundary = false;
	}

	void setExternal(bool external) {
		cell_is_external = external;
	}

	static int getNx() {
		return Nx;
	}
	static int getNy() {
		return Ny;
	}
	static int getNz() {
		return Nz;
	}
	static void setNx(int Nx) {
		Cell::Nx = Nx;
	}
	static void setNy(int Ny) {
		Cell::Ny = Ny;
	}
	static void setNz(int Ny) {
		Cell::Nz = Ny;
	}

};

#endif /* CELL_H_ */
