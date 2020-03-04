#ifndef CELL_H_
#define CELL_H_

#include <vector>
#include "vec3d.h"

class Scene;
class Particle;
class Boundary_planar;
class Boundary_axissymmetric;

typedef std::vector<Vec3d> pointData;

struct Bounds {
	// lower bound coordinates:
	float x1 = 0;
	float y1 = 0;
	float z1 = 0;
	// upper bound coordinates:
	float x2 = 0;
	float y2 = 0;
	float z2 = 0;
};

class Cell {

	static Scene *scene; // TODO make constant

	static int Nx, Ny, Nz;

	Bounds bounds;
	Bounds bounds_for_display; // scaled for avoiding overlap of edges during display

	Vec3d center = { 0, 0, 0 };
	pointData corners;
	pointData face_centers;
	pointData edge_centers; // TODO: check if it improves performance when they are not stored

	std::vector<int> particle_IDs; // TODO reserve the expected size
	std::vector<int> boundary_IDs_planar;
	std::vector<int> boundary_IDs_axis_symmetric;

	float half_diagonal; // center to corner distance
	float test;

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
	const pointData& getCorners() const {
		return corners;
	}
	const pointData& getFaceCenters() const {
		return face_centers;
	}
	const pointData& getEdgeCenters() const {
		return edge_centers;
	}
	pointData getAllPoints() const {
		pointData all;
		all.push_back(center);
		all.insert(all.end(), corners.begin(), corners.end());
		all.insert(all.end(), face_centers.begin(), face_centers.end());
		all.insert(all.end(), edge_centers.begin(), edge_centers.end());
		return all;
	}
	float getHalfDiagonal() const {
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

	inline bool isInternal() {
		return !cell_is_external;
	}

	void setCellWithBoundary() {
		cell_with_boundary = true;
	}
	void setCellWithoutBoundary() {
		cell_with_boundary = false;
	}

	void setExternal() {
		cell_is_external = true;
	}

	void setInternal() {
		cell_is_external = false;
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
