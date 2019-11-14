#ifndef CELL_H_
#define CELL_H_

#include <vector>
#include "Vec3d.h"

class Scene;
class Particle;
class Boundary_planar;
class Boundary_axis_symmetric;

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

	static int Nx, Ny, Nz;

	Bounds bounds;
	Bounds bounds_display; // scaled for avoiding overlap of edges during display

	Vec3d center = {0,0,0};

	std::vector<int> particle_IDs; // TODO reserve expected size
	std::vector<int> boundary_IDs_planar;
	std::vector<int> boundary_IDs_axis_symmetric;

	double r = 0; // center to corner distance

	bool cell_with_boundary = false;

public:
	//Cell() {};
	Cell(const Vec3d& center, const Vec3d& dX);

	void init(const Scene&);
	void shrink();
	void update();

	void clear();
	void populate(std::vector<Particle>& particles);
	bool contains(const Particle&);
	bool contains(const Boundary&);
	void addParticle(const Particle&);
	void addBoundaryPlanar(const Boundary_planar&);
	void addBoundaryAxiSym(const Boundary_axis_symmetric&);

	void draw2D();

	const std::vector<int>& getBoundaryIDsAxisSymmetric() const { return boundary_IDs_axis_symmetric; }
	const std::vector<int>& getBoundaryIDsPlanar() const {return boundary_IDs_planar;}
	const Bounds& getBounds() const {return bounds;}
	const Vec3d& getCenter() const {return center;}
	const std::vector<int>& getParticleIDs() const {return particle_IDs;}

	bool hasBoundary() {return cell_with_boundary;}

	void setCellWithBoundary() {cell_with_boundary = true;}
	void setCellWithoutBoundary() {cell_with_boundary = false;}

	static int getNx() {return Nx;}
	static int getNy() {return Ny;}
	static int getNz() {return Nz;}
	static void setNx(int Nx) {Cell::Nx = Nx;}
	static void setNy(int Ny) {Cell::Ny = Ny;}
	static void setNz(int Ny) {Cell::Nz = Ny;}

};

#endif /* CELL_H_ */
