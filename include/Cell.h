#ifndef CELL_H_
#define CELL_H_

#include <vector>
#include "Vec3d.h"

class Scene;
class Particle;
class Boundary_planar;
class Boundary_axis_symmetric;

struct Bounds {
	double x = 0; // lower bound coordinate
	double y = 0;
	double z = 0;
	double X = 0; // upper bound coordinate
	double Y = 0;
	double Z = 0;
};

class Cell {

	static int Nx, Ny, Nz;

	Bounds bounds;
	Bounds bounds_display; // scaled for avoiding overlap of edges during display

	Vec3d center_ = {0,0,0};

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

	const Vec3d center() const {return center_;}

	const std::vector<int>& getBoundaryIDsAxisSymmetric() const { return boundary_IDs_axis_symmetric; }

	const std::vector<int>& getBoundaryIDsPlanar() const {return boundary_IDs_planar;}

	const Bounds& getBounds() const {return bounds;}

	const Vec3d& getCenter() const {return center_;}

	const std::vector<int>& getParticleIDs() const {return particle_IDs;}

	bool hasBoundary() {return cell_with_boundary;}

	void setCellWithBoundary() {cell_with_boundary = true;}
	void setCellWithoutBoundary() {cell_with_boundary = false;}

	static int getNx() {return Nx;}
	static int getNy() {return Ny;}
	static int getNz() {return Nz;}
	static void setNx(int _nx) {Nx = _nx;}
	static void setNy(int _ny) {Ny = _ny;}
	static void setNz(int _nz) {Nz = _nz;}

};



#endif /* CELL_H_ */
