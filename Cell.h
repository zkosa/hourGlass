#ifndef CELL_H_
#define CELL_H_

#include <vector>
#include "Vec3d.h"

class Scene;
class Particle;
class Boundary_planar;
class Boundary_axis_symmetric;

struct Bounds {
	double x; // lower bound coordinate
	double X; // upper bound coordinate
	double y;
	double Y;
	double z;
	double Z;
};

class Cell {

	Bounds bounds;

	Vec3d center_ = {0.5*(bounds.x+bounds.X), 0.5*(bounds.y+bounds.Y), 0.5*(bounds.z+bounds.Z)};

	std::vector<int> particle_IDs; // TODO reserve expected size
	std::vector<int> boundary_IDs_planar;
	std::vector<int> boundary_IDs_axis_symmetric;

public:
	Cell() {};
	Cell(const Vec3d& center, const Vec3d& dX);

	void init(const Scene&);
	void shrink();
	void update();

	void clear();
	void populate(std::vector<Particle>& particles);
	bool contains(const Particle&);
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
};



#endif /* CELL_H_ */
