#ifndef CELL_H_
#define CELL_H_

#include <vector>
#include "vec3d.h"
#include "cuda.h"

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

	Vec3d center;
	static Vec3d dX; // cell edge sizes

	std::vector<int> particle_IDs;
	//std::vector<int> boundary_IDs_planar;
	//std::vector<int> boundary_IDs_axis_symmetric;

	bool cell_with_boundary = false;
	bool cell_is_external = false;

public:
	//Cell() {};
	Cell(const Vec3d &center);

	static void connectScene(Scene *scene) {
		Cell::scene = scene;
	}

	void clear();
	void populate(std::vector<Particle> &particles);
	void populateCuda(const Particle* device_particle_ptr, int N);
	bool contains(const Particle&) const;
	__device__
	bool containsCuda(const Particle*) const;
	template<typename Boundary_T>
	bool containsBoundary(const Boundary_T&) const;
	void addParticle(const Particle&);
	// functionality is different from the CPU version!
	// TODO: reflect it in name, consider making it a free function
	__device__
	void addParticleCuda(	const Particle *p,
							int *particle_IDs_in_cell,
							int *particle_ID_counter
						);
	__device__
	void addParticleCudaMultiCell(	const Particle *p,
							const int *IN_number_of_particleIDs, // for each cell
							int *OUT_particle_IDs_in_cells, // particles of all cells
							int cell_index,
							int *OUT_particle_ID_counter
						  );

	void size() const;
	void draw2D() const;
/*
	const std::vector<int>& getBoundaryIDsAxisSymmetric() const {
		return boundary_IDs_axis_symmetric;
	}
	const std::vector<int>& getBoundaryIDsPlanar() const {
		return boundary_IDs_planar;
	}*/
	const Bounds& getBounds() const {
		return bounds;
	}
//	__host__ __device__
	const Vec3d& getCenter() const {
		return center;
	}

	// they are not stored, because they are needed only during cell classification after creation:
	pointData getCorners() const;
	pointData getFaceCenters() const;
	pointData getEdgeCenters() const;

	pointData getAllPoints() const;

	__host__ __device__
	const std::vector<int>& cGetParticleIDs() const {
		return particle_IDs;
	}
	__host__ __device__
	std::vector<int>& getParticleIDs() {
		// creating a non-const getter via overloading the const getter
		return const_cast<std::vector<int>&>( // casting away return value constantness
				const_cast<const Cell*>(this) // casting away object constantness
				->cGetParticleIDs() );
	}

	bool hasBoundary() const {
		return cell_with_boundary;
	}

	inline bool isExternal() const {
		return cell_is_external;
	}

	inline bool isInternal() const {
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
	static Vec3d getDX() {
		return Cell::dX;
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
	static void setDX(Vec3d dX) {
		Cell::dX = dX;
	}
	static float getHalfDiagonal() {
		return 	abs(0.5 * Cell::dX);
	}

	static Vec3d average(const pointData& pd);

};

#endif /* CELL_H_ */
