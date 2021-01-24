#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "vec3d.h"
#include "constants.h"
#include "cuda.h"

//class Boundary;
//class Boundary_axissymmetric;
class Scene;

// use global, "namespaced" variables to mimic static class member variables on the GPU device:
namespace static_container {
namespace Particle {
	__device__
	extern float drag_coefficient_global;
	__device__
	extern float restitution_coeff_global;
}
}


class Particle {
private:

	Vec3d pos { 0.0, 1.0, 0.0 };
	Vec3d vel { 0.0, 0.0, 0.0 };
	Vec3d acc = gravity;

	int ID = -1;

	static Vec3d force_field;
	static Scene *scene;

	static int last_ID;
	static constexpr float density = 2700.0; // kg/m3
	static constexpr float density_medium = 1.0; // air kg/m3
	static float uniform_radius;
	float radius = uniform_radius;

	CUDA_HOSTDEV
	float volume() const;
	CUDA_HOSTDEV
	float mass() const;
	CUDA_HOSTDEV
	float A() const;
	CUDA_HOSTDEV
	float CdA() const;
	CUDA_HOSTDEV
	float CoR() const;

	CUDA_HOSTDEV
	Vec3d apply_forces();

public:
//	CUDA_HOSTDEV
//	Particle() = default;
	CUDA_HOSTDEV
	Particle(Vec3d _pos, float _r = Particle::uniform_radius) :
			pos(_pos), radius(_r) {
	}
	CUDA_HOSTDEV
	Particle(Vec3d _pos, Vec3d _vel, float _r = Particle::uniform_radius) :
			pos(_pos), vel(_vel), radius(_r) {
	}
	CUDA_HOSTDEV
	Particle(Vec3d _pos, Vec3d _vel, int _ID, float _r =
			Particle::uniform_radius) :
			pos(_pos), vel(_vel), ID(_ID), radius(_r) {
	}

	CUDA_HOSTDEV
	void advance(float dt);
	CUDA_HOSTDEV
	inline void move(const Vec3d &movement) {
		pos += movement;
	}
	float kineticEnergy() const;
	float potentialEnergy() const;
	float energy() const;
	Vec3d impulse() const;

	void info() const;
	void draw2D() const;
	void drawNow2D() const;

	inline float distance(const Particle &other) const {
		return abs(pos - other.pos);
	}

	template<typename Boundary_T>
	void collideToWall(const Boundary_T &wall);
	template<typename Boundary_T>
	__device__
	void collideToWall(const Boundary_T *wall);
	void collideToParticle(Particle &other);
	void collideToParticle_checkBoundary(Particle &other);
	CUDA_HOSTDEV
	void correctVelocity(const Vec3d &position_correction);
	void exchangeImpulse(Particle &other);
	template<typename Boundary_T>
	bool overlapWithWall(const Boundary_T &wall) const;
	bool overlapWithWalls() const;
	template<typename Boundary_T>
	Vec3d overlapVectorWithWall(const Boundary_T &wall);
	Vec3d overlapVectorWithWalls();
	void size() const;

	void setID(int ID) {
		this->ID = ID;
	}
	void setX(float x) {
		this->pos.x = x;
	}
	void setPos(Vec3d pos) {
		this->pos = pos;
	}
	void setR(float r) {
		this->radius = r;
	}
	void setV(Vec3d v) {
		this->vel = v;
	}
	static void connectScene(Scene *scene) {
		Particle::scene = scene;
	}

	CUDA_HOSTDEV
	static void setCd(const float _drag_coefficient);
	CUDA_HOSTDEV
	static void setRestitutionCoefficient(const float restitution_coefficient);

	static void setUniformRadius(float _uniform_radius) {
		uniform_radius = _uniform_radius;
	}
	static void resetLastID() {
		last_ID = -1;
	}
	static void incrementLastID() {
		last_ID += 1;
	}

	CUDA_HOSTDEV
	int getID() const {
		return ID;
	}
	CUDA_HOSTDEV
	float getX() const {
		return pos.x;
	}
	CUDA_HOSTDEV
	float getY() const {
		return pos.y;
	}
	CUDA_HOSTDEV
	float getZ() const {
		return pos.z;
	}
	CUDA_HOSTDEV
	float getR() const {
		return radius;
	}
	float getM() const {
		return mass();
	}
	Vec3d getV() const {
		return vel;
	}
	CUDA_HOSTDEV
	Vec3d getPos() const {
		return pos;
	}
	CUDA_HOSTDEV
	const Vec3d* cGetPos() const {
		return &pos;
	}
	Vec3d getAcceleration() const {
		return acc;
	}
	float terminalVelocity() const;
	float maxFreeFallVelocity() const; // in the domain, no drag
	float maxVelocity() const;
	float timeStepLimit() const;

	// static getters can not be qualified as const according to the standard
	// (they do not modify any instance of the class)
	CUDA_HOSTDEV
	static float getCd();
	CUDA_HOSTDEV
	static float getRestitutionCoeff();

	static float getUniformRadius() {
		return uniform_radius;
	}
	static int getLastID() {
		return last_ID;
	}

};

inline bool operator==(const Particle& a, const Particle& b) {
	return (
			a.getPos() == b.getPos() &&
			a.getV() == b.getV() &&
			a.getAcceleration() == b.getAcceleration() &&
			a.getID() == b.getID() &&
			a.getR() == b.getR()
			);
}

__global__
void particles_advance(float dt, Particle *particles, int number_of_particles);

#endif /* PARTICLE_H_ */
