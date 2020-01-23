#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "Vec3d.h"
#include "Constants.h"

class Boundary;
class Boundary_planar;
class Boundary_axis_symmetric;
class Scene;

class Particle {
private:

	Vec3d pos { 0.0, 1.0, 0.0 };
	Vec3d vel { 0.0, 0.0, 0.0 };

	int ID = -1;

	static Vec3d force_field;
	static Scene *scene;

	static constexpr float density = 2700.; // kg/m3
	static constexpr float density_medium = 1.; // air kg/m3
	static constexpr float restitution_coeff = 0.5;
	static float drag_coefficient; // non-const can not be initialized in the declaration
	static float uniform_radius;
	float radius = uniform_radius;
	float volume() const {
		return radius * radius * radius * pi * 4.0 / 3.0;
	}
	float mass() const {
		return volume() * density;
	}
	float A() const {
		return radius * radius * pi;
	}
	float CdA() const {
		return drag_coefficient * A();
	}
	float CoR() const {
		return restitution_coeff;
	}

	Vec3d apply_forces();

public:
	Particle();
	Particle(Vec3d _pos, float _r = Particle::uniform_radius) :
			pos(_pos), radius(_r) {
	}
	Particle(Vec3d _pos, Vec3d _vel, float _r = Particle::uniform_radius) :
			pos(_pos), vel(_vel), radius(_r) {
	}
	Particle(Vec3d _pos, Vec3d _vel, int _ID, float _r =
			Particle::uniform_radius) :
			pos(_pos), vel(_vel), ID(_ID), radius(_r) {
	}
	~Particle();

	void advance(float dt);
	float kineticEnergy();
	float potentialEnergy();
	float energy();
	Vec3d impulse();

	void info();
	void draw2D();
	void drawNow2D();

	inline float distance(const Particle &other) const {
		return abs(pos - other.pos);
	}

	void collideToWall(const Boundary &wall);
	void collideToParticle(Particle &other);
	void collideToParticle_checkBoundary(Particle &other);
	void correctVelocity(const Vec3d &position_correction);
	void exchangeImpulse(Particle &other);
	bool overlapWithWall(const Boundary &wall) const;
	bool overlapWithWalls() const;
	Vec3d overlapVectorWithWall(const Boundary &wall);
	void size() const;

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

	static void setCd(const float _drag_coefficient) {
		drag_coefficient = _drag_coefficient;
	}
	static void setUniformRadius(float _uniform_radius) {
		uniform_radius = _uniform_radius;
	}

	int getID() const {
		return ID;
	}
	float getX() const {
		return pos.x;
	}
	float getY() const {
		return pos.y;
	}
	float getZ() const {
		return pos.z;
	}
	float getR() const {
		return radius;
	}
	float getM() const {
		return mass();
	}
	Vec3d getV() const {
		return vel;
	}
	Vec3d getPos() const {
		return pos;
	}
	float terminalVelocity();
	float maxFreeFallVelocity(); // in the domain, no drag
	float maxVelocity();
	float timeStepLimit();

	// static getters can not be qualified as const according to the standard
	// (they do not modify any instance of the class)
	static float getCd() {
		return drag_coefficient;
	}
	static float getUniformRadius() {
		return uniform_radius;
	}

};

#endif /* PARTICLE_H_ */
