#ifndef PARTICLE_H_
#define PARTICLE_H_

#include "vec3d.h"
#include "constants.h"

class Boundary;
class Scene;

class Particle {
private:

	Vec3d pos { 0.0f, 1.0f, 0.0f };
	Vec3d vel { 0.0f, 0.0f, 0.0f };
	Vec3d acc = gravity;

	int ID = -1;

	static Vec3d force_field;
	static Scene *scene;

	static int last_ID;
	static constexpr float density = 2700.0f; // kg/m3
	static constexpr float density_medium = 1.0f; // air kg/m3
	static float restitution_coeff;
	static float drag_coefficient; // non-const can not be initialized in the declaration
	static float uniform_radius;
	float radius = uniform_radius;
	float volume() const {
		return radius * radius * radius * pi * 4.0f / 3.0f;
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

	void advance(float dt);
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

	void collideToWall(const Boundary &wall);
	void collideToParticle(Particle &other);
	void collideToParticle_checkBoundary(Particle &other);
	void correctVelocity(const Vec3d &position_correction);
	void exchangeImpulse(Particle &other);
	bool overlapWithWall(const Boundary &wall) const;
	bool overlapWithWalls() const;
	Vec3d overlapVectorWithWall(const Boundary &wall);
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

	static void setCd(const float _drag_coefficient) {
		drag_coefficient = _drag_coefficient;
	}
	static void setRestitutionCoefficient(const float restitution_coefficient) {
		restitution_coeff = restitution_coefficient;
	}
	static void setUniformRadius(float _uniform_radius) {
		uniform_radius = _uniform_radius;
	}
	static void resetLastID() {
		last_ID = -1;
	}
	static void incrementLastID() {
		last_ID += 1;
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
	Vec3d getAcceleration() const {
		return acc;
	}
	float terminalVelocity() const;
	float maxFreeFallVelocity() const; // in the domain, no drag
	float maxVelocity() const;
	float timeStepLimit() const;

	// static getters can not be qualified as const according to the standard
	// (they do not modify any instance of the class)
	static float getCd() {
		return drag_coefficient;
	}
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

#endif /* PARTICLE_H_ */
