#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <iostream>
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

	static constexpr double density = 2700.; // kg/m3
	static constexpr double density_medium = 1.; // air kg/m3
	static constexpr double restitution_coeff = 0.5;
	static double drag_coefficient; // non-const can not be initialized in the declaration
	static double uniform_radius;
	double radius = uniform_radius;
	double volume() const {
		return radius * radius * radius * pi * 4.0 / 3.0;
	}
	double mass() const {
		return volume() * density;
	}
	double A() const {
		return radius * radius * pi;
	}
	double CdA() const {
		return drag_coefficient * A();
	}
	double CoR() const {
		return restitution_coeff;
	}
	// particle is in a cell which contains a boundary, therefore it must be considered for collision to boundary
	bool check_boundary = true;

	Vec3d apply_forces();

public:
	Particle();
	Particle(Vec3d _pos, double _r = Particle::uniform_radius) :
			pos(_pos), radius(_r) {
	}
	Particle(Vec3d _pos, Vec3d _vel, double _r = Particle::uniform_radius) :
			pos(_pos), vel(_vel), radius(_r) {
	}
	Particle(Vec3d _pos, Vec3d _vel, int _ID, double _r =
			Particle::uniform_radius) :
			pos(_pos), vel(_vel), ID(_ID), radius(_r) {
	}
	~Particle();

	void advance(double dt);
	double kineticEnergy();
	double potentialEnergy();
	double energy();
	Vec3d impulse();

	void info();
	void draw2D();
	void drawNow2D();

	double distance(const Particle &other) const;

	void collideToWall(const Boundary &wall);
	void collideToParticle(Particle &other);
	void collideToParticle_checkBoundary(Particle &other);
	void correctVelocity(const Vec3d &position_correction);
	void exchangeImpulse(Particle &other);
	bool overlapWithWall(const Boundary &wall) const;
	bool overlapWithWalls() const;
	Vec3d overlapVectorWithWall(const Boundary &wall);
	void size() {
		std::cout << "Size of particle object: " << sizeof(*this) << std::endl;
	}

	void setX(double x) {
		this->pos.x = x;
	}
	void setPos(Vec3d pos) {
		this->pos = pos;
	}
	void setR(double r) {
		this->radius = r;
	}
	void setV(Vec3d v) {
		this->vel = v;
	}
	static void connectScene(Scene *scene) {
		Particle::scene = scene;
	}

	static void setCd(const double _drag_coefficient) {
		drag_coefficient = _drag_coefficient;
	}
	static void setUniformRadius(double _uniform_radius) {
		uniform_radius = _uniform_radius;
	}

	int getID() const {
		return ID;
	}
	double getX() const {
		return pos.x;
	}
	double getY() const {
		return pos.y;
	}
	double getZ() const {
		return pos.z;
	}
	double getR() const {
		return radius;
	}
	double getM() const {
		return mass();
	}
	Vec3d getV() const {
		return vel;
	}
	Vec3d getPos() const {
		return pos;
	}
	double terminalVelocity();
	double maxFreeFallVelocity(); // in the domain, no drag
	double maxVelocity();
	double timeStepLimit();

	// static getters can not be qualified as const according to the standard
	// (they do not modify any instance of the class)
	static double getCd() {
		return drag_coefficient;
	}
	static double getUniformRadius() {
		return uniform_radius;
	}

};

#endif /* PARTICLE_H_ */
