#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <iostream>
#include "Vec3d.h"
#include "Constants.h"
#include "Boundary_planar.h"
#include <GLFW/glfw3.h>  // version 3.3 is installed to Dependencies


class Particle {
private:
	Vec3d pos {0.0, 1.0, 0.0};
	Vec3d vel {0.0, 0.0, 0.0};
	Vec3d old_pos {0.0, 1.0, 0.0};
	Vec3d old_vel {0.0, 0.0, 0.0};

	std::string last_collision = "";

	Vec3d acc = gravity;
	GLFWwindow* window =0;

	const double Cd = 0.5;
	const double density = 2700; // kg/m3
	double radius = 0.05;
	double volume() const {return radius*radius*radius * pi * 4.0 / 3.0;}
	double mass() const {return volume()*density;}
	double A() const {return radius*radius * pi;}
	double CdA() const {return Cd * A();}

	Vec3d apply_forces();

public:
	Particle();
	~Particle();

	void update(double dt);
	double kinetic_energy();
	double potential_energy();
	double energy();
	Vec3d impulse();

	void info();
	void draw2D();
	void drawNow2D();

	double distance(class Particle& other);

	void collide_wall(class Boundary_planar& ground);
	void collide_particle(class Particle& other);
	Vec3d findPlace(class Particle& other);

	void setX(double _x) {pos.x = _x;}
	void setPos(Vec3d _pos) {pos = _pos;}
	void setR(double _r) {radius = _r;}
	void setV(Vec3d _v) {vel = _v;}
	void setWindow(GLFWwindow* _window) {window = _window;}

	double getX() const {return pos.x;}
	double getY() const {return pos.y;}
	double getZ() const {return pos.z;}
	double getR() const {return radius;}
	double getM() const {return mass();}
	Vec3d getV() const {return vel;}
	Vec3d getPos() const {return pos;}

	void debug() const;
};

#endif /* PARTICLE_H_ */
