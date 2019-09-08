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
	Vec3d acc = gravity;

	const double Cd = 0;
	const double density = 3.0;
	double radius = 0.005;
	double volume() {return radius*radius*radius * pi * 4.0 / 3.0;}
	double mass() {return volume()*density;}
	double A() {return radius*radius * pi;}
	double CdA() {return Cd * A();}

	Vec3d apply_forces();

public:
	Particle();
	~Particle();

	void update(double dt);
	double kinetic_energy();
	double potential_energy();
	double energy();
	void info();
	void draw2D();

	void bounce_back(class Boundary_planar ground);

	void setX(double _x) { pos.x = _x;}
	void setR(double _r) { radius = _r;}

	double getX() const {return pos.x;}
	double getY() const {return pos.y;}
	double getZ() const {return pos.z;}
	double getR() const {return radius;}
	Vec3d getPos() const {return pos;}
};

#endif /* PARTICLE_H_ */
