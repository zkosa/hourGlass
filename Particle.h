#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <iostream>
#include "Vec3d.h"

const double gravity = 9.81;
const double pi = 3.14159;

class Particle {
private:
	Vec3d pos {0.0, 1.0, 0.0};
	Vec3d vel {0.0, 0.0, 0.0};
	Vec3d acc {0.0, -gravity, 0.0};

	const double density = 3.0;
	const double radius = 0.001;
	const double volume = radius*radius*radius * pi * 4.0 / 3.0;
	const double mass = volume*density;
	const double A = radius*radius * pi;
	const double Cd = 0.5;
	const double CdA = Cd * A;

	Vec3d apply_forces();

public:
	Particle();
	~Particle();

	void update(double dt);
	double kinetic_energy();
	double potential_energy();
	double energy();
	void info();

	double getX() {return pos.x;}
	double getY() {return pos.y;}
	double getZ() {return pos.z;}
};

#endif /* PARTICLE_H_ */
