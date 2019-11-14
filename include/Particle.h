#ifndef PARTICLE_H_
#define PARTICLE_H_

#include <iostream>
#include "Vec3d.h"
#include "Constants.h"
//#include "Boundary_planar.h"
#include <GLFW/glfw3.h>  // version 3.3 is installed to Dependencies

class Boundary;
class Boundary_planar;
class Boundary_axis_symmetric;
class Scene;

class Particle {
private:

	Vec3d pos {0.0, 1.0, 0.0};
	Vec3d vel {0.0, 0.0, 0.0};
	Vec3d old_pos {0.0, 1.0, 0.0};
	Vec3d old_vel {0.0, 0.0, 0.0};

	int ID = -1;

	std::string last_collision = "";

	static Vec3d acc;
	static Scene* scene;

	static constexpr double density = 2700; // kg/m3
	static constexpr double density_medium = 1; // air kg/m3
	static constexpr double restitution_coeff = 0.5; // now used only between walls and particles
	static double Cd; // non-const can not be initialized in the declaration
	static double uniform_radius;
	double radius = uniform_radius;
	double volume() const {return radius*radius*radius * pi * 4.0 / 3.0;}
	double mass() const {return volume()*density;}
	double A() const {return radius*radius * pi;}
	double CdA() const {return Cd * A();}
	double CoR() const {return restitution_coeff;}
	// particle is in a cell which contains a boundary, therefore it must be considered for collision to boundary
	bool check_boundary = true;

	Vec3d apply_forces();

public:
	Particle();
	Particle(Vec3d _pos, double _r=Particle::uniform_radius) : pos(_pos), radius(_r) {};
	Particle(Vec3d _pos, Vec3d _vel, double _r=Particle::uniform_radius) : pos(_pos), vel(_vel), radius(_r) {};
	Particle(Vec3d _pos, Vec3d _vel, int _ID, double _r=Particle::uniform_radius) : pos(_pos), vel(_vel), ID(_ID), radius(_r) {};
	~Particle();

	void advance(double dt);
	double kinetic_energy();
	double potential_energy();
	double energy();
	Vec3d impulse();

	void info();
	void draw2D();
	void drawNow2D();

	double distance(const Particle& other) const;

	void collide_wall(Boundary& wall);
	void collide_particle(Particle& other);
	void collide_particle(Particle& other, Boundary_planar& b_pl, Boundary_axis_symmetric& b_ax);
	bool overlap_wall(const Boundary& wall);
	bool overlap_walls();
	Vec3d overlapVect_wall(const Boundary& wall);

	Vec3d findPlace(Particle& other);

	void setX(double x) {this->pos.x = x;}
	void setPos(Vec3d pos) {this->pos = pos;}
	void setR(double r) {this->radius = r;}
	void setV(Vec3d v) {this->vel = v;}
	void connectScene(Scene* scene) {this->scene = scene;}
	void setCheckBoundary(bool is_near) {check_boundary = is_near;};

	static void setCd(const double _Cd) {Cd = _Cd;}
	static void setUniformRadius(double _uniform_radius) {uniform_radius = _uniform_radius;}

	int getID() const {return ID;}
	double getX() const {return pos.x;}
	double getY() const {return pos.y;}
	double getZ() const {return pos.z;}
	double getR() const {return radius;}
	double getM() const {return mass();}
	Vec3d getV() const {return vel;}
	Vec3d getPos() const {return pos;}

	// static getters can not be qualified as const according to the standard
	// (they do not modify any instance of the class)
	static double getCd() {return Cd;}
	static double energy_of_all_particles_combined();
	static double getUniformRadius() {return uniform_radius;}

};

#endif /* PARTICLE_H_ */
