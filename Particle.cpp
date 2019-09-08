#include "Particle.h"


Particle::Particle(){};

Particle::~Particle(){};

void Particle::update(double dt) {
	// velocity Verlet integration

    Vec3d new_pos = pos + vel*dt + acc*(dt*dt*0.5);
    Vec3d new_acc = apply_forces();
    Vec3d new_vel = vel + (acc+new_acc)*(dt*0.5);

    pos = new_pos;
    vel = new_vel;
    acc = new_acc;
}

double Particle::kinetic_energy() {
	return vel * vel * (mass()/2);
}

double Particle::potential_energy() {
	//return mass * (gravity * (pos + Vec3d(0,1,0)));
	return mass() * g * (pos.y + 1);
}

double Particle::energy() {
	return kinetic_energy() + potential_energy();
}

Vec3d Particle::apply_forces(){
	double density_medium = 0.001;
    Vec3d grav_acc = gravity;
    Vec3d drag_force = 0.5 * density_medium * CdA() * (vel * abs(vel)); // D = 0.5 * (rho * C * Area * vel^2)
    Vec3d drag_acc = drag_force / mass(); // a = F/m

    return grav_acc - drag_acc;
}

void Particle::info() {
	std::cout << "---------------------------" << std::endl;
	std::cout << "pos: " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
	std::cout << "vel: " << vel.x << ", " << vel.y << ", " << vel.z << std::endl;
	std::cout << "acc: " << acc.x << ", " << acc.y << ", " << acc.z << std::endl;
	std::cout << "energy: " << energy() << "\t= "<< potential_energy() << "\t+ " << kinetic_energy() << std::endl;
}

void Particle::draw2D() {
	int triangleAmount = 20; //# of triangles used to draw circle

	GLfloat display_radius = radius; //radius
	GLfloat twicePi = 2.0f * pi;

	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(pos.x, pos.y); // center of circle
	for(int i = 0; i <= triangleAmount; i++) {
		glVertex2f(
				pos.x + (display_radius * cos(i *  twicePi / triangleAmount)),
				pos.y + (display_radius * sin(i * twicePi / triangleAmount))
		);
	}
	glEnd();
}

void Particle::bounce_back(Boundary_planar wall) {

	Vec3d n = wall.getNormal();

	Vec3d pos_corr {0,0,0};
	if ( abs(n*vel) > SMALL) // not parallel, and moving
		pos_corr = (radius - wall.distance(*this)) / abs(n*vel) * vel *(-1); // move along the OLD! velocity vector
	else {
		pos_corr = (radius - wall.distance(*this)) * n; // move in surface normal direction
	}

	// move back to the position when it touched the boundary:
	pos = pos + pos_corr;

	// correct the velocity to conserve energy (dissipation work is not considered!)
	vel = std::sqrt(vel*vel + 2*gravity*pos_corr)  * norm(vel);

	// revert the wall normal velocity component
	vel = vel - 2*(vel*n)*n;
}
