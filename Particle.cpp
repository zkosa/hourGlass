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
	return vel * vel * (mass/2);
}

double Particle::potential_energy() {
	return mass * gravity * pos.y;
}

double Particle::energy() {
	return kinetic_energy() + potential_energy();
}

Vec3d Particle::apply_forces(){
	double density_medium = 0.001;
    Vec3d grav_acc = Vec3d{0.0, -gravity, 0.0 }; // 9.81m/s^2 down in the Z-axis
    Vec3d drag_force = 0.5 * density_medium * CdA * (vel * abs(vel)); // D = 0.5 * (rho * C * Area * vel^2)
    Vec3d drag_acc = drag_force / mass; // a = F/m

    return grav_acc - drag_acc;
}

void Particle::info() {
	std::cout << "---------------------------" << std::endl;
	std::cout << "pos: " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
	std::cout << "vel: " << vel.x << ", " << vel.y << ", " << vel.z << std::endl;
	std::cout << "acc: " << acc.x << ", " << acc.y << ", " << acc.z << std::endl;
	std::cout << "energy: " << energy() << std::endl;
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

void Particle::bounce_back(Boundary_planar ground) {
	vel = vel - 2*(vel*ground.getNormal())*ground.getNormal(); // revert the surface normal component of velocity
}
