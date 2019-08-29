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

Vec3d Particle::apply_forces(){
    Vec3d grav_acc = Vec3d{0.0, 0.0, -9.81 }; // 9.81m/s^2 down in the Z-axis
    Vec3d drag_force = 0.5 * density * CdA * (vel * abs(vel)); // D = 0.5 * (rho * C * Area * vel^2)
    Vec3d drag_acc = drag_force / mass; // a = F/m

    return grav_acc - drag_acc;
}

void Particle::info() {
	std::cout << "---------------------------" << std::endl;
	std::cout << "pos: " << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
	std::cout << "vel: " << vel.x << ", " << vel.y << ", " << vel.z << std::endl;
	std::cout << "acc: " << acc.x << ", " << acc.y << ", " << acc.z << std::endl;
}


