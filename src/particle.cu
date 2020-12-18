#include "particle.h"
#include "cuda.h"

CUDA_HOSTDEV
float Particle::volume() const {
	return radius * radius * radius * pi * 4.0 / 3.0;
}
CUDA_HOSTDEV
float Particle::mass() const {
	return volume() * density;
}
CUDA_HOSTDEV
float Particle::A() const {
	return radius * radius * pi;
}
CUDA_HOSTDEV
float Particle::CdA() const {
	return 0.5 * A();  // TODO: use getCd()
}
CUDA_HOSTDEV
float Particle::CoR() const {
	return Particle::getRestitutionCoeff();
}

CUDA_HOSTDEV
Vec3d Particle::apply_forces() {
	const Vec3d drag_force = -0.5 * density_medium * CdA() * (vel * abs(vel));
	const Vec3d drag_acc = drag_force / mass();

	return Vec3d(0.0f, -g, 0.0f) + drag_acc; // TODO: readd gravity
}

CUDA_HOSTDEV
void Particle::advance(float dt) {
	// velocity Verlet integration:
	const Vec3d new_pos = pos + vel * dt + acc * dt * dt * 0.5;
	const Vec3d new_acc = apply_forces();
	const Vec3d new_vel = vel + 0.5 * (acc + new_acc) * dt;

	pos = new_pos;
	vel = new_vel;
	acc = new_acc;
}

__global__
void particles_advance(float dt, Particle *particles, int number_of_particles) {
	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < number_of_particles;
		i += blockDim.x * gridDim.x)
	{
		particles[i].advance(dt);
	}
}
