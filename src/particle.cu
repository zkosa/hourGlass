#include "particle.h"
#include "boundary_axissymmetric.h"
#include "boundary_planar.h"
#include "cuda.h"

__device__
float static_container::Particle::drag_coefficient_global = 0.5;
__device__
float static_container::Particle::restitution_coeff_global = 0.5;

__device__
float Particle::getRestitutionCoeff() {
	return static_container::Particle::restitution_coeff_global;
}
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

template<typename Boundary_T>
__device__
void Particle::collideToWall(const Boundary_T *wall) {

	const Vec3d n = wall->getNormal(this);

	Vec3d pos_corr { 0, 0, 0 };
	if (std::abs(n * vel) > SMALL && wall->isPlanar()) { // not parallel, and moving
		// Move outwards along the incoming velocity vector so,
		// that the normal correction component equals to the overlap,
		// This doesn't ensure overlap-less corrected position for curved surfaces,
		// so it is performed only for planar boundaries
		pos_corr = (radius - wall->distanceSigned(this)) / std::abs(n * vel) * vel * (-1);
	} else {
		// If there is no wall normal movement,
		// move in surface normal direction to the touching position
		pos_corr = (radius - wall->distanceSigned(this)) * n;
	}

	// move back to the position when it touched the boundary:
	this->move(pos_corr);

	// correct the velocity to conserve energy (dissipation work is not considered!)
	correctVelocity(pos_corr);

	// revert the wall normal velocity component
	vel = vel - (1 + Particle::getRestitutionCoeff()) * (vel * n) * n;
}

// explicitly instantiating the template instances
template __device__ void Particle::collideToWall<Boundary_axissymmetric>(const Boundary_axissymmetric*);
template __device__ void Particle::collideToWall<Boundary_planar>(const Boundary_planar*);

__device__
void Particle::collideToParticle(Particle *other) {
	Vec3d n = other->pos - this->pos; // distance vector, pointing towards the other particle

	const float distance = abs(n);

	// do not do anything with distant particles:
	if (distance > this->getR() + other->getR()) {
		return;
	}

	n = norm(n); // normalize

	// move back to the positions where they just touched the other:
	const Vec3d pos_corr = -0.5 * n * (this->getR() + other->getR() - distance);
	this->move(pos_corr);
	other->move(-pos_corr);

	correctVelocity(pos_corr);
	other->correctVelocity(-pos_corr);

	exchangeImpulse(other);
}

__host__ __device__
void Particle::correctVelocity(const Vec3d &pos_corr) {
	// correct the velocity to conserve energy (dissipation work is not considered!)
	if (vel * vel + 2 * Vec3d(0.0f, -g, 0.0f) * pos_corr >= 0.0) {
		vel = std::sqrt(vel * vel + 2 * Vec3d(0.0f, -g, 0.0f) * pos_corr) * norm(vel);
	} else {
		vel = -std::sqrt(-(vel * vel + 2 * Vec3d(0.0f, -g, 0.0f) * pos_corr)) * norm(vel);
	}
}

__device__
void Particle::exchangeImpulse(Particle *other) {
	Vec3d n = other->pos - this->pos; // distance vector, pointing towards the other particle
	n = norm(n); // normalize

	const Vec3d vel_old = vel; // store it for the other particle
	vel = vel_old - n * (n * vel_old)
			+ (mass() - other->mass()) / (mass() + other->mass()) * n
					* (vel_old * n)
			+ 2 * other->mass() / (mass() + other->mass()) * n
					* (other->getV() * n);

	other->setV(
			other->getV() - n * (other->getV() * n)
					+ 2 * mass() / (other->getM() + mass()) * n * (vel_old * n)
					+ (other->mass() - mass()) / (other->mass() + mass()) * n
							* (other->getV() * n));
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
