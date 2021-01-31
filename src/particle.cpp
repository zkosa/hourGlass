#include <iostream>
#include "particle.h"
#include "scene.h"
#include <QOpenGLWidget>
//#include <device_launch_parameters.h> // just for proper indexing, nvcc includes it anyhow

// static members:

int Particle::last_ID = -1;

Vec3d Particle::force_field = GRAVITY;

Scene *Particle::scene = nullptr;

float Particle::uniform_radius = 0.005f;

CUDA_HOSTDEV
void Particle::setCd(const float _drag_coefficient) {
	static_container::Particle::drag_coefficient_global = _drag_coefficient;
}
CUDA_HOSTDEV
void Particle::setRestitutionCoefficient(const float restitution_coefficient) {
	static_container::Particle::restitution_coeff_global = restitution_coefficient;
}
CUDA_HOSTDEV
float Particle::getCd() {
	return static_container::Particle::drag_coefficient_global;
}

float Particle::kineticEnergy() const {
	return vel * vel * (mass() / 2);
}

float Particle::potentialEnergy() const {
	//return mass * (gravity * (pos + Vec3d(0,1,0)));
	return mass() * g * (pos.y + 1);
}

float Particle::energy() const {
	return kineticEnergy() + potentialEnergy();
}

Vec3d Particle::impulse() const {
	return mass() * vel;
}

void Particle::info() const {
	std::cout << "---------------------------" << std::endl;
	std::cout << "ID: " << ID << std::endl;
	std::cout << "pos: "; pos.print();
	std::cout << "vel: "; vel.print();
	std::cout << "acc: "; acc.print();
	std::cout << "energy: " << energy() << " = "
			  << potentialEnergy() << " + " << kineticEnergy() << std::endl;
}

void Particle::draw2D() const {
	const GLfloat display_radius = radius; //radius
	constexpr GLfloat twicePi = 2.0f * pi;

	int number_of_triangles; //# of triangles used to draw circle
	if (display_radius < 0.002) {
		number_of_triangles = 6;
	} else if (display_radius < 0.01) {
		number_of_triangles = 10;
	} else if (display_radius < 0.05) {
		number_of_triangles = 15;
	} else {
		number_of_triangles = 20;
	}

	glColor4f(0.86, 0.72, 0.39, 1); // "sand"
	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(pos.x, pos.y); // center of circle
	for (int i = 0; i <= number_of_triangles; i++) {
		glVertex2f(
				pos.x
						+ (display_radius
								* cos(i * twicePi / number_of_triangles)),
				pos.y
						+ (display_radius
								* sin(i * twicePi / number_of_triangles)));
	}
	glEnd();
	glColor4f(1, 1, 1, 1); // reset color
}

template<typename Boundary_T>
void Particle::collideToWall(const Boundary_T &wall) {

	const Vec3d n = wall.getNormal(*this);

	Vec3d pos_corr { 0, 0, 0 };
	if (std::abs(n * vel) > SMALL && wall.isPlanar()) { // not parallel, and moving
		// Move outwards along the incoming velocity vector so,
		// that the normal correction component equals to the overlap,
		// This doesn't ensure overlap-less corrected position for curved surfaces,
		// so it is performed only for planar boundaries
		pos_corr = (radius - wall.distanceSigned(*this)) / std::abs(n * vel) * vel * (-1);
	} else {
		// If there is no wall normal movement,
		// move in surface normal direction to the touching position
		pos_corr = (radius - wall.distanceSigned(*this)) * n;
	}

	// move back to the position when it touched the boundary:
	this->move(pos_corr);

	// correct the velocity to conserve energy (dissipation work is not considered!)
	correctVelocity(pos_corr);

	// revert the wall normal velocity component
	vel = vel - (1 + Particle::getRestitutionCoeff()) * (vel * n) * n;
}

template void Particle::collideToWall<Boundary_axissymmetric>(const Boundary_axissymmetric&);
template void Particle::collideToWall<Boundary_planar>(const Boundary_planar&);

void Particle::collideToParticle(Particle &other) {

	Vec3d n = other.pos - this->pos; // distance vector, pointing towards the other particle

	const float distance = abs(n);

	// do not do anything with distant particles:
	if (distance > this->getR() + other.getR()) {
		return;
	}

	n = norm(n); // normalize

	// move back to the positions where they just touched the other:
	const Vec3d pos_corr = -0.5 * n * (this->getR() + other.getR() - distance);
	this->move(pos_corr);
	other.move(-pos_corr);

	correctVelocity(pos_corr);
	other.correctVelocity(-pos_corr);

	exchangeImpulse(other);
}

void Particle::collideToParticle_checkBoundary(Particle &other) {

	Vec3d n = other.pos - this->pos; // distance vector, pointing towards the other particle

	const float distance = abs(n);

	// do not do anything with distant particles:
	if (distance > this->getR() + other.getR()) {
		return;
	}

	n = norm(n); // normalize

	// necessary correction to reach touching position
	Vec3d pos_corr = -0.5 * n * (this->getR() + other.getR() - distance);

	// create temporary particles with the planned correction, for overlap checking
	Particle tmp_particle(*this);
	tmp_particle.setPos(pos + pos_corr);
	const bool this_overlaps = tmp_particle.overlapWithWalls();

	Particle tmp_other_particle(other);
	tmp_other_particle.setPos(other.getPos() - pos_corr);
	const bool other_overlaps = tmp_other_particle.overlapWithWalls();

	if (!this_overlaps && !other_overlaps) {
		// apply the correction to both!
		// (move back to the positions where they touched each other)
		this->move(pos_corr);
		other.move(-pos_corr);
	} else if (this_overlaps && !other_overlaps) {
		// move both particles also with the value of the overlap of the overlapping particle
		// mutual overlap of the particles becomes resolved, but
		// the non-overlapping particle can become overlapping with a wall in narrow channels and corners!
		const Vec3d overlap = tmp_particle.overlapVectorWithWalls(); // prospected overlap after correction with pos_corr
		this->move(pos_corr - overlap);
		other.move(-pos_corr - overlap);
	} else if (other_overlaps && !this_overlaps) {
		// move both particles also with the value of the overlap of the overlapping particle
		// mutual overlap of the particles becomes resolved, but
		// the non-overlapping particle can become overlapping with a wall in narrow channels and corners
		const Vec3d overlap = tmp_other_particle.overlapVectorWithWalls(); // prospected overlap after correction with pos_corr
		this->move(pos_corr - overlap);
		other.move(-pos_corr - overlap);
	} else { // both particles overlap with walls
		// move both particles also with the value of their prospected overlap with the all
		// mutual overlap is not necessarily resolved
		this->move(pos_corr - this->overlapVectorWithWalls());
		other.move(-pos_corr - other.overlapVectorWithWalls());
	}

	correctVelocity(pos_corr); // TODO: use the actually used one! (0/1/2)
	other.correctVelocity(-pos_corr);

	// perform the actual collision after the positions and velocities are corrected to touching position
	exchangeImpulse(other);
}

void Particle::exchangeImpulse(Particle &other) {

	Vec3d n = other.pos - this->pos; // distance vector, pointing towards the other particle
	n = norm(n); // normalize

	const Vec3d vel_old = vel; // store it for the other particle
	vel = vel_old - n * (n * vel_old)
			+ (mass() - other.mass()) / (mass() + other.mass()) * n
					* (vel_old * n)
			+ 2 * other.mass() / (mass() + other.mass()) * n
					* (other.getV() * n);

	other.setV(
			other.getV() - n * (other.getV() * n)
					+ 2 * mass() / (other.getM() + mass()) * n * (vel_old * n)
					+ (other.mass() - mass()) / (other.mass() + mass()) * n
							* (other.getV() * n));
}

template<typename Boundary_T>
bool Particle::overlapWithWall(const Boundary_T &wall) const {
	if (wall.distance(*this) - radius < 0) {
		return true;
	} else {
		return false;
	}
}

template bool Particle::overlapWithWall<Boundary_axissymmetric>(const Boundary_axissymmetric&) const;

bool Particle::overlapWithWalls() const {
	// TODO return pointers to the actually overlapped walls?
	for (const auto &b : scene->getBoundariesPlanar()) {
		if (overlapWithWall(b)) {
			return true;
		}
	}
	for (const auto &b : scene->getBoundariesAxiSym()) {
		if (overlapWithWall(b)) {
			return true;
		}
	}
	return false;
}

template<typename Boundary_T>
Vec3d Particle::overlapVectorWithWall(const Boundary_T &wall) {
	// TODO: simplify
	return (wall.distance(*this) - radius) * wall.getNormal(*this);
}

template Vec3d Particle::overlapVectorWithWall<Boundary_axissymmetric>(const Boundary_axissymmetric&);
template Vec3d Particle::overlapVectorWithWall<Boundary_planar>(const Boundary_planar&);

Vec3d Particle::overlapVectorWithWalls() {
	// returns the overlapVector to the first boundary that is overlapped
	// start with planar boundaries, because those are cheaper to be checked

	for(auto b: scene->getBoundariesPlanar()) {
		if (overlapWithWall(b)) {
			return overlapVectorWithWall(b);
		}
	}
	for(auto b: scene->getBoundariesAxiSym()) {
		if (overlapWithWall(b)) {
			return overlapVectorWithWall(b);
		}
	}

	// if no overlapping boundary has been found:
	return Vec3d::null;
}

void Particle::size() const {
	std::cout << "Size of particle object: " << sizeof(*this) << std::endl;
}

float Particle::terminalVelocity() const {
	// equilibrium velocity, where drag cancels the gravitation force
	return std::sqrt(2 * mass() * abs(GRAVITY) / CdA() / density_medium);
}

float Particle::maxFreeFallVelocity() const {
	// The velocity which can be reached by gravitational acceleration within the domain.
	// domain height along the gravity vector:
	const float h = std::abs(scene->getBoundingBox().diagonal() * norm(GRAVITY));
	return std::sqrt(2 * abs(GRAVITY) * h);
}

float Particle::maxVelocity() const {
	return std::min(terminalVelocity(), maxFreeFallVelocity());
}

float Particle::timeStepLimit() const {
	// the traveled distance during one time step is a radius length:
	return radius / maxVelocity();
}
