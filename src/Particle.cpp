#include "Particle.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"
#include "Scene.h"

Particle::Particle() {
}

Particle::~Particle() {
}

Vec3d Particle::acc = gravity;

Scene *Particle::scene = nullptr;

double Particle::Cd = 0.5; // non-constexpr static members must be initialized in the definition

double Particle::uniform_radius = 0.005;

void Particle::advance(double dt) {
	// velocity Verlet integration:
	Vec3d new_pos = pos + vel * dt + acc * (dt * dt * 0.5);
	Vec3d new_acc = apply_forces();
	Vec3d new_vel = vel + (acc + new_acc) * (dt * 0.5);

	pos = new_pos;
	vel = new_vel;
	acc = new_acc;
}

double Particle::kinetic_energy() {
	return vel * vel * (mass() / 2);
}

double Particle::potential_energy() {
	//return mass * (gravity * (pos + Vec3d(0,1,0)));
	return mass() * g * (pos.y + 1);
}

double Particle::energy() {
	return kinetic_energy() + potential_energy();
}

Vec3d Particle::impulse() {
	return mass() * vel;
}

Vec3d Particle::apply_forces() {
	Vec3d drag_force = -0.5 * density_medium * CdA() * (vel * abs(vel));
	Vec3d drag_acc = drag_force / mass(); // a = F/m

	return gravity + drag_acc;
}

void Particle::info() {
	std::cout << "---------------------------" << std::endl;
	std::cout << "pos: " << pos.x << ", " << pos.y << ", " << pos.z
			<< std::endl;
	std::cout << "vel: " << vel.x << ", " << vel.y << ", " << vel.z
			<< std::endl;
	std::cout << "acc: " << acc.x << ", " << acc.y << ", " << acc.z
			<< std::endl;
	std::cout << "energy: " << energy() << "\t= " << potential_energy()
			<< "\t+ " << kinetic_energy() << std::endl;
}

void Particle::draw2D() {
	GLfloat display_radius = radius; //radius
	GLfloat twicePi = 2.0f * pi;

	int triangleAmount; //# of triangles used to draw circle
	if (display_radius < 0.002) {
		triangleAmount = 6;
	} else if (display_radius < 0.01) {
		triangleAmount = 10;
	} else if (display_radius < 0.05) {
		triangleAmount = 15;
	} else {
		triangleAmount = 20;
	}

	glBegin(GL_TRIANGLE_FAN);
	glVertex2f(pos.x, pos.y); // center of circle
	for (int i = 0; i <= triangleAmount; i++) {
		glVertex2f(pos.x + (display_radius * cos(i * twicePi / triangleAmount)),
				pos.y + (display_radius * sin(i * twicePi / triangleAmount)));
	}
	glEnd();
}

double Particle::distance(const Particle &other) const {
	return abs(pos - other.pos);
}

void Particle::collide_wall(const Boundary &wall) {

	Vec3d n = wall.getNormal(*this);

	Vec3d pos_corr { 0, 0, 0 };
	if (abs(n * vel) > SMALL) { // not parallel, and moving
		pos_corr = (radius - wall.distance(*this)) / abs(n * vel) * vel * (-1); // move along the OLD! velocity vector
	} else {
		pos_corr = (radius - wall.distance(*this)) * n; // move in surface normal direction
	}

	// move back to the position when it touched the boundary:
	pos = pos + pos_corr;

	// correct the velocity to conserve energy (dissipation work is not considered!)
	if (vel * vel + 2 * gravity * pos_corr >= 0.0) {
		vel = std::sqrt(vel * vel + 2 * gravity * pos_corr) * norm(vel);
	} else {
		vel = -std::sqrt(-1 * (vel * vel + 2 * gravity * pos_corr)) * norm(vel);
	}

	// revert the wall normal velocity component
	vel = vel - (1 + Particle::restitution_coeff) * (vel * n) * n;
}

void Particle::collide_particle(Particle &other) {

	Vec3d n = other.pos - this->pos; // distance vector, pointing towards the other particle

	double distance = abs(n);
	n = norm(n); // normalize

	Vec3d pos_corr = -0.5 * n * (this->getR() + other.getR() - distance);
	// move back to the position when it touched the other:
	pos = pos + pos_corr;
	other.setPos(other.getPos() - pos_corr);

	correct_velocity(pos_corr); // TODO: use the actually used one! (0/1/2)
	other.correct_velocity(-1*pos_corr);

	exchange_impulse(other);
}

void Particle::collide_particle_check_boundary(Particle &other) {

	Vec3d n = other.pos - this->pos; // distance vector, pointing towards the other particle

	double distance = abs(n);
	n = norm(n); // normalize

	Vec3d pos_corr = -0.5 * n * (this->getR() + other.getR() - distance);

	// create temporary particles for overlap checking
	Particle tmp_particle(*this);
	tmp_particle.setPos(pos + pos_corr);
	Particle tmp_other_particle(other);
	tmp_other_particle.setPos(other.getPos() - pos_corr);

	// TODO: store overlapping properties (free/bounded by wall)
	if (!tmp_particle.overlap_walls() && !tmp_other_particle.overlap_walls()) {
		// move back to the position when it touched the other:
		pos = pos + pos_corr;
		other.setPos(other.getPos() - pos_corr);
	} else if (tmp_particle.overlap_walls()
			&& !tmp_other_particle.overlap_walls()) {
		other.setPos(other.getPos() - 2 * pos_corr); // correct where there is no wall
	} else if (tmp_other_particle.overlap_walls()
			&& !tmp_particle.overlap_walls()) {
		pos = pos + 2 * pos_corr; // correct where there is no wall
	} else { // both particles overlap with walls
		// TODO: implement correction
		pos = pos + pos_corr;
		other.setPos(other.getPos() - pos_corr);
	}

	correct_velocity(pos_corr); // TODO: use the actually used one! (0/1/2)
	other.correct_velocity(-1*pos_corr);

	exchange_impulse(other);

}

void Particle::correct_velocity(const Vec3d& pos_corr) {
	// correct the velocity to conserve energy (dissipation work is not considered!)
	if (vel * vel + 2 * gravity * pos_corr >= 0.0) {
		vel = std::sqrt(vel * vel + 2 * gravity * pos_corr) * norm(vel);
	} else {
		vel = -std::sqrt(-1 * (vel * vel + 2 * gravity * pos_corr)) * norm(vel);
	}
}

void Particle::exchange_impulse(Particle &other) {

	Vec3d n = other.pos - this->pos; // distance vector, pointing towards the other particle
	n = norm(n); // normalize

	Vec3d vel_old = vel; // store it for the other particle
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

bool Particle::overlap_wall(const Boundary &wall) const {
	if (wall.distance(*this) - radius < 0) {
		return true;
	} else {
		return false;
	}
}

bool Particle::overlap_walls() const {
	// TODO return pointers to the actually overlapped walls?
	for (const auto &b : scene->getBoundariesPlanar()) {
		if (overlap_wall(b)) {
			return true;
		}
	}
	for (const auto &b : scene->getBoundariesAxiSym()) {
		if (overlap_wall(b)) {
			return true;
		}
	}
	return false;
}

Vec3d Particle::overlapVect_wall(const Boundary &wall) {
	return (wall.distance(*this) - radius) * wall.getNormal(*this);
}

Vec3d Particle::findPlace(Particle &other) {
	// gives a direction in case of null vectors (coinciding particles!)
	// TODO: choose later from available place (scene must be known!)
	return Vec3d { 1, 0, 0 };
}
