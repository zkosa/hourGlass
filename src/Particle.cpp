#include "Particle.h"
#include "Boundary_planar.h"
#include "Boundary_axis_symmetric.h"


Particle::Particle(){};

Particle::~Particle(){};

double Particle::Cd = 0.5; // non-constexpr static members must be initialized in the definition

double Particle::uniform_radius = 0.005;

void Particle::advance(double dt) {
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

Vec3d Particle::impulse() {
	return mass()*vel;
}

Vec3d Particle::apply_forces(){

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
	int triangleAmount = 10; //# of triangles used to draw circle

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

void Particle::drawNow2D() {
	Particle::draw2D();
	glfwSwapBuffers(window);
}

double Particle::distance(class Particle& other) {
	return abs(pos - other.pos);
}

void Particle::collide_wall(Boundary_planar& wall) {
	old_pos = pos;
	old_vel = vel;

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
	if (vel*vel + 2*gravity*pos_corr >= 0.0) {
		vel = std::sqrt(vel*vel + 2*gravity*pos_corr)  * norm(vel);
	}
	else {
		vel = -std::sqrt(-1*(vel*vel + 2*gravity*pos_corr) )  * norm(vel);
	}

	// revert the wall normal velocity component
	vel = vel - (1 + this->CoR())*(vel*n)*n;

}

void Particle::collide_wall(Boundary_axis_symmetric& wall) {

	Vec3d n = wall.getNormal(*this);

	Vec3d pos_corr {0,0,0};
	if ( abs(n*vel) > SMALL) // not parallel, and moving
		pos_corr = (radius - wall.distance(*this)) / abs(n*vel) * vel *(-1); // move along the OLD! velocity vector
	else {
		pos_corr = (radius - wall.distance(*this)) * n; // move in surface normal direction
	}

	// move back to the position when it touched the boundary:
	pos = pos + pos_corr;

	// correct the velocity to conserve energy (dissipation work is not considered!)
	if (vel*vel + 2*gravity*pos_corr >= 0.0) {
		vel = std::sqrt(vel*vel + 2*gravity*pos_corr)  * norm(vel);
	}
	else {
		vel = -std::sqrt(-1*(vel*vel + 2*gravity*pos_corr) )  * norm(vel);
	}

	// revert the wall normal velocity component
	vel = vel - (1 + this->CoR())*(vel*n)*n;
}

void Particle::collide_particle(class Particle& other) {
	old_pos = pos;
	old_vel = vel;
	other.old_pos = other.pos;
	other.old_vel = other.vel;
	Particle old_other = other;

	Vec3d n = other.pos - this->pos; // pointing towards the other

	double distance = abs(n);
	n = norm(n); // normalize

	Vec3d pos_corr = -0.5 * n * (this->getR() + other.getR() - distance);
	// move back to the position when it touched the other:
	pos = pos + pos_corr;
	//drawNow2D();
	other.setPos(other.getPos() - pos_corr);
	//other.drawNow2D();

	// correct the velocity to conserve energy (dissipation work is not considered!)
	if (vel*vel + 2*gravity*pos_corr >= 0.0){
		vel = std::sqrt(vel*vel + 2*gravity*pos_corr)  * norm(vel);
	}
	else {
		vel = -std::sqrt( -1*(vel*vel + 2*gravity*pos_corr) )  * norm(vel);
	}
	//drawNow2D();
	if (other.getV()*other.getV() + 2*gravity*pos_corr >= 0.0) {
		other.setV(std::sqrt(other.getV()*other.getV() + 2*gravity*pos_corr) * norm(other.getV())) ;
	}
	else {
		other.setV(-std::sqrt( -1*(other.getV()*other.getV() + 2*gravity*pos_corr) ) * norm(other.getV())) ;
	}
	//other.drawNow2D();

	// impulse exchange
	Vec3d vel_old = vel;
	vel = vel_old - n*(n*vel_old) + (mass()-other.mass())/(mass() + other.mass())*n*(vel_old*n) + 2*other.mass()/(mass()+other.mass())*n*(other.getV()*n);
	//drawNow2D();
	other.setV(  other.getV() - n*(other.getV()*n)  +  2*mass()/(other.getM()+mass())*n*(vel_old*n) + (other.mass()-mass())/(other.mass()+mass())*n*(other.getV()*n) );
	//other.drawNow2D();

}

Vec3d Particle::findPlace(class Particle& other) {
	// gives a direction in case of null vectors (coinciding particles!)
	// TODO: choose later from available place (scene must be known!)
	return Vec3d {1,0,0};
}

void Particle::debug() const{
	if (pos.large()) {
		std::cout << "WARNING: large pos: ";  pos.print(); // place breakpoint here
	}
	if (vel.large()) {
		std::cout << "WARNING: large vel: ";  vel.print(); // place breakpoint here
	}

}
