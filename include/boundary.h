#ifndef BOUNDARY_H_
#define BOUNDARY_H_

class Particle;
class Vec3d;

class Boundary {

	bool temporary = false; // TODO: rather derive temporary boundaries

protected:
	bool planar = false; // for type checking: set to true in constructor of planar

public:

	virtual bool operator==(const Boundary &other) const = 0; // TODO: add tests!
	virtual void draw2D() = 0;
	virtual float distance(const Particle &particle) const = 0;
	virtual float distanceSigned(const Particle &particle) const = 0;
	virtual Vec3d getNormal(const Particle &particle) const = 0;

	bool isTemporary() const {
		return temporary;
	}
	void setTemporary() {
		temporary = true;
	}
	bool isPlanar() const {
		return planar;
	}

};

#endif /* BOUNDARY_H_ */
