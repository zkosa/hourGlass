#ifndef BOUNDINGBOX_H_
#define BOUNDINGBOX_H_

class Vec3d;
class Scene;

class BoundingBox {
	Vec3d corner1;
	Vec3d corner2;

public:

	BoundingBox(Scene &scene);

	Vec3d center() const;
	Vec3d diagonal() const;
	float volume() const;

	const Vec3d& getCorner1() const {
		return corner1;
	}
	const Vec3d& getCorner2() const {
		return corner2;
	}

};

#endif /* BOUNDINGBOX_H_ */
