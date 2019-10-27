#ifndef MINIMUM_H_
#define MINIMUM_H_

#include <functional>
#include "Vec3d.h"
#include "VecAxiSym.h"
#include "Boundary_axis_symmetric.h"

class Minimum {
	std::function<double(double, double, double)> function; // to be minimized (distance squared)
	std::function<double(double)> contour;

	double X0;
	double R0;

	double Xmin=0; // result
	double Rmin=0; // result
	double distance=0; // result

public:

	Minimum(std::function<double(double, double, double)> function_, double X0_, double R0_) :
		function(function_),
		X0(X0_),
		R0(R0_)
		{};

	Minimum(const Boundary_axis_symmetric& wall, double X0_, double R0_) :
		function(wall.getDistance2Fun()),
		contour(wall.getContourFun()),
		X0(X0_),
		R0(R0_)
		{};

	void search(double starting_value=0);

	double getDistance() {return distance;}
	VecAxiSym getContactPointInRadialCoord();

};

#endif /* MINIMUM_H_ */
