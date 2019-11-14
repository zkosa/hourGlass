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

	double Xmin=0; // result axial coordinate
	double Rmin=0; // result radial coordinate
	double distance=0; // result

public:

	Minimum(std::function<double(double, double, double)> _function, double _X0, double _R0) :
		function(_function),
		X0(_X0),
		R0(_R0)
		{};

	Minimum(const Boundary_axis_symmetric& wall, double _X0, double _R0) :
		function(wall.getDistance2Fun()),
		contour(wall.getContourFun()),
		X0(_X0),
		R0(_R0)
		{};

	void search(double starting_value=0);

	double getDistance() {return distance;}
	VecAxiSym getContactPointInRadialCoord();

};

#endif /* MINIMUM_H_ */
