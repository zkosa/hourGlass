#ifndef MINIMUM_H_
#define MINIMUM_H_

#include <functional>
#include "boundary_axissymmetric.h"
#include "vecaxisym.h"

class Minimum {
	// std::function enables passing the functions to other objects
	std::function<float(float, float, float)> function; // the function to be minimized (distance squared)
	std::function<float(float)> contour;

	float X0;
	float R0;

	float Xmin = 0; // result axial coordinate
	float Rmin = 0; // result radial coordinate
	float distance = 0; // result

public:

	Minimum(std::function<float(float, float, float)> _function, float _X0,
			float _R0) :
			function(_function), X0(_X0), R0(_R0) {
	}

	Minimum(const Boundary_axissymmetric &wall, float _X0, float _R0) :
			function(wall.getDistance2Fun()), contour(wall.getContourFun()), X0(
					_X0), R0(_R0) {
	}

	void search(float starting_value = 0);

	float getDistance() {
		return distance;
	}
	VecAxiSym getContactPointInRadialCoord();

};

#endif /* MINIMUM_H_ */
