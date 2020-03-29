#ifndef MINIMUM_H_
#define MINIMUM_H_

#include <functional>

class Minimum {
	// std::function enables passing the functions to other objects
	std::function<float(float)> function;

	static float tolerance; // result tolerance
	static constexpr int max_iter = 2000; // max number of Newton iterations
	static constexpr float delta = 1e-3; // step size for numerical differentiation

public:

	Minimum(std::function<float(float)> _function) :
			function(_function) {
	}

	float findRoot(float starting_value = 0) const;
	static float getTolerance() {
		return tolerance;
	}
	static void setTolerance(float tolerance) {
		Minimum::tolerance = tolerance;
	}

};

#endif /* MINIMUM_H_ */
