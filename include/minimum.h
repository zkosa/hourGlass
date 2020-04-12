#ifndef MINIMUM_H_
#define MINIMUM_H_

#include <functional>

class Minimum {
	// std::function enables passing the functions to other objects
	std::function<float(float)> function;

	// starting value for the Newton iterations:
	float guess = 0;
	// result tolerance:
	static float tolerance;
	// max number of Newton iterations:
	static constexpr int max_iter = 2000;
	// step size for numerical differentiation
	// limits of float are reached with values smaller than 1e-3
	static constexpr float delta = 1e-3;

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
	void setInitialGuess(float guess) {
		this->guess = guess;
	}
};

#endif /* MINIMUM_H_ */
