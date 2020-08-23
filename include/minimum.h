#ifndef MINIMUM_H_
#define MINIMUM_H_

#include <functional>

class Minimum {
	// std::function enables passing the functions to other objects
	const std::function<float(float)> function;

	// starting value for the Newton iterations:
	float guess = 0;
	// result tolerance:
	static float tolerance;
	// step size for numerical differentiation
	// limits of float are reached with values smaller than 1e-3
	static constexpr float delta = 1e-3;
	// max number of Newton iterations:
	static constexpr int max_iter = 100;
	// number of actually performed iterations
	int performed_iterations = 0;

public:

	Minimum(std::function<float(float)> _function) :
			function(_function) {
	}

	float findRoot(float starting_value);
	float findRoot() {
		// use the default, or previously user specified starting value
		return findRoot(guess);
	};

	static float getTolerance() {
		return tolerance;
	}
	static void setTolerance(float tolerance) {
		Minimum::tolerance = tolerance;
	}
	void setInitialGuess(float guess) {
		this->guess = guess;
	}
	int getPerformedIterations() {
		return performed_iterations;
	}
	bool numberOfIterReached();
};

#endif /* MINIMUM_H_ */
