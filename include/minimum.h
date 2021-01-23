#ifndef MINIMUM_H_
#define MINIMUM_H_

#include "cuda.h"
//#include <functional>
#include "functionhandler.h"

//// for function pointer:
//typedef float				// function return value type
//		(*function_t) 		// type name
//		(float);			// function argument type(s)

class MinimumDistance;

class Minimum {
	// std::function enables passing the functions to other objects
	//const std::function<float(float)> function;
	//const function_t function_ptr;

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
	MinimumDistance* function_owner;
	constFunctionHandler<MinimumDistance> functionHandler_func_to_be_minimized;

	__host__ __device__
	Minimum(MinimumDistance* function_owner_, constFunctionHandler<MinimumDistance> functionHandler_func_to_be_minimized_)
		: function_owner(function_owner_), functionHandler_func_to_be_minimized(functionHandler_func_to_be_minimized_)
	{
		//initializer(functionHandler_func_to_be_minimized_);
	};

	__host__ __device__
	float findRoot(float starting_value);
	__host__ __device__
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
	__host__ __device__
	void setInitialGuess(float guess) {
		this->guess = guess;
	}
	int getPerformedIterations() {
		return performed_iterations;
	}
	__host__ __device__
	bool numberOfIterReached();
};

#endif /* MINIMUM_H_ */
