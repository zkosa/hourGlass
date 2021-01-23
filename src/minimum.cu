#include "minimum.h"
#include "minimumdistance.h"

float Minimum::tolerance = 1e-5f;

__host__ __device__
float Minimum::findRoot(float starting_value) {
	// Newton-method

	// initialization:
	float X = starting_value;
	float X_new = X;
	float func_prime, func_prime_prime;
	performed_iterations = 0;
	constFunctionHandler<MinimumDistance> handler = function_owner->functionHandler_distance2;

	do {
		performed_iterations++;
		X = X_new;

		func_prime = ((function_owner->*handler)(X + delta) - (function_owner->*handler)(X - delta)) / (2 * delta);
		func_prime_prime = ((function_owner->*handler)(X + 2 * delta)
				- 2 * (function_owner->*handler)(X) + (function_owner->*handler)(X - 2 * delta))
				/ (4 * 1e-3f * 1e-3f); // TODO: use the constant 1e-3f???? elta

		X_new = X - func_prime / func_prime_prime;
		//CUDA_HELLO;
	} while (std::abs(X_new - X) >= 1e-5f && !numberOfIterReached()); // TODO: create "static like variable" for tolerance

	return X_new;
}

__host__ __device__
bool Minimum::numberOfIterReached() {
	if (performed_iterations < 100) {// TODO: use the constant max_iter 1e-3f???? elta
		return false;
	} else {
		// TODO: add warning!
		return true;
	}
}
