#include "minimum.h"

float Minimum::tolerance = 1e-5f;

float Minimum::findRoot(float starting_value) {
	// Newton-method

	// initialization:
	float X = starting_value;
	float X_new = X;
	float func_prime, func_prime_prime;
	performed_iterations = 0;

	do {
		performed_iterations++;
		X = X_new;

		func_prime = ((function_owner->*functionHandler_func_to_be_minimized)(X + delta) - (function_owner->*functionHandler_func_to_be_minimized)(X - delta)) / (2 * delta);
		func_prime_prime = ((function_owner->*functionHandler_func_to_be_minimized)(X + 2 * delta)
				- 2 * (function_owner->*functionHandler_func_to_be_minimized)(X) + (function_owner->*functionHandler_func_to_be_minimized)(X - 2 * delta))
				/ (4 * delta * delta);

		X_new = X - func_prime / func_prime_prime;
	} while (std::abs(X_new - X) >= tolerance && !numberOfIterReached());

	return X_new;
}

bool Minimum::numberOfIterReached() {
	if (performed_iterations < max_iter) {
		return false;
	} else {
		// TODO: add warning!
		return true;
	}
}
