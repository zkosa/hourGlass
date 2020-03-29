#include "minimum.h"

float Minimum::tolerance = 1e-5;

float Minimum::findRoot(float starting_value) const {
	// Newton-method

	// initialization:
	float X = starting_value;
	float X_new = X;
	float func_prime, func_prime_prime;
	int counter = 0;

	do {
		counter++;
		X = X_new;

		func_prime = (function(X + delta) - function(X - delta)) / (2 * delta);
		func_prime_prime = (function(X + 2 * delta)
				- 2 * function(X) + function(X - 2 * delta))
				/ (4 * delta * delta);

		X_new = X - func_prime / func_prime_prime;
	} while (std::abs(X_new - X) >= tolerance && counter < max_iter);

	return X_new;
}
