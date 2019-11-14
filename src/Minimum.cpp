#include "Minimum.h"
#include <cmath>

void Minimum::search(double starting_value) {
	double X = starting_value;
	double X_new = X;
	double delta = 1e-3;
	double dist2_prime, dist2_prime_prime;
	double epsilon = 1e-5;
	int counter = 0;
	int max_iter = 2000;

	do {
		counter++;
		X = X_new;

		dist2_prime =
				(function(X + delta, X0, R0) - function(X - delta, X0, R0))
						/ (2 * delta);
		dist2_prime_prime = (function(X + 2 * delta, X0, R0)
				- 2 * function(X, X0, R0) + function(X - 2 * delta, X0, R0))
				/ (4 * delta * delta);

		X_new = X - dist2_prime / dist2_prime_prime;
	} while (std::abs(X_new - X) >= epsilon && counter < max_iter);

	Xmin = X_new;
	Rmin = contour(Xmin);

	distance = std::sqrt(function(X_new, X0, R0));
}

VecAxiSym Minimum::getContactPointInRadialCoord() {
	this->search(X0);
	return VecAxiSym { Xmin, Rmin };
}
