#ifndef VECAXISYM_H_
#define VECAXISYM_H_

// container with axial and radial coordinates for axis-symmetric vector calculations

struct VecAxiSym {
	double X;
	double R;

	VecAxiSym(double X_, double R_) : X(X_), R(R_ ) {}
};


#endif /* VECAXISYM_H_ */
