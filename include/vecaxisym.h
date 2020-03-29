#ifndef VECAXISYM_H_
#define VECAXISYM_H_

// container with axial and radial coordinates for axis-symmetric vector calculations

struct VecAxiSym {
	float X;
	float R;

	VecAxiSym(float X_, float R_) :
			X(X_), R(R_) {
	}

	std::ostream& print() const {
		return std::cout << this;
	}
};

inline bool operator==(const VecAxiSym &a, const VecAxiSym &b) {
	return ( a.X == b.X && a.R == b.R );
}

inline std::ostream& operator<<(std::ostream &out, const VecAxiSym &a) {
	out << "(X: " << a.X << ", R:" << a.R << ")" << std::endl;
	return out;
}

#endif /* VECAXISYM_H_ */
