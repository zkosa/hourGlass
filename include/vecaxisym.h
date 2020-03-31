#ifndef VECAXISYM_H_
#define VECAXISYM_H_

// container with axial and radial coordinates for axis-symmetric vector calculations

struct VecAxiSym {
	float X;
	float R;

	VecAxiSym(float X_, float R_) :
			X(X_), R(R_) {
	}

	void print(std::ostream& os = std::cout) const {
		os << this << std::endl;; // TODO: fix and add test
	}
};

inline bool operator==(const VecAxiSym &a, const VecAxiSym &b) {
	return ( a.X == b.X && a.R == b.R );
}

inline VecAxiSym operator-(const VecAxiSym &a, const VecAxiSym &b) {
	return VecAxiSym( a.X - b.X, a.R - b.R );
}

inline std::ostream& operator<<(std::ostream &out, const VecAxiSym &a) {
	out << "(X: " << a.X << ", R: " << a.R << ")";
	return out;
}

inline float abs(const VecAxiSym& a) {
	return std::sqrt(a.X*a.X + a.R*a.R);
}

#endif /* VECAXISYM_H_ */
