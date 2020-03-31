#ifndef VECAXISYM_H_
#define VECAXISYM_H_

// container with axial and radial coordinates for axis-symmetric vector calculations

struct VecAxiSym {
	// coordinates in axial coordinate system:
	float axial;
	float radial;

	VecAxiSym(float axial, float radial) :
			axial(axial), radial(radial) {
	}

	std::ostream& print(std::ostream& os = std::cout) const {
		return os << *this << std::endl;
	}

	// declare non-member function:
	friend std::ostream& operator<<(std::ostream &out, const VecAxiSym &a);
};

inline bool operator==(const VecAxiSym &a, const VecAxiSym &b) {
	return ( a.axial == b.axial && a.radial == b.radial );
}

inline VecAxiSym operator-(const VecAxiSym &a, const VecAxiSym &b) {
	return VecAxiSym( a.axial - b.axial, a.radial - b.radial );
}

inline std::ostream& operator<<(std::ostream &out, const VecAxiSym &a) {
	out << "(axial: " << a.axial << ", radial: " << a.radial << ")";
	return out;
}

inline float abs(const VecAxiSym& a) {
	return std::sqrt(a.axial*a.axial + a.radial*a.radial);
}

#endif /* VECAXISYM_H_ */
