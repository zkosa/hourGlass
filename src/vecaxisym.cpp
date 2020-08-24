#include "vecaxisym.h"
#include <iostream>

std::ostream& VecAxiSym::print() const {
	return std::cout << *this << std::endl;
}

std::ostream& operator<<(std::ostream &out, const VecAxiSym &a) {
	out << "(axial: " << a.axial << ", radial: " << a.radial << ")";
	return out;
}
