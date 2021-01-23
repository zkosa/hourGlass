#ifndef INCLUDE_FUNCTIONHANDLER_H_
#define INCLUDE_FUNCTIONHANDLER_H_

// function handlers are used because std::function is not supported in CUDA
template<typename T> using constFunctionHandler = float(T::*)(float) const;

/* HOW TO CALL:
 * constFunctionHandler<Boundary_axissymmetric> handler = boundaries_ax_ptr->functionHandler_contour;
 * float res = (boundaries_ax_ptr->*handler)(0.0f);
 */


#endif /* INCLUDE_FUNCTIONHANDLER_H_ */
