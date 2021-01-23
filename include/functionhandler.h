#ifndef INCLUDE_FUNCTIONHANDLER_H_
#define INCLUDE_FUNCTIONHANDLER_H_

// function handlers are used because std::function is not supported in CUDA
template<typename T> using constFunctionHandler = float(T::*)(float) const;


#endif /* INCLUDE_FUNCTIONHANDLER_H_ */
