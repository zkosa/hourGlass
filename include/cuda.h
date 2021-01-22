#ifndef CUDA_H_
#define CUDA_H_

#include <cstdio>
#include <iostream>
#include <cstdlib> // for exit
#include <cuda_runtime_api.h> // just for proper indexing, nvcc includes it anyhow


#define CUDA_HOSTDEV __host__ __device__
#define CHECK_CUDA(cuda_call) { cudaAssert((cuda_call), __FILE__, __LINE__); }
#define CHECK_CUDA_POST { cudaAssert((cudaPeekAtLastError()), __FILE__, __LINE__); }
#define CHECK_CUDA_POINTER(ptr) { cudaPointerAttributes attributes; printf("----\n"); printf(#ptr); printf(":\n"); CHECK_CUDA( cudaPointerGetAttributes(&attributes, ptr) ); cudaPrintPointerAttributes(attributes); }

inline void cudaAssert(cudaError_t error_code, const char *file, int line, bool abort=true)
{
	if (error_code != cudaSuccess) {

		fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(error_code), file, line);

		if (abort) {
			exit(error_code);
		}
	}
}

inline void cudaPrintPointerAttributes(cudaPointerAttributes attributes) {
	std::cout << "attributes.device:        " << attributes.device << std::endl;
	std::cout << "attributes.devicePointer: " << attributes.devicePointer << std::endl;
	std::cout << "attributes.hostPointer:   " << attributes.hostPointer << std::endl;
	std::cout << "attributes.type:          " << attributes.type << std::endl;
}

/*
template<class T>
inline void CHECK_POINTER(T* ptr) {
	cudaPointerAttributes* attributes = nullptr;
	CHECK_CUDA( cudaPointerGetAttributes(attributes, ptr) );
	cudaPrintPointerAttributes(attributes);
}
*/


#endif /* CUDA_H_ */
