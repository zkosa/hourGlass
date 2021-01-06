#ifndef CUDA_H_
#define CUDA_H_

#include <cuda_runtime_api.h> // just for proper indexing, nvcc includes it anyhow
#define CUDA_HOSTDEV __host__ __device__
#define CHECK_CUDA(cuda_call) { cudaAssert((cuda_call), __FILE__, __LINE__); }
#define CHECK_CUDA_POST if (cudaPeekAtLastError() != cudaSuccess) printf("The error is %s", cudaGetErrorString(cudaGetLastError()));;


inline void cudaAssert(cudaError_t error_code, const char *file, int line, bool abort=true)
{
	if (error_code != cudaSuccess) {

		fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(error_code), file, line);

		if (abort) {
			exit(error_code);
		}
	}
}

#endif /* CUDA_H_ */
