#include <iostream>
#include "test.cuh"
#include <cuda_runtime_api.h> // just for proper indexing, nvcc includes it anyhow

bool areCudaDevicesAvailable() {
	int _; // ignored variable
	cudaError_t error_code = cudaGetDeviceCount( &_ );

	return (error_code == cudaSuccess);
}

void printCudaVersion() {
	std::cout << "CUDA Compiled version: " << __CUDACC_VER_MAJOR__ << "."
										   << __CUDACC_VER_MINOR__
										   <<  ", Build: " << __CUDACC_VER_BUILD__ << std::endl;

	int runtime_version;
	cudaRuntimeGetVersion(&runtime_version);
	std::cout << "CUDA runtime version: " << runtime_version << std::endl;

	int driver_version;
	cudaDriverGetVersion(&driver_version);
	std::cout << "CUDA driver version: " << driver_version << std::endl;
}


void printGpuDeviceInfo() {

	int number_of_gpus = 0;
	cudaError_t error_code = cudaGetDeviceCount( &number_of_gpus );
	std::cout << "CUDA device count: " << number_of_gpus << std::endl;

	if (error_code != cudaSuccess) {
		std::cout << "No GPU device available" << std::endl;
		std::cout << "cudaGetDeviceCount error code: " << error_code << std::endl;
	}
	else {
		cudaDeviceProp device_properties;
		// Iterate over the GPUs
		for (int device = 0; device < number_of_gpus;  device++)
		{
			cudaGetDeviceProperties(&device_properties, device);
			std::cout << "--------------------------------------------" << std::endl;
			std::cout << "CUDA device name: " << device_properties.name << std::endl;
			std::cout << "Integrated: " << device_properties.integrated << std::endl;
			std::cout << "Clock rate " << device_properties.clockRate << std::endl;

			auto maxGridSize = device_properties.maxGridSize;
			std::cout << "maxGridSize: " << maxGridSize[0] << " x "
										 << maxGridSize[1] << " x "
										 << maxGridSize[2] << std::endl;

			std::cout << "maxThreadsPerBlock: " << device_properties.maxThreadsPerBlock << std::endl;

			auto maxThreadsDim = device_properties.maxThreadsDim;
			std::cout << "maxThreadsDim: " << maxThreadsDim[0] << " x "
										   << maxThreadsDim[1] << " x "
										   << maxThreadsDim[2] << std::endl;
		}
	}
}
