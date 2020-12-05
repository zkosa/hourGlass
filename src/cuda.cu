#include "particle.h"
#include "cell.h"
#include "cuda.h"
#include <stdio.h> // for printing from the device
#include <iostream>

__device__
bool Cell::containsCuda(const Particle *p) const {
	float r = p->getR();

	return (p->getX() + r > bounds.x1 && p->getX() - r < bounds.x2)
			&& (p->getY() + r > bounds.y1 && p->getY() - r < bounds.y2)
			&& (p->getZ() + r > bounds.z1 && p->getZ() - r < bounds.z2);
}

__device__
void Cell::addParticleCuda(const Particle *p, int *particle_IDs_in_cell, int *index_counter) {
	int old_index = atomicAdd(index_counter, 1);
	particle_IDs_in_cell[old_index] = p->getID();
}

__global__
void get_number_of_particles_in_cell(int number_of_particles, const Particle *p, Cell *c, int *number_of_particle_IDs) {
	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < number_of_particles;
		i += blockDim.x * gridDim.x)
	{
		if (c->containsCuda(p + i)) {
			atomicAdd(number_of_particle_IDs, 1);
		}
	}
}

__global__
void get_particle_IDs_in_cell(int number_of_particles, const Particle *p, Cell *c, int *particle_IDs_in_cell, int *index_counter) {
	// grid-stride loop
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < number_of_particles;
		i += blockDim.x * gridDim.x)
	{
		if (c->containsCuda(p + i)) {
			c->addParticleCuda(p + i, particle_IDs_in_cell, index_counter);
		}
	}
}

__host__
void Cell::populateCuda(Particle* device_particles_ptr, int N_particles) {

	Cell* device_cell_ptr;
	cudaMalloc((void **)&device_cell_ptr, sizeof(Cell));
	cudaMemcpy(device_cell_ptr, this, sizeof(Cell), cudaMemcpyHostToDevice);


// calculate the number of outputs:
	int *device_number_of_particle_IDs = 0;
	cudaMalloc((void **)&device_number_of_particle_IDs, sizeof(int));
	cudaMemset(device_number_of_particle_IDs, 0, sizeof(int));


	int threads = 256; // recommended first value, must not be larger than 1024
	int blocks = ceil(float(N_particles)/threads);
	// calling function to be run on the GPU:
	get_number_of_particles_in_cell<<<blocks,threads>>>(N_particles, device_particles_ptr, device_cell_ptr, device_number_of_particle_IDs);
	cudaDeviceSynchronize();

	int host_number_of_particle_IDs = 0;
	cudaMemcpy( &host_number_of_particle_IDs,
				device_number_of_particle_IDs,
				sizeof(int),
				cudaMemcpyDeviceToHost
				);
	cudaDeviceSynchronize();

// allocate memory and get the particles after getting to know the size:
	int *device_index_counter;
	cudaMalloc((void **)&device_index_counter, sizeof(int));
	cudaMemset(device_index_counter, 0, sizeof(int));

	int *device_particle_IDs_in_cell;
	int max_number_of_particles_in_the_cell = host_number_of_particle_IDs; // use the exact, calculated value
	cudaMalloc((void **)&device_particle_IDs_in_cell, max_number_of_particles_in_the_cell*sizeof(int));
	cudaMemset(device_particle_IDs_in_cell, -1, max_number_of_particles_in_the_cell*sizeof(int));// zero could be a particle ID, so use something obviously not particle id


	get_particle_IDs_in_cell<<<blocks,threads>>>(N_particles, device_particles_ptr, device_cell_ptr, device_particle_IDs_in_cell, device_index_counter);
	cudaDeviceSynchronize(); // TODO: try to move it one layer higher (from within cell to within scene level, to reduce number of synchronizations)

// check
	int host_number_of_particle_IDs_second_kernel;
	cudaMemcpy( &host_number_of_particle_IDs_second_kernel,
				device_index_counter,
				sizeof(int),
				cudaMemcpyDeviceToHost
				);

	if (host_number_of_particle_IDs != host_number_of_particle_IDs_second_kernel) {
		std::cout << "number of particle IDs does not match between the two kernels: "
				<< host_number_of_particle_IDs << " != "
				<< host_number_of_particle_IDs_second_kernel << std::endl;

		std::exit(EXIT_FAILURE); // makes it untestable???
	}

// copy the resultant cell IDs into the host Cell::particle_IDs vectors
	particle_IDs.resize(host_number_of_particle_IDs);
	cudaMemcpy( particle_IDs.data(),
				device_particle_IDs_in_cell,
				host_number_of_particle_IDs * sizeof(int),
				cudaMemcpyDeviceToHost
				);

	cudaFree(device_cell_ptr);
	cudaFree(device_particle_IDs_in_cell);
	cudaFree(device_number_of_particle_IDs);

	//cudaDeviceReset();
}
