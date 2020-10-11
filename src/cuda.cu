#include "particle.h"
#include "cell.h"
#include "cuda.h"
#include <stdio.h>

__device__
bool Cell::containsCuda(const Particle *p) {
	float r = p->getR();

	return (p->getX() + r > bounds.x1 && p->getX() - r < bounds.x2)
			&& (p->getY() + r > bounds.y1 && p->getY() - r < bounds.y2)
			&& (p->getZ() + r > bounds.z1 && p->getZ() - r < bounds.z2);
}

__device__
void Cell::addParticleCuda(const Particle *p, int *particle_IDs_in_cell, int *number_of_particle_IDs) {
	int old_index = atomicAdd(number_of_particle_IDs, 1);
	particle_IDs_in_cell[old_index] = p->getID();
}

__global__
void get_particle_IDs_in_cell(int number_of_particles, const Particle *p, Cell *c, int *particle_IDs_in_cell, int *number_of_particle_IDs) {

	// grid-stride loop, handling even more processes than
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < number_of_particles;
		i += blockDim.x * gridDim.x)
	{
		if (c->containsCuda(p + i)) {
			c->addParticleCuda(p + i, particle_IDs_in_cell, number_of_particle_IDs);
		}

	}
}

__host__
void Cell::populateCuda(Particle* device_particles_ptr, int N_particles) {

	Cell* device_cell_ptr;
	cudaMalloc((void **)&device_cell_ptr, sizeof(Cell));
	cudaMemcpy(device_cell_ptr, this, sizeof(Cell), cudaMemcpyHostToDevice);

// outputs:
	int *device_particle_IDs_in_cell;
	// TODO: use a two step method to reduce the amount of cudaMallocs via calculating the number of particle IDs (see my 009-populate-array-on-GPU two_step branch)
	int max_number_of_particles_in_the_cell = N_particles; // very conservative and impractical guess!
	cudaMalloc((void **)&device_particle_IDs_in_cell, max_number_of_particles_in_the_cell*sizeof(int));
	cudaMemset(device_particle_IDs_in_cell, -1, max_number_of_particles_in_the_cell*sizeof(int));// zero could be a particle ID, so use something obviously not particle id

	int *device_number_of_particle_IDs = 0;
	cudaMalloc((void **)&device_number_of_particle_IDs, sizeof(int));
	cudaMemset(device_number_of_particle_IDs, 0, sizeof(int));


	int threads = 256; // recommended first value, must not be larger than 1024
	int blocks = ceil(float(N_particles)/threads);
	// calling function to be run on the GPU:
	get_particle_IDs_in_cell<<<blocks,threads>>>(N_particles, device_particles_ptr, device_cell_ptr, device_particle_IDs_in_cell, device_number_of_particle_IDs);
	cudaDeviceSynchronize(); // TODO: try to move it one layer higher (from within cell to within scene level, to educe number of synchronizations)

	int host_number_of_particle_IDs = -999;
	cudaMemcpy( &host_number_of_particle_IDs,
				device_number_of_particle_IDs,
				sizeof(int),
				cudaMemcpyDeviceToHost
				);

	int host_particle_IDs_in_cell[host_number_of_particle_IDs];

	cudaMemcpy( &host_particle_IDs_in_cell[0],
				device_particle_IDs_in_cell,
				host_number_of_particle_IDs * sizeof(int),
				cudaMemcpyDeviceToHost
				);
/*
	cudaMemcpy( this,
				device_cell_ptr,
				sizeof(Cell),
				cudaMemcpyDeviceToHost
				);
*/

	// copy the array of particle IDs (collected via CUDA) into the cells vector:
	size_t number_of_elements = sizeof(host_particle_IDs_in_cell) / sizeof(int);
	particle_IDs.resize(number_of_elements);
	std::copy(host_particle_IDs_in_cell + 0,
			  host_particle_IDs_in_cell + number_of_elements,
			  particle_IDs.begin()
			 );

	cudaFree(device_cell_ptr);
	cudaFree(device_particle_IDs_in_cell);
	cudaFree(device_number_of_particle_IDs);

	//cudaDeviceReset();
}
