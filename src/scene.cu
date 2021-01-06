#include "scene.h"
#include "mainwindow.h"
#include <iostream>
#include <device_launch_parameters.h> // just for proper indexing, nvcc includes it anyhow

__global__
void get_number_of_particles_per_cell(
		int number_of_particles, const Particle *p,
		int number_of_cells, const Cell *c,
		int *number_of_particle_IDs_per_cell
		)
{
	// nested (2D) grid-stride loop
	int index_particle = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_particle = blockDim.x * gridDim.x;
	int index_cell = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_cell = blockDim.y * gridDim.y;

	for (int i_c = index_cell;
		i_c < number_of_cells;
		i_c += stride_cell)
	{
		for (int i_p = index_particle;
			i_p < number_of_particles;
			i_p += stride_particle)
		{
			if ((c + i_c)->containsCuda(p + i_p)) {
				atomicAdd(number_of_particle_IDs_per_cell + i_c, 1);
			}
		}
	}
}

__global__
void get_particle_IDs_in_cells(
		int number_of_particles, const Particle *p,
		int number_of_cells, Cell *c,
		const int *IN_number_of_particleIDs, // per cell, as input???
		int *OUT_particle_IDs_in_cells,
		int *OUT_particle_ID_counter // per cell, for counting
		)
{
	// nested (2D) grid-stride loop
	int index_particle = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_particle = blockDim.x * gridDim.x;
	int index_cell = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_cell = blockDim.y * gridDim.y;

	for (int i_c = index_cell;
		i_c < number_of_cells;
		i_c += stride_cell)
	{
		for (int i_p = index_particle;
			i_p < number_of_particles;
			i_p += stride_particle)
		{
			if ((c + i_c)->containsCuda(p + i_p)) {
				(c + i_c)->addParticleCudaMultiCell(p + i_p, IN_number_of_particleIDs, OUT_particle_IDs_in_cells, i_c, OUT_particle_ID_counter);
			}
		}
	}
}

void Scene::populateCellsCuda() {

	// this->clearCells(); // will we need something like this?

	int N_cells = cells.size();
	int N_particles = particles.size();
//	for(int i=0; i<N_cells; i++) {
//		device_cells_ptr[i].populateCuda(device_particles_ptr, N_particles);
//	}


	dim3 block(16,16);
	dim3 grid((N_cells*N_particles+15)/16,  (N_cells*N_particles+15)/16);

	int *device_number_of_particle_IDs_per_cell;
	CHECK_CUDA( cudaMalloc((void **)&device_number_of_particle_IDs_per_cell, sizeof(int)*N_cells) );
	CHECK_CUDA( cudaMemset(device_number_of_particle_IDs_per_cell, 0, sizeof(int)*N_cells) );

	get_number_of_particles_per_cell<<<grid,block>>>(
			N_particles,
			device_particles_ptr,
			N_cells,
			device_cells_ptr,
			device_number_of_particle_IDs_per_cell);

	//cudaDeviceSynchronize();

	std::vector<int> host_number_of_particle_IDs_per_cell(N_cells);
	CHECK_CUDA( cudaMemcpy( host_number_of_particle_IDs_per_cell.data(),
				device_number_of_particle_IDs_per_cell,
				sizeof(int)*N_cells,
				cudaMemcpyDeviceToHost
				) );
//	for (auto const& n : host_number_of_particle_IDs_per_cell) {
//			std::cout << n << '\t';
//	} std::cout << std::endl;

/*
	for (auto& ids: host_number_of_particle_IDs_per_cell) {
		std::cout << ids << '\t';
	}
	std::cout << std::endl;
	for (auto& p: particles) {
		std::cout << p.getX() << ',' << p.getY() <<'\t';
	}
	std::cout << std::endl;
	for (auto& c: cells) {
		std::cout << c.getCenter().x << ','<< c.getCenter().y <<'\t';
	}
	std::cout << std::endl;
*/
	int total_number_of_IDs_in_cells = std::accumulate(
			host_number_of_particle_IDs_per_cell.begin(),
			host_number_of_particle_IDs_per_cell.end(),
			0);

	int *device_particle_IDs_per_cell;
	CHECK_CUDA( cudaMalloc((void **)&device_particle_IDs_per_cell, sizeof(int)*total_number_of_IDs_in_cells) );
	CHECK_CUDA( cudaMemset(device_particle_IDs_per_cell, 0, sizeof(int)*total_number_of_IDs_in_cells) );

	int *device_indices_counter;
	CHECK_CUDA( cudaMalloc((void **)&device_indices_counter, sizeof(int)*N_cells) );
	CHECK_CUDA( cudaMemset(device_indices_counter, 0, sizeof(int)*N_cells) );

	get_particle_IDs_in_cells<<<grid,block>>>(
			N_particles, device_particles_ptr,
			N_cells, device_cells_ptr,
			device_number_of_particle_IDs_per_cell, // input
			device_particle_IDs_per_cell, // output
			device_indices_counter // output, for debugging
			);

/*
#define CHECK
#ifdef CHECK
	std::vector<int> host_number_of_particle_IDs_per_cell_second_kernel(N_cells);
	CHECK_CUDA( cudaMemcpy( host_number_of_particle_IDs_per_cell_second_kernel.data(),
				device_indices_counter,
				sizeof(int)*N_cells,
				cudaMemcpyDeviceToHost
				) );

	int total_number_of_IDs_in_cells_second_kernel = std::accumulate(
			host_number_of_particle_IDs_per_cell_second_kernel.begin(),
			host_number_of_particle_IDs_per_cell_second_kernel.end(),
			0);

	if (total_number_of_IDs_in_cells != total_number_of_IDs_in_cells_second_kernel) {
		std::cout << "number of particle IDs does not match between the two kernels: "
				<< total_number_of_IDs_in_cells << " != "
				<< total_number_of_IDs_in_cells_second_kernel << std::endl;

		std::exit(EXIT_FAILURE); // causes trouble in testing
	} else {
		std::cout << "first and second count gives the same number: " << total_number_of_IDs_in_cells << ": fine" << std::endl;
	}
#endif
#undef CHECK
*/
/*
	// TODO: transfer the results to the right place...
	// store the results here until no better solution has been implemented
	//worst case: copy back to the host Cell objects (DeviceToHost copy, not preferred, but currently needed)
	// ideally device to device copy
	std::vector<std::vector<int>> cell_particle_IDs(N_cells);
	int array_index = 0;
	int cell_ID = 0;
	for (auto const & number_of_particles_in_cell : host_number_of_particle_IDs_per_cell) {
		int chunk = number_of_particles_in_cell;
		cell_particle_IDs[cell_ID].resize(chunk);
		CHECK_CUDA( cudaMemcpy( cell_particle_IDs[cell_ID].data(),
					device_particle_IDs_per_cell + array_index,
					sizeof(int)*chunk,
					cudaMemcpyDeviceToHost
					) );
		array_index = array_index + chunk;
		cell_ID = cell_ID + 1;
	}

	std::cout<< "----" << std::endl;
	for (auto const& IDs: cell_particle_IDs) {
		for (auto const& ID : IDs) {
				std::cout << ID << '\t';
		}
		std::cout << std::endl;
	}
	std::cout<< "----" << std::endl;
*/
	// copy the collected particle IDs into the cells in the device
	// it is useful for testing, but the target is to keep everything on the device!
	int array_index = 0;
	for (int cell_ID=0; cell_ID<N_cells; cell_ID++) {
		size_t number_of_elements = host_number_of_particle_IDs_per_cell[cell_ID];
		cells[cell_ID].getParticleIDs().resize(number_of_elements);
		CHECK_CUDA( cudaMemcpy( cells[cell_ID].getParticleIDs().data(),
					device_particle_IDs_per_cell + array_index,
					sizeof(int)*number_of_elements,
					cudaMemcpyDeviceToHost
					) );
		array_index = array_index + number_of_elements;
	}

}

void Scene::advanceCuda() {
	if (benchmark_mode && simulation_time >= benchmark_simulation_time) { // in benchmark mode the simulation time is fixed
		if (viewer != nullptr) {
			// compilation difficulties when linking Qt to device code!
			// error "You must build your code with position independent code if Qt was built with -reduce-relocations.
			viewer->wrapStopButtonClicked();
		} else { // do not call the GUI stuff when we are GUI-less
			setFinished();
		}
	} else {
		simulation_time += time_step;
		int N_particles = particles.size();
		int threads = 256; // recommended first value, must not be larger than 1024
		int blocks = ceil(float(N_particles)/threads);
		// launching kernel on the GPU:
		particles_advance<<<blocks,threads>>>(time_step, device_particles_ptr, N_particles);
	}
	advanceCounter();
}

void Scene::collideWithBoundariesCellsCuda() {

}

void Scene::collideParticlesCellsCuda() {

}
