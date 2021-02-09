#include "scene.h"
#include "mainwindow.h"
#include <device_launch_parameters.h> // just for proper indexing, nvcc includes it anyhow


void Scene::hostToDevice() {
	// TODO: lock data on host!

	int N_particles = particles.size();
	CHECK_CUDA( cudaMalloc((void **)&device_particles_ptr, N_particles*sizeof(Particle)) );
	CHECK_CUDA( cudaMemcpy(device_particles_ptr, &particles[0],
				N_particles*sizeof(Particle),
				cudaMemcpyHostToDevice) );

	int N_cells = cells.size();
	CHECK_CUDA( cudaMalloc((void **)&device_cells_ptr, N_cells*sizeof(Cell)) );
	CHECK_CUDA( cudaMemcpy( device_cells_ptr, &cells[0],
							N_cells*sizeof(Cell),
							cudaMemcpyHostToDevice) );

	int N_boundaries_ax = boundaries_ax.size();
	CHECK_CUDA( cudaMalloc((void **)&device_boundaries_ax_ptr, N_boundaries_ax*sizeof(Boundary_axissymmetric)) );
	CHECK_CUDA( cudaMemcpy( device_boundaries_ax_ptr,
							&boundaries_ax[0],
							N_boundaries_ax*sizeof(Boundary_axissymmetric),
							cudaMemcpyHostToDevice) );
	// address of function handle is not valid on the device --> recreate it:
	initializeFunctionHandle<<<1,1>>>(device_boundaries_ax_ptr); CHECK_CUDA_POST

	int N_boundaries_pl = boundaries_pl.size();
	CHECK_CUDA( cudaMalloc((void **)&device_boundaries_pl_ptr, N_boundaries_pl*sizeof(Boundary_planar)) );
	CHECK_CUDA( cudaMemcpy( device_boundaries_pl_ptr,
							&boundaries_pl[0],
							N_boundaries_pl*sizeof(Boundary_planar),
							cudaMemcpyHostToDevice) );
}

void Scene::deviceToHost() {

	// copy the particles back for display purposes
	int N_particles = particles.size();
	//CHECK_CUDA_POINTER( device_particles_ptr );
	CHECK_CUDA( cudaMemcpy( particles.data(),
				device_particles_ptr,
				N_particles*sizeof(Particle),
				cudaMemcpyDeviceToHost
				) );

	// TODO: protect against overwriting
	/* it has been copied in Scene::populateCellsCuda() */
//	int N_cells = cells.size();
//	CHECK_CUDA( cudaMemcpy( cells.data(),
//				device_cells_ptr,
//				N_cells*sizeof(Cell), // TODO: how does it know the changed amount of particle IDS, stored in a vector? (resize particle_IDs?)
//				cudaMemcpyDeviceToHost
//				) );

	// boundaries do not change (can it be enforced?) --> no need to copy


	CHECK_CUDA( cudaFree(device_particles_ptr) );
	CHECK_CUDA( cudaFree(device_cells_ptr) );
	CHECK_CUDA( cudaFree(device_boundaries_ax_ptr) );
	CHECK_CUDA( cudaFree(device_boundaries_pl_ptr) );

	// TODO: unlock data on host
}

void Scene::deviceToHost_Cells() {

	std::vector<int> host_number_of_particle_IDs_per_cell(N_cells);
	CHECK_CUDA( cudaMemcpy( host_number_of_particle_IDs_per_cell.data(),
				device_number_of_particle_IDs_per_cell,
				sizeof(int)*N_cells,
				cudaMemcpyDeviceToHost
				) );

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

__global__
void get_number_of_particles_per_cell(
		int number_of_particles, const Particle *p,
		int number_of_cells, const Cell *c,
		int *number_of_particle_IDs_per_cell
		)
{
	// nested (2D) grid-stride loop
	int index_cell = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_cell = blockDim.x * gridDim.x;
	int index_particle= blockIdx.y * blockDim.y + threadIdx.y;
	int stride_particle = blockDim.y * gridDim.y;

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
		const int *IN_number_of_particleIDs, // per cell, as input
		int *OUT_particle_IDs_in_cells,
		int *OUT_particle_ID_counter // per cell, for counting
		)
{
	// nested (2D) grid-stride loop
	int index_cell = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_cell = blockDim.x * gridDim.x;
	int index_particle = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_particle = blockDim.y * gridDim.y;

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

	// this->clearCells(); // do we need something like this?

	int N_particles = particles.size();

	dim3 threads(1, std::min(N_cells, 1024)); // all cells are within a block with usual number of cells
	dim3 blocks((N_particles + threads.x - 1)/threads.x, (N_cells + threads.y - 1)/threads.y);
	// std::cout << blocks.x << "x" << blocks.y << " X " << threads.x << "x" << threads.y << std::endl;

	CHECK_CUDA( cudaMalloc((void **)&device_number_of_particle_IDs_per_cell, sizeof(int)*N_cells) );
	CHECK_CUDA( cudaMemset(device_number_of_particle_IDs_per_cell, 0, sizeof(int)*N_cells) );

	get_number_of_particles_per_cell<<<blocks,threads>>>(
			N_particles,
			device_particles_ptr,
			N_cells,
			device_cells_ptr,
			device_number_of_particle_IDs_per_cell); CHECK_CUDA_POST


	std::vector<int> host_number_of_particle_IDs_per_cell(N_cells);
	CHECK_CUDA( cudaMemcpy( host_number_of_particle_IDs_per_cell.data(),
				device_number_of_particle_IDs_per_cell,
				sizeof(int)*N_cells,
				cudaMemcpyDeviceToHost
				) );

	total_number_of_IDs_in_cells = std::accumulate(
			host_number_of_particle_IDs_per_cell.begin(),
			host_number_of_particle_IDs_per_cell.end(),
			0);

	CHECK_CUDA( cudaMalloc((void **)&device_particle_IDs_per_cell, sizeof(int)*total_number_of_IDs_in_cells) );
	CHECK_CUDA( cudaMemset(device_particle_IDs_per_cell, 0, sizeof(int)*total_number_of_IDs_in_cells) );

	int *device_indices_counter;
	CHECK_CUDA( cudaMalloc((void **)&device_indices_counter, sizeof(int)*N_cells) );
	CHECK_CUDA( cudaMemset(device_indices_counter, 0, sizeof(int)*N_cells) );

	get_particle_IDs_in_cells<<<blocks,threads>>>(
			N_particles, device_particles_ptr,
			N_cells, device_cells_ptr,
			device_number_of_particle_IDs_per_cell, // input
			device_particle_IDs_per_cell, // output
			device_indices_counter // output, for debugging
			); CHECK_CUDA_POST
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

__global__
void collide_with_boundaries(
		Particle *p, int number_of_particles,
		const Boundary_axissymmetric *boundaries_ax_ptr, int N_boundaries_ax,
		const Boundary_planar *boundaries_pl_ptr, int N_boundaries_pl
		)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i_p = index; i_p < number_of_particles; i_p += stride ) {
		for (int i_bax = 0; i_bax<N_boundaries_ax; i_bax += 1) {
			if ((boundaries_ax_ptr + i_bax)->distanceDev((p + i_p)) < (p + i_p)->getR()) {
				(p + i_p)->collideToWall(boundaries_ax_ptr + i_bax);
				// to_be_collided.emplace_back(p, b); // more sophisticated method is used on the CPU!!!
			}
		}
		for (int i_bpl = 0; i_bpl<N_boundaries_pl; i_bpl += 1) {
			if ((boundaries_pl_ptr + i_bpl)->distanceDev((p + i_p)) < (p + i_p)->getR()) {
				(p + i_p)->collideToWall(boundaries_pl_ptr + i_bpl);
				// to_be_collided.emplace_back(p, b); // more sophisticated method is used on the CPU!!!
			}
		}
	}
}

void Scene::collideWithBoundariesCellsCuda() {
	// we could make use of the getIDsOfParticlesInCellsWithBoundary();
	// and collide only in cells with boundaries

	int N_particles = particles.size();
	int N_boundaries_ax = boundaries_ax.size();
	int N_boundaries_pl = boundaries_pl.size();

	dim3 threads(std::min(N_particles, 256), 1); // all cells are within the same block with usual number of cells
	dim3 blocks((N_particles + threads.x - 1)/threads.x, 1);
	//std::cout << blocks.x << "x" << blocks.y << " X " << threads.x << "x" << threads.y << std::endl;
	collide_with_boundaries<<<blocks, threads>>>(
			device_particles_ptr, N_particles,
			device_boundaries_ax_ptr, N_boundaries_ax,
			device_boundaries_pl_ptr, N_boundaries_pl
			); CHECK_CUDA_POST

}

__global__
void collide_particles_cellwise(
		int number_of_particles, Particle *p,
		int number_of_cells, const int *number_of_particleIDs_per_cell,
		int number_of_particleIDs, const int *particle_IDs_in_cells
		)
{
	int index_current = blockIdx.x * blockDim.x + threadIdx.x;
	int stride_current = blockDim.x * gridDim.x;
	int index_other = blockIdx.y * blockDim.y + threadIdx.y;
	int stride_other = blockDim.y * gridDim.y;

	int starting_index = 0;
	int i_p_current;
	int number_of_particleIDs_in_current_cell;
	for (int i_c = 0; i_c < number_of_cells; i_c++) {
		number_of_particleIDs_in_current_cell = *(number_of_particleIDs_per_cell + i_c);
		for (int i_p = index_current; i_p < number_of_particleIDs_in_current_cell; i_p+=stride_current) {
			i_p_current = *(particle_IDs_in_cells + starting_index + i_p);
			for (int i_p_other = index_other; i_p_other < number_of_particleIDs_in_current_cell; i_p_other+=stride_other) {
				(p + i_p_current)->collideToParticle(p + *(particle_IDs_in_cells + starting_index + i_p_other));
			}
		}
		starting_index += number_of_particleIDs_in_current_cell;
	}
}

int roundToNextIntPowerOfTwo(float f) {
	return std::pow(2, std::ceil(std::log2(f)));
}

void Scene::collideParticlesCellsCuda() {

	int N_particles = particles.size();

	// estimating the square root of the necessary threads x blocks
	// the grid stride loop handles the error in estimation
	int n = roundToNextIntPowerOfTwo(N_particles/float(N_cells));
	// use equal x and y dimensions:
	dim3 threads(16, 16);
	dim3 blocks((n + threads.x - 1)/threads.x, (n + threads.y - 1)/threads.y);

	collide_particles_cellwise<<<blocks,threads>>>(
			N_particles, device_particles_ptr,
			N_cells, device_number_of_particle_IDs_per_cell,
			total_number_of_IDs_in_cells, device_particle_IDs_per_cell
			);  CHECK_CUDA_POST
}
