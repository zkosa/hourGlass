#include "scene.h"
//#include "mainwindow.h"

void Scene::hostToDevice() {
	// TODO: lock data on host!

	int N_particles = particles.size();
	cudaMalloc((void **)&device_particles_ptr, N_particles*sizeof(Particle));
	cudaMemcpy(device_particles_ptr, &particles[0], N_particles*sizeof(Particle), cudaMemcpyHostToDevice);

	int N_cells = cells.size();
	cudaMalloc((void **)&device_cells_ptr, N_cells*sizeof(Cell));
	cudaMemcpy(device_cells_ptr, &cells[0], N_cells*sizeof(Cell), cudaMemcpyHostToDevice);

	int N_boundaries_ax = boundaries_ax.size();
	cudaMalloc((void **)&device_boundaries_ax_ptr, N_boundaries_ax*sizeof(Boundary_axissymmetric));
	cudaMemcpy(device_boundaries_ax_ptr, &boundaries_ax[0], N_boundaries_ax*sizeof(Boundary_axissymmetric), cudaMemcpyHostToDevice);

	int N_boundaries_pl = boundaries_pl.size();
	cudaMalloc((void **)&device_boundaries_pl_ptr, N_boundaries_pl*sizeof(Boundary_planar));
	cudaMemcpy(device_boundaries_pl_ptr, &boundaries_pl[0], N_boundaries_pl*sizeof(Boundary_planar), cudaMemcpyHostToDevice);

}

void Scene::deviceToHost() {

	// copy the particles back for display purposes
	int N_particles = particles.size();
	cudaMemcpy( particles.data(),
				device_particles_ptr,
				N_particles*sizeof(Particle),
				cudaMemcpyDeviceToHost
				);

	// cell geometry does not change, particle_IDs are not needed on host --> no need to copy

	// boundaries do not change (can it be enforced???) --> no need to copy


	cudaFree(device_particles_ptr);
	cudaFree(device_cells_ptr);
	cudaFree(device_boundaries_ax_ptr);
	cudaFree(device_boundaries_pl_ptr);

	// TODO: unlock data on host
}

void Scene::populateCellsCuda() {

}

void Scene::advanceCuda() {
	if (benchmark_mode && simulation_time >= benchmark_simulation_time) { // in benchmark mode the simulation time is fixed
		/*
		 //#error "You must build your code with position independent code if Qt was built with -reduce-relocations.
		if (viewer != nullptr) {
			viewer->wrapStopButtonClicked();
		} else { // do not call the GUI stuff when we are GUI-less
			setFinished();
		}
		*/

		setFinished(); // fix
		printf("The benchmark has been finished.\n");
	} else {
		simulation_time += time_step;
		for (auto &p : particles) {
			//p.advance(time_step);
		}
	}
	advanceCounter();
}

void Scene::collideWithBoundariesCellsCuda() {

}

void Scene::collideParticlesCellsCuda() {

}
