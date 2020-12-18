#include "scene.h"
#include <iostream>

void Scene::populateCellsCuda() {

}

void Scene::advanceCuda() {
	if (benchmark_mode && simulation_time >= benchmark_simulation_time) { // in benchmark mode the simulation time is fixed

//		 //#error "You must build your code with position independent code if Qt was built with -reduce-relocations.
//		if (viewer != nullptr) {
//			viewer->wrapStopButtonClicked();
//		} else { // do not call the GUI stuff when we are GUI-less
//			setFinished();
//		}


		setFinished(); // fix
		printf("The benchmark has been finished.\n");
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
