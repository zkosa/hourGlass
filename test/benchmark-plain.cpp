#include "scene.h"
#include "cuda.h"

// Benchmark without google benchmark library:
// More straightforward syntax, easier debugging

int main() {
	Scene scene;
	cudaFree(nullptr);
	scene.setBenchmarkMode(true);
	// make the benchmark shorter for easier GPU profiling
	//scene.setBenchmarkSimulationTime(5 * scene.getTimeStep());

	scene.applyDefaults();
	int N = scene.getDefaults().number_of_particles;
	scene.setNumberOfParticles(N);
	scene.addParticles(scene.getNumberOfParticles());

	// We can use less cells for easier debugging
	int Nx = scene.getDefaults().Nx; // 2
	int Ny = Nx;
	Cell::setNx(Nx);
	Cell::setNy(Ny);

	scene.createCells();
	scene.populateCells();
	//scene.resolveConstraintsOnInitCells(5);

	cudaDeviceSynchronize();

	scene.setRunning();
	while ( scene.isRunning() ) {
		//scene.calculatePhysics();
		scene.calculatePhysicsCuda();
	}

}

