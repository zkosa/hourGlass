#include "scene.h"
#include "cuda.h"


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

	int Nx = scene.getDefaults().Nx;
	//Nx = 2; // use less cells, for easier GPU profiling
	int Ny = Nx;
	Cell::setNx(Nx);
	Cell::setNy(Ny);

	scene.createCells();
	scene.populateCells();
	scene.resolveConstraintsOnInitCells(5);
	scene.populateCells();

	cudaDeviceSynchronize();
	scene.setRunning();
	while ( scene.isRunning() ) {
		//scene.calculatePhysics();
		scene.calculatePhysicsCuda();
	}
}

