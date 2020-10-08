#include "test.cuh"
#include "scene.h"

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER // it deactivates the tolerance!!!
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuda-TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( cuda_CudaDevicesAvailable_test )
{
	// sometimes CUDA looses sight of the GPU (Ubuntu 20.04)
	// a restart can help to resolve the issue
	BOOST_TEST( areCudaDevicesAvailable() );
}

BOOST_AUTO_TEST_CASE( cuda_info_test )
{
	printCudaVersion();
	printGpuDeviceInfo();

	//cudaSetDevice(0);
}

void prepareTest(Scene& scene) {
	Particle::connectScene(&scene);
	Cell::connectScene(&scene);

	Cell::setNx(2);
	Cell::setNy(1);
	Cell::setNz(1);
	scene.setNumberOfParticles(3);
	Particle::setUniformRadius(0.005f);

	Particle::resetLastID();
	float y_position = 0.5f;
	scene.addParticles(scene.getNumberOfParticles(), y_position, Particle::getUniformRadius(), false);

	scene.createCells();
}

std::ostream& operator<<(std::ostream& os, const std::vector<int> &input)
{
	for (auto const& i: input) {
		os << i << " ";
	}
	return os;
}

void printResults(Scene& scene) {
	std::cout << scene.getCells().size() << std::endl;
	for (size_t i=0; i<scene.getCells().size(); i++) {
		std::cout << i << " "
				<< scene.getCells()[i].getCenter() << " : "
				<< scene.getCells()[i].getParticleIDs().size() << " : "
				<< scene.getCells()[i].getParticleIDs() << std::endl;
	}

	for (size_t i=0; i<scene.getParticles().size(); i++) {
		std::cout << i << " " <<scene.getParticles()[i].getPos() << std::endl;
	}
}

bool compareResults(Scene& scene1, Scene& scene2) {

	std::vector<std::vector<int>> particle_IDs_scene1;
	for (auto const& c : scene1.getCells())  {
		particle_IDs_scene1.push_back(c.getParticleIDs());
	}
	std::vector<std::vector<int>> particle_IDs_scene2;
	for (auto const& c : scene2.getCells())  {
		particle_IDs_scene2.push_back(c.getParticleIDs());
	}

	return (particle_IDs_scene1 == particle_IDs_scene2);
}

BOOST_AUTO_TEST_CASE( cuda_populateCells_test )
{

	std::cout << "Serial ------- " << std::endl;

	Scene scene;
	prepareTest(scene);
	scene.populateCells();
	printResults(scene);

	std::cout << "CUDA ------- " << std::endl;

	Scene sceneCuda;
	prepareTest(sceneCuda);
	sceneCuda.populateCellsCuda();
	printResults(sceneCuda);

	BOOST_TEST( compareResults(scene, sceneCuda) );
}
