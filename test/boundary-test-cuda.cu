#include "devtools.h"
#include "boundary_planar.h"
#include "boundary_axissymmetric.h"
#include "particle.h"
#include "scene.h"
#include "cuda.h"

// fix necessary before BOOST 1.72
#if defined(__NVCC__)
#define BOOST_PP_VARIADICS 1
#endif

//#define BOOST_TEST_TOOLS_UNDER_DEBUGGER // it deactivates the tolerance!!!
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE boundary-cuda-TEST
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/array.hpp>


static const boost::array< float, 3 > horizontal_offset_data{-1e-3f, 0.0f, 1e-3f};

BOOST_DATA_TEST_CASE( collide_planar_arbivel_GPU_test, horizontal_offset_data, offset_ )
{
	// Check if a particle on the wall is properly moved back to the domain
	// Arbitrary velocity vector

	float midpoint_offset = offset_;
	float corner = 0.999f;
	Boundary_planar ground(Vec3d(-1, -corner, 0), Vec3d(1, -corner, 0),
					Vec3d(-1, -corner, 1));

	auto tolerance = boost::test_tools::tolerance(1e-6f);
	auto tolerance_large = boost::test_tools::tolerance(2e-5f);

	Vec3d point(0.0f, -corner + midpoint_offset, 0.0f);
	Vec3d vel(2.0f, -3.0f, 0.0f);

	Particle p(point, vel);

	// check that the point and the particle are really at the expected distance from the surface before collision:
	BOOST_TEST_REQUIRE( ground.distance(point) == std::abs(midpoint_offset), tolerance_large );
	BOOST_TEST_REQUIRE( ground.distance(p) == std::abs(midpoint_offset), tolerance_large );
	BOOST_TEST_REQUIRE( ground.distanceSigned(point) == midpoint_offset, tolerance_large );
	BOOST_TEST_REQUIRE( ground.distanceSigned(p) == midpoint_offset, tolerance_large );

	// prepare for collision on the GPU:

	std::vector<Boundary_planar> boundaries_pl;
	boundaries_pl.push_back(ground);
	int N_boundaries_pl = boundaries_pl.size();
	Boundary_planar* device_boundaries_pl_ptr = nullptr;
	CHECK_CUDA( cudaMalloc((void **)&device_boundaries_pl_ptr, N_boundaries_pl*sizeof(Boundary_planar)) );
	CHECK_CUDA( cudaMemcpy( device_boundaries_pl_ptr,
							&boundaries_pl[0],
							N_boundaries_pl*sizeof(Boundary_planar),
							cudaMemcpyHostToDevice) );

	std::vector<Particle> particles;
	particles.push_back(p);
	int N_particles = particles.size();
	Particle* device_particles_ptr = nullptr;
	CHECK_CUDA( cudaMalloc((void **)&device_particles_ptr, N_particles*sizeof(Particle)) );
	CHECK_CUDA( cudaMemcpy(device_particles_ptr, &particles[0],
				N_particles*sizeof(Particle),
				cudaMemcpyHostToDevice) );

	// collide on the GPU:

	dim3 threads(std::min(N_particles, 256), 1); // all cells are within a block with usual number of cells
	dim3 blocks((N_particles + threads.x - 1)/threads.x, 1);
	//std::cout << blocks.x << "x" << blocks.y << " X " << threads.x << "x" << threads.y << std::endl;

	collide_with_boundaries<<<blocks, threads>>>(
			device_particles_ptr, N_particles,
			nullptr, 0,
			device_boundaries_pl_ptr, N_boundaries_pl
			);	 CHECK_CUDA_POST

	// copy the results to the host

	CHECK_CUDA( cudaMemcpy( particles.data(),
				device_particles_ptr,
				N_particles*sizeof(Particle),
				cudaMemcpyDeviceToHost
				) );

	CHECK_CUDA( cudaMemcpy( boundaries_pl.data(),
			device_boundaries_pl_ptr,
				N_particles*sizeof(Boundary_planar),
				cudaMemcpyDeviceToHost
				) );

	// check that the point is at touching distance (radius) after collision:
	BOOST_TEST_REQUIRE( ground.distance(particles[0]) == particles[0].getR(), tolerance );

	CHECK_CUDA( cudaFree(device_particles_ptr) );
	CHECK_CUDA( cudaFree(device_boundaries_pl_ptr) );
}
