#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE cuda-TEST
#include <boost/test/unit_test.hpp>
#include "test.cuh"


BOOST_AUTO_TEST_CASE( cuda_test )
{
	printCudaVersion();
	printGpuDeviceInfo();
}
