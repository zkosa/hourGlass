#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE passing-TEST
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE( passing_test )
{
	BOOST_REQUIRE(true);
}
