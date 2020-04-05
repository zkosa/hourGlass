#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE failing-TEST
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_CASE( failing_test )
{
	BOOST_REQUIRE(false);
}
