#include <iostream>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE boosttest-TEST
#include <boost/test/unit_test.hpp>

// these are rather "experiments", than tests
// fails do not necessarily indicate a problem
// disable the test by default

// looking for the default tolerance
BOOST_AUTO_TEST_CASE( float_test_test )
{
	BOOST_TEST( 1.0f == 1.0f + 1.2e-9f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-8f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-7f ); // this fails first when no tol --> default tolerance is 1e-7?
	BOOST_TEST( 1.0f == 1.0f + 1.2e-6f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-5f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-4f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-3f );
}

// tolerance is relative (and not in percent)
BOOST_AUTO_TEST_CASE( float_test_tolerance_test, * boost::unit_test::tolerance(1e-4f) )
{
	BOOST_TEST( 1.0f == 1.0f + 1.2e-9f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-8f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-7f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-6f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-5f );
	BOOST_TEST( 1.0f == 1.0f + 1.2e-4f ); // this fails first --> tolerance is not in percent
	BOOST_TEST( 1.0f == 1.0f + 1.2e-3f );
}

// test with different assertions
BOOST_AUTO_TEST_CASE( float_test_tolerance_assertions_test, * boost::unit_test::tolerance(1e-4f) )
{
	// BOOST_TEST takes the tolerance settings from the BOOST_AUTO_TEST_CASE
	BOOST_TEST( 1.0f == 1.0f + 1.2e-6f ); // passes
	BOOST_TEST_REQUIRE( 1.0f == 1.0f + 1.2e-6f ); // passes
	// it does not work for other test types, such as BOOST_CHECK
	BOOST_CHECK( 1.0f == 1.0f + 1.2e-6f ); // fails
	BOOST_CHECK_EQUAL( 1.0f, 1.0f + 1.2e-6f ); // fails
}

// test with different types
BOOST_AUTO_TEST_CASE( float_test_tolerance_types_test, * boost::unit_test::tolerance(1e-4f) )
{
	// tolerance effects only the specified type
	BOOST_TEST( 1.0f == 1.0f + 1.2e-6f ); // passes
	BOOST_TEST( 1.0  == 1.0f + 1.2e-6f ); // fails
	BOOST_TEST( 1.0f == 1.0  + 1.2e-6  ); // fails
	BOOST_TEST( 1.0  == 1.0  + 1.2e-6  ); // fails
	BOOST_TEST( 1    == 1.0  + 1.2e-6  ); // fails
	BOOST_TEST( 1    == 1    + 1.2e-6  ); // fails
	BOOST_TEST( 1.0f == 1   + 1.2e-6f  ); // passes
}

// test with zero
BOOST_AUTO_TEST_CASE( float_test_tolerance_zero_test, * boost::unit_test::tolerance(1e-4f) )
{
	BOOST_TEST( 0.0f == 0.0f + 0.2e-6f ); // passes
	BOOST_TEST( 0.0  == 0.0f + 0.2e-6f ); // fails
	BOOST_TEST( 0.0f == 0.0  + 0.2e-6  ); // fails
	BOOST_TEST( 0.0  == 0.0  + 0.2e-6  ); // fails
	BOOST_TEST( 0    == 0.0  + 0.2e-6  ); // fails
	BOOST_TEST( 0    == 0    + 0.2e-6  ); // fails
	BOOST_TEST( 0.0f == 0    + 0.2e-6f ); // passes
}


// test with specific tolerance settings per test
BOOST_AUTO_TEST_CASE( float_test_tolerance_per_test_test )
{
	BOOST_TEST( 1.0f == 1.0f + 1.2e-6f ); // fails
	BOOST_TEST( 1.0f == 1.0f + 1.2e-6f, boost::test_tools::tolerance(1e-4f) ); // passes
	BOOST_TEST_REQUIRE( 1.0f == 1.0f + 1.2e-6f, boost::test_tools::tolerance(1e-4f) ); // passes}
}
