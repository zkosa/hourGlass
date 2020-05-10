#include "vec3d.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE vec3d-TEST
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>


BOOST_AUTO_TEST_CASE( vec3d_test, * boost::unit_test::tolerance(1e-6f) )
{
	Vec3d nullv{0,0,0};
	Vec3d i{1,0,0};
	i = Vec3d::i;
	Vec3d j{0,1,0};
	Vec3d k{0,0,1};
	Vec3d v1{1,1,1};
	Vec3d v2{2,2,2};

	BOOST_REQUIRE_EQUAL( v1+v2, Vec3d(3,3,3) );
	BOOST_REQUIRE_EQUAL( v2+v1, Vec3d(3,3,3) );

	BOOST_REQUIRE_EQUAL( 2*v2-v2, v2 );

	BOOST_REQUIRE_EQUAL( v1*1, v1 );
	BOOST_REQUIRE_EQUAL( v1*2, v2 );
	BOOST_REQUIRE_EQUAL( 2*v1, v2 );
	BOOST_REQUIRE_EQUAL( v1*v2, 6 );
	BOOST_REQUIRE_EQUAL( v2*v1, 6 );

	BOOST_REQUIRE_EQUAL( v2/2, 0.5*v2 );
	BOOST_REQUIRE_EQUAL( v2/2, v1 );
	BOOST_REQUIRE_EQUAL( v2*2, v2/0.5 );
	BOOST_CHECK_THROW( v2/0, std::invalid_argument );

	BOOST_REQUIRE_EQUAL( v2.isLarge(), false );
	BOOST_REQUIRE_EQUAL( (1e25*v2).isLarge(), true );
	BOOST_REQUIRE_EQUAL( v2.isSmall(), false );
	BOOST_REQUIRE_EQUAL( (1e-25*v2).isSmall(), true );

	BOOST_REQUIRE_EQUAL( nullv, Vec3d::null );
	BOOST_REQUIRE_EQUAL( abs(nullv), 0 );
	BOOST_REQUIRE_EQUAL( norm(nullv), nullv );
	BOOST_REQUIRE_EQUAL( abs(norm(nullv)), 0 );
	BOOST_REQUIRE_EQUAL( abs(Vec3d{1,0,0}), 1 );
	BOOST_REQUIRE_EQUAL( crossProduct(i,j), k );

	BOOST_REQUIRE_EQUAL( Vec3d::i, i );

	float x = 1;
	float y = 2;
	float z = 3;
	Vec3d v3{x,y,z};
	Vec3d minv3{-x,-y,-z};

	BOOST_REQUIRE_EQUAL( -v3, minv3 );
	BOOST_REQUIRE_EQUAL( v3, -minv3 );

	x = std::sqrt(2);
	y = 1;
	z = std::sqrt(2);
	Vec3d v4{x,y,z};
	BOOST_REQUIRE_EQUAL( v4.toYAxial(), VecAxiSym(y, std::sqrt(x*x + z*z)) );

	// the absolute values must be equal independently of coordinate system:
	BOOST_TEST_REQUIRE( abs(v1) == abs(v1.toYAxial()) );
	BOOST_TEST_REQUIRE( abs(v2) == abs(v2.toYAxial()) );
	BOOST_TEST_REQUIRE( abs(v3) == abs(v3.toYAxial()) );
	BOOST_TEST_REQUIRE( abs(v4) == abs(v4.toYAxial()) );
	BOOST_TEST_REQUIRE( abs(v4) == abs(v4.toYAxial()) ); // passes with global 1e-4f, fails with 1e-7f
	//BOOST_CHECK_EQUAL( abs(v4), abs(v4.toYAxial()) ); // fails, independently of global tolerance


	boost::test_tools::output_test_stream output_test;

	output_test << v3;
	BOOST_TEST( output_test.is_equal( "(1, 2, 3)" ) );

	// checking repeated execution:
	output_test << v3;
	BOOST_TEST( output_test.is_equal( "(1, 2, 3)" ) );

	output_test << v3.toYAxial();
	BOOST_TEST( output_test.is_equal( "(axial: 2, radial: 3.16228)" ) );

	v3.print();
	// TODO: implement test
	v3.toYAxial().print(); // TODO: fix
	// TODO: implement test

	Vec3d v5 = v3;
	Vec3d v6 = v5;
	v6 += v5;
	BOOST_TEST_REQUIRE( 2*v5 == v6 );

	v6 -= v5;
	BOOST_TEST_REQUIRE( v5 == v6 );

	v6 *= 2;
	BOOST_TEST_REQUIRE( 2*v5 == v6 );

	v6 /= 2;
	BOOST_TEST_REQUIRE( v5 == v6 );

	BOOST_CHECK_THROW( v6 /= 0;, std::invalid_argument );
}
