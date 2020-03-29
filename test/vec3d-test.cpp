#include "vec3d.h"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE vec3d-TEST
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( vec3d_test )
{
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

	BOOST_REQUIRE_EQUAL( v2.large(), false );
	BOOST_REQUIRE_EQUAL( (v2*1e25).large(), true );

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
}
