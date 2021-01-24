#include "cell.h"
#include "boundary_planar.h"
#include "boundary_axissymmetric.h"

#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE cell-TEST
#include <boost/test/unit_test.hpp>

#define FLOAT_TEST_PRECISION 1e-6f


BOOST_AUTO_TEST_CASE( helpers_test, * boost::unit_test::tolerance(FLOAT_TEST_PRECISION) )
{
	Vec3d i(Vec3d::i);
	pointData pd_is{i, i, i, i};
	pointData pd_is2{i, i, -i, -i};

	BOOST_TEST( Cell::average(pd_is) == i );
	BOOST_REQUIRE_EQUAL( Cell::average(pd_is2), Vec3d(0,0,0) );
}

BOOST_AUTO_TEST_CASE( construction_test, * boost::unit_test::tolerance(FLOAT_TEST_PRECISION) )
{
	Vec3d center(0,0,0);
	Vec3d dX(0.1,0.1,0.1);
	Cell::setDX(dX);

	Cell c(center);

	// prints the size of a cell object
	c.size();

	BOOST_REQUIRE_EQUAL( c.getCenter(), center );
	BOOST_REQUIRE_EQUAL( c.getHalfDiagonal(),  abs(dX)/2.0f );
	BOOST_REQUIRE_EQUAL( abs(c.getCorners()[0] - c.getCorners()[7]),  abs(dX) );

}

BOOST_AUTO_TEST_CASE( average_test, * boost::unit_test::tolerance(FLOAT_TEST_PRECISION) )
{
	Vec3d center(0,0,0);
	Vec3d dX(0.1,0.1,0.1);
	Cell::setDX(dX);

	Cell c(center);
	Cell c2(center+dX);

	BOOST_TEST( abs(Cell::average(c.getCorners())) == abs(center) );
	BOOST_TEST( Cell::average(c.getCorners()).x == center.x );
	BOOST_TEST( Cell::average(c.getCorners()).y == center.y );
	BOOST_TEST( Cell::average(c.getCorners()).z == center.z );

	BOOST_TEST( abs(Cell::average(c2.getCorners())) == abs(center+dX) );
	BOOST_TEST( Cell::average(c2.getCorners()).x == (center+dX).x );
	BOOST_TEST( Cell::average(c2.getCorners()).y == (center+dX).y );
	BOOST_TEST( Cell::average(c2.getCorners()).z == (center+dX).z );


	BOOST_TEST( abs(Cell::average(c.getFaceCenters())) == abs(center) );
	BOOST_TEST( Cell::average(c.getFaceCenters()).x == center.x );
	BOOST_TEST( Cell::average(c.getFaceCenters()).y == center.y );
	BOOST_TEST( Cell::average(c.getFaceCenters()).z == center.z );

	BOOST_TEST( abs(Cell::average(c2.getFaceCenters())) == abs(center+dX) );
	BOOST_TEST( Cell::average(c2.getFaceCenters()).x == (center+dX).x );
	BOOST_TEST( Cell::average(c2.getFaceCenters()).y == (center+dX).y );
	BOOST_TEST( Cell::average(c2.getFaceCenters()).z == (center+dX).z );


	BOOST_TEST( abs(Cell::average(c.getEdgeCenters())) == abs(center) );
	BOOST_TEST( Cell::average(c.getEdgeCenters()).x == center.x );
	BOOST_TEST( Cell::average(c.getEdgeCenters()).y == center.y );
	BOOST_TEST( Cell::average(c.getEdgeCenters()).z == center.z );

	BOOST_TEST( abs(Cell::average(c2.getEdgeCenters())) == abs(center+dX) );
	BOOST_TEST( Cell::average(c2.getEdgeCenters()).x == (center+dX).x );
	BOOST_TEST( Cell::average(c2.getEdgeCenters()).y == (center+dX).y );
	BOOST_TEST( Cell::average(c2.getEdgeCenters()).z == (center+dX).z );

	BOOST_TEST( abs(Cell::average(c.getAllPoints())) == abs(center) );
	BOOST_TEST( Cell::average(c.getAllPoints()).x == center.x );
	BOOST_TEST( Cell::average(c.getAllPoints()).y == center.y );
	BOOST_TEST( Cell::average(c.getAllPoints()).z == center.z );

	BOOST_TEST( abs(Cell::average(c2.getAllPoints())) == abs(center+dX) );
	BOOST_TEST( Cell::average(c2.getAllPoints()).x == (center+dX).x );
	BOOST_TEST( Cell::average(c2.getAllPoints()).y == (center+dX).y );
	BOOST_TEST( Cell::average(c2.getAllPoints()).z == (center+dX).z );
}

BOOST_AUTO_TEST_CASE( boundary_planar_test )
{
	float corner = 0.999;
	Boundary_planar ground(
			Vec3d(-1, -corner, 0),
			Vec3d(1, -corner, 0),
			Vec3d(-1, -corner, 1)
			);

	Vec3d center(0,-corner,0);
	Vec3d dX(0.1,0.1,0.1);
	Cell::setDX(dX);

	Cell c_crossing(center);
	Cell c_touching(center + 0.5 * dX.y * Vec3d::j);
	Cell c_above(   center + 1.5 * dX.y * Vec3d::j);

	BOOST_REQUIRE_EQUAL( c_crossing.containsBoundary( ground ) , true );
	BOOST_REQUIRE_EQUAL( c_touching.containsBoundary( ground ) , true );
	BOOST_REQUIRE_EQUAL( c_above.containsBoundary( ground ) , false );
}

BOOST_AUTO_TEST_CASE( boundary_axissymmetric_test )
{
	Boundary_axissymmetric glass;

	Vec3d center(0,0,0);

	Vec3d dX_small(0.01,0.01,0.01);
	Cell::setDX(dX_small);
	Cell c_small(center);

	BOOST_REQUIRE_EQUAL( c_small.containsBoundary( glass ), false );


	Vec3d dX_large(0.2,0.2,0.2);
	Cell::setDX(dX_large);
	Cell c_large(center);

	BOOST_REQUIRE_EQUAL( c_large.containsBoundary( glass ), true );


	Vec3d dX_touching(2*0.07,2*0.07,2*0.07);
	Cell::setDX(dX_touching);
	Cell c_touching(center);

	BOOST_REQUIRE_EQUAL( c_touching.containsBoundary( glass ), true );

	// repeated execution fails
/*
	BOOST_REQUIRE_EQUAL( c_small.contains( glass ), false );
	BOOST_REQUIRE_EQUAL( c_large.contains( glass ), true );
*/
}
