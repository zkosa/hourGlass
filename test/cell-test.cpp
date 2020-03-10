#include "cell.h"

#define BOOST_TEST_DYN_LINK

#define BOOST_TEST_MODULE cell-TEST
#include <boost/test/unit_test.hpp>

#define FLOAT_TEST_PRECISION 1e-6f


BOOST_AUTO_TEST_CASE( helpers_test, * boost::unit_test::tolerance(FLOAT_TEST_PRECISION) )
{
	Vec3d i(Vec3d::i);
	pointData pd_is{i, i, i, i};
	pointData pd_is2{i, i, -1*i, -1*i};

	BOOST_TEST( Cell::average(pd_is) == i );
	BOOST_REQUIRE_EQUAL( Cell::average(pd_is2), Vec3d(0,0,0) );
}

BOOST_AUTO_TEST_CASE( construction_test, * boost::unit_test::tolerance(FLOAT_TEST_PRECISION) )
{
	Vec3d center(0,0,0);
	Vec3d dX(0.1,0.1,0.1);

	Cell c(center, dX);

	BOOST_REQUIRE_EQUAL( c.getCenter(), center );
	BOOST_REQUIRE_EQUAL( c.getHalfDiagonal(),  abs(dX)/2.0 );
	BOOST_REQUIRE_EQUAL( abs(c.getCorners()[0] - c.getCorners()[7]),  abs(dX) );

}

BOOST_AUTO_TEST_CASE( average_test, * boost::unit_test::tolerance(FLOAT_TEST_PRECISION) )
{
	Vec3d center(0,0,0);
	Vec3d dX(0.1,0.1,0.1);

	Cell c(center, dX);
	Cell c2(center+dX, dX);

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

