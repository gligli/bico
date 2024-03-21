#include "../point/squaredl2metric.h"
#include "../point/point.h"

using namespace CluE;

SquaredL2Metric* SquaredL2Metric::clone() const
{
	return new SquaredL2Metric(*this);
}

double SquaredL2Metric::dissimilarity(Point const& p1, Point const& p2, double& minDist) const
{
    return p1.squaredL2distanceMin(p2, minDist);
}
