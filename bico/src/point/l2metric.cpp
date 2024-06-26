#include "../point/l2metric.h"
#include "../point/point.h"

using namespace CluE;

L2Metric* L2Metric::clone() const
{
	return new L2Metric(*this);
}

double L2Metric::dissimilarity(Point const& p1, Point const& p2, double& minDist) const
{
    return p1.l2distance(p2);
}
