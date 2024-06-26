#ifndef POINT_H
#define POINT_H

#include <iostream>
#include <vector>

#include "../base/weightedobject.h"

namespace CluE
{

/**
 * @brief Weighted point of arbitrary dimension.
 * 
 * @ingroup pointrelated_classes
 */
class Point : public WeightedObject
{
public:
	/**
	 * @brief Constructs a weighted point.
	 */
	Point(size_t dimension = 0, double pointWeight = 1.0):coordinates(std::vector<double>(dimension)),weight(pointWeight)
	{
	}

	/**
	 * @brief Constructs a weighted point.
	 */
	Point(std::vector<double> coords, double pointWeight = 1.0):coordinates(coords),weight(pointWeight)
	{
	}

	/**
	 * @brief Constructs a point of gravity.
	 * @throw InvalidArgumentException [0] Can't consolidate points with different dimensions!
	 */
	Point(std::vector<Point*> const&);
	
	/**
	 * @brief Copy constructor
	 */
	Point(Point const& p):coordinates(p.coordinates),weight(p.weight)
	{
	}

	virtual ~Point()
	{
	}
	
	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	Point& operator+=(Point const & x);
	
	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	Point& operator-=(Point const & x);
	
	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	Point operator+(Point const & x) const;
	
	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	Point operator-(Point const & x) const;
	
	/**
	 * Accesses one particular coordinate entry.
	 */
	double& operator[](size_t index)
	{
		return this->coordinates[index];
	}
	
	/**
	 * Returns one particular coordinate entry.
	 */
	double operator[](size_t index) const
	{
		return this->coordinates[index];
	}
	
	size_t dimension() const
	{
		return this->coordinates.size();
	}
	
	virtual double getWeight() const
	{
		return this->weight;
	}
	
	virtual void setWeight(double w)
	{
		this->weight = w;
	}
	
	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	double squaredL1distance(Point const&) const;
	
	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	double l1distance(Point const&) const;

	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	double squaredL2distance(Point const&) const;


	double squaredL2distanceMin(Point const& p, double& minDist) const;
	
	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	double l2distance(Point const&) const;

	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	double lpdistance(Point const&, double p) const;

	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 */
	double squaredLpDistance(Point const&, double p) const;

	/**
	 * @throw InvalidArgumentException [0] Incompatible dimensions!
	 * @throw InvalidRuntimeConfigurationException [1] Point has coordinate <= 0.
	 */
	double kullbackleibler(Point const&) const;

	void debug(size_t index, std::string valFormat="%7.3f", size_t elementsBeforeNewLine=10) const
	{
        printf("  p_%ld = [", index);
        for (size_t i = 0; i < dimension(); i++)
        {
            if (i % elementsBeforeNewLine == 0)
                printf("\n     ");
            printf(valFormat.c_str(), this->coordinates[i]);
			printf(" ");

        }
        printf("\n  ], dimensions = %ld\n", dimension());
	}

	void debugNonZero(size_t index, std::string valFormat="%7.3f", size_t elementsBeforeNewLine=10) const
	{
		printf("  p_%ld, dimensions = %ld\n", index, dimension());
        for (size_t i = 0; i < dimension(); i++)
        {
			if (this->coordinates[i] == 0.0) 
				continue;
            
			printf("  p[%5ld] = ", i);
            printf(valFormat.c_str(), this->coordinates[i]);
			printf("\n");
        }
	}
	
private:
	std::vector<double> coordinates;
	double weight;
};

std::ostream& operator<<(std::ostream&, Point const&);

Point operator*(double scalar, Point const & vec);

/**
 * @throw InvalidArgumentException [0] Incompatible dimensions!
 */
double operator*(Point const & vec1, Point const & vec2);

}

#endif