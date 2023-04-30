#ifndef RANDOMNESS_H
#define RANDOMNESS_H

#include <random>
#include <time.h>

#include "randomgenerator.h"

namespace CluE
{
/**
 * @brief Random number generator.
 *
 * @ingroup helper_classes
 */
class Randomness
{
private:	
	// TODO Use mt19937_64 ?
	std::mt19937 mt19937Generator;

public:
	RandomGenerator getRandomGenerator()
	{
		return RandomGenerator(&mt19937Generator);
	}
	
	Randomness(uint_fast32_t seed)
	{
		mt19937Generator.seed(seed);
	}
};

}

#endif
