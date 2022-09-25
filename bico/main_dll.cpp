#ifdef _MSC_VER
	#include "oneapi/tbb/tbbmalloc_proxy.h"
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <ctime>
#include <time.h>

#include <boost/algorithm/string.hpp>

#include "src/point/l2metric.h"
#include "src/point/squaredl2metric.h"
#include "src/point/point.h"
#include "src/point/pointweightmodifier.h"
#include "src/clustering/bico.h"
#include "src/misc/randomness.h"
#include "src/misc/randomgenerator.h"
#include "src/datastructure/proxysolution.h"
#include "src/point/pointcentroid.h"
#include "src/point/pointweightmodifier.h"
#include "src/point/realspaceprovider.h"

using namespace CluE;

class api_ptr_t {
	public:
	Bico<Point>* bico;
	ProxySolution<Point>* sol;
	int64_t dimension;

	inline api_ptr_t(void) {bico = NULL; sol = NULL; dimension = -1;};
};

extern "C"
{
#ifdef _MSC_VER
	#define DLL_API __declspec(dllexport)

	#define API_FP_PRE() \
			unsigned int _cFP; \
			_controlfp_s(&_cFP, 0, 0); \
			_set_controlfp(0x1f, 0x1f);

	#define API_FP_POST() \
			_set_controlfp(_cFP, 0x1f); \
			_clearfp();
#else			
	int main(int argc, char** argv)
	{
		return 0;
	}

	#define DLL_API __attribute__((dllexport))
	#define API_FP_PRE()
	#define API_FP_POST()
#endif

	DLL_API void* __stdcall bico_create(int64_t dimension, int64_t npoints, int64_t k, int64_t nrandproj, int64_t coresetsize, int32_t randomSeed)
	{
		API_FP_PRE();

		api_ptr_t* ab = new api_ptr_t();
		ab->bico = new Bico<Point>(dimension, npoints, k, nrandproj, coresetsize, new SquaredL2Metric(), new PointWeightModifier(), randomSeed);
		ab->dimension = dimension;

		API_FP_POST();

		return ab;
	}

	DLL_API void __stdcall bico_destroy(void* bico)
	{
		API_FP_PRE();

		api_ptr_t* ab = (api_ptr_t*)bico;

		delete ab->bico;
		delete ab->sol;
		delete ab;

		API_FP_POST();
	}

	DLL_API void __stdcall bico_insert_line(void* bico, double* line, double weight)
	{
		API_FP_PRE();

		api_ptr_t* ab = (api_ptr_t*)bico;

		Point p(ab->dimension, weight);

		double* l = line;
		for(int i = 0; i < ab->dimension; ++i)
		  p[i] = *l++;

		*ab->bico << p;

		API_FP_POST();
	}

	DLL_API int64_t __stdcall bico_get_results(void* bico, double* centroids, double* weights)
	{
		API_FP_PRE();

		api_ptr_t* ab = (api_ptr_t*)bico;

		ab->sol = ab->bico->compute();

		size_t res = ab->sol->proxysets[0].size();

		double* c = centroids;
		double* w = weights;
		for (size_t i = 0; i < res; ++i)
		{
		  *w++ = ab->sol->proxysets[0][i].getWeight();

		  for (size_t j = 0; j < ab->sol->proxysets[0][i].dimension(); ++j)
		  {
			*c++ = ab->sol->proxysets[0][i][j];
		  }
		}

		API_FP_POST();

		return res;
	}
}