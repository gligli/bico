#ifndef BICO_H
#define BICO_H

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "../base/streamingalgorithm.h"
#include "../base/dissimilaritymeasure.h"
#include "../base/solutionprovider.h"
#include "../base/weightmodifier.h"
#include "../base/partitionprovider.h"
#include "../clustering/cfrentry.h"
#include "../datastructure/proxysolution.h"
#include "../evaluation/kmeansevaluator.h"
#include "../exception/invalidruntimeconfigurationexception.h"
#include "../misc/randomness.h"

#define OMP_SCHEDULE static,256

namespace CluE
{

/**
 * @brief Fast computation of k-means coresets in a data stream
 *
 * BICO maintains a tree which is inspired by the clustering tree of BIRCH,
 * a SIGMOD Test of Time award-winning clustering algorithm.
 * Each node in the tree represents a subset of these points. Instead of
 * storing all points as individual objects, only the number of points,
 * the sum and the squared sum of the subset's points are stored as key features
 * of each subset. Points are inserted into exactly one node.
 * A detailed description of BICO can be found here:
 * * Hendrik Fichtenberger, Marc Gillé, Melanie Schmidt, Chris Schwiegelshohn,
 *   Christian Sohler: BICO: BIRCH Meets Coresets for k-Means Clustering.
 *   ESA 2013: 481-492
 * In this implementation, the nearest neighbour search on the first level
 * of the tree ist sped up by projecting all points to random 1-d subspaces.
 * The first estimation of the optimal clustering cost is computed in a
 * buffer phase at the beginning of the algorithm.
 */
template<typename T> class Bico : public StreamingAlgorithm<T>
{
private:

    /**
     * @brief Class representing a node in BICO's tree
     */
    class BicoNode
    {
    public:
        typedef std::pair<CFREntry<T>, BicoNode*> FeaturePair;
        typedef std::list<FeaturePair> FeatureList;

        /**
         * Constructs a node for BICO's tree
         * @param outer Parent BICO instance
         */
        BicoNode(Bico<T>& outer) :
        objectId(outer.nodeIdCounter),
        outer(outer),
        features()
        {
            ++outer.nodeIdCounter;
        }

        /**
         * @brief Delete all nodes
         */
        void clear()
        {
            for (auto it = features.begin(); it != features.end(); ++it)
                delete it->second;
        }

        /**
         * Inserts a CFREntry into this node
         * @param feature CFREntry to be inserted
         * @return Iterator pointing to inserted CFREntry
         */
        typename FeatureList::iterator insert(CFREntry<T> const & feature)
        {
            return features.insert(features.end(),
                                   FeaturePair(feature, new BicoNode(outer)));
        }

        /**
         * Iterator pointing at the first CFREntry
         * @return Begin iterator
         */
        typename FeatureList::iterator begin()
        {
            return features.begin();
        }

        /**
         * Iterator pointing behind the last CFREntry
         * @return End iterator
         */
        typename FeatureList::iterator end()
        {
            return features.end();
        }

        /**
         * Number of contained CFREntries
         * @return Number of elements
         */
        size_t size()
        {
            return features.size();
        }

        /**
         * Indicates if node is empty
         * @return Indicator
         */
        bool empty()
        {
            return features.empty();
        }

        /**
         * Returns an iterator to the CFREntry in this node whose reference point
         * is nearest to a fixed point
         * @param element Fixed point
         * @param level Level of this node
         * @return Nearest CFREntry
         */
        typename FeatureList::iterator nearest(T const & element, int level)
        {
            typename FeatureList::iterator minIt = features.end();
            // Nearest neighbour search based on projections in level 1
            if (level == 1 && outer.L > 0)
            {
                // Project point and calculate projection bucket number
                double val = outer.project(element, 0);
                int bucket_number = outer.calcBucketNumber(0, val);
                int mini = 0;
                int bucket_min = bucket_number;
                int mins;

                if ((bucket_number < 0) || (bucket_number >= outer.buckets[0].size()))
                {
                    // The bucket does not exist (yet)
                    mins = 0;
                }
                else
                {
                    // Search for the projection with smallest bucket size
                    mins = outer.buckets[mini][bucket_min].size();
                    for (int i = 1; i < outer.L; i++)
                    {
                        val = outer.project(element, i);
                        bucket_number = outer.calcBucketNumber(i, val);
                        if ((bucket_number >= 0) && (bucket_number < outer.buckets[i].size()))
                        {
                            size_t s = outer.buckets[i][bucket_number].size();
                            if (s < mins)
                            {
                                mins = s;
                                bucket_min = bucket_number;
                                mini = i;
                            }
                        }
                        else
                        {
                            mins = 0;
                            bucket_min = bucket_number;
                            mini = i;
                            break;
                        }
                    }

                }

                bucket_number = bucket_min;
                int rnd = mini;

                if (bucket_number < 0)
                {
                    // Bucket does not exist => create one
                    outer.allocateBucket(rnd, true);
                }
                else if (bucket_number >= outer.buckets[rnd].size())
                {
                    // Bucket does not exist => create one
                    outer.allocateBucket(rnd, false);
                }
                else
                {
                    // Bucket does exist => search nearest point in bucket
                    double minDist = -1;
                    std::vector<T*> prm_buf;
                    std::vector<double> dissim_buf;

                    prm_buf.reserve(outer.buckets[rnd][bucket_number].size());
                    dissim_buf.reserve(outer.buckets[rnd][bucket_number].size());

                    for (auto it = outer.buckets[rnd][bucket_number].begin(); it != outer.buckets[rnd][bucket_number].end(); ++it)
                    {
                        prm_buf.push_back(&(*it)->first.representative);
                    }

                    double sharedMinDist = std::numeric_limits<double>::infinity();

                    #pragma omp parallel for shared(sharedMinDist) schedule(OMP_SCHEDULE)
                    for (intptr_t i = 0; i < prm_buf.size(); ++i)
                    {
                        double localMinDist = sharedMinDist;
                        dissim_buf[i] = outer.measure->dissimilarity(*prm_buf[i], element, localMinDist);
                        if (dissim_buf[i] < localMinDist)
                          sharedMinDist = dissim_buf[i];
                    }

                    intptr_t pos = 0;
                    for (auto it = outer.buckets[rnd][bucket_number].begin(); it != outer.buckets[rnd][bucket_number].end(); ++it)
                    {
                        double tmpDist = dissim_buf[pos];
                        if (tmpDist < minDist || minDist == -1)
                        {
                            minDist = tmpDist;
                            minIt = (*it);
                        }
                        ++pos;
                    }

                }
            }
                // Simple nearest neighbour search in all other levels
            else
            {
                double minDist = -1;
                std::vector<T*> prm_buf;
                std::vector<double> dissim_buf;

                prm_buf.reserve(features.size());
                dissim_buf.reserve(features.size());

                for (auto it = features.begin(); it != features.end(); ++it)
                {
                    prm_buf.push_back(&it->first.representative);
                }

                double sharedMinDist = std::numeric_limits<double>::infinity();

                #pragma omp parallel for shared(sharedMinDist) schedule(OMP_SCHEDULE)
                for (intptr_t i = 0; i < prm_buf.size(); ++i)
                {
                    double localMinDist = sharedMinDist;
                    dissim_buf[i] = outer.measure->dissimilarity(*prm_buf[i], element, localMinDist);
                    if (dissim_buf[i] < localMinDist)
                      sharedMinDist = dissim_buf[i];
                }

                intptr_t pos = 0;
                for (auto it = features.begin(); it != features.end(); ++it)
                {
                    double tmpDist = dissim_buf[pos];
                    if (tmpDist < minDist || minDist == -1)
                    {
                        minDist = tmpDist;
                        minIt = it;
                    }
                    ++pos;
                }
            }

            return minIt;
        }

        /**
         * Removes a specified CFREntry
         * @param pos Position of the CFREntry to be removed
         */
        void erase(typename FeatureList::iterator pos)
        {
            features.erase(pos);
        }

        /**
         * Inserts all CFREntries of this node into a given FeatureList
         * @param to Destination of insertion
         * @param pos Position of insertion
         */
        void spliceAllTo(BicoNode* to, typename FeatureList::iterator pos)
        {
            to->features.splice(pos, features);
        }

        /**
         * Inserts one CFREntry of this node into a given FeatureList
         * @param it CFREntry to be inserted
         * @param to Destination of insertion
         * @param pos Postion of insertion
         */
        void spliceElementTo(typename FeatureList::iterator it, BicoNode* to, typename FeatureList::iterator pos)
        {
            to->features.splice(pos, features, it);
        }

        /**
         * Returns the unique object id
         * @return Object id
         */
        int id()
        {
            return objectId;
        }

    private:
        /**
         * @brief Unique object id
         */
        int objectId;

        /**
         * @brief Parent BICO instance
         */
        Bico<T>& outer;

        /**
         * List of all contained CFREntries
         */
        FeatureList features;
    };

public:
    /**
     * @brief Constructs BICO for points of type T
     * T can be an arbitrary type but it has to fulfil the requirements
     * of CFREntry.
     * 
     * @param dimension Dimension of the data
     * @param n Size of the data
     * @param k Number of desired centeres
     * @param p Number of random projections used for nearest neighbour search
     *          in the first level
     * @param nMax Maximum coreset size
     * @param measure Implementation of the squared L2 metric for T
     * @param weightModifier Class to read and modify weight of T
     * @param seed RNG random seed
     */
    Bico(size_t dimension, size_t n, size_t k, size_t p, size_t nMax,
         DissimilarityMeasure<T>* measure, WeightModifier<T>* weightModifier, uint_fast32_t seed);

    /**
     * @brief Returns a coreset of all point read so far
     * @return Coreset
     */
    virtual ProxySolution<T>* compute();

    /**
     * @brief Read a point
     * Insert the point into BICO's tree
     * 
     * @param element Point of type T
     * @return This BICO instance
     */
    virtual Bico<T>& operator<<(T const & element);

    /**
     * @brief Write the tree as GraphViz source into a stream
     * @param os Output stream
     */
    void print(std::ostream& os);

    void setRebuildProperties(size_t interval, double initial, double grow);

private:
    /**
     * @brief Inserts an element into a BicoNode at a certain level
     * @param node BicoNode to be inserted into
     * @param level Level of this BicoNode
     * @param element Elemente to be inserted
     */
    void insert(BicoNode* node, int level, T const & element);

    /**
     * @brief Allocates a new bucket
     * @param bucket Number of projection
     * @param left Push front bucket (instead of push back)
     */
    void allocateBucket(int bucket, bool left);

    /**
     * Calculates the bucket number for a given value
     * @param rnd Number of projections
     * @param val Value
     * @return Bucket number
     */
    int calcBucketNumber(int rnd, double val);

    /**
     * @brief Create initial buckets
     */
    void buildBuckets();

    /**
     * @brief Projects a point onto a projection line
     * @param point Point
     * @param i Number of projection line
     * @return Projected point
     */
    double project(T point, int i);

    /**
     * @brief Rebuilds the tree
     */
    void rebuild();

    /**
     * Rebuilds the first level
     * @param parent New root
     * @param child Old root
     */
    void rebuildFirstLevel(BicoNode* parent, BicoNode* child);

    /**
     * Recursive rebuilding of the tree
     * @param node Some node to be rebuilded
     * @param level Level of this node
     */
    void rebuildTraversMerge(BicoNode* node, int level);

    /**
     * @brief Recursive computation of the coreset
     * @param node Some node to be processed
     * @param solution ProxySolution containing the coreset
     */
    void computeTraverse(BicoNode* node, ProxySolution<T>* solution);

    /**
     * @brief Returns the threshold for a given level
     * @param level Level
     * @return Threshold at this level
     */
    double getT(int level);

    /**
     * @brief Returns the radius for a given level
     * @param level Level
     * @return Radius at this level
     */
    double getR(int level);

    /**
     * Writes a BicoNode as GraphViz source into a stream
     * @param os Output stream
     * @param node Some BicoNode
     */
    void print(std::ostream& os, BicoNode* node);

    /**
     * @brief Number of centers
     */
    size_t k;
    /**
     * @brief Number of projections
     */
    size_t L;

    /**
     * @brief Random projection vectors
     */
    std::vector<std::vector<double >> rndprojections;

    /**
     * @brief Buckets for nearest neighbour search in first level
     */
    std::vector<std::vector<std::vector<typename BicoNode::FeatureList::iterator >> > buckets;
    /**
     * @brief Bucket borders
     */
    std::vector<std::pair<double, double >> borders;
    /**
     * @brief Width of buckets
     */
    std::vector<int> bucket_radius;

    /**
     * @brief Counter for unique BicoNode object ids
     */
    int nodeIdCounter;

    /**
     * @brief Buffer for DissimilarityMeasure
     */
    std::unique_ptr<DissimilarityMeasure<T >> measure;

    /**
     * @brief Buffer for WeightModifier
     */
    std::unique_ptr<WeightModifier<T >> weightModifier;

    /**
     * @brief Maximum coreset size
     */
    size_t maxNumOfCFs;

    /**
     * @brief Current coreset size
     */
    size_t curNumOfCFs;

    /**
     * @brief Dimension of the input points
     */
    size_t dimension;

    /**
     * @brief Current estimation of the optimal clustering cost
     */
    double optEst;

    double optEst_initial;
    double optEst_grow;
    size_t rebuildInterval;
    size_t rebuildPos;

    /**
     * @brief Extreme values used for constructing the nearest neighbour buckets
     */
    std::vector<double> maxVal;

    /**
     * @brief Root node of BICO's tree
     */
    BicoNode* root;

    /**
     * @brief Buffer phase indicator
     */
    bool bufferPhase;

    /**
     * @brief Buffer phase's buffer
     */
    std::vector<T> buffer;

    /**
     * @brief Current number of rebuilding
     */
    int numOfRebuilds;

    double minDist;
    size_t pairwise_different;
};

template<typename T> Bico<T>::Bico(size_t dim, size_t n, size_t k, size_t p, size_t nMax,
                                   DissimilarityMeasure<T>* measure, WeightModifier<T>* weightModifier, uint_fast32_t seed) :
nodeIdCounter(0),
measure(measure->clone()),
weightModifier(weightModifier->clone()),
maxNumOfCFs(nMax),
curNumOfCFs(0),
k(k),
L(p),
optEst(-1),
optEst_initial(16.0),
optEst_grow(2.0),
rebuildInterval(1),
rebuildPos(0),
root(new BicoNode(*this)),
bufferPhase(true),
numOfRebuilds(0),
buffer(),
dimension(dim)
{
    minDist = std::numeric_limits<double>::infinity();
    pairwise_different = 0;
  
    Randomness r(seed);
    RandomGenerator rg = r.getRandomGenerator();
    std::vector<double> rndpoint(dimension);
    rndprojections.resize(L);
    bucket_radius.resize(L);
    maxVal.resize(L);
    double norm;
    std::normal_distribution<double> realDist(0.0, 1.0);
    for (int i = 0; i < L; i++)
    {
        maxVal[i] = -INFINITY;
        norm = 0.0;
        for (int j = 0; j < dimension; j++)
        {
            rndpoint[j] = realDist(rg);
            norm += rndpoint[j] * rndpoint[j];
        }
        norm = std::sqrt(norm);
        for (int j = 0; j < dimension; j++)
        {
            rndpoint[j] /= norm;
        }
        rndprojections[i] = rndpoint;
    }
    buckets.resize(L);
    buffer.reserve(maxNumOfCFs + 1);
}

template<typename T> int Bico<T>::calcBucketNumber(int rnd, double val)
{
    return (int) floor((val - borders[rnd].first) / bucket_radius[rnd]);
}

template<typename T> void Bico<T>::buildBuckets()
{
    double Size = 0;
    for (int i = 0; i < L; i++)
    {
        // Compute new bucket size
        if (buckets[i].size() == 1)
        {
            Size = 1;
        }
        else
        {
            bucket_radius[i] = (int) ceil(sqrt(getR(1)));
            Size = (int) ceil((borders[i].second - borders[i].first) / (double) bucket_radius[i]);
        }
        for (int l = 0; l < buckets[i].size(); l++) buckets[i][l].clear();
        // Create new buckets
        buckets[i].clear();
        buckets[i].resize((int) ceil(Size));
    }
}

template<typename T> void Bico<T>::allocateBucket(int bucket, bool left)
{
    if (left)
    {
        // Push front bucket
        borders[bucket].first = 2 * borders[bucket].first - borders[bucket].second;
        std::vector < std::vector<typename BicoNode::FeatureList::iterator >> a(2 * buckets[bucket].size());
        for (int i = 0; i < buckets[bucket].size(); i++)
        {
            a[buckets[bucket].size() + i] = buckets[bucket][i];
        }
        for (int l = 0; l < buckets[bucket].size(); l++) buckets[bucket][l].clear();
        buckets[bucket].clear();
        buckets[bucket] = a;
    }
    else
    {
        // Push back bucket
        borders[bucket].second = 2 * borders[bucket].second - borders[bucket].first;
        std::vector < std::vector<typename BicoNode::FeatureList::iterator >> a(2 * buckets[bucket].size());
        for (int i = 0; i < buckets[bucket].size(); i++)
        {
            a[i] = buckets[bucket][i];
        }
        for (int l = 0; l < buckets[bucket].size(); l++) buckets[bucket][l].clear();
        buckets[bucket].clear();
        buckets[bucket] = a;
    }
}

template<typename T> double Bico<T>::project(T point, int i)
{
    double ip = 0.0;
    for (int j = 0; j < dimension; j++)
    {
        ip += point[j]*(rndprojections[i][j]);
    }
    return ip;
}

template<typename T> ProxySolution<T>* Bico<T>::compute()
{
    // Rebuild up to k clusters (gli)
    while (curNumOfCFs > k)
    {
        rebuild();
    }

    ProxySolution<T>* result = new ProxySolution<T>();
        result->proxysets.push_back(std::vector<T>());
        result->proxysets[0].reserve(curNumOfCFs);
        computeTraverse(root, result);
    return result;
}

template<typename T> void Bico<T>::computeTraverse(BicoNode* node, ProxySolution<T>* solution)
{
    for (auto it = node->begin(); it != node->end(); ++it)
    {
        T point(it->first.cog());
        weightModifier->setWeight(point, it->first.number);
        solution->proxysets[0].push_back(point);
        computeTraverse(it->second, solution);
    }
}

template<typename T> Bico<T>& Bico<T>::operator<<(T const & element)
{
    if (bufferPhase)
    {
        // Find nearest neighbor

        std::vector<double> dissim_buf;
        dissim_buf.reserve(buffer.size());

        double sharedMinDist = minDist;
        
        #pragma omp parallel for shared(sharedMinDist) schedule(OMP_SCHEDULE)
        for (intptr_t i = 0; i < buffer.size(); ++i)
        {
          double localMinDist = sharedMinDist;
          dissim_buf[i] = measure->dissimilarity(buffer[i], element, localMinDist);
          if (dissim_buf[i] < localMinDist)
              sharedMinDist = dissim_buf[i];
        }

        for (intptr_t i = 0; i < buffer.size(); ++i)
        {
            double tmpDist = dissim_buf[i];
            if (tmpDist > 0)
            {
                ++pairwise_different;
                if (tmpDist < this->minDist)
                    minDist = tmpDist;
            }
        }

        for (int i = 0; i < L; i++)
        {
            double val = std::abs(project(element, i));
            if (val > maxVal[i] || maxVal[i] == -INFINITY)
            {
                maxVal[i] = val;
            }
        }

        buffer.push_back(element);

        // Enough pairwise different elements to estimate optimal cost?
        if (pairwise_different >= maxNumOfCFs + 1)
        {
            optEst = optEst_initial * minDist;
            int radius = (int) ceil(sqrt(getR(1)));
            borders.resize(L);
            for (int i = 0; i < L; i++)
            {
                borders[i].first = -maxVal[i];
                borders[i].second = maxVal[i];
                bucket_radius[i] = radius;
            }
            buildBuckets();

            // Insert buffered elements into tree
            for (auto it = buffer.begin(); it != buffer.end(); ++it)
                insert(root, 1, *it);
            buffer.resize(0);
            bufferPhase = false;
        }
    }
    else
        insert(root, 1, element);
    return *this;
}

template<typename T> void Bico<T>::insert(BicoNode* node, int level, T const & element)
{

    if (optEst < 0)
        throw (InvalidRuntimeConfigurationException(0, "Estimated optimal cost invalid"));

    // Determine nearest clustering feature in current node
    typename BicoNode::FeatureList::iterator nearest(node->nearest(element, level));

    double r = getR(level);

    // Construct new clustering feature if element is too far away from
    // nearest clustering feature or insert element into nearest
    if (node->empty() || nearest == node->end()
            || measure->dissimilarity(nearest->first.representative, element, r) > r)
    {
        CFREntry<T> feature(1, element, element * element, element);
        typename BicoNode::FeatureList::iterator itele = node->insert(feature);

        if (level == 1)
        {
            for (int i = 0; i < L; i++)
            {
                double val = project(element, i);
                int bucket_number = calcBucketNumber(i, val);

                if (bucket_number < 0)
                {
                    while (bucket_number < 0)
                    {
                        allocateBucket(i, true);
                        bucket_number = calcBucketNumber(i, val);
                    }
                }
                else if (bucket_number >= buckets[i].size())
                {
                    while (bucket_number >= buckets[i].size())
                    {
                        allocateBucket(i, false);
                        bucket_number = calcBucketNumber(i, val);
                    }
                }
                buckets[i][bucket_number].push_back(itele);
            }
        }

        ++curNumOfCFs;
    }
    else
    {
        T center(nearest->first.cog());
        // Insert element into (a copy of) nearest and compute cost
        // for insertion at current level
        CFEntry<T> testFeature(nearest->first);
        testFeature.insert(element);
        double tfCost = testFeature.kMeansCost(center);

        // Insert element either to current level (if cost remains small)
        // or to higher level
        if (tfCost < getT(level))
        {
            nearest->first.insert(element);
        }
        else
        {
            insert(nearest->second, level + 1, element);
        }
    }

    // Rebuild?
    if (++rebuildPos >= rebuildInterval)
    {
        rebuildPos = 0;
        
        while (curNumOfCFs > maxNumOfCFs)
        {
            rebuild();
        }
    }
}

template<typename T> void Bico<T>::rebuild()
{
    // Rebuild first level
    BicoNode * oldRoot(this->root);
    this->root = new BicoNode(*this);
    rebuildFirstLevel(this->root, oldRoot);
    oldRoot->clear();
    delete oldRoot;

    // Traverse through tree and merge
    for (auto it = this->root->begin(); it != this->root->end(); ++it)
        rebuildTraversMerge(it->second, 2);
}

template<typename T> void Bico<T>::rebuildFirstLevel(BicoNode* parent, BicoNode* child)
{
    optEst *= optEst_grow;
    ++numOfRebuilds;

    buildBuckets();

    // The current element it may be spliced around the tree, so nextIt
    // will maintain an iterator pointing at the next element in child
    auto nextIt = child->begin();
    for (auto it = child->begin(); it != child->end(); it = nextIt)
    {
        ++nextIt;

        double r = getR(1);

        // Determine clustering feature in parent that is nearest to child
        typename BicoNode::FeatureList::iterator nearest(parent->nearest(it->first.representative, 1));
        if (parent->empty() || nearest == parent->end()
                || measure->dissimilarity(nearest->first.representative, it->first.representative, r) > r)
        {
            // Move it from child to parent
            child->spliceElementTo(it, parent, parent->end());

            for (int i = 0; i < L; i++)
            {
                double val = project(it->first.representative, i);
                int bucket_number = calcBucketNumber(i, val);
                if (bucket_number < 0)
                {
                    while (bucket_number < 0)
                    {
                        allocateBucket(i, true);
                        bucket_number = calcBucketNumber(i, val);
        }
                }
                else if (bucket_number >= buckets[i].size())
                {
                    while (bucket_number >= buckets[i].size())
                    {
                        allocateBucket(i, false);
                        bucket_number = calcBucketNumber(i, val);
                    }
                }
                buckets[i][bucket_number].push_back(it);
            }
        }
        else
        {
            CFEntry<T> testFeature(it->first);
            testFeature += nearest->first;
            if (testFeature.kMeansCost(nearest->first.representative) <= getT(1))
            {
                // Insert it into nearest
                nearest->first += it->first;
                // Make children of it children of nearest
                it->second->spliceAllTo(nearest->second, nearest->second->end());
                // Delete (now) empty child node of it
                it->second->clear();
                delete it->second;
                // Remove it from tree and delete it
                child->erase(it);
                --curNumOfCFs;
            }
            else
            {
                // Make it a child of nearest
                child->spliceElementTo(it, nearest->second, nearest->second->end());
            }
        }
    }
}

template<typename T> void Bico<T>::rebuildTraversMerge(BicoNode* node, int level)
{
    for (auto parentIt = node->begin(); parentIt != node->end(); ++parentIt)
    {
        if (!parentIt->second->empty())
        {
            auto nextIt = parentIt->second->begin();
            for (auto childIt = parentIt->second->begin(); childIt != parentIt->second->end(); childIt = nextIt)
            {
                ++nextIt;

                T center(parentIt->first.cog());
                // Insert element into (a copy of) nearest and compute cost
                // for insertion at current level
                CFEntry<T> testFeature(parentIt->first + childIt->first);
                double tfCost = testFeature.kMeansCost(center);

                // Merge if possible
                if (tfCost < getT(level))
                {
                    parentIt->first += childIt->first;
                    childIt->second->spliceAllTo(parentIt->second, parentIt->second->end());
                    childIt->second->clear();
                    delete childIt->second;
                    parentIt->second->erase(childIt);
                    --curNumOfCFs;
                }
                else
                {
                    rebuildTraversMerge(childIt->second, level + 1);
                }
            }
        }
    }
}

template<typename T> double Bico<T>::getT(int level)
{
    return optEst;
}

template<typename T> double Bico<T>::getR(int level)
{
    return getT(level) / (double(1 << (3 + level)));
}

template<typename T> void Bico<T>::print(std::ostream& os)
{
    os << "digraph G{\n";
    os << "node [shape=record];\n";
    print(os, root);
    os << "}\n";
}

template<typename T> void Bico<T>::print(std::ostream& os, BicoNode* node)
{
    int id = node->id();

    os << "node" << id << "[label=\"";
    int fvalue = 0;
    os << node->id() << "|";
    for (auto it = node->begin(); it != node->end(); ++it)
    {
        if (fvalue > 0)
            os << "|";
        os << "<f" << fvalue << "> " << it->first.number << "," << it->first.representative
                << "\\n" << it->first.LS << "," << it->first.SS;
        fvalue++;
    }
    os << "\"];\n";

    fvalue = 0;
    for (auto it = node->begin(); it != node->end(); ++it)
    {
        print(os, it->second);
        os << "node" << id << ":f" << fvalue << " -> node" << it->second->id() << ";\n";
        fvalue++;
    }
}

template<typename T> void Bico<T>::setRebuildProperties(size_t interval, double initial, double grow)
{
  rebuildInterval = interval;
  rebuildPos = 0;

  if (!std::isnan(initial))
      optEst_initial = initial;
  
  if (!std::isnan(grow))
      optEst_grow = grow;
}


}

#endif
