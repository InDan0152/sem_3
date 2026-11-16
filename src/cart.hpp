
#pragma once

#include "model.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

using namespace std;

// Узел CART
struct CartNode {
    bool isLeaf = true;
    size_t featureIndex = 0;
    double threshold = 0.0;
    double prediction = 0.0;

    vector<double> classProbabilities;

    unique_ptr<CartNode> left;
    unique_ptr<CartNode> right;
};

struct CartSplitDescription {
    size_t depth = 0;
    size_t featureIndex = 0;
    double threshold = 0.0;
};

struct CartTreeSummary {
    size_t nodeCount = 0;
    size_t leafCount = 0;
    size_t depth = 0;
    vector<CartSplitDescription> splits;
};

class CartRegressor : public IRegressor {
private:
    unique_ptr<CartNode> root;

    size_t maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;
    size_t featureCount = 0;

    bool fitted = false;

    unique_ptr<CartNode> buildNode(
        const Matrix& features,
        span<const double> target,
        const vector<size_t>& objectIds,
        const vector<size_t>& featureIds,
        size_t depth
    );

public:
    CartRegressor(
        size_t max_depth = 5,
        size_t min_samples_split = 2,
        size_t min_samples_leaf = 1
    );

    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    vector<double> predict(
        const Matrix& features
    ) const override;

    CartTreeSummary summary() const;
};

class CartClassifier : public IClassifier {
private:
    unique_ptr<CartNode> root;

    size_t maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;
    size_t featureCount = 0;

    vector<double> classLabels;

    bool fitted = false;

    unique_ptr<CartNode> buildNode(
        const Matrix& features,
        span<const double> target,
        const vector<size_t>& objectIds,
        const vector<size_t>& featureIds,
        size_t depth
    );

public:
    CartClassifier(
        size_t max_depth = 5,
        size_t min_samples_split = 2,
        size_t min_samples_leaf = 1
    );

    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    vector<double> predict(
        const Matrix& features
    ) const override;

    Matrix predict_proba(
        const Matrix& features
    ) const override;

    span<const double> classes() const override;

    CartTreeSummary summary() const;
};
