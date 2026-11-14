#pragma once

#include "model.hpp"

#include <cstddef>
#include <span>
#include <vector>

using namespace std;

// Критерий выбора разбиения.
enum class SplitCriterion {
    Variance, // Для регрессии
    Gini      // Для классификации
};

// Решающий пень — дерево решений глубины 1.
class DecisionStump : public IModel {
private:
    SplitCriterion criterion;
    size_t minSamplesLeaf;

    bool fitted = false;

    size_t featureCount = 0;
    size_t bestFeature = 0;
    double bestThreshold = 0.0;

    double leftPrediction = 0.0;
    double rightPrediction = 0.0;
    double constantPrediction = 0.0;

    bool hasSplit = false;

public:
    DecisionStump(
        SplitCriterion criterion,
        size_t min_samples_leaf
    );

    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    vector<double> predict(
        const Matrix& features
    ) const override;

    size_t split_feature() const;
    double split_threshold() const;
};