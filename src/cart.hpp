
#pragma once

#include "model.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

using namespace std;

/**
 * @brief Режим выбора признаков для поиска разбиения.
 */
enum class FeatureMode {
    All,
    Sqrt,
    Count
};

/**
 * @brief Настройки случайного выбора признаков.
 */
struct FeatureSearchParams {
    FeatureMode mode = FeatureMode::All;
    size_t count = 1;
};

/**
 * @brief Узел дерева CART.
 */
struct CartNode {
    bool isLeaf = true;
    size_t featureIndex = 0;
    double threshold = 0.0;
    double prediction = 0.0;

    vector<double> classProbabilities;

    unique_ptr<CartNode> left;
    unique_ptr<CartNode> right;
};

/**
 * @brief Описание одного внутреннего разбиения в сводке дерева.
 */
struct CartSplitDescription {
    size_t depth = 0;
    size_t featureIndex = 0;
    double threshold = 0.0;
};

/**
 * @brief Сводные размеры и разбиения обученного дерева.
 */
struct CartTreeSummary {
    size_t nodeCount = 0;
    size_t leafCount = 0;
    size_t depth = 0;
    vector<CartSplitDescription> splits;
};

/**
 * @brief Дерево CART для регрессии.
 */
class CartRegressor : public IRegressor {
private:
    unique_ptr<CartNode> root;

    optional<size_t> maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;
    size_t featureCount = 0;

    FeatureSearchParams featureSearch;
    uint32_t randomState;

    bool fitted = false;

    unique_ptr<CartNode> buildNode(
        const Matrix& features,
        span<const double> target,
        const vector<size_t>& objectIds,
        const vector<size_t>& featureIds,
        span<const size_t> counts,
        size_t depth,
        mt19937& generator
    );

public:
    CartRegressor(
        optional<size_t> max_depth = 5,
        size_t min_samples_split = 2,
        size_t min_samples_leaf = 1,
        FeatureSearchParams feature_search = {},
        uint32_t random_state = 0
    );

    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    void fit_with_counts(
        const Matrix& features,
        span<const double> target,
        span<const size_t> counts
    );

    vector<double> predict(
        const Matrix& features
    ) const override;

    CartTreeSummary summary() const;
};

/**
 * @brief Дерево CART для классификации.
 */
class CartClassifier : public IClassifier {
private:
    unique_ptr<CartNode> root;

    optional<size_t> maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;
    size_t featureCount = 0;

    FeatureSearchParams featureSearch;
    uint32_t randomState;

    vector<double> classLabels;

    bool fitted = false;

    unique_ptr<CartNode> buildNode(
        const Matrix& features,
        span<const double> target,
        const vector<size_t>& objectIds,
        const vector<size_t>& featureIds,
        span<const size_t> counts,
        size_t depth,
        mt19937& generator
    );

public:
    CartClassifier(
        optional<size_t> max_depth = 5,
        size_t min_samples_split = 2,
        size_t min_samples_leaf = 1,
        FeatureSearchParams feature_search = {},
        uint32_t random_state = 0
    );

    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    void fit_with_counts(
        const Matrix& features,
        span<const double> target,
        span<const size_t> counts
    );

    vector<double> predict(
        const Matrix& features
    ) const override;

    Matrix predict_proba(
        const Matrix& features
    ) const override;

    span<const double> classes() const override;

    CartTreeSummary summary() const;
};
