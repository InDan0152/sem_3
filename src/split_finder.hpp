
#pragma once

#include "matrix.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

enum class SplitCriterion {
    Variance,
    Gini
};

struct Split {
    std::size_t featureIndex;
    double threshold;
    long double score;
};

struct FeatureSplitResult {
    bool isConstant = false;
    std::optional<Split> best;
};

class SplitFinder {
public:
    static std::optional<Split> findBestSplit(
        const Matrix& features,
        std::span<const double> target,
        const std::vector<std::size_t>& objectIds,
        const std::vector<std::size_t>& featureIds,
        SplitCriterion criterion,
        std::size_t minSamplesLeaf,
        std::span<const std::size_t> counts = {}
    );

    static std::optional<Split> findBestSplit(
        const Matrix& features,
        std::span<const double> target,
        const std::vector<std::size_t>& objectIds,
        SplitCriterion criterion,
        std::size_t minSamplesLeaf,
        std::span<const std::size_t> counts = {}
    );

    static FeatureSplitResult findForFeature(
        const Matrix& features,
        std::span<const double> target,
        const std::vector<std::size_t>& objectIds,
        std::size_t feature,
        SplitCriterion criterion,
        std::size_t minSamplesLeaf,
        std::span<const std::size_t> counts = {}
    );
};
