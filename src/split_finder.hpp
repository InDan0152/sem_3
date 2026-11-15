
#pragma once

#include "matrix.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

// Критерий выбора разбиения.
enum class SplitCriterion {
    Variance, // Для регрессии
    Gini      // Для классификации
};

// Результат поиска разбиения.
struct Split {
    std::size_t featureIndex;
    double threshold;
    long double score;
};

// Общий поиск оптимального разбиения.
class SplitFinder {
public:
    static std::optional<Split> findBestSplit(
        const Matrix& features,
        std::span<const double> target,
        const std::vector<std::size_t>& objectIds,
        const std::vector<std::size_t>& featureIds,
        SplitCriterion criterion,
        std::size_t minSamplesLeaf
    );
};
