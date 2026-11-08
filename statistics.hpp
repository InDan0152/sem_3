
#pragma once

#include "dataset.hpp"
#include <vector>
#include <string>
#include <cstddef>

struct HistogramBin {
    double lowerBound = 0.0;
    double upperBound = 0.0;
    size_t frequency = 0;
};

struct NumericalStatistics {
    size_t missingCount = 0;
    size_t validCount = 0;

    bool hasValues = false;

    double min = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double variance = 0.0;

    double q05 = 0.0;
    double q25 = 0.0;
    double q50 = 0.0;
    double q75 = 0.0;
    double q95 = 0.0;

    vector<HistogramBin> histogram;
};

struct CategoryFrequency {
    std::string category;
    size_t frequency = 0;
};

struct CategoricalStatistics {
    size_t missingCount = 0;
    size_t validCount = 0;

    std::vector<CategoryFrequency> categories;
};

bool calculateCategoricalStatistics(
    const Column& column,
    CategoricalStatistics& statistics
);

bool calculateNumericalStatistics(
    const Column& column,
    NumericalStatistics& statistics,
    size_t binCount = 10
);