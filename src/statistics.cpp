#include "statistics.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <span>

using namespace std;

void calculateHistogram(
    const vector<double>& values,
    NumericalStatistics& statistics,
    size_t binCount
);

namespace {

//средн арифм
double calculateMean(span<const double> values) {
    double sum = 0.0;

    for (double value : values) {
        sum += value;
    }

    return sum / values.size();
}

//генеральная дисперсия
double calculateVariance(span<const double> values, double mean) {
    double squaredDifferences = 0.0;

    for (double value : values) {
        squaredDifferences += (value - mean) * (value - mean);
    }

    return squaredDifferences / values.size();
}

//квантиль
double calculateQuantile(span<const double> values, double p) {
    double position = p * (values.size() - 1);

    size_t lowerIndex = static_cast<size_t>(position);
    size_t upperIndex = lowerIndex + 1;

    if (upperIndex >= values.size()) {
        return values.back();
    }

    double fraction = position - lowerIndex;

    return values[lowerIndex] +
        fraction * (values[upperIndex] - values[lowerIndex]);
}

}//namespace


bool calculateNumericalStatistics(
    const Column& column,
    NumericalStatistics& statistics,
    size_t binCount
) {
    if (column.type != FeatureType::Numerical) {
        return false;
    }


    vector<double> values;

    statistics = NumericalStatistics{};

    for (const auto& value : column.numericalValues) {
        if (value.has_value()) {
            values.push_back(value.value());
        }
        else {
            ++statistics.missingCount;
        }
    }

    statistics.validCount = values.size();


    if (values.empty()) {
        return true;
    }

    statistics.hasValues = true;

// сорт временной копии
    sort(values.begin(), values.end());

    statistics.min = values.front();
    statistics.max = values.back();


    statistics.mean = calculateMean(values);
    statistics.variance = calculateVariance(values, statistics.mean);

//квантили
    statistics.q05 = calculateQuantile(values, 0.05);
    statistics.q25 = calculateQuantile(values, 0.25);
    statistics.q50 = calculateQuantile(values, 0.50);
    statistics.q75 = calculateQuantile(values, 0.75);
    statistics.q95 = calculateQuantile(values, 0.95);

    calculateHistogram(values, statistics, binCount);

    return true;
}

    bool calculateCategoricalStatistics(const Column& column,CategoricalStatistics& statistics) {
    if (column.type != FeatureType::Categorical) {return false;}

    statistics = CategoricalStatistics{};

    map<string, size_t> frequencies;
    // todo: map vs unordered_map, что находится по факту внутри?

    for (const auto& value : column.categoricalValues) {
        if (value.has_value()) {
            ++frequencies[value.value()];
        }
        else {
            ++statistics.missingCount;
        }
    }

    statistics.validCount =
        column.categoricalValues.size() - statistics.missingCount;


    for (const auto& [category, frequency] : frequencies) {
        statistics.categories.push_back({
            category,
            frequency
        });
    }

    return true;
}

void calculateHistogram(const vector<double>& values,NumericalStatistics& statistics,size_t binCount) {
    statistics.histogram.clear();

    if (values.empty() || binCount == 0) {
        return;
    }

    double minValue = values.front();
    double maxValue = values.back();

    if (minValue == maxValue) {
        HistogramBin bin;
        bin.lowerBound = minValue;
        bin.upperBound = maxValue;
        bin.frequency = values.size();

        statistics.histogram.push_back(bin);
        return;
    }

    double binWidth = (maxValue - minValue) / binCount;

    for (size_t i = 0; i < binCount; ++i) {
        HistogramBin bin;

        bin.lowerBound = minValue + i * binWidth;
        bin.upperBound = minValue + (i + 1) * binWidth;
        bin.frequency = 0;

        statistics.histogram.push_back(bin);
    }

    for (double value : values) {
        size_t index = static_cast<size_t>(
            (value - minValue) / binWidth
        );

        if (index >= binCount) {
            index = binCount - 1;
        }

        ++statistics.histogram[index].frequency;
    }
}