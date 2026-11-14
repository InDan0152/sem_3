#pragma once

#include "dataset.hpp"

#include <string>
#include <vector>

using namespace std;

enum class NumericalMissingStrategy {
    Mean,
    Median,
    Zero
};

enum class CategoricalMissingStrategy {
    MissingCategory
};

enum class ScalingStrategy {
    MinMax,
    Standard,
    Robust
}; // todo: зачем нужно масштабирование?

// todo: во что превращает компилятор лямбда функцию и что такое раскрутка стека?

enum class EncodingStrategy {
    OneHot,
    Ordinal
};

struct PreprocessingConfig {
    NumericalMissingStrategy numericalMissing =
        NumericalMissingStrategy::Median;

    CategoricalMissingStrategy categoricalMissing =
        CategoricalMissingStrategy::MissingCategory;

    ScalingStrategy scaling = ScalingStrategy::MinMax;
    EncodingStrategy encoding = EncodingStrategy::OneHot;
};

struct ProcessedDataset {
    vector<vector<double>> data;
    vector<string> featureNames;
};

bool preprocessDataset(
    const Dataset& dataset,
    ProcessedDataset& processedDataset,
    const PreprocessingConfig& config = {}
);

bool saveProcessedDataset(
    const ProcessedDataset& processedDataset,
    const string& filename
);

bool saveProcessedDatasetBinary(
    const ProcessedDataset& processedDataset,
    const string& filename
);

bool loadProcessedDatasetBinary(
    const string& filename,
    ProcessedDataset& processedDataset
);

bool compareProcessedDatasets(
    const ProcessedDataset& first,
    const ProcessedDataset& second
);
