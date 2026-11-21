#pragma once

#include "dataset.hpp"

#include <string>
#include <vector>

using namespace std;

// Стратегии заполнения пропусков в числовых признаках.
enum class NumericalMissingStrategy {
    Mean,
    Median,
    Zero
};

// Стратегии заполнения пропусков в категориальных признаках.
enum class CategoricalMissingStrategy {
    MissingCategory
};

// Стратегии масштабирования числовых признаков.
enum class ScalingStrategy {
    MinMax,
    Standard,
    Robust
};

// Способы кодирования категориальных признаков.
enum class EncodingStrategy {
    OneHot,
    Ordinal
};

// Настройки предобработки.
struct PreprocessingConfig {
    NumericalMissingStrategy numericalMissing =
        NumericalMissingStrategy::Median;

    CategoricalMissingStrategy categoricalMissing =
        CategoricalMissingStrategy::MissingCategory;

    ScalingStrategy scaling = ScalingStrategy::MinMax;
    EncodingStrategy encoding = EncodingStrategy::OneHot;
};

// Результат предобработки.
struct ProcessedDataset {
    vector<vector<double>> data;
    vector<string> featureNames;
};

// Параметры преобразования числового признака.
struct NumericalTransformParams {
    double imputationValue = 0.0;
    double center = 0.0;
    double scale = 1.0;
};

// Параметры преобразования одного признака.
struct FeatureTransformParams {
    string name;
    FeatureType type;

    NumericalTransformParams numericalParams;
    vector<string> categories;
};

// Класс для обучения и применения предобработки.
class DatasetPreprocessor {
private:
    PreprocessingConfig config;
    vector<FeatureTransformParams> featureParams;
    vector<string> outputFeatureNames;
    bool fitted = false;

public:
    explicit DatasetPreprocessor(const PreprocessingConfig& config = {});

    // Рассчитывает параметры предобработки по обучающим данным.
    void fit(const Dataset& dataset);

    // Применяет ранее рассчитанные параметры.
    ProcessedDataset transform(const Dataset& dataset) const;

    // Обучает и сразу преобразует один датасет.
    ProcessedDataset fitTransform(const Dataset& dataset);
};

// Предобработка датасета.
void preprocessDataset(
    const Dataset& dataset,
    ProcessedDataset& processedDataset,
    const PreprocessingConfig& config = {}
);

// Сохранение обработанного датасета в CSV.
void saveProcessedDataset(
    const ProcessedDataset& processedDataset,
    const string& filename
);

// Сохранение обработанного датасета в бинарном формате.
void saveProcessedDatasetBinary(
    const ProcessedDataset& processedDataset,
    const string& filename
);

// Загрузка обработанного датасета из бинарного формата.
void loadProcessedDatasetBinary(
    const string& filename,
    ProcessedDataset& processedDataset
);

// Сравнение двух обработанных датасетов.
bool compareProcessedDatasets(
    const ProcessedDataset& first,
    const ProcessedDataset& second
);
