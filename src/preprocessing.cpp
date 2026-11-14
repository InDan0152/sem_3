#include "preprocessing.hpp"
#include "statistics.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <algorithm>
#include <iostream>
#include <set>

using namespace std;

namespace {

bool fillNumericalMissing(
    const Column& column,
    vector<double>& values,
    NumericalMissingStrategy strategy
) {
    if (column.type != FeatureType::Numerical) {
        return false;
    }

    values.clear();

    double replacementValue = 0.0;

    // Для среднего и медианы вычисляем статистику
    // по исходным непустым значениям.
    if (strategy == NumericalMissingStrategy::Mean ||
        strategy == NumericalMissingStrategy::Median) {

        NumericalStatistics statistics;

        if (!calculateNumericalStatistics(column, statistics)) {
            cerr << "Ошибка при расчёте статистики признака "
                 << column.name << '\n';
            return false;
        }

        if (!statistics.hasValues) {
            cerr << "Ошибка: числовой признак " << column.name
                 << " полностью пустой\n";
            return false;
        }

        if (strategy == NumericalMissingStrategy::Mean) {
            replacementValue = statistics.mean;
        }
        else {
            replacementValue = statistics.q50;
        }
    }
    else if (strategy != NumericalMissingStrategy::Zero) {
        return false;
    }

    // Заполняем пропуски выбранным значением.
    for (const auto& value : column.numericalValues) {
        if (value.has_value()) {
            values.push_back(value.value());
        }
        else {
            values.push_back(replacementValue);
        }
    }

    return true;
}

bool fillCategoricalMissing(
    const Column& column,
    vector<string>& values,
    CategoricalMissingStrategy strategy
) {
    if (column.type != FeatureType::Categorical ||
        strategy != CategoricalMissingStrategy::MissingCategory) {
        return false;
    }

    values.clear();
    const string missingCategory = "__MISSING__";

    for (const auto& value : column.categoricalValues) {
        if (value.has_value()) {
            values.push_back(value.value());
        }
        else {
            values.push_back(missingCategory);
        }
    }

    return true;
}

bool scaleNumericalValues(
    vector<double>& values,
    ScalingStrategy strategy
) {
    if (values.empty()) {
        return true;
    }

    if (strategy == ScalingStrategy::MinMax) {
        const auto minMax =
            minmax_element(values.begin(), values.end());

        const double minValue = *minMax.first;
        const double maxValue = *minMax.second;

        if (minValue == maxValue) {
            for (double& value : values) {
                value = 0.0;
            }
            return true;
        }

        for (double& value : values) {
            value = (value - minValue) / (maxValue - minValue);
        }
    }
    else if (strategy == ScalingStrategy::Standard) {
        double mean = 0.0;

        for (double value : values) {
            mean += value;
        }

        mean /= static_cast<double>(values.size());

        double variance = 0.0;

        for (double value : values) {
            variance += (value - mean) * (value - mean);
        }

        variance /= static_cast<double>(values.size());

        const double standardDeviation = sqrt(variance);

        if (standardDeviation == 0.0) {
            for (double& value : values) {
                value = 0.0;
            }
            return true;
        }

        for (double& value : values) {
            value = (value - mean) / standardDeviation;
        }
    }
    else if (strategy == ScalingStrategy::Robust) {
        vector<double> sortedValues = values;
        sort(sortedValues.begin(), sortedValues.end());

        const size_t n = sortedValues.size();

        const auto calculateQuantile =
            [&sortedValues, n](double p) {
                const double position =
                    p * static_cast<double>(n - 1);

                const size_t lowerIndex =
                    static_cast<size_t>(position);

                const size_t upperIndex =
                    min(lowerIndex + 1, n - 1);

                const double fraction =
                    position - static_cast<double>(lowerIndex);

                return sortedValues[lowerIndex] * (1.0 - fraction)
                    + sortedValues[upperIndex] * fraction;
            };

        const double median = calculateQuantile(0.5);
        const double q25 = calculateQuantile(0.25);
        const double q75 = calculateQuantile(0.75);

        const double interquartileRange = q75 - q25;

        if (interquartileRange == 0.0) {
            for (double& value : values) {
                value = 0.0;
            }
            return true;
        }

        for (double& value : values) {
            value = (value - median) / interquartileRange;
        }
    }
    else {
        return false;
    }

    return true;
}

bool encodeCategoricalValues(const Column& column,const vector<string>& values,
vector<vector<double>>& data,vector<string>& featureNames,EncodingStrategy strategy
) {
    // Получаем уникальные категории в отсортированном порядке.
    const set<string> categoriesSet(values.begin(), values.end());

    vector<string> categories(
        categoriesSet.begin(),
        categoriesSet.end()
    );

    if (strategy == EncodingStrategy::OneHot) {

        for (const string& category : categories) {
            featureNames.push_back(column.name + "_" + category);

            for (size_t i = 0; i < values.size(); i++) {
                data[i].push_back(
                    values[i] == category ? 1.0 : 0.0
                );
            }
        }
    }
    else if (strategy == EncodingStrategy::Ordinal) {

        // Один исходный признак превращается в один числовой столбец.
        featureNames.push_back(column.name);

        for (size_t i = 0; i < values.size(); i++) {
            auto it = lower_bound(
                categories.begin(),
                categories.end(),
                values[i]
            );

            if (it == categories.end() || *it != values[i]) {
                return false;
            }

            const size_t categoryCode =
                static_cast<size_t>(distance(categories.begin(), it));

            data[i].push_back(static_cast<double>(categoryCode));
        }
    }
    else {
        return false;
    }

    return true;
} 
}


bool preprocessDataset(
    const Dataset& dataset,
    ProcessedDataset& processedDataset,
    const PreprocessingConfig& config
) {
    if (dataset.columns.empty()) {
        cerr << "Ошибка: датасет не содержит признаков\n";
        return false;
    }

    size_t recordCount = 0;

    if (dataset.columns[0].type == FeatureType::Numerical) {
        recordCount = dataset.columns[0].numericalValues.size();
    }
    else {
        recordCount = dataset.columns[0].categoricalValues.size();
    }


    for (const Column& column : dataset.columns) {
        const size_t columnSize =
            (column.type == FeatureType::Numerical)
                ? column.numericalValues.size()
                : column.categoricalValues.size();

        if (columnSize != recordCount) {
            cerr << "Ошибка: разное количество записей в признаках\n";
            return false;
        }
    }

    processedDataset.data.assign(recordCount, {});
    processedDataset.featureNames.clear();

    for (const Column& column : dataset.columns) {
        if (column.type == FeatureType::Numerical) {
            vector<double> values;

            if (!fillNumericalMissing(
                    column, values, config.numericalMissing)) {
                cerr << "Ошибка заполнения числового признака "
                     << column.name << '\n';
                return false;
            }

            if (!scaleNumericalValues(values, config.scaling)) {
                cerr << "Ошибка: неподдерживаемая стратегия масштабирования\n";
                return false;
            }

            processedDataset.featureNames.push_back(column.name);

            for (size_t i = 0; i < recordCount; i++) {
                processedDataset.data[i].push_back(values[i]);
            }
        }
        else {
            vector<string> values;

            if (!fillCategoricalMissing(
                    column, values, config.categoricalMissing)) {
                cerr << "Ошибка заполнения категориального признака "
                     << column.name << '\n';
                                    return false;
            }

            if (!encodeCategoricalValues(
                    column,
                    values,
                    processedDataset.data,
                    processedDataset.featureNames,
                    config.encoding)) {
                cerr << "Ошибка: неподдерживаемая стратегия кодирования\n";
                return false;
            }
        }
    }

    return true;
}

bool saveProcessedDataset(
    const ProcessedDataset& processedDataset,
    const string& filename
) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Ошибка: не удалось создать файл " << filename << endl;
        return false;
    }


    for (size_t j = 0; j < processedDataset.featureNames.size(); j++) {
        if (j > 0) file << ';';
        file << processedDataset.featureNames[j];
    }
    file << '\n';


    for (const auto& row : processedDataset.data) {
        for (size_t j = 0; j < row.size(); j++) {
            if (j > 0) file << ';';
            file << row[j];
        }
        file << endl;
    }

    file.close();
    return true;
}

bool saveProcessedDatasetBinary(
    const ProcessedDataset& processedDataset,
    const string& filename
) {
    ofstream file(filename, ios::binary);

    if (!file.is_open()) {
        cerr << "Ошибка: не удалось создать бинарный файл\n";
        return false;
    }

    uint64_t rows = processedDataset.data.size();
    uint64_t columns = processedDataset.featureNames.size();


    for (const auto& row : processedDataset.data) {
        if (row.size() != columns) {
            cerr << "Ошибка: неправильный размер строки матрицы\n";
            return false;
        }
    }

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));

    uint32_t storageOrder = 1;
    file.write(reinterpret_cast<const char*>(&storageOrder), sizeof(storageOrder));

    for (const string& name : processedDataset.featureNames) {
        uint64_t length = name.size();

        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(name.data(), length);
    }

    for (const auto& row : processedDataset.data) {
        file.write(
            reinterpret_cast<const char*>(row.data()),
            row.size() * sizeof(double)
        );
    }

    if (!file) {
        cerr << "Ошибка при записи бинарного файла\n";
        return false;
    }

    return true;
}

bool loadProcessedDatasetBinary(
    const string& filename,
    ProcessedDataset& processedDataset
) {
    ifstream file(filename, ios::binary);

    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть бинарный файл\n";
        return false;
    }

    uint64_t rows;
    uint64_t columns;
    uint32_t storageOrder;

    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    file.read(reinterpret_cast<char*>(&storageOrder), sizeof(storageOrder));

    if (!file || storageOrder != 1) {
        cerr << "Ошибка: неправильный заголовок файла\n";
        return false;
    }

    if (rows > numeric_limits<size_t>::max() ||
        columns > numeric_limits<size_t>::max()) {
        cerr << "Ошибка: слишком большие размеры матрицы\n";
        return false;
    }

    if (columns != 0 &&
        rows > numeric_limits<size_t>::max() / columns) {
        cerr << "Ошибка: переполнение размера матрицы\n";
        return false;
    }

    processedDataset.data.assign(
        static_cast<size_t>(rows),
        vector<double>(static_cast<size_t>(columns))
    );

    processedDataset.featureNames.clear();

    for (uint64_t i = 0; i < columns; i++) {
        uint64_t length;

        file.read(reinterpret_cast<char*>(&length), sizeof(length));

        if (!file || length > numeric_limits<size_t>::max()) {
            cerr << "Ошибка при чтении имени признака\n";
            return false;
        }

        string name(static_cast<size_t>(length), '\0');

        file.read(name.data(), static_cast<streamsize>(length));

        if (!file) {
            cerr << "Ошибка при чтении имени признака\n";
            return false;
        }

        processedDataset.featureNames.push_back(name);
    }

    for (auto& row : processedDataset.data) {
        file.read(
            reinterpret_cast<char*>(row.data()),
            static_cast<streamsize>(row.size() * sizeof(double))
        );

        if (!file) {
            cerr << "Ошибка при чтении значений матрицы\n";
            return false;
        }
    }

    return true;
}

bool compareProcessedDatasets(
    const ProcessedDataset& first,
    const ProcessedDataset& second
) {
    if (first.featureNames != second.featureNames)
        return false;

    if (first.data.size() != second.data.size())
        return false;

    for (size_t i = 0; i < first.data.size(); i++) {
        if (first.data[i].size() != second.data[i].size())
            return false;

        for (size_t j = 0; j < first.data[i].size(); j++) {
            if (first.data[i][j] != second.data[i][j])
                return false;
        }
    }

    return true;
}
