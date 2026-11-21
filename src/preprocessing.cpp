#include "preprocessing.hpp"
#include "statistics.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <algorithm>
#include <set>
#include <stdexcept>
#include <numeric>

using namespace std;

namespace {

void fillNumericalMissing(
    const Column& column,
    vector<double>& values,
    NumericalMissingStrategy strategy
) {
    if (column.type != FeatureType::Numerical) {
        throw invalid_argument("Ожидался числовой признак: " + column.name);
    }

    values.clear();

    double replacementValue = 0.0;

    // Для среднего и медианы вычисляем статистику
    // по исходным непустым значениям.
    if (strategy == NumericalMissingStrategy::Mean ||
        strategy == NumericalMissingStrategy::Median) {

        NumericalStatistics statistics;

        if (!calculateNumericalStatistics(column, statistics)) {
            throw runtime_error(
                "Не удалось рассчитать статистику числового признака " +
                column.name
            );
        }

        if (!statistics.hasValues) {
            throw invalid_argument(
                "Числовой признак " + column.name + " полностью пустой"
            );
        }

        if (strategy == NumericalMissingStrategy::Mean) {
            replacementValue = statistics.mean;
        }
        else {
            replacementValue = statistics.q50;
        }
    }
    else if (strategy != NumericalMissingStrategy::Zero) {
        throw invalid_argument("Неизвестная стратегия заполнения пропусков");
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

}

void fillCategoricalMissing(
    const Column& column,
    vector<string>& values,
    CategoricalMissingStrategy strategy
) {
    if (column.type != FeatureType::Categorical) {
        throw invalid_argument("Ожидался категориальный признак: " + column.name);
    }

    if (strategy != CategoricalMissingStrategy::MissingCategory) {
        throw invalid_argument("Неизвестная стратегия заполнения категорий");
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

}

void scaleNumericalValues(
    vector<double>& values,
    ScalingStrategy strategy
) {
    if (values.empty()) {
        return;
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
            return;
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
            return;
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
            return;
        }

        for (double& value : values) {
            value = (value - median) / interquartileRange;
        }
    }
    else {
        throw invalid_argument("Неизвестная стратегия масштабирования");
    }
}

void encodeCategoricalValues(const Column& column,const vector<string>& values,
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
                throw runtime_error("Не удалось найти категорию для кодирования");
            }

            const size_t categoryCode =
                static_cast<size_t>(distance(categories.begin(), it));

            data[i].push_back(static_cast<double>(categoryCode));
        }
    }
    else {
        throw invalid_argument("Неизвестная стратегия кодирования");
    }
} 
}

// Вычисляет медиану числового набора данных.
static double calculateMedian(vector<double> values) {
    if (values.empty()) {
        throw invalid_argument(
            "Невозможно вычислить медиану пустого набора данных"
        );
    }

    sort(values.begin(), values.end());

    size_t middle = values.size() / 2;

    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }

    return values[middle];
}

// Вычисляет среднее арифметическое.
static double calculateMean(const vector<double>& values) {
    if (values.empty()) {
        throw invalid_argument(
            "Невозможно вычислить среднее пустого набора данных"
        );
    }

    double sum = accumulate(values.begin(), values.end(), 0.0);

    return sum / static_cast<double>(values.size());
}

void preprocessDataset(
    const Dataset& dataset,
    ProcessedDataset& processedDataset,
    const PreprocessingConfig& config
) {
    if (dataset.columns.empty()) {
        throw invalid_argument("Датасет не содержит признаков");
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
            throw invalid_argument("Разное количество записей в признаках");
        }
    }

    processedDataset.data.assign(recordCount, {});
    processedDataset.featureNames.clear();

    for (const Column& column : dataset.columns) {
        if (column.type == FeatureType::Numerical) {
            vector<double> values;
            fillNumericalMissing(column, values, config.numericalMissing);
            scaleNumericalValues(values, config.scaling);

            processedDataset.featureNames.push_back(column.name);

            for (size_t i = 0; i < recordCount; i++) {
                processedDataset.data[i].push_back(values[i]);
            }
        }
        else {
            vector<string> values;
            fillCategoricalMissing(column, values, config.categoricalMissing);
            encodeCategoricalValues(
                column,
                values,
                processedDataset.data,
                processedDataset.featureNames,
                config.encoding
            );
        }
    }
}

void saveProcessedDataset(
    const ProcessedDataset& processedDataset,
    const string& filename
) {
    ofstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Не удалось создать файл: " + filename);
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
        file << '\n';
    }

    if (!file) {
        throw runtime_error("Ошибка при записи файла: " + filename);
    }
}

void saveProcessedDatasetBinary(
    const ProcessedDataset& processedDataset,
    const string& filename
) {
    ofstream file(filename, ios::binary);

    if (!file.is_open()) {
        throw runtime_error("Не удалось создать бинарный файл: " + filename);
    }

    const uint64_t rows = processedDataset.data.size();
    const uint64_t columns = processedDataset.featureNames.size();

    for (const auto& row : processedDataset.data) {
        if (row.size() != columns) {
            throw invalid_argument("Неправильный размер строки матрицы");
        }
    }

    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));

    const uint32_t storageOrder = 1;
    file.write(reinterpret_cast<const char*>(&storageOrder), sizeof(storageOrder));

    for (const string& name : processedDataset.featureNames) {
        const uint64_t length = name.size();
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(name.data(), static_cast<streamsize>(length));
    }

    for (const auto& row : processedDataset.data) {
        file.write(
            reinterpret_cast<const char*>(row.data()),
            static_cast<streamsize>(row.size() * sizeof(double))
        );
    }

    if (!file) {
        throw runtime_error("Ошибка при записи бинарного файла");
    }
}

void loadProcessedDatasetBinary(
    const string& filename,
    ProcessedDataset& processedDataset
) {
    ifstream file(filename, ios::binary);

    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть бинарный файл: " + filename);
    }

    uint64_t rows = 0;
    uint64_t columns = 0;
    uint32_t storageOrder = 0;

    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns), sizeof(columns));
    file.read(reinterpret_cast<char*>(&storageOrder), sizeof(storageOrder));

    if (!file || storageOrder != 1) {
        throw invalid_argument("Неправильный заголовок бинарного файла");
    }

    if (rows > numeric_limits<size_t>::max() ||
        columns > numeric_limits<size_t>::max()) {
        throw length_error("Слишком большие размеры матрицы");
    }

    if (columns != 0 &&
        rows > numeric_limits<size_t>::max() / columns) {
        throw length_error("Переполнение размера матрицы");
    }

    processedDataset.data.assign(
        static_cast<size_t>(rows),
        vector<double>(static_cast<size_t>(columns))
    );

    processedDataset.featureNames.clear();
    processedDataset.featureNames.reserve(static_cast<size_t>(columns));

    for (uint64_t i = 0; i < columns; i++) {
        uint64_t length = 0;
        file.read(reinterpret_cast<char*>(&length), sizeof(length));

        if (!file || length > numeric_limits<size_t>::max() ||
            length > static_cast<uint64_t>(numeric_limits<streamsize>::max())) {
            throw runtime_error("Ошибка при чтении имени признака");
        }

        string name(static_cast<size_t>(length), '\0');
        file.read(name.data(), static_cast<streamsize>(length));

        if (!file) {
            throw runtime_error("Ошибка при чтении имени признака");
        }

        processedDataset.featureNames.push_back(move(name));
    }

    for (auto& row : processedDataset.data) {
        const size_t byteCount = row.size() * sizeof(double);
        if (byteCount > static_cast<size_t>(numeric_limits<streamsize>::max())) {
            throw length_error("Слишком большой размер строки матрицы");
        }

        file.read(
            reinterpret_cast<char*>(row.data()),
            static_cast<streamsize>(byteCount)
        );

        if (!file) {
            throw runtime_error("Ошибка при чтении значений матрицы");
        }
    }
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

DatasetPreprocessor::DatasetPreprocessor(const PreprocessingConfig& config) : config(config) {}


void DatasetPreprocessor::fit(const Dataset& dataset) {
    fitted = false;
    // Проверяем, что набор данных не пустой.
    if (dataset.columns.empty()) {
        throw invalid_argument("Набор данных пуст");
    }

    // Определяем количество объектов.
    const Column& firstColumn = dataset.columns.front();

    size_t rowCount =
        firstColumn.type == FeatureType::Numerical
            ? firstColumn.numericalValues.size()
            : firstColumn.categoricalValues.size();

    if (rowCount == 0) {
        throw invalid_argument(
            "Набор данных не содержит объектов"
        );
    }

    // Очищаем предыдущие параметры.
    featureParams.clear();
    outputFeatureNames.clear();

    // Обрабатываем каждый признак отдельно.
    for (const Column& column : dataset.columns) {
        FeatureTransformParams params;

        params.name = column.name;
        params.type = column.type;

        if (column.type == FeatureType::Numerical) {
            // Проверяем размер числового столбца.
            if (column.numericalValues.size() != rowCount) {
                throw invalid_argument(
                    "Некорректное количество значений в столбце: "
                    + column.name
                );
            }

            // Собираем существующие числовые значения.
            vector<double> values;

            for (const auto& value : column.numericalValues) {
                if (value.has_value()) {
                    values.push_back(*value);
                }
            }

            if (values.empty()) {
                throw invalid_argument(
                    "Числовой признак полностью состоит из пропусков: "
                    + column.name
                );
            }

            // Определяем значение для заполнения пропусков.
            switch (config.numericalMissing) {
                case NumericalMissingStrategy::Mean:
                    params.numericalParams.imputationValue =
                        calculateMean(values);
                    break;

                case NumericalMissingStrategy::Median:
                    params.numericalParams.imputationValue =
                        calculateMedian(values);
                    break;

                case NumericalMissingStrategy::Zero:
                    params.numericalParams.imputationValue = 0.0;
                    break;
            }

            // Вычисляем параметры масштабирования.
            double center = 0.0;
            double scale = 1.0;

            switch (config.scaling) {
                case ScalingStrategy::MinMax: {
                    auto minMax = minmax_element(
                        values.begin(), values.end()
                    );

                    center = *minMax.first;
                    scale = *minMax.second - center;
                    break;
                }

                case ScalingStrategy::Standard: {
                    center = calculateMean(values);

                    double sumSquares = 0.0;

                    for (double value : values) {
                        double difference = value - center;
                        sumSquares += difference * difference;
                    }

                    scale = sqrt(
                        sumSquares / static_cast<double>(values.size())
                    );
                    break;
                }

                case ScalingStrategy::Robust: {
                    // Сортируем значения для вычисления квартилей.
                    vector<double> sortedValues = values;
                    sort(sortedValues.begin(), sortedValues.end());

                    const size_t middle = sortedValues.size() / 2;

                    // Вычисляем медиану.
                    center = calculateMedian(sortedValues);

                    // Разделяем значения на нижнюю и верхнюю половины.
                    vector<double> lower(
                        sortedValues.begin(),
                        sortedValues.begin() + middle
                    );

                    vector<double> upper(
                        sortedValues.begin() +
                            (sortedValues.size() % 2 == 0
                                ? middle
                                : middle + 1),
                        sortedValues.end()
                    );

                    // Вычисляем межквартильный размах.
                    if (!lower.empty() && !upper.empty()) {
                        double q1 = calculateMedian(lower);
                        double q3 = calculateMedian(upper);

                        scale = q3 - q1;
                    } else {
                        scale = 1.0;
                    }

                    break;
                }
            }

            // Защита от деления на ноль.
            if (scale == 0.0) {
                scale = 1.0;
            }

            params.numericalParams.center = center;
            params.numericalParams.scale = scale;

            outputFeatureNames.push_back(column.name);
        } else {
            // Проверяем размер категориального столбца.
            if (column.categoricalValues.size() != rowCount) {
                throw invalid_argument(
                    "Некорректное количество значений в столбце: "
                    + column.name
                );
            }

            // Запоминаем категории, встречающиеся в обучающих данных.
            for (const auto& value : column.categoricalValues) {
                string category = value.has_value()
                    ? *value
                    : "__MISSING__";

                if (find(
                    params.categories.begin(),
                    params.categories.end(),
                    category
                ) == params.categories.end()) {
                    params.categories.push_back(category);
                }
            }

            // Формируем имена новых признаков.
            if (config.encoding == EncodingStrategy::OneHot) {
                for (const string& category : params.categories) {
                    outputFeatureNames.push_back(
                        column.name + "_" + category
                    );
                }
            } else {
                outputFeatureNames.push_back(column.name);
            }
        }

        featureParams.push_back(params);
    }

    // Отмечаем, что параметры предобработки рассчитаны.
    fitted = true;
}


ProcessedDataset DatasetPreprocessor::transform(const Dataset& dataset) const {
    // Проверяем, что предварительное обучение выполнено.
    if (!fitted) {
        throw invalid_argument(
            "Предобработчик ещё не обучен. Сначала вызовите fit()"
        );
    }

    // Проверяем количество признаков.
    if (dataset.columns.size() != featureParams.size()) {
        throw invalid_argument(
            "Количество признаков не совпадает с обучающей выборкой"
        );
    }

    // Определяем количество объектов.
    const Column& firstColumn = dataset.columns.front();

    size_t rowCount =
        firstColumn.type == FeatureType::Numerical
            ? firstColumn.numericalValues.size()
            : firstColumn.categoricalValues.size();

    // Проверяем размеры и порядок признаков.
    for (size_t i = 0; i < dataset.columns.size(); ++i) {
        const Column& column = dataset.columns[i];
        const FeatureTransformParams& params = featureParams[i];

        if (column.name != params.name ||
            column.type != params.type) {
            throw invalid_argument(
                "Структура признаков не совпадает с обучающей выборкой"
            );
        }

        size_t columnSize =
            column.type == FeatureType::Numerical
                ? column.numericalValues.size()
                : column.categoricalValues.size();

        if (columnSize != rowCount) {
            throw invalid_argument(
                "Некорректное количество значений в столбце: "
                + column.name
            );
        }
    }

    ProcessedDataset result;

    result.featureNames = outputFeatureNames;
    result.data.assign(rowCount, {});

    // Обрабатываем каждый признак.
    for (size_t j = 0; j < dataset.columns.size(); ++j) {
        const Column& column = dataset.columns[j];
        const FeatureTransformParams& params = featureParams[j];

        if (column.type == FeatureType::Numerical) {
            // Заполняем пропуски сохранённым значением.
            for (size_t i = 0; i < rowCount; ++i) {
                double value = column.numericalValues[i].has_value()
                    ? *column.numericalValues[i]
                    : params.numericalParams.imputationValue;

                // Масштабируем с параметрами обучающей выборки.
                value = (value - params.numericalParams.center) /
                        params.numericalParams.scale;

                result.data[i].push_back(value);
            }
        } else {
            // Обрабатываем категориальный признак.
            for (size_t i = 0; i < rowCount; ++i) {
                string category = column.categoricalValues[i].has_value()
                    ? *column.categoricalValues[i]
                    : "__MISSING__";

                auto it = find(
                    params.categories.begin(),
                    params.categories.end(),
                    category
                );

                if (config.encoding == EncodingStrategy::OneHot) {
                    // Для неизвестной категории все значения будут равны 0.
                    for (const string& knownCategory : params.categories) {
                        result.data[i].push_back(
                            category == knownCategory ? 1.0 : 0.0
                        );
                    }
                } else {
                    // Для неизвестной категории используем код -1.
                    if (it == params.categories.end()) {
                        result.data[i].push_back(-1.0);
                    } else {
                        size_t categoryCode = static_cast<size_t>(
                            distance(params.categories.begin(), it)
                        );

                        result.data[i].push_back(
                            static_cast<double>(categoryCode)
                        );
                    }
                }
            }
        }
    }

    return result;
}

ProcessedDataset DatasetPreprocessor::fitTransform(const Dataset& dataset) {
    fit(dataset);
    return transform(dataset);
}
