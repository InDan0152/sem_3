#include "csv_loader.hpp"
#include "statistics.hpp"
#include "preprocessing.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

using namespace std;

// Тест 1. Разбор CSV
bool testCsvParsing() {
    vector<string> result = parseCsvLine("25;unknown;admin.", ';');

    return result.size() == 3 &&
           result[0] == "25" &&
           result[1] == "unknown" &&
           result[2] == "admin.";
}

// Тест 2. Определение типов признаков
bool testFeatureTypes() {
    const string filename = "test_types.csv";
    ofstream file(filename);

    if (!file.is_open()) return false;

    file << "age;job\n";
    file << "25;admin.\n";
    file << "30;services\n";
    file.close();

    Dataset dataset;
    CsvConfig config;

    if (!loadCsv(filename, dataset, config)) return false;

    return dataset.columns.size() == 2 &&
           dataset.columns[0].type == FeatureType::Numerical &&
           dataset.columns[1].type == FeatureType::Categorical;
}

// Тест 3. Расчёт квантилей
bool testQuantiles() {
    Column column;
    column.name = "test";
    column.type = FeatureType::Numerical;
    column.numericalValues = {1.0, 2.0, 3.0, 4.0};

    NumericalStatistics statistics;

    if (!calculateNumericalStatistics(column, statistics))
        return false;

    return abs(statistics.q25 - 1.75) < 1e-9 &&
           abs(statistics.q50 - 2.5) < 1e-9 &&
           abs(statistics.q75 - 3.25) < 1e-9;
}

// Тест 4. One-Hot Encoding
bool testOneHotEncoding() {
    Dataset dataset;

    Column column;
    column.name = "color";
    column.type = FeatureType::Categorical;

    column.categoricalValues = {
        string("red"),
        string("blue"),
        nullopt
    };

    dataset.columns.push_back(column);

    ProcessedDataset processedDataset;

    if (!preprocessDataset(dataset, processedDataset))
        return false;

    if (processedDataset.data.size() != 3 ||
        processedDataset.featureNames.size() != 3) {
        return false;
    }

    const vector<string> expectedNames = {
        "color___MISSING__",
        "color_blue",
        "color_red"
    };

    if (processedDataset.featureNames != expectedNames)
        return false;

    const vector<vector<double>> expectedData = {
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}
    };

    return processedDataset.data == expectedData;
}
bool testBinarySaveLoad() {
    ProcessedDataset original;

    original.featureNames = {
        "age",
        "job_admin.",
        "job_services"
    };

    original.data = {
        {0.0, 1.0, 0.0},
        {0.5, 0.0, 1.0},
        {1.0, 1.0, 0.0}
    };

    const string filename = "test_binary.bin";

    if (!saveProcessedDatasetBinary(original, filename))
        return false;

    ProcessedDataset loaded;

    if (!loadProcessedDatasetBinary(filename, loaded))
        return false;

    return compareProcessedDatasets(original, loaded);
}

bool testInvalidCsv() {
    const string filename = "test_invalid.csv";

    ofstream file(filename);

    if (!file.is_open()) {
        return false;
    }

    file << "age;job;marital\n";
    file << "25;admin.\n"; // Не хватает одного поля

    file.close();

    Dataset dataset;
    CsvConfig config;

    // Загрузчик должен обнаружить ошибку.
    return !loadCsv(filename, dataset, config);
}

bool testScalingStrategies() {
    Dataset dataset;

    Column column;
    column.name = "value";
    column.type = FeatureType::Numerical;
    column.numericalValues = {1.0, 2.0, 3.0};

    dataset.columns.push_back(column);

    const vector<ScalingStrategy> strategies = {
        ScalingStrategy::MinMax,
        ScalingStrategy::Standard,
        ScalingStrategy::Robust
    };

    for (const auto strategy : strategies) {
        PreprocessingConfig config;
        config.scaling = strategy;

        ProcessedDataset result;

        if (!preprocessDataset(dataset, result, config))
            return false;

        if (result.data.size() != 3 ||
            result.featureNames.size() != 1)
            return false;

        if (strategy == ScalingStrategy::MinMax) {
            if (abs(result.data[0][0] - 0.0) > 1e-9 ||
                abs(result.data[1][0] - 0.5) > 1e-9 ||
                abs(result.data[2][0] - 1.0) > 1e-9)
                return false;
        }

        if (strategy == ScalingStrategy::Standard) {
            if (abs(result.data[0][0] + sqrt(1.5)) > 1e-9 ||
                abs(result.data[1][0]) > 1e-9 ||
                abs(result.data[2][0] - sqrt(1.5)) > 1e-9)
                return false;
        }

        if (strategy == ScalingStrategy::Robust) {
            if (abs(result.data[0][0] + 1.0) > 1e-9 ||
                abs(result.data[1][0]) > 1e-9 ||
                abs(result.data[2][0] - 1.0) > 1e-9)
                return false;
        }
    }

    return true;
}

// Тест 8. Заполнение пропусков средним
bool testMeanImputation() {
    Dataset dataset;

    Column column;
    column.name = "value";
    column.type = FeatureType::Numerical;
    column.numericalValues = {
        1.0, nullopt, 9.0, 10.0
    };

    dataset.columns.push_back(column);

    PreprocessingConfig config;
    config.numericalMissing = NumericalMissingStrategy::Mean;

    ProcessedDataset result;

    if (!preprocessDataset(dataset, result, config))
        return false;

    return result.data.size() == 4 &&
           abs(result.data[1][0] - 17.0 / 27.0) < 1e-9;
}

// Тест 9. Ordinal Encoding
bool testOrdinalEncoding() {
    Dataset dataset;

    Column column;
    column.name = "color";
    column.type = FeatureType::Categorical;

    column.categoricalValues = {
        string("red"),
        string("blue"),
        nullopt
    };

    dataset.columns.push_back(column);

    PreprocessingConfig config;
    config.encoding = EncodingStrategy::Ordinal;

    ProcessedDataset result;

    if (!preprocessDataset(dataset, result, config))
        return false;

    const vector<vector<double>> expected = {
        {2.0},
        {1.0},
        {0.0}
    };

    return result.featureNames.size() == 1 &&
           result.featureNames[0] == "color" &&
           result.data == expected;
}

int main() {
    if (!testCsvParsing()) {
        cerr << "Тест CSV не пройден\n";
        return 1;
    }
    cout << "Тест CSV пройден\n";

    if (!testFeatureTypes()) {
        cerr << "Тест типов признаков не пройден\n";
        return 1;
    }
    cout << "Тест типов признаков пройден\n";

    if (!testQuantiles()) {
        cerr << "Тест квантилей не пройден\n";
        return 1;
    }
    cout << "Тест квантилей пройден\n";

    if (!testOneHotEncoding()) {
        cerr << "Тест One-Hot Encoding не пройден\n";
        return 1;
    }
    cout << "Тест One-Hot Encoding пройден\n";

    if (!testBinarySaveLoad()) {
        cerr << "Тест бинарного сохранения не пройден\n";
        return 1;
    }

    cout << "Тест бинарного сохранения пройден\n";

    if (!testInvalidCsv()) {
        cerr << "Тест некорректного CSV не пройден\n";
        return 1;
    }

    cout << "Тест некорректного CSV пройден\n";

    if (!testScalingStrategies()) {
    cerr << "Тест масштабирования не пройден\n";
    return 1;
    }
    cout << "Тест масштабирования пройден\n";

    if (!testMeanImputation()) {
        cerr << "Тест заполнения пропусков не пройден\n";
        return 1;
    }
    cout << "Тест заполнения пропусков пройден\n";

    if (!testOrdinalEncoding()) {
        cerr << "Тест Ordinal Encoding не пройден\n";
        return 1;
    }
    cout << "Тест Ordinal Encoding пройден\n";

    cout << "\nВсе тесты пройдены\n";
    return 0;
}