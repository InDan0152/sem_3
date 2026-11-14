#include "csv_loader.hpp"
#include "statistics.hpp"
#include "preprocessing.hpp"
#include "supervised_dataset.hpp"
#include "dataset_splitter.hpp"

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

// Тест 10. Выделение целевого столбца
bool testSupervisedDataset() {
    Dataset dataset;

    // Создаём числовой признак.
    Column age;
    age.name = "age";
    age.type = FeatureType::Numerical;
    age.numericalValues = {25.0, 30.0, 35.0};

    // Создаём категориальный признак.
    Column job;
    job.name = "job";
    job.type = FeatureType::Categorical;
    job.categoricalValues = {
        string("admin"),
        string("services"),
        string("admin")
    };

    // Создаём целевой столбец.
    Column target;
    target.name = "y";
    target.type = FeatureType::Categorical;
    target.categoricalValues = {
        string("no"),
        string("yes"),
        string("no")
    };

    dataset.columns = {age, job, target};

    // Задаём соответствие категорий числам.
    map<string, double> mapping = {
        {"no", 0.0},
        {"yes", 1.0}
    };

    SupervisedDataset result =
        prepareSupervisedDataset(dataset, "y", mapping);

    // Проверяем количество признаков и объектов.
    if (result.features.columns.size() != 2 ||
        result.targets.size() != 3) {
        return false;
    }

    // Проверяем, что целевой столбец исключён из признаков.
    if (result.features.columns[0].name != "age" ||
        result.features.columns[1].name != "job") {
        return false;
    }

    // Проверяем правильность преобразования целевых значений.
    const vector<double> expectedTargets = {
        0.0, 1.0, 0.0
    };

    return result.targets == expectedTargets;
}

// Тест 11. Проверка отсутствующего целевого столбца
bool testMissingTargetColumn() {
    Dataset dataset;

    Column age;
    age.name = "age";
    age.type = FeatureType::Numerical;
    age.numericalValues = {25.0, 30.0};

    dataset.columns = {age};

    try {
        prepareSupervisedDataset(dataset, "y");
    } catch (const invalid_argument&) {
        return true;
    }

    return false;
}

// Тест 12. Проверка пропущенных значений в целевом столбце
bool testMissingTargetValues() {
    Dataset dataset;

    Column age;
    age.name = "age";
    age.type = FeatureType::Numerical;
    age.numericalValues = {25.0, 30.0};

    Column target;
    target.name = "y";
    target.type = FeatureType::Categorical;
    target.categoricalValues = {
        string("no"),
        nullopt
    };

    dataset.columns = {age, target};

    map<string, double> mapping = {
        {"no", 0.0},
        {"yes", 1.0}
    };

    try {
        prepareSupervisedDataset(dataset, "y", mapping);
    } catch (const invalid_argument&) {
        return true;
    }

    return false;
}

// Тест 13. Проверка неизвестной категории
bool testUnknownTargetCategory() {
    Dataset dataset;

    Column age;
    age.name = "age";
    age.type = FeatureType::Numerical;
    age.numericalValues = {25.0, 30.0};

    Column target;
    target.name = "y";
    target.type = FeatureType::Categorical;
    target.categoricalValues = {
        string("no"),
        string("yes")
    };

    dataset.columns = {age, target};

    // Намеренно не добавляем категорию "yes".
    map<string, double> mapping = {
        {"no", 0.0}
    };

    try {
        prepareSupervisedDataset(dataset, "y", mapping);
    } catch (const invalid_argument&) {
        return true;
    }

    return false;
}

// Тест 14. Проверка отсутствия признаков
bool testNoFeatures() {
    Dataset dataset;

    // В наборе данных присутствует только целевой столбец.
    Column target;
    target.name = "y";
    target.type = FeatureType::Numerical;
    target.numericalValues = {0.0, 1.0};

    dataset.columns = {target};

    try {
        prepareSupervisedDataset(dataset, "y");
    } catch (const invalid_argument&) {
        return true;
    }

    return false;
}


// Тест 15. Проверка размеров обучающей и тестовой выборок
bool testDatasetSplitSizes() {
    Dataset dataset;

    Column age;
    age.name = "age";
    age.type = FeatureType::Numerical;

    Column target;
    target.name = "y";
    target.type = FeatureType::Numerical;

    for (int i = 0; i < 100; ++i) {
        age.numericalValues.push_back(static_cast<double>(i));
        target.numericalValues.push_back(static_cast<double>(i % 2));
    }

    dataset.columns = {age, target};

    SupervisedDataset supervised =
        prepareSupervisedDataset(dataset, "y");

    DatasetSplit split = splitDataset(supervised, 0.2, 42);

    return split.train.targets.size() == 80 &&
           split.test.targets.size() == 20;
}

// Тест 16. Проверка воспроизводимости разделения
bool testDatasetSplitReproducibility() {
    Dataset dataset;

    Column age;
    age.name = "age";
    age.type = FeatureType::Numerical;

    Column target;
    target.name = "y";
    target.type = FeatureType::Numerical;

    for (int i = 0; i < 100; ++i) {
        age.numericalValues.push_back(static_cast<double>(i));
        target.numericalValues.push_back(static_cast<double>(i));
    }

    dataset.columns = {age, target};

    SupervisedDataset supervised =
        prepareSupervisedDataset(dataset, "y");

    DatasetSplit first = splitDataset(supervised, 0.2, 42);
    DatasetSplit second = splitDataset(supervised, 0.2, 42);

    return first.train.targets == second.train.targets &&
           first.test.targets == second.test.targets;
}

// Тест 17. Проверка некорректной доли тестовой выборки
bool testInvalidTestRatio() {
    SupervisedDataset dataset;

    Column age;
    age.name = "age";
    age.type = FeatureType::Numerical;
    age.numericalValues = {20.0, 30.0};

    dataset.features.columns = {age};
    dataset.targets = {0.0, 1.0};

    try {
        splitDataset(dataset, 1.0);
    } catch (const invalid_argument&) {
        return true;
    }

    return false;
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

      if (!testSupervisedDataset()) {
        cerr << "Тест выделения целевого столбца не пройден\n";
        return 1;
    }

    cout << "Тест выделения целевого столбца пройден\n";

    
    if (!testMissingTargetColumn()) {
        cerr << "Тест отсутствующего целевого столбца не пройден\n";
        return 1;
    }
    cout << "Тест отсутствующего целевого столбца пройден\n";

    if (!testMissingTargetValues()) {
        cerr << "Тест пропущенных значений не пройден\n";
        return 1;
    }
    cout << "Тест пропущенных значений пройден\n";

    if (!testUnknownTargetCategory()) {
        cerr << "Тест неизвестной категории не пройден\n";
        return 1;
    }
    cout << "Тест неизвестной категории пройден\n";

    if (!testNoFeatures()) {
        cerr << "Тест отсутствия признаков не пройден\n";
        return 1;
    }
    cout << "Тест отсутствия признаков пройден\n";

        if (!testDatasetSplitSizes()) {
        cerr << "Тест размеров выборок не пройден\n";
        return 1;
    }
    cout << "Тест размеров выборок пройден\n";

    if (!testDatasetSplitReproducibility()) {
        cerr << "Тест воспроизводимости разделения не пройден\n";
        return 1;
    }
    cout << "Тест воспроизводимости разделения пройден\n";

    if (!testInvalidTestRatio()) {
        cerr << "Тест некорректной доли выборки не пройден\n";
        return 1;
    }
    cout << "Тест некорректной доли выборки пройден\n";

    cout << "\nВсе тесты пройдены\n";
    return 0;
}