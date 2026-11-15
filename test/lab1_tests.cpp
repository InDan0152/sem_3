#include "csv_loader.hpp"
#include "statistics.hpp"
#include "preprocessing.hpp"
#include "supervised_dataset.hpp"
#include "dataset_splitter.hpp"
#include "matrix.hpp"
#include "distance.hpp"
#include "neighbor_search.hpp"
#include "knn.hpp"
#include "exceptions.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <algorithm>

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

// Тест Matrix 1. Создание матрицы заданного размера.
bool testMatrixCreation() {
    Matrix matrix(3, 2);

    if (matrix.rows() != 3 || matrix.columns() != 2) {
        return false;
    }

    // Все элементы новой матрицы должны быть равны нулю.
    for (size_t i = 0; i < matrix.rows(); ++i) {
        for (double value : matrix[i]) {
            if (value != 0.0) {
                return false;
            }
        }
    }

    return true;
}

// Тест Matrix 2. Создание матрицы из двумерного вектора.
bool testMatrixFromVector() {
    vector<vector<double>> values = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Matrix matrix(values);

    return matrix.rows() == 2 &&
           matrix.columns() == 3 &&
           matrix[0][0] == 1.0 &&
           matrix[0][2] == 3.0 &&
           matrix[1][0] == 4.0 &&
           matrix[1][2] == 6.0;
}

// Тест Matrix 3. Проверка строк разной длины.
bool testMatrixInvalidRows() {
    vector<vector<double>> values = {
        {1.0, 2.0},
        {3.0}
    };

    try {
        Matrix matrix(values);
    }
    catch (const invalid_argument&) {
        return true;
    }

    return false;
}

// Тест Matrix 4. Доступ к строкам через span.
bool testMatrixSpanAccess() {
    Matrix matrix(2, 3);

    matrix[0][0] = 10.0;
    matrix[0][1] = 20.0;
    matrix[1][2] = 30.0;

    span<double> firstRow = matrix[0];

    // Изменяем элемент через span.
    firstRow[1] = 50.0;

    // Проверяем, что изменилась исходная матрица.
    return matrix[0][0] == 10.0 &&
           matrix[0][1] == 50.0 &&
           matrix[1][2] == 30.0 &&
           firstRow.size() == 3;
}

// Тест Matrix 5. Проверка выхода за границы через at().
bool testMatrixOutOfRange() {
    Matrix matrix(2, 3);

    try {
        matrix.at(2);
    }
    catch (const out_of_range&) {
        return true;
    }

    return false;
}

// Тест Distance 1. Евклидово расстояние.
bool testEuclideanDistance() {
    EuclideanDistance distance;

    vector<double> a = {1.0, 2.0, 3.0};
    vector<double> b = {4.0, 6.0, 3.0};

    double result = distance.compute(a, b);

    return abs(result - 5.0) < 1e-9;
}

// Тест Distance 2. Манхэттенское расстояние.
bool testManhattanDistance() {
    ManhattanDistance distance;

    vector<double> a = {1.0, 2.0, 3.0};
    vector<double> b = {4.0, 6.0, 3.0};

    double result = distance.compute(a, b);

    return abs(result - 7.0) < 1e-9;
}

// Тест Distance 3. Расстояние Чебышёва.
bool testChebyshevDistance() {
    ChebyshevDistance distance;

    vector<double> a = {1.0, 2.0, 3.0};
    vector<double> b = {4.0, 6.0, 3.0};

    double result = distance.compute(a, b);

    return abs(result - 4.0) < 1e-9;
}

// Тест Distance 4. Проверка несовпадающих размерностей.
bool testDistanceDimensionMismatch() {
    EuclideanDistance distance;

    vector<double> a = {1.0, 2.0, 3.0};
    vector<double> b = {4.0, 6.0};

    try {
        distance.compute(a, b);
    }
    catch (const invalid_argument&) {
        return true;
    }

    return false;
}

// Тест Neighbor Search 1. Поиск двух ближайших соседей.
bool testKNearestNeighbors() {
    vector<vector<double>> values = {
        {0.0},
        {2.0},
        {5.0},
        {9.0}
    };

    Matrix trainFeatures(values);

    vector<double> query = {3.0};

    EuclideanDistance distance;

    vector<Neighbor> neighbors = findKNearestNeighbors(
        trainFeatures,
        query,
        2,
        distance
    );

    return neighbors.size() == 2 &&
           neighbors[0].index == 1 &&
           neighbors[1].index == 2 &&
           abs(neighbors[0].distance - 1.0) < 1e-9 &&
           abs(neighbors[1].distance - 2.0) < 1e-9;
}

// Тест Neighbor Search 2. Детерминированный порядок при равных расстояниях.
bool testKNearestNeighborsEqualDistances() {
    vector<vector<double>> values = {
        {-1.0},
        {1.0},
        {3.0}
    };

    Matrix trainFeatures(values);

    vector<double> query = {0.0};

    EuclideanDistance distance;

    vector<Neighbor> neighbors = findKNearestNeighbors(
        trainFeatures,
        query,
        2,
        distance
    );

    return neighbors.size() == 2 &&
           neighbors[0].index == 0 &&
           neighbors[1].index == 1 &&
           abs(neighbors[0].distance - 1.0) < 1e-9 &&
           abs(neighbors[1].distance - 1.0) < 1e-9;
}

// Тест Neighbor Search 3. Проверка k = 0.
bool testKNearestNeighborsZeroK() {
    vector<vector<double>> values = {
        {1.0},
        {2.0}
    };

    Matrix trainFeatures(values);

    vector<double> query = {1.5};

    EuclideanDistance distance;

    try {
        findKNearestNeighbors(
            trainFeatures,
            query,
            0,
            distance
        );
    }
    catch (const invalid_argument&) {
        return true;
    }

    return false;
}

// Тест Neighbor Search 4. Проверка k больше количества объектов.
bool testKNearestNeighborsTooLargeK() {
    vector<vector<double>> values = {
        {1.0},
        {2.0}
    };

    Matrix trainFeatures(values);

    vector<double> query = {1.5};

    EuclideanDistance distance;

    try {
        findKNearestNeighbors(
            trainFeatures,
            query,
            3,
            distance
        );
    }
    catch (const invalid_argument&) {
        return true;
    }

    return false;
}

// Тест Neighbor Search 5. Несовпадение размерностей.
bool testKNearestNeighborsDimensionMismatch() {
    vector<vector<double>> values = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix trainFeatures(values);

    vector<double> query = {1.0};

    EuclideanDistance distance;

    try {
        findKNearestNeighbors(
            trainFeatures,
            query,
            1,
            distance
        );
    }
    catch (const invalid_argument&) {
        return true;
    }

    return false;
}


// Тест kNN 1. Регрессия с одним соседом.
bool testKnnRegressorK1() {
    Matrix features({
        {0.0},
        {2.0},
        {5.0}
    });

    vector<double> targets = {10.0, 20.0, 30.0};

    KnnRegressor model(
        1,
        make_unique<EuclideanDistance>()
    );

    model.fit(features, targets);

    vector<double> predictions = model.predict(features);

    return predictions.size() == 3 &&
           abs(predictions[0] - 10.0) < 1e-9 &&
           abs(predictions[1] - 20.0) < 1e-9 &&
           abs(predictions[2] - 30.0) < 1e-9;
}

// Тест kNN 2. Регрессия с равными весами.
bool testKnnRegressorUniform() {
    vector<vector<double>> trainValues = {
        {0.0},
        {3.0}
    };
    Matrix trainFeatures(trainValues);

    vector<double> targets = {0.0, 12.0};

    vector<vector<double>> testValues = {
        {1.0}
    };
    Matrix testFeatures(testValues);


    KnnRegressor model(
        2,
        make_unique<EuclideanDistance>(),
        Weighting::Uniform
    );

    model.fit(trainFeatures, targets);

    vector<double> predictions = model.predict(testFeatures);

    return predictions.size() == 1 &&
           abs(predictions[0] - 6.0) < 1e-9;
}

// Тест kNN 3. Регрессия с обратным взвешиванием расстояния.
bool testKnnRegressorDistanceWeighting() {
    vector<vector<double>> trainValues = {
        {0.0},
        {3.0}
    };
    Matrix trainFeatures(trainValues);

    vector<double> targets = {0.0, 12.0};

    vector<vector<double>> testValues = {
        {1.0}
    };
    Matrix testFeatures(testValues);

    KnnRegressor model(
        2,
        make_unique<EuclideanDistance>(),
        Weighting::Distance
    );

    model.fit(trainFeatures, targets);

    vector<double> predictions = model.predict(testFeatures);

    // Расстояния: 1 и 2.
    // Веса: 1 и 0.5.
    // Ответ: (0 * 1 + 12 * 0.5) / 1.5 = 4.
    return predictions.size() == 1 &&
           abs(predictions[0] - 4.0) < 1e-9;
}

// Тест kNN 4. Классификация.
bool testKnnClassifier() {
    Matrix trainFeatures({
        {0.0},
        {2.0},
        {4.0}
    });

    vector<double> targets = {1.0, 0.0, 0.0};

    Matrix testFeatures({
        {3.0}
    });

    KnnClassifier model(
        3,
        make_unique<EuclideanDistance>()
    );

    model.fit(trainFeatures, targets);

    vector<double> predictions = model.predict(testFeatures);

    return predictions.size() == 1 &&
           predictions[0] == 0.0;
}

// Тест kNN 5. Обработка нулевого расстояния.
bool testKnnZeroDistance() {
    vector<vector<double>> trainValues = {
        {0.0},
        {2.0}
    };
    Matrix trainFeatures(trainValues);

    vector<double> targets = {10.0, 20.0};

    vector<vector<double>> testValues = {
        {0.0}
    };
    Matrix testFeatures(testValues);

    KnnRegressor model(
        2,
        make_unique<EuclideanDistance>(),
        Weighting::Distance
    );

    model.fit(trainFeatures, targets);

    vector<double> predictions = model.predict(testFeatures);

    // Совпавший объект должен иметь единственное влияние на ответ.
    return predictions.size() == 1 &&
           abs(predictions[0] - 10.0) < 1e-9;
}


// Тест Exception 1. Несовпадение размерностей в метрике.
bool testDimensionMismatchException() {
    EuclideanDistance distance;

    vector<double> a = {1.0, 2.0};
    vector<double> b = {1.0};

    try {
        distance.compute(a, b);
    }
    catch (const DimensionMismatchError&) {
        return true;
    }
    catch (...) {
        return false;
    }

    return false;
}

// Тест Exception 2. Некорректное количество соседей.
bool testInvalidKException() {
    try {
        KnnRegressor model(
            0,
            make_unique<EuclideanDistance>()
        );
    }
    catch (const InvalidParameterError&) {
        return true;
    }
    catch (...) {
        return false;
    }

    return false;
}

// Тест Exception 3. Отсутствующая метрика.
bool testNullDistanceException() {
    try {
        KnnRegressor model(
            1,
            nullptr
        );
    }
    catch (const InvalidParameterError&) {
        return true;
    }
    catch (...) {
        return false;
    }

    return false;
}

// Тест Exception 4. Предсказание до обучения.
bool testPredictBeforeFitException() {
    KnnRegressor model(
        1,
        make_unique<EuclideanDistance>()
    );

    Matrix testFeatures(1, 2);

    try {
        model.predict(testFeatures);
    }
    catch (const ModelUsageError&) {
        return true;
    }
    catch (...) {
        return false;
    }

    return false;
}

// Тест Exception 5. Обучение на пустой выборке.
bool testEmptyTrainingDataException() {
    KnnRegressor model(
        1,
        make_unique<EuclideanDistance>()
    );

    Matrix trainFeatures(0, 2);
    vector<double> targets;

    try {
        model.fit(trainFeatures, targets);
    }
    catch (const InvalidParameterError&) {
        return true;
    }
    catch (...) {
        return false;
    }

    return false;
}

// Тест Exception 6. Несовпадение количества объектов и ответов.
bool testTargetSizeMismatchException() {
    KnnRegressor model(
        1,
        make_unique<EuclideanDistance>()
    );

    Matrix trainFeatures(2, 2);
    vector<double> targets = {10.0};

    try {
        model.fit(trainFeatures, targets);
    }
    catch (const DimensionMismatchError&) {
        return true;
    }
    catch (...) {
        return false;
    }

    return false;
}

// Тест Exception 7. Несовпадение признаков при предсказании.
bool testPredictionDimensionMismatchException() {
    KnnRegressor model(
        1,
        make_unique<EuclideanDistance>()
    );

    Matrix trainFeatures(2, 2);
    vector<double> targets = {10.0, 20.0};

    model.fit(trainFeatures, targets);

    Matrix testFeatures(1, 3);

    try {
        model.predict(testFeatures);
    }
    catch (const DimensionMismatchError&) {
        return true;
    }
    catch (...) {
        return false;
    }

    return false;
}


// Тест Neighbor Search 6.
// Сравнение поиска через кучу с полной сортировкой.
bool testKNearestNeighborsAgainstFullSort() {
    vector<vector<double>> values = {
        {0.0, 0.0},
        {1.0, 1.0},
        {2.0, 2.0},
        {-1.0, -1.0},
        {3.0, 3.0},
        {1.0, -1.0},
        {-1.0, 1.0},
        {5.0, 5.0}
    };

    Matrix trainFeatures(values);

    vector<double> query = {0.0, 0.0};

    EuclideanDistance distance;

    // Независимо вычисляем расстояния до всех объектов.
    vector<Neighbor> reference;

    for (size_t i = 0; i < trainFeatures.rows(); ++i) {
        double currentDistance =
            distance.compute(trainFeatures[i], query);

        reference.push_back({currentDistance, i});
    }

    // Полная сортировка используется только в тесте.
    sort(
        reference.begin(),
        reference.end(),
        [](const Neighbor& a, const Neighbor& b) {
            if (a.distance != b.distance) {
                return a.distance < b.distance;
            }

            return a.index < b.index;
        }
    );

    // Проверяем результаты для разных значений k.
    for (size_t k = 1; k <= trainFeatures.rows(); ++k) {
        vector<Neighbor> actual = findKNearestNeighbors(
            trainFeatures,
            query,
            k,
            distance
        );

        if (actual.size() != k) {
            return false;
        }

        for (size_t i = 0; i < k; ++i) {
            if (actual[i].index != reference[i].index) {
                return false;
            }

            if (abs(actual[i].distance - reference[i].distance) > 1e-9) {
                return false;
            }
        }
    }

    return true;
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

    // Тесты матрицы.
    if (!testMatrixCreation()) {
        cerr << "Тест создания матрицы не пройден\n";
        return 1;
    }
    cout << "Тест создания матрицы пройден\n";

    if (!testMatrixFromVector()) {
        cerr << "Тест создания матрицы из вектора не пройден\n";
        return 1;
    }
    cout << "Тест создания матрицы из вектора пройден\n";

    if (!testMatrixInvalidRows()) {
        cerr << "Тест некорректных строк матрицы не пройден\n";
        return 1;
    }
    cout << "Тест некорректных строк матрицы пройден\n";

    if (!testMatrixSpanAccess()) {
        cerr << "Тест доступа через span не пройден\n";
        return 1;
    }
    cout << "Тест доступа через span пройден\n";

    if (!testMatrixOutOfRange()) {
        cerr << "Тест выхода за границы матрицы не пройден\n";
        return 1;
    }
    cout << "Тест выхода за границы матрицы пройден\n";

        // Тесты метрик расстояния.

    if (!testEuclideanDistance()) {
        cerr << "Тест евклидова расстояния не пройден\n";
        return 1;
    }
    cout << "Тест евклидова расстояния пройден\n";

    if (!testManhattanDistance()) {
        cerr << "Тест манхэттенского расстояния не пройден\n";
        return 1;
    }
    cout << "Тест манхэттенского расстояния пройден\n";

    if (!testChebyshevDistance()) {
        cerr << "Тест расстояния Чебышёва не пройден\n";
        return 1;
    }
    cout << "Тест расстояния Чебышёва пройден\n";

    if (!testDistanceDimensionMismatch()) {
        cerr << "Тест несовпадающих размерностей не пройден\n";
        return 1;
    }
    cout << "Тест несовпадающих размерностей пройден\n";

        // Тесты поиска ближайших соседей.

    if (!testKNearestNeighbors()) {
        cerr << "Тест поиска ближайших соседей не пройден\n";
        return 1;
    }
    cout << "Тест поиска ближайших соседей пройден\n";

    if (!testKNearestNeighborsEqualDistances()) {
        cerr << "Тест равных расстояний не пройден\n";
        return 1;
    }
    cout << "Тест равных расстояний пройден\n";

    if (!testKNearestNeighborsZeroK()) {
        cerr << "Тест нулевого количества соседей не пройден\n";
        return 1;
    }
    cout << "Тест нулевого количества соседей пройден\n";

    if (!testKNearestNeighborsTooLargeK()) {
        cerr << "Тест слишком большого количества соседей не пройден\n";
        return 1;
    }
    cout << "Тест слишком большого количества соседей пройден\n";

    if (!testKNearestNeighborsDimensionMismatch()) {
        cerr << "Тест размерностей при поиске соседей не пройден\n";
        return 1;
    }
    cout << "Тест размерностей при поиске соседей пройден\n";


    // Тесты kNN.

    if (!testKnnRegressorK1()) {
        cerr << "Тест регрессии kNN при k = 1 не пройден\n";
        return 1;
    }
    cout << "Тест регрессии kNN при k = 1 пройден\n";

    if (!testKnnRegressorUniform()) {
        cerr << "Тест регрессии kNN с равными весами не пройден\n";
        return 1;
    }
    cout << "Тест регрессии kNN с равными весами пройден\n";

    if (!testKnnRegressorDistanceWeighting()) {
        cerr << "Тест взвешенной регрессии kNN не пройден\n";
        return 1;
    }
    cout << "Тест взвешенной регрессии kNN пройден\n";

    if (!testKnnClassifier()) {
        cerr << "Тест классификации kNN не пройден\n";
        return 1;
    }
    cout << "Тест классификации kNN пройден\n";

    if (!testKnnZeroDistance()) {
        cerr << "Тест нулевого расстояния kNN не пройден\n";
        return 1;
    }
    cout << "Тест нулевого расстояния kNN пройден\n";

        if (!testDimensionMismatchException()) {
    cerr << "Тест исключения несовпадения размерностей не пройден\n";
    return 1;
}
cout << "Тест исключения несовпадения размерностей пройден\n";

if (!testInvalidKException()) {
    cerr << "Тест исключения некорректного k не пройден\n";
    return 1;
}
cout << "Тест исключения некорректного k пройден\n";

if (!testNullDistanceException()) {
    cerr << "Тест исключения отсутствующей метрики не пройден\n";
    return 1;
}
cout << "Тест исключения отсутствующей метрики пройден\n";

if (!testPredictBeforeFitException()) {
    cerr << "Тест исключения предсказания до обучения не пройден\n";
    return 1;
}
cout << "Тест исключения предсказания до обучения пройден\n";

if (!testEmptyTrainingDataException()) {
    cerr << "Тест исключения пустой обучающей выборки не пройден\n";
    return 1;
}
cout << "Тест исключения пустой обучающей выборки пройден\n";

if (!testTargetSizeMismatchException()) {
    cerr << "Тест исключения несовпадения размеров target не пройден\n";
    return 1;
}
cout << "Тест исключения несовпадения размеров target пройден\n";

if (!testPredictionDimensionMismatchException()) {
    cerr << "Тест исключения размерностей при предсказании не пройден\n";
    return 1;
}
cout << "Тест исключения размерностей при предсказании пройден\n";

// Сравнение поиска через кучу с полной сортировкой.
if (!testKNearestNeighborsAgainstFullSort()) {
    cerr << "Тест сравнения поиска соседей с полной сортировкой не пройден\n";
    return 1;
}
cout << "Тест сравнения поиска соседей с полной сортировкой пройден\n";


    cout << "\nВсе тесты пройдены\n";
    return 0;
}
