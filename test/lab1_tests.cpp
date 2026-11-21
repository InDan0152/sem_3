#include "csv_loader.hpp"
#include "statistics.hpp"
#include "preprocessing.hpp"
#include "supervised_dataset.hpp"
#include "dataset_splitter.hpp"
#include "matrix.hpp"
#include "exceptions.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <algorithm>

using namespace std;

// Вспомогательные функции тестов: ожидаемые исключения проверяются отдельно.
bool tryLoadCsv(const string& filename, Dataset& dataset, const CsvConfig& config) {
    try {
        loadCsv(filename, dataset, config);
        return true;
    } catch (const exception&) {
        return false;
    }
}

bool tryPreprocess(
    const Dataset& dataset,
    ProcessedDataset& processedDataset,
    const PreprocessingConfig& config = {}
) {
    try {
        preprocessDataset(dataset, processedDataset, config);
        return true;
    } catch (const exception&) {
        return false;
    }
}


// Тест 1. Разбор CSV
bool testCsvParsing() {
    const vector<string> result = parseCsvLine("25;unknown;admin.", ';');

    bool malformedThrows = false;
    try {
        (void)parseCsvLine("25;\"admin.", ';');
    } catch (const invalid_argument& error) {
        malformedThrows =
            string(error.what()).find("незакрытую кавычку") != string::npos;
    }

    return result.size() == 3 &&
           result[0] == "25" &&
           result[1] == "unknown" &&
           result[2] == "admin." &&
           malformedThrows;
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

    if (!tryLoadCsv(filename, dataset, config)) return false;

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
    if (!tryPreprocess(dataset, processedDataset)) return false;

    if (processedDataset.data.size() != 3 ||
        processedDataset.featureNames.size() != 3) {
        return false;
    }

    const vector<string> expectedNames = {
        "color___MISSING__",
        "color_blue",
        "color_red"
    };
    if (processedDataset.featureNames != expectedNames) return false;

    const vector<vector<double>> expectedData = {
        {0.0, 0.0, 1.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0}
    };
    return processedDataset.data == expectedData;
}
bool testBinarySaveLoad() {
    ProcessedDataset original;
    original.featureNames = {"age", "job_admin.", "job_services"};
    original.data = {
        {0.0, 1.0, 0.0},
        {0.5, 0.0, 1.0},
        {1.0, 1.0, 0.0}
    };

    const string filename = "test_binary.bin";
    ProcessedDataset loaded;

    try {
        saveProcessedDatasetBinary(original, filename);
        loadProcessedDatasetBinary(filename, loaded);
    } catch (const exception&) {
        return false;
    }

    return compareProcessedDatasets(original, loaded);
}

bool testInvalidCsv() {
    const string filename = "test_invalid.csv";
    {
        ofstream file(filename);
        if (!file.is_open()) return false;
        file << "age;job;marital\n";
        file << "25;admin.\n";
    }

    Dataset dataset;
    CsvConfig config;
    ostringstream capturedOutput;
    streambuf* previousErrorBuffer = cerr.rdbuf(capturedOutput.rdbuf());

    bool invalidFieldCountReported = false;
    try {
        loadCsv(filename, dataset, config);
    } catch (const invalid_argument& error) {
        invalidFieldCountReported =
            string(error.what()).find("строке 2") != string::npos;
    }

    {
        ofstream malformedQuoteFile(filename, ios::trunc);
        malformedQuoteFile << "age;job\n";
        malformedQuoteFile << "25;\"admin.\n";
        malformedQuoteFile.close();
    }

    bool malformedQuoteReported = false;
    try {
        loadCsv(filename, dataset, config);
    } catch (const invalid_argument& error) {
        const string message = error.what();
        malformedQuoteReported =
            message.find("строке 2") != string::npos &&
            message.find("незакрытую кавычку") != string::npos;
    }

    cerr.rdbuf(previousErrorBuffer);

    return invalidFieldCountReported &&
           malformedQuoteReported &&
           capturedOutput.str().empty();
}

bool testPreprocessingErrorReporting() {
    Dataset dataset;
    Column emptyNumericalColumn;
    emptyNumericalColumn.name = "empty_value";
    emptyNumericalColumn.type = FeatureType::Numerical;
    emptyNumericalColumn.numericalValues = {nullopt};
    dataset.columns.push_back(emptyNumericalColumn);

    ProcessedDataset processedDataset;
    PreprocessingConfig config;
    ostringstream capturedOutput;
    streambuf* previousErrorBuffer = cerr.rdbuf(capturedOutput.rdbuf());

    bool exceptionReported = false;
    try {
        preprocessDataset(dataset, processedDataset, config);
    } catch (const invalid_argument& error) {
        exceptionReported =
            string(error.what()).find("empty_value") != string::npos;
    }

    cerr.rdbuf(previousErrorBuffer);
    return exceptionReported && capturedOutput.str().empty();
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

        if (!tryPreprocess(dataset, result, config))
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

    if (!tryPreprocess(dataset, result, config))
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

    if (!tryPreprocess(dataset, result, config))
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
    span<double> checkedRow = matrix.at(1);
    checkedRow[0] = 40.0;

    // Проверяем, что изменилась исходная матрица.
    return matrix[0][0] == 10.0 &&
           matrix[0][1] == 50.0 &&
           matrix[1][0] == 40.0 &&
           matrix[1][2] == 30.0 &&
           firstRow.size() == 3;
}

bool testMatrixConstSpanAccess() {
    Matrix matrix({
        {1.0, 2.0},
        {3.0, 4.0}
    });
    const Matrix& constMatrix = matrix;

    span<const double> row = constMatrix[1];
    span<const double> checkedRow = constMatrix.at(0);
    return row.size() == 2 && row[0] == 3.0 && row[1] == 4.0 &&
           checkedRow.size() == 2 && checkedRow[0] == 1.0;
}

bool testMatrixZeroColumns() {
    Matrix matrix(2, 0);
    const Matrix& constMatrix = matrix;

    span<double> mutableRow = matrix[0];
    span<const double> constRow = constMatrix[1];
    span<double> checkedMutableRow = matrix.at(1);
    span<const double> checkedConstRow = constMatrix.at(0);

    bool mutableAtThrows = false;
    bool constAtThrows = false;
    try {
        (void)matrix.at(2);
    }
    catch (const out_of_range&) {
        mutableAtThrows = true;
    }
    try {
        (void)constMatrix.at(2);
    }
    catch (const out_of_range&) {
        constAtThrows = true;
    }

    return mutableRow.empty() && constRow.empty() &&
           checkedMutableRow.empty() && checkedConstRow.empty() &&
           mutableAtThrows && constAtThrows;
}

bool testMatrixColumnMajorSpanErrors() {
    Matrix matrix(
        {{1.0, 2.0}, {3.0, 4.0}},
        MatrixLayout::ColumnMajor
    );
    const Matrix& constMatrix = matrix;

    bool mutableOperatorThrows = false;
    bool constOperatorThrows = false;
    bool mutableAtThrows = false;
    bool constAtThrows = false;

    try {
        (void)matrix[0];
    }
    catch (const NonContiguousSliceError&) {
        mutableOperatorThrows = true;
    }

    try {
        (void)constMatrix[0];
    }
    catch (const NonContiguousSliceError&) {
        constOperatorThrows = true;
    }

    try {
        (void)matrix.at(0);
    }
    catch (const NonContiguousSliceError&) {
        mutableAtThrows = true;
    }

    try {
        (void)constMatrix.at(0);
    }
    catch (const NonContiguousSliceError&) {
        constAtThrows = true;
    }

    return mutableOperatorThrows && constOperatorThrows &&
           mutableAtThrows && constAtThrows;
}

// at(row) проверяет границы раньше, чем проверяет непрерывность строки.
bool testMatrixOutOfRange() {
    Matrix matrix(2, 3);
    const Matrix& constMatrix = matrix;

    bool mutableThrows = false;
    bool constThrows = false;

    try {
        matrix.at(2);
    }
    catch (const out_of_range&) {
        mutableThrows = true;
    }

    try {
        constMatrix.at(2);
    }
    catch (const out_of_range&) {
        constThrows = true;
    }

    Matrix columnMajor(2, 3, MatrixLayout::ColumnMajor);
    const Matrix& constColumnMajor = columnMajor;
    bool columnMajorThrowsForInvalidRow = false;
    bool constColumnMajorThrowsForInvalidRow = false;
    try {
        columnMajor.at(2);
    }
    catch (const out_of_range&) {
        columnMajorThrowsForInvalidRow = true;
    }
    try {
        constColumnMajor.at(2);
    }
    catch (const out_of_range&) {
        constColumnMajorThrowsForInvalidRow = true;
    }

    return mutableThrows && constThrows && columnMajorThrowsForInvalidRow &&
           constColumnMajorThrowsForInvalidRow;
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

    if (!testPreprocessingErrorReporting()) {
        cerr << "Тест передачи ошибок предобработки не пройден\n";
        return 1;
    }
    cout << "Тест передачи ошибок предобработки пройден\n";

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

    if (!testMatrixConstSpanAccess()) {
        cerr << "Тест константного доступа через span не пройден\n";
        return 1;
    }
    cout << "Тест константного доступа через span пройден\n";

    if (!testMatrixZeroColumns()) {
        cerr << "Тест матрицы с нулём столбцов не пройден\n";
        return 1;
    }
    cout << "Тест матрицы с нулём столбцов пройден\n";

    if (!testMatrixColumnMajorSpanErrors()) {
        cerr << "Тест span для ColumnMajor не пройден\n";
        return 1;
    }
    cout << "Тест span для ColumnMajor пройден\n";

    if (!testMatrixOutOfRange()) {
        cerr << "Тест выхода за границы матрицы не пройден\n";
        return 1;
    }
    cout << "Тест выхода за границы матрицы пройден\n";

    cout << "\nВсе тесты пройдены\n";
    return 0;
}
