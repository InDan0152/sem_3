#include <sstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <cxxopts.hpp>
#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>

#include "csv_loader.hpp"
#include "statistics.hpp"
#include "preprocessing.hpp"
#include "exceptions.hpp"
#include "supervised_dataset.hpp"
#include "dataset_splitter.hpp"
#include "matrix.hpp"
#include "distance.hpp"
#include "knn.hpp"

using namespace std;

bool parseFillStrategy(
    const string& value,
    NumericalMissingStrategy& strategy
) {
    if (value == "mean") {
        strategy = NumericalMissingStrategy::Mean;
    }
    else if (value == "median") {
        strategy = NumericalMissingStrategy::Median;
    }
    else if (value == "zero") {
        strategy = NumericalMissingStrategy::Zero;
    }
    else {
        return false;
    }

    return true;
}
bool parseScalingStrategy(
    const string& value,
    ScalingStrategy& strategy
) {
    if (value == "minmax") {
        strategy = ScalingStrategy::MinMax;
    }
    else if (value == "standard") {
        strategy = ScalingStrategy::Standard;
    }
    else if (value == "robust") {
        strategy = ScalingStrategy::Robust;
    }
    else {
        return false;
    }

    return true;
}

bool parseEncodingStrategy(const string& value, EncodingStrategy& strategy) {
    if (value == "onehot") {
        strategy = EncodingStrategy::OneHot;
    }
    else if (value == "ordinal") {
        strategy = EncodingStrategy::Ordinal;
    }
    else {
        return false;
    }

    return true;
}

unique_ptr<IDistance> createDistance(const string& value) {
    if (value == "euclidean") {
        return make_unique<EuclideanDistance>();
    }
    else if (value == "manhattan") {
        return make_unique<ManhattanDistance>();
    }
    else if (value == "chebyshev") {
        return make_unique<ChebyshevDistance>();
    }

    throw invalid_argument(
        "Неизвестная метрика расстояния: " + value
    );
}

// Запуск kNN-классификации.
int runKnnClassification(
    const Dataset& dataset, const string& targetName,
    const PreprocessingConfig& preprocessingConfig,size_t k,
    const string& metricName,const string& weightingName) {
    // Преобразуем категориальный целевой признак в числа.
    const map<string, double> targetMapping = {
        {"no", 0.0},
        {"yes", 1.0}
    };

    // Отделяем целевой признак от остальных признаков.
    SupervisedDataset supervisedDataset =
        prepareSupervisedDataset(
            dataset,
            targetName,
            targetMapping
        );

    // Разделяем данные на обучающую и тестовую выборки.
    DatasetSplit split = splitDataset(
        supervisedDataset,
        0.2,
        42
    );

    cout << "\nРазделение датасета завершено.\n";
    cout << "Обучающих объектов: "
         << split.train.targets.size() << '\n';
    cout << "Тестовых объектов: "
         << split.test.targets.size() << '\n';

    // Обучаем предобработку только на обучающих данных.
    DatasetPreprocessor preprocessor(preprocessingConfig);

    preprocessor.fit(split.train.features);

    // Применяем одинаковые параметры к обеим выборкам.
    ProcessedDataset trainProcessed =
        preprocessor.transform(split.train.features);

    ProcessedDataset testProcessed =
        preprocessor.transform(split.test.features);

    // Преобразуем обработанные данные в матрицы.
    Matrix trainMatrix(trainProcessed.data);
    Matrix testMatrix(testProcessed.data);

    cout << "\nПредобработка завершена.\n";
    cout << "Количество признаков после преобразования: "
         << trainMatrix.columns() << '\n';

    // Создаём метрику расстояния.
unique_ptr<IDistance> distance = createDistance(metricName);

// Выбираем способ взвешивания.
Weighting weighting;

if (weightingName == "uniform") {
    weighting = Weighting::Uniform;
}
else if (weightingName == "distance") {
    weighting = Weighting::Distance;
}
else {
    throw invalid_argument(
        "Неизвестный способ взвешивания: " + weightingName
    );
}

// Создаём классификатор.
KnnClassifier model(
    k,
    move(distance),
    weighting
);

cout << "\nПараметры модели:\n";
cout << "Количество соседей: " << k << '\n';
cout << "Метрика: " << metricName << '\n';
cout << "Взвешивание: " << weightingName << '\n';

    model.fit(
        trainMatrix,
        span<const double>(split.train.targets)
    );

    cout << "\nМодель kNN обучена.\n";

    // Получаем прогнозы для тестовой выборки.
    const vector<double> predictions =
        model.predict(testMatrix);

    // Выводим первые десять прогнозов.
    const size_t outputCount = min(
        size_t{10},
        predictions.size()
    );

    cout << "\nРезультаты классификации:\n";
    cout << "№ | Реальное значение | Прогноз\n";
    cout << "--------------------------------\n";

    for (size_t i = 0; i < outputCount; ++i) {
        cout << i + 1 << " | "
             << (split.test.targets[i] == 1.0 ? "yes" : "no")
             << " | "
             << (predictions[i] == 1.0 ? "yes" : "no")
             << '\n';
    }

    return 0;
}

// Запуск kNN-регрессии.
int runKnnRegression(
    const Dataset& dataset,
    const string& targetName,
    const PreprocessingConfig& preprocessingConfig,
    size_t k,
    const string& metricName,
    const string& weightingName
) {
    // Отделяем числовой целевой признак от остальных признаков.
    SupervisedDataset supervisedDataset =
        prepareSupervisedDataset(dataset, targetName);

    // Разделяем данные на обучающую и тестовую выборки.
    DatasetSplit split = splitDataset(
        supervisedDataset,
        0.2,
        42
    );

    cout << "\nРазделение датасета завершено.\n";
    cout << "Обучающих объектов: "
         << split.train.targets.size() << '\n';
    cout << "Тестовых объектов: "
         << split.test.targets.size() << '\n';

    // Обучаем предобработку только на обучающих данных.
    DatasetPreprocessor preprocessor(preprocessingConfig);
    preprocessor.fit(split.train.features);

    ProcessedDataset trainProcessed =
        preprocessor.transform(split.train.features);

    ProcessedDataset testProcessed =
        preprocessor.transform(split.test.features);

    Matrix trainMatrix(trainProcessed.data);
    Matrix testMatrix(testProcessed.data);

    cout << "\nПредобработка завершена.\n";
    cout << "Количество признаков после преобразования: "
         << trainMatrix.columns() << '\n';

    // Создаём метрику расстояния.
    unique_ptr<IDistance> distance = createDistance(metricName);

    // Выбираем способ взвешивания.
    Weighting weighting;

    if (weightingName == "uniform") {
        weighting = Weighting::Uniform;
    }
    else if (weightingName == "distance") {
        weighting = Weighting::Distance;
    }
    else {
        throw invalid_argument(
            "Неизвестный способ взвешивания: " + weightingName
        );
    }

    // Создаём регрессор.
    KnnRegressor model(
        k,
        move(distance),
        weighting
    );

    cout << "\nПараметры модели:\n";
    cout << "Количество соседей: " << k << '\n';
    cout << "Метрика: " << metricName << '\n';
    cout << "Взвешивание: " << weightingName << '\n';

    model.fit(
        trainMatrix,
        span<const double>(split.train.targets)
    );

    cout << "\nМодель kNN-регрессии обучена.\n";

    // Получаем прогнозы.
    const vector<double> predictions =
        model.predict(testMatrix);
    // Рассчитываем качество регрессии.
    double absoluteErrorSum = 0.0;
    double squaredErrorSum = 0.0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        const double error =
            predictions[i] - split.test.targets[i];

        absoluteErrorSum += abs(error);
        squaredErrorSum += error * error;
    }

    if (predictions.empty()) {
        throw runtime_error(
            "Невозможно рассчитать метрики: тестовая выборка пуста"
        );
    }

    const double mae =
        absoluteErrorSum / predictions.size();

    const double rmse =
        sqrt(squaredErrorSum / predictions.size());

    cout << "\nОценка качества регрессии:\n";
    cout << fixed << setprecision(3);
    cout << "MAE:  " << mae << '\n';
    cout << "RMSE: " << rmse << '\n';

    const size_t outputCount = min(
        size_t{10},
        predictions.size()
    );

    cout << "\nРезультаты регрессии:\n";
    cout << "№ | Реальное значение | Прогноз\n";
    cout << "--------------------------------\n";

    cout << fixed << setprecision(2);

    for (size_t i = 0; i < outputCount; ++i) {
        cout << i + 1 << " | "
             << split.test.targets[i] << " | "
             << predictions[i] << '\n';
    }

    return 0;
}

int main(int argc, char* argv[]) {
    system("chcp 65001 > nul");
     try {

const string defaultInput =
    (filesystem::path(__FILE__).parent_path().parent_path() /
     "bank-additional/bank-additional/bank-additional.csv").string();


const string defaultOutput =
    (filesystem::path(__FILE__).parent_path().parent_path() /
     "bank-additional-processed.csv").string();


cxxopts::Options options("lab1", "Анализ и предобработка датасета");

options.add_options()
    ("h,help", "Показать справку")
    ("i,input", "Путь к входному CSV-файлу",
        cxxopts::value<string>()->default_value(defaultInput))
    ("o,output", "Путь для сохранения обработанного CSV-файла",
        cxxopts::value<string>()->default_value(defaultOutput))
    ("b,bins", "Количество интервалов гистограммы",
        cxxopts::value<int>()->default_value("10"))
    ("f,fill", "Способ заполнения пропусков: mean, median, zero",
        cxxopts::value<string>()->default_value("median"))
    ("s,scale", "Способ масштабирования: minmax, standard, robust",
        cxxopts::value<string>()->default_value("minmax"))
    ("e,encode", "Способ кодирования: onehot, ordinal",
        cxxopts::value<string>()->default_value("onehot"))
    ("knn", "Запустить режим kNN-классификации",
        cxxopts::value<bool>()->default_value("false"))
    ("target", "Название целевого признака",
        cxxopts::value<string>()->default_value("y"))
    ("neighbors", "Количество ближайших соседей",
        cxxopts::value<size_t>()->default_value("5"))
    ("metric", "Метрика расстояния: euclidean, manhattan, chebyshev",
        cxxopts::value<string>()->default_value("euclidean"))
    ("weighting", "Способ взвешивания: uniform, distance",
        cxxopts::value<string>()->default_value("uniform"))
    ("regression", "Запустить режим kNN-регрессии",
        cxxopts::value<bool>()->default_value("false"));

cxxopts::ParseResult arguments;

try {
    arguments = options.parse(argc, argv);
}
catch (const cxxopts::exceptions::exception& error) {
    cerr << "Ошибка параметров: " << error.what() << '\n';
    cerr << "Используйте --help для просмотра справки.\n";
    return 1;
}


if (arguments.count("help")) {
    cout << options.help() << '\n';
    return 0;
}


const string filename = arguments["input"].as<string>();
const filesystem::path outputPath =
    arguments["output"].as<string>();

const string fillValue = arguments["fill"].as<string>();

PreprocessingConfig preprocessingConfig;

if (!parseFillStrategy(fillValue,preprocessingConfig.numericalMissing)) {
    cerr << "Ошибка: неизвестная стратегия заполнения: "<< fillValue << endl;
    cerr << "Допустимые значения: mean, median, zero" << endl;
    return 1;
}


const string scaleValue = arguments["scale"].as<string>();

if (!parseScalingStrategy(
        scaleValue,
        preprocessingConfig.scaling
    )) {
    cerr << "Ошибка: неизвестный способ масштабирования: "
         << scaleValue << '\n';
    cerr << "Допустимые значения: minmax, standard, robust\n";
    return 1;
}


const string encodeValue = arguments["encode"].as<string>();

if (!parseEncodingStrategy(
        encodeValue,
        preprocessingConfig.encoding
    )) {
    cerr << "Ошибка: неизвестный способ кодирования: "
         << encodeValue << '\n';
    cerr << "Допустимые значения: onehot, ordinal\n";
    return 1;
}


int binsValue = arguments["bins"].as<int>();

if (binsValue <= 0) {
    cerr << "Ошибка: количество интервалов должно быть больше нуля.\n";
    return 1;
}

size_t binCount = static_cast<size_t>(binsValue);

    const CsvConfig config{ ';', "unknown", true };

    Dataset dataset;

    if (!loadCsv(filename, dataset, config)) {
        return 1;
    }

    const bool runClassification = arguments["knn"].as<bool>();
const bool runRegression = arguments["regression"].as<bool>();

if (runClassification && runRegression) {
    cerr << "Ошибка: выберите только один режим kNN.\n";
    return 1;
}

if (runClassification || runRegression) {
    const string targetName =
        arguments["target"].as<string>();

    const size_t k =
        arguments["neighbors"].as<size_t>();

    const string metricName =
        arguments["metric"].as<string>();

    const string weightingName =
        arguments["weighting"].as<string>();

    if (runClassification) {
        return runKnnClassification(
            dataset,
            targetName,
            preprocessingConfig,
            k,
            metricName,
            weightingName
        );
    }

    return runKnnRegression(
        dataset,
        targetName,
        preprocessingConfig,
        k,
        metricName,
        weightingName
    );
}

    size_t recordCount = 0;

    if (!dataset.columns.empty()) {
        const auto& firstColumn = dataset.columns[0];

        if (firstColumn.type == FeatureType::Numerical) {
            recordCount = firstColumn.numericalValues.size();
        }
        else {
            recordCount = firstColumn.categoricalValues.size();
        }
    }

    cout << "CSV-файл успешно загружен\n\n";

    cout << "Объектов: " << recordCount << endl;
    cout << "Признаков: " << dataset.columns.size() << "\n\n";

    cout << "# | Признак | Тип | Пропуски" << endl;
    cout << "-----------------------------------------------" << endl;

    size_t totalMissing = 0;
    size_t featuresWithMissing = 0;

    for (size_t i = 0; i < dataset.columns.size(); ++i) {
        const auto& column = dataset.columns[i];

        size_t missingCount = 0;

        if (column.type == FeatureType::Numerical) {
            for (const auto& value : column.numericalValues) {
                if (!value.has_value()) {
                    ++missingCount;
                }
            }
        }
        else {
            for (const auto& value : column.categoricalValues) {
                if (!value.has_value()) {
                    ++missingCount;
                }
            }
        }

        totalMissing += missingCount;

        if (missingCount > 0) {
            ++featuresWithMissing;
        }

        double missingPercentage = 0.0;

        if (recordCount > 0) {
            missingPercentage =
                static_cast<double>(missingCount) / recordCount * 100.0;
        }

        cout << i << " | " << column.name << " | "
             << (column.type == FeatureType::Numerical
                 ? "числовой" : "категориальный")
             << " | " << missingCount;

        if (missingCount > 0) {
            cout << " (" << fixed << setprecision(2)
                 << missingPercentage << "%)";
        }

        cout << '\n';
    }

    cout << "\nВсего пропусков: " << totalMissing
         << " в " << featuresWithMissing << " признаках" << endl;


 //цикл выбора признака
    while (true) {
        cout << "\nВыберите признак для анализа (0-"
             << dataset.columns.size() - 1 << ").\n";
        cout << "Введите -1 для выхода.\n\n";

        cout << "Ваш выбор: ";

        string input;
        getline(cin, input);

        stringstream inputStream(input);
        int choice;
        char extra;

        if (!(inputStream >> choice) || (inputStream >> extra)) {
            cout << "Ошибка: введите целое число.\n";
            continue;
        }


        if (input == "-1") {
            string answer;

            cout << "\nСохранить преобразованный датасет? (y/n): ";
            getline(cin, answer);

            if (answer == "y" || answer == "Y" ||
                answer == "д" || answer == "Д") {

                ProcessedDataset processedDataset;

                if (preprocessDataset(dataset,processedDataset,preprocessingConfig)) {
                    if (saveProcessedDataset(processedDataset,outputPath.string())) {
                        cout << "Датасет успешно сохранён: "
                            << outputPath << '\n';

                        cout << "Количество объектов: "
                            << processedDataset.data.size() << '\n';

                        cout << "Количество признаков после обработки: "
                            << processedDataset.featureNames.size() << '\n';

        filesystem::path binaryPath = outputPath;
        binaryPath.replace_extension(".bin");

        if (saveProcessedDatasetBinary(
            processedDataset,
            binaryPath.string()
        )) {
            cout << "\nБинарный файл сохранён: "
                << binaryPath << '\n';

            ProcessedDataset loadedDataset;

            if (loadProcessedDatasetBinary(
                binaryPath.string(),
                loadedDataset
            )) {
                if (compareProcessedDatasets(
                    processedDataset,
                    loadedDataset
                )) {
                    cout << "Проверка бинарного файла пройдена успешно" ;
                }
                else {
                    cout << "Ошибка: данные после загрузки отличаются" << endl;
                }
            }
        }

        cout << "\nРазмеры файлов:" << endl;

        cout << "CSV: "
            << filesystem::file_size(outputPath)
            << " байт" << endl;

        cout << "Binary: "
            << filesystem::file_size(binaryPath)
            << " байт" << endl;
                    }
                }
            }

            break;
        }

        if (choice < 0 ||
            choice >= static_cast<int>(dataset.columns.size())) {
            cout << "Ошибка: такого признака нет." << endl;
            continue;
        }

        const auto& column = dataset.columns[choice];

//анализ числового признака
        if (column.type == FeatureType::Numerical) {
            NumericalStatistics stats;

            calculateNumericalStatistics(column, stats, binCount);

            cout << "\nСтатистика признака " << choice
                 << ": " << column.name << " (числовой)\n\n";

            cout << "Пропусков: " << stats.missingCount << endl;

            if (stats.hasValues) {
                cout << fixed << setprecision(3);

                cout << "Min: " << stats.min
                     << "    Max: " << stats.max << endl;

                cout << "Mean: " << stats.mean
                     << "    Var: " << stats.variance << endl;

                cout << "\n";

                cout << "q05: " << stats.q05
                     << "\nq25: " << stats.q25
                     << "\nq50: " << stats.q50
                     << "\nq75: " << stats.q75
                     << "\nq95: " << stats.q95 << endl;

//числовая гистограмма
                cout << "\nГистограмма:\n";

                size_t maxFrequency = 0;

                for (const auto& bin : stats.histogram) {
                    if (bin.frequency > maxFrequency) {
                        maxFrequency = bin.frequency;
                    }
                }

                for (size_t i = 0; i < stats.histogram.size(); ++i) {
                    const auto& bin = stats.histogram[i];

                    cout << "[" << bin.lowerBound << ", "
                         << bin.upperBound
                         << (i + 1 == stats.histogram.size() ? "] " : ") ");

                    size_t barLength = 0;

                    if (maxFrequency > 0) {
                        barLength = static_cast<size_t>(
                            30.0 * bin.frequency / maxFrequency
                        );
                    }

                    for (size_t j = 0; j < barLength; ++j) {
                        cout << '#';
                    }

                    cout << " " << bin.frequency << endl;
                }
            }
            else {
                cout << "Нет данных для расчёта статистики.\n";
            }
        }
//анализ категориального признака
        else {
            CategoricalStatistics categoryStats;

            calculateCategoricalStatistics(column, categoryStats);

            cout << "\nСтатистика признака " << choice
                 << ": " << column.name << " (категориальный)\n\n";

            cout << "Пропусков: "
                 << categoryStats.missingCount << '\n';

            cout << "Количество категорий: "
                 << categoryStats.categories.size() << "\n\n";

            cout << "Категории:"<< endl;

            for (const auto& category : categoryStats.categories) {
                cout << category.category
                     << " | " << category.frequency << endl;
            }
//категориальная гистограмма
            cout << "\nГистограмма:"<< endl;

            size_t maxFrequency = 0;

            for (const auto& category : categoryStats.categories) {
                if (category.frequency > maxFrequency) {
                    maxFrequency = category.frequency;
                }
            }

            for (const auto& category : categoryStats.categories) {
                cout << left << setw(25) << category.category;

                size_t barLength = 0;

                if (maxFrequency > 0) {
                    barLength = static_cast<size_t>(
                        30.0 * category.frequency / maxFrequency
                    );
                }

                for (size_t i = 0; i < barLength; ++i) {
                    cout << '#';
                }

                cout << " " << category.frequency << endl;
            }
        }
    }

        return 0;
    }
    catch (const std::exception& error) {
        cerr << "\nОшибка выполнения программы: "
             << error.what() << '\n';

        return 1;
    }
    catch (...) {
        cerr << "\nПроизошла неизвестная ошибка.\n";
        return 1;
    }
}