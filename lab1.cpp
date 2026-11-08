#include <sstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <cxxopts.hpp>
#include "csv_loader.hpp"
#include "statistics.hpp"
#include "preprocessing.hpp"

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

int main(int argc, char* argv[]) {
    system("chcp 65001 > nul");


const string defaultInput =
    (filesystem::path(__FILE__).parent_path() /
     "bank-additional/bank-additional/bank-additional.csv").string();


const string defaultOutput =
    (filesystem::path(__FILE__).parent_path() /
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
        cxxopts::value<string>()->default_value("onehot"));

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
