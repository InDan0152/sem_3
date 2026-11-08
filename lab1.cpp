
#include <iostream>
#include <filesystem>
#include "csv_loader.hpp"

using namespace std;

int main() {
    system("chcp 65001 > nul");

    const string filename = (filesystem::path(__FILE__).parent_path()
        / "bank-additional/bank-additional/bank-additional.csv").string();

    const CsvConfig config{
        ';',
        "unknown",
        true
    };

    Dataset dataset;

    if (!loadCsv(filename, dataset, config)) {
        return 1;
    }

    // Определяем количество записей.
    size_t recordCount = 0;

    if (!dataset.columns.empty()) {
        const auto& firstColumn = dataset.columns[0];

        if (firstColumn.type == FeatureType::Numerical) {
            recordCount = firstColumn.numericalValues.size();
        } else {
            recordCount = firstColumn.categoricalValues.size();
        }
    }

    cout << "CSV-файл успешно загружен\n";
    cout << "Количество признаков: " << dataset.columns.size() << endl;
    cout << "Количество записей: " << recordCount << endl;

    int z = 1;

    if (z == 1) {
        cout << "Информация о столбцах:" << endl;

        size_t totalMissing = 0;

        for (const auto& column : dataset.columns) {
            size_t missingCount = 0;

            if (column.type == FeatureType::Numerical) {
                for (const auto& value : column.numericalValues) {
                    if (!value.has_value()) {
                        ++missingCount;
                    }
                }
            } else {
                for (const auto& value : column.categoricalValues) {
                    if (!value.has_value()) {
                        ++missingCount;
                    }
                }
            }

            totalMissing += missingCount;

            cout << column.name << " | ";

            if (column.type == FeatureType::Numerical) {
                cout << "числовой";
            } else {
                cout << "категориальный";
            }

            cout << " | пропусков: " << missingCount << endl;
        }

        cout << "Всего пропусков: " << totalMissing << endl;
    }

    return 0;
}
