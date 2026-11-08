
#include <iostream>
#include <filesystem>
#include <iomanip>
#include "csv_loader.hpp"

using namespace std;

int main() {
    system("chcp 65001 > nul");

    const string filename = (filesystem::path(__FILE__).parent_path()
        / "bank-additional/bank-additional/bank-additional.csv").string();

    const CsvConfig config{ ';', "unknown", true };

    Dataset dataset;

    if (!loadCsv(filename, dataset, config)) {
        return 1;
    }

    // Определяем количество объектов
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

            cout << i << " | "<< column.name << " | "<< (column.type == FeatureType::Numerical
         ? "числовой" : "категориальный") << " | " << missingCount;

    if (missingCount > 0) {cout << " (" << fixed << setprecision(2) << missingPercentage << "%)";}

cout << '\n';

    }

    cout << "\nВсего пропусков: " << totalMissing
         << " в " << featuresWithMissing << " признаках" << endl;

    return 0;
}
