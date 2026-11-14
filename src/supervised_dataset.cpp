
#include "supervised_dataset.hpp"

#include <stdexcept>
#include <unordered_set>

using namespace std;

SupervisedDataset prepareSupervisedDataset(
    const Dataset& dataset,
    const string& targetName,
    const map<string, double>& categoricalTargetMapping
) {
    // Проверяем, что исходный набор данных не пустой.
    if (dataset.columns.empty()) {
        throw invalid_argument("Набор данных пуст");
    }

    if (targetName.empty()) {
        throw invalid_argument(
            "Имя целевого столбца не может быть пустым"
        );
    }

    // Определяем количество объектов по первому столбцу.
    const Column& firstColumn = dataset.columns.front();

    const size_t rowCount =
        firstColumn.type == FeatureType::Numerical
            ? firstColumn.numericalValues.size()
            : firstColumn.categoricalValues.size();

    if (rowCount == 0) {
        throw invalid_argument(
            "Набор данных не содержит объектов"
        );
    }

    // Проверяем уникальность имён и размеры столбцов.
    unordered_set<string> columnNames;
    const Column* targetColumn = nullptr;

    for (const Column& column : dataset.columns) {
        if (!columnNames.insert(column.name).second) {
            throw invalid_argument(
                "Обнаружено повторяющееся имя столбца: " + column.name
            );
        }

        if (column.type == FeatureType::Numerical) {
            if (column.numericalValues.size() != rowCount) {
                throw invalid_argument(
                    "Некорректное количество значений в столбце: "
                    + column.name
                );
            }
        } else {
            if (column.categoricalValues.size() != rowCount) {
                throw invalid_argument(
                    "Некорректное количество значений в столбце: "
                    + column.name
                );
            }
        }

        if (column.name == targetName) {
            targetColumn = &column;
        }
    }

    // Проверяем, найден ли целевой столбец.
    if (targetColumn == nullptr) {
        throw invalid_argument(
            "Целевой столбец не найден: " + targetName
        );
    }

    SupervisedDataset result;

    // Переносим целевой столбец в вектор targets.
    if (targetColumn->type == FeatureType::Numerical) {
        for (const auto& value : targetColumn->numericalValues) {
            if (!value.has_value()) {
                throw invalid_argument(
                    "Целевой столбец содержит пропущенные значения"
                );
            }

            result.targets.push_back(*value);
        }
    } else {
        // Для категориального целевого столбца требуется явное
        // соответствие категорий числовым значениям.
        if (categoricalTargetMapping.empty()) {
            throw invalid_argument(
                "Для категориального целевого столбца "
                "необходимо задать числовое соответствие категорий"
            );
        }

        for (const auto& value : targetColumn->categoricalValues) {
            if (!value.has_value()) {
                throw invalid_argument(
                    "Целевой столбец содержит пропущенные значения"
                );
            }

            auto it = categoricalTargetMapping.find(*value);

            if (it == categoricalTargetMapping.end()) {
                throw invalid_argument(
                    "Неизвестная категория целевого столбца: " + *value
                );
            }

            result.targets.push_back(it->second);
        }
    }

    // Все столбцы, кроме целевого, становятся признаками.
    for (const Column& column : dataset.columns) {
        if (column.name != targetName) {
            result.features.columns.push_back(column);
        }
    }

    // Проверяем, что после выделения целевого столбца
    // остался хотя бы один признак.
    if (result.features.columns.empty()) {
        throw invalid_argument(
            "В наборе данных должен быть хотя бы один признак"
        );
    }

    return result;
}
