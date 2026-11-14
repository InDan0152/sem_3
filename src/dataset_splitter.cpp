
#include "dataset_splitter.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace std;

// Добавляет выбранный объект в соответствующую выборку.
static void addRow(
    const SupervisedDataset& source,
    SupervisedDataset& destination,
    size_t rowIndex
) {
    // Копируем целевое значение.
    destination.targets.push_back(source.targets[rowIndex]);

    // Копируем значения каждого признака.
    for (size_t i = 0; i < source.features.columns.size(); ++i) {
        const Column& sourceColumn = source.features.columns[i];

        if (sourceColumn.type == FeatureType::Numerical) {
            destination.features.columns[i].numericalValues.push_back(
                sourceColumn.numericalValues[rowIndex]
            );
        } else {
            destination.features.columns[i].categoricalValues.push_back(
                sourceColumn.categoricalValues[rowIndex]
            );
        }
    }
}

DatasetSplit splitDataset(
    const SupervisedDataset& dataset,
    double testRatio,
    uint32_t seed
) {
    // Проверяем корректность доли тестовой выборки.
    if (testRatio <= 0.0 || testRatio >= 1.0) {
        throw invalid_argument(
            "Доля тестовой выборки должна быть больше 0 и меньше 1"
        );
    }

    // Проверяем наличие признаков.
    if (dataset.features.columns.empty()) {
        throw invalid_argument(
            "Набор данных не содержит признаков"
        );
    }

    // Проверяем наличие целевых значений.
    if (dataset.targets.empty()) {
        throw invalid_argument(
            "Набор данных не содержит целевых значений"
        );
    }

    const size_t rowCount = dataset.targets.size();

    // Для разделения необходимы хотя бы два объекта.
    if (rowCount < 2) {
        throw invalid_argument(
            "Для разделения необходимо минимум два объекта"
        );
    }

    // Проверяем соответствие размеров всех столбцов.
    for (const Column& column : dataset.features.columns) {
        size_t columnSize =
            column.type == FeatureType::Numerical
                ? column.numericalValues.size()
                : column.categoricalValues.size();

        if (columnSize != rowCount) {
            throw invalid_argument(
                "Количество значений признака не совпадает "
                "с количеством целевых значений: " + column.name
            );
        }
    }

    // Определяем количество объектов для тестовой выборки.
    size_t testSize = static_cast<size_t>(
        round(static_cast<double>(rowCount) * testRatio)
    );

    // В обеих выборках должен остаться хотя бы один объект.
    testSize = max<size_t>(1, testSize);
    testSize = min(testSize, rowCount - 1);

    // Создаём индексы всех объектов.
    vector<size_t> indices(rowCount);
    iota(indices.begin(), indices.end(), 0);

    // Перемешиваем индексы с помощью генератора случайных чисел.
    mt19937 generator(seed);
    shuffle(indices.begin(), indices.end(), generator);

    DatasetSplit result;

    // Копируем структуру столбцов без значений.
    for (const Column& column : dataset.features.columns) {
        Column trainColumn;
        trainColumn.name = column.name;
        trainColumn.type = column.type;

        Column testColumn;
        testColumn.name = column.name;
        testColumn.type = column.type;

        result.train.features.columns.push_back(trainColumn);
        result.test.features.columns.push_back(testColumn);
    }

    // Распределяем объекты по обучающей и тестовой выборкам.
    for (size_t i = 0; i < rowCount; ++i) {
        size_t rowIndex = indices[i];

        if (i < testSize) {
            addRow(dataset, result.test, rowIndex);
        } else {
            addRow(dataset, result.train, rowIndex);
        }
    }

    return result;
}
