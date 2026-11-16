
#include "dataset_splitter.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace std;


static void addRow(const SupervisedDataset& source,SupervisedDataset& destination,size_t rowIndex) {

    destination.targets.push_back(source.targets[rowIndex]);

//копи значения каждого признака
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
    if (testRatio <= 0.0 || testRatio >= 1.0) {
        throw invalid_argument(
            "Доля тестовой выборки должна быть больше 0 и меньше 1"
        );
    }

    if (dataset.features.columns.empty()) {
        throw invalid_argument(
            "Набор данных не содержит признаков"
        );
    }

    if (dataset.targets.empty()) {
        throw invalid_argument(
            "Набор данных не содержит целевых значений"
        );
    }

    const size_t rowCount = dataset.targets.size();

    if (rowCount < 2) {
        throw invalid_argument(
            "Для разделения необходимо минимум два объекта"
        );
    }

//соосветствие размеров столбцов 
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

    size_t testSize = static_cast<size_t>(round(static_cast<double>(rowCount) * testRatio));

    testSize = max<size_t>(1, testSize);
    testSize = min(testSize, rowCount - 1);

    vector<size_t> indices(rowCount);
    iota(indices.begin(), indices.end(), 0);

    mt19937 generator(seed);
    shuffle(indices.begin(), indices.end(), generator);

    DatasetSplit result;

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

//распределяем объекты по обучающей и тестовой выборкам
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
