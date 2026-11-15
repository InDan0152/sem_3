
#include "nadaraya_watson.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

using namespace std;

// Преобразование строки матрицы в непрерывный вектор.
// Поддерживает RowMajor и ColumnMajor.
static vector<double> getMatrixRow(
    const Matrix& matrix,
    size_t rowIndex
) {
    auto row = matrix(rowIndex, nullopt);

    vector<double> result(row.size());

    for (size_t i = 0; i < row.size(); ++i) {
        result[i] = row(i);
    }

    return result;
}

// Конструктор общей части NW.
NadarayaWatsonSearch::NadarayaWatsonSearch(
    double bandwidth,
    unique_ptr<IDistance> distance
)
    : bandwidth(bandwidth),
      distance(move(distance))
{
    if (!isfinite(this->bandwidth) || this->bandwidth <= 0.0) {
        throw InvalidParameterError(
            "Ширина окна h должна быть конечной и положительной"
        );
    }

    if (!this->distance) {
        throw InvalidParameterError(
            "Метрика расстояния не задана"
        );
    }
}

// Обучение общей части NW.
void NadarayaWatsonSearch::fit(
    const Matrix& features,
    span<const double> target
) {
    if (features.rows() == 0) {
        throw InvalidParameterError(
            "Нельзя обучить модель на пустой выборке"
        );
    }

    if (features.columns() == 0) {
        throw InvalidParameterError(
            "Обучающая выборка не содержит признаков"
        );
    }

    if (features.rows() != target.size()) {
        throw DimensionMismatchError(
            "Количество объектов и целевых значений не совпадает"
        );
    }

    // Проверяем корректность входных данных.
    for (size_t i = 0; i < features.rows(); ++i) {
        for (size_t j = 0; j < features.columns(); ++j) {
            if (!isfinite(features(i, j))) {
                throw InvalidParameterError(
                    "Матрица признаков содержит неконечное значение"
                );
            }
        }

        if (!isfinite(target[i])) {
            throw InvalidParameterError(
                "Целевые значения содержат неконечное значение"
            );
        }
    }

    Matrix newFeatures = features;
    vector<double> newTargets(target.begin(), target.end());

    trainFeatures = move(newFeatures);
    trainTargets = move(newTargets);

    fitted = true;
}

// Вычисление нормированных гауссовых весов.
vector<double> NadarayaWatsonSearch::getWeights(
    span<const double> query
) const {
    if (!fitted) {
        throw ModelUsageError(
            "Модель ещё не обучена"
        );
    }

    if (query.size() != trainFeatures.columns()) {
        throw DimensionMismatchError(
            "Количество признаков объекта не совпадает с обучающей выборкой"
        );
    }

    for (double value : query) {
        if (!isfinite(value)) {
            throw InvalidParameterError(
                "Вектор признаков содержит неконечное значение"
            );
        }
    }

    vector<double> distances(trainFeatures.rows());

    double minDistance = numeric_limits<double>::infinity();

    for (size_t i = 0; i < trainFeatures.rows(); ++i) {
        vector<double> trainRow = getMatrixRow(trainFeatures, i);

        double currentDistance = distance->compute(
            query,
            trainRow
        );

        if (!isfinite(currentDistance) || currentDistance < 0.0) {
            throw InvalidParameterError(
                "Метрика вернула некорректное расстояние"
            );
        }

        distances[i] = currentDistance;
        minDistance = min(minDistance, currentDistance);
    }

    vector<double> weights(distances.size());

    double weightSum = 0.0;

    for (size_t i = 0; i < distances.size(); ++i) {
        double currentDistance = distances[i];

        if (currentDistance == minDistance) {
            weights[i] = 1.0;
        } else {
            // Разность квадратов расстояний:
            // (d^2 - d_min^2) / (2 * h^2).
            // Такая форма позволяет избежать переполнения
            // при непосредственном возведении расстояний в квадрат.
            double scaledDifference =
                (currentDistance - minDistance) / bandwidth;

            double scaledSum =
                (currentDistance / bandwidth +
                 minDistance / bandwidth) / 2.0;

            double exponent = scaledDifference * scaledSum;

            weights[i] = exp(-exponent);
        }

        weightSum += weights[i];
    }

    if (!isfinite(weightSum) || weightSum <= 0.0) {
        throw ModelUsageError(
            "Не удалось вычислить нормированные веса"
        );
    }

    for (double& weight : weights) {
        weight /= weightSum;
    }

    return weights;
}

// Получение целевого значения обучающего объекта.
double NadarayaWatsonSearch::getTarget(size_t index) const {
    if (!fitted) {
        throw ModelUsageError(
            "Модель ещё не обучена"
        );
    }

    return trainTargets.at(index);
}

// Получение количества признаков.
size_t NadarayaWatsonSearch::featureCount() const {
    if (!fitted) {
        throw ModelUsageError(
            "Модель ещё не обучена"
        );
    }

    return trainFeatures.columns();
}

// Проверка состояния обучения.
bool NadarayaWatsonSearch::isFitted() const noexcept {
    return fitted;
}

// Конструктор регрессора.
NadarayaWatsonRegressor::NadarayaWatsonRegressor(
    double bandwidth,
    unique_ptr<IDistance> distance
)
    : search(bandwidth, move(distance))
{
}

// Обучение регрессора.
void NadarayaWatsonRegressor::fit(
    const Matrix& features,
    span<const double> target
) {
    search.fit(features, target);
}

// Предсказание регрессора.
vector<double> NadarayaWatsonRegressor::predict(
    const Matrix& features
) const {
    if (!search.isFitted()) {
        throw ModelUsageError(
            "Регрессор ещё не обучен"
        );
    }

    if (features.columns() != search.featureCount()) {
        throw DimensionMismatchError(
            "Количество признаков тестовой выборки не совпадает с обучающей"
        );
    }

    vector<double> predictions;
    predictions.reserve(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        vector<double> query = getMatrixRow(features, i);

        vector<double> weights = search.getWeights(query);

        double prediction = 0.0;

        for (size_t j = 0; j < weights.size(); ++j) {
            prediction += weights[j] * search.getTarget(j);
        }

        predictions.push_back(prediction);
    }

    return predictions;
}

// Конструктор классификатора.
NadarayaWatsonClassifier::NadarayaWatsonClassifier(
    double bandwidth,
    unique_ptr<IDistance> distance
)
    : search(bandwidth, move(distance))
{
}

// Обучение классификатора.
void NadarayaWatsonClassifier::fit(
    const Matrix& features,
    span<const double> target
) {
    if (target.empty()) {
        throw InvalidParameterError(
            "Нельзя обучить классификатор на пустой выборке"
        );
    }

    vector<double> newClasses(target.begin(), target.end());

    for (double value : newClasses) {
        if (!isfinite(value)) {
            throw InvalidParameterError(
                "Метки классов должны быть конечными числами"
            );
        }
    }

    sort(newClasses.begin(), newClasses.end());

    newClasses.erase(
        unique(newClasses.begin(), newClasses.end()),
        newClasses.end()
    );

    search.fit(features, target);

    classLabels = move(newClasses);
}

// Получение вероятностей принадлежности объектов классам.
Matrix NadarayaWatsonClassifier::predict_proba(
    const Matrix& features
) const {
    if (!search.isFitted()) {
        throw ModelUsageError(
            "Классификатор ещё не обучен"
        );
    }

    if (features.columns() != search.featureCount()) {
        throw DimensionMismatchError(
            "Количество признаков тестовой выборки не совпадает с обучающей"
        );
    }

    Matrix probabilities(features.rows(), classLabels.size());

    for (size_t i = 0; i < features.rows(); ++i) {
        vector<double> query = getMatrixRow(features, i);

        vector<double> weights = search.getWeights(query);

        for (size_t j = 0; j < weights.size(); ++j) {
            double target = search.getTarget(j);

            auto it = lower_bound(
                classLabels.begin(),
                classLabels.end(),
                target
            );

            size_t classIndex =
                static_cast<size_t>(it - classLabels.begin());

            probabilities(i, classIndex) += weights[j];
        }
    }

    return probabilities;
}

// Предсказание класса.
vector<double> NadarayaWatsonClassifier::predict(
    const Matrix& features
) const {
    Matrix probabilities = predict_proba(features);

    vector<double> predictions;
    predictions.reserve(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        size_t bestClass = 0;

        for (size_t j = 1; j < classLabels.size(); ++j) {
            if (probabilities(i, j) > probabilities(i, bestClass)) {
                bestClass = j;
            }
        }

        predictions.push_back(classLabels[bestClass]);
    }

    return predictions;
}

// Получение списка классов.
span<const double> NadarayaWatsonClassifier::classes() const {
    if (!search.isFitted()) {
        throw ModelUsageError(
            "Классификатор ещё не обучен"
        );
    }

    return classLabels;
}
