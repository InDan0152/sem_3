
#include "knn.hpp"
#include "exceptions.hpp"

#include <map>
#include <utility>

using namespace std;

// Конструктор общей части kNN.
KnnSearch::KnnSearch(
    size_t neighborsCount,
    unique_ptr<IDistance> distanceMetric,
    Weighting weightingMethod
)
    : k(neighborsCount),
      distance(move(distanceMetric)),
      weighting(weightingMethod)
{
    if (k == 0) {
        throw InvalidParameterError(
            "Количество соседей k должно быть больше нуля"
        );
    }

    if (!distance) {
        throw InvalidParameterError(
            "Метрика расстояния не задана"
        );
    }
}

// Обучение общей части kNN.
void KnnSearch::fit(
    const Matrix& features,
    span<const double> target
)
{
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

    if (k > features.rows()) {
        throw InvalidParameterError(
            "Количество соседей k превышает размер обучающей выборки"
        );
    }

    // Сначала подготавливаем новые данные.
    // Если копирование завершится ошибкой, старое состояние сохранится.
    Matrix newFeatures = features;
    vector<double> newTargets(target.begin(), target.end());

    trainFeatures = move(newFeatures);
    trainTargets = move(newTargets);
    fitted = true;
}

// Поиск соседей для одного объекта.
vector<Neighbor> KnnSearch::findNeighbors(
    span<const double> query
) const
{
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

    return findKNearestNeighbors(
        trainFeatures,
        query,
        k,
        *distance
    );
}

// Получение весов найденных соседей.
vector<double> KnnSearch::getWeights(
    const vector<Neighbor>& neighbors
) const
{
    vector<double> weights(neighbors.size(), 1.0);

    if (weighting == Weighting::Uniform) {
        return weights;
    }

    // Проверяем, есть ли соседи с нулевым расстоянием.
    bool hasZeroDistance = false;

    for (const Neighbor& neighbor : neighbors) {
        if (neighbor.distance == 0.0) {
            hasZeroDistance = true;
            break;
        }
    }

    // Если совпадения есть, учитываем только их.
    if (hasZeroDistance) {
        for (size_t i = 0; i < neighbors.size(); ++i) {
            weights[i] =
                (neighbors[i].distance == 0.0) ? 1.0 : 0.0;
        }

        return weights;
    }

    // Иначе используем веса, обратные расстоянию.
    for (size_t i = 0; i < neighbors.size(); ++i) {
        weights[i] = 1.0 / neighbors[i].distance;
    }

    return weights;
}

// Получение целевого значения обучающего объекта.
double KnnSearch::getTarget(size_t index) const
{
    if (!fitted) {
        throw ModelUsageError(
            "Модель ещё не обучена"
        );
    }

    return trainTargets.at(index);
}

// Получение количества признаков.
size_t KnnSearch::featureCount() const
{
    if (!fitted) {
        throw ModelUsageError(
            "Модель ещё не обучена"
        );
    }

    return trainFeatures.columns();
}

// Проверка состояния обучения.
bool KnnSearch::isFitted() const noexcept
{
    return fitted;
}

// Конструктор регрессора.
KnnRegressor::KnnRegressor(
    size_t k,
    unique_ptr<IDistance> distance,
    Weighting weighting
)
    : search(k, move(distance), weighting)
{
}

// Обучение регрессора.
void KnnRegressor::fit(
    const Matrix& features,
    span<const double> target
)
{
    search.fit(features, target);
}

// Предсказание регрессора.
vector<double> KnnRegressor::predict(
    const Matrix& features
) const
{
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
        vector<Neighbor> neighbors =
            search.findNeighbors(features[i]);

        vector<double> weights =
            search.getWeights(neighbors);

        double weightedSum = 0.0;
        double weightSum = 0.0;

        for (size_t j = 0; j < neighbors.size(); ++j) {
            weightedSum +=
                weights[j] * search.getTarget(neighbors[j].index);

            weightSum += weights[j];
        }

        if (weightSum == 0.0) {
            throw ModelUsageError(
                "Сумма весов соседей равна нулю"
            );
        }

        predictions.push_back(weightedSum / weightSum);
    }

    return predictions;
}

// Конструктор классификатора.
KnnClassifier::KnnClassifier(
    size_t k,
    unique_ptr<IDistance> distance,
    Weighting weighting
)
    : search(k, move(distance), weighting)
{
}

// Обучение классификатора.
void KnnClassifier::fit(
    const Matrix& features,
    span<const double> target
)
{
    search.fit(features, target);
}

// Предсказание классификатора.
vector<double> KnnClassifier::predict(
    const Matrix& features
) const
{
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

    vector<double> predictions;
    predictions.reserve(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        vector<Neighbor> neighbors =
            search.findNeighbors(features[i]);

        vector<double> weights =
            search.getWeights(neighbors);

        // Для каждого класса суммируем голоса соседей.
        map<double, double> votes;

        for (size_t j = 0; j < neighbors.size(); ++j) {
            double target =
                search.getTarget(neighbors[j].index);

            votes[target] += weights[j];
        }

        // Выбираем класс с наибольшей суммой голосов.
        // При равенстве побеждает меньшая метка класса.
        double bestClass = votes.begin()->first;
        double bestVote = votes.begin()->second;

        for (const auto& [classLabel, vote] : votes) {
            if (vote > bestVote) {
                bestClass = classLabel;
                bestVote = vote;
            }
        }

        predictions.push_back(bestClass);
    }

    return predictions;
}
