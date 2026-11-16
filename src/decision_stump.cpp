
#include "decision_stump.hpp"
#include "exceptions.hpp"

#include <cmath>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

using namespace std;

// Конструктор решающего пня.
DecisionStump::DecisionStump(
    SplitCriterion criterion,
    size_t min_samples_leaf
)
    : criterion(criterion),
      minSamplesLeaf(min_samples_leaf)
{
    if (minSamplesLeaf == 0) {
        throw InvalidParameterError(
            "Минимальное количество объектов в листе должно быть больше нуля"
        );
    }

    if (criterion != SplitCriterion::Variance &&
        criterion != SplitCriterion::Gini) {
        throw InvalidParameterError(
            "Неизвестный критерий разбиения"
        );
    }
}

// Обучение решающего пня.
void DecisionStump::fit(
    const Matrix& features,
    span<const double> target
) {
    const size_t n = features.rows();
    const size_t d = features.columns();

    if (n == 0 || target.empty()) {
        throw InvalidParameterError(
            "Нельзя обучить модель на пустой выборке"
        );
    }

    if (d == 0) {
        throw InvalidParameterError(
            "Матрица признаков не должна быть пустой"
        );
    }

    if (n != target.size()) {
        throw DimensionMismatchError(
            "Количество объектов не совпадает с количеством целевых значений"
        );
    }

    // Проверяем входные данные.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            if (!isfinite(features(i, j))) {
                throw InvalidParameterError(
                    "Матрица содержит некорректные значения"
                );
            }
        }

        if (!isfinite(target[i])) {
            throw InvalidParameterError(
                "Целевые значения содержат некорректные значения"
            );
        }
    }

    fitted = false;
    hasSplit = false;
    featureCount = d;

    // Вычисляем константное предсказание.
    if (criterion == SplitCriterion::Variance) {
        long double sum = 0.0L;

        for (double value : target) {
            sum += static_cast<long double>(value);
        }

        constantPrediction = static_cast<double>(
            sum / static_cast<long double>(n)
        );
    }
    else {
        map<double, size_t> frequencies;

        for (double value : target) {
            ++frequencies[value];
        }

        size_t maxFrequency = 0;

        for (const auto& [value, frequency] : frequencies) {
            if (frequency > maxFrequency) {
                maxFrequency = frequency;
                constantPrediction = value;
            }
        }
    }

    // Формируем индексы всех объектов.
    vector<size_t> objectIds(n);
    iota(objectIds.begin(), objectIds.end(), 0);

    // Формируем индексы всех признаков.
    vector<size_t> featureIds(d);
    iota(featureIds.begin(), featureIds.end(), 0);

    // Используем общий компонент поиска разбиения.
    optional<Split> bestSplit = SplitFinder::findBestSplit(
        features,
        target,
        objectIds,
        featureIds,
        criterion,
        minSamplesLeaf
    );

    // Если разбиение не найдено, оставляем константное предсказание.
    if (!bestSplit.has_value()) {
        fitted = true;
        return;
    }

    bestFeature = bestSplit->featureIndex;
    bestThreshold = bestSplit->threshold;
    hasSplit = true;

// Вычисляем предсказания левого и правого листьев.
    long double leftSum = 0.0L;
    long double rightSum = 0.0L;

    size_t leftCount = 0;
    size_t rightCount = 0;

    map<double, size_t> leftFrequencies;
    map<double, size_t> rightFrequencies;

    for (size_t i = 0; i < n; ++i) {
        if (features(i, bestFeature) <= bestThreshold) {
            ++leftCount;
            leftSum += static_cast<long double>(target[i]);

            if (criterion == SplitCriterion::Gini) {
                ++leftFrequencies[target[i]];
            }
        }
        else {
            ++rightCount;
            rightSum += static_cast<long double>(target[i]);

            if (criterion == SplitCriterion::Gini) {
                ++rightFrequencies[target[i]];
            }
        }
    }

    if (criterion == SplitCriterion::Variance) {
        leftPrediction = static_cast<double>(
            leftSum / static_cast<long double>(leftCount)
        );

        rightPrediction = static_cast<double>(
            rightSum / static_cast<long double>(rightCount)
        );
    }
    else {
        size_t maxLeftFrequency = 0;

        for (const auto& [value, frequency] : leftFrequencies) {
            if (frequency > maxLeftFrequency) {
                maxLeftFrequency = frequency;
                leftPrediction = value;
            }
        }

        size_t maxRightFrequency = 0;

        for (const auto& [value, frequency] : rightFrequencies) {
            if (frequency > maxRightFrequency) {
                maxRightFrequency = frequency;
                rightPrediction = value;
            }
        }
    }

    fitted = true;
}

// todo: рассказать, как работает пень решений (какие решения для оптимальности алгоритма там применяются)

// Предсказание
vector<double> DecisionStump::predict(
    const Matrix& features
) const {
    if (!fitted) {
        throw ModelUsageError(
            "Нельзя выполнить предсказание до обучения модели"
        );
    }

    if (features.columns() != featureCount) {
        throw DimensionMismatchError(
            "Количество признаков не совпадает с обучающей выборкой"
        );
    }

    vector<double> predictions(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        for (size_t j = 0; j < features.columns(); ++j) {
            if (!isfinite(features(i, j))) {
                throw InvalidParameterError(
                    "Матрица для предсказания содержит некорректные значения"
                );
            }
        }

        if (!hasSplit) {
            predictions[i] = constantPrediction;
        }
        else if (features(i, bestFeature) <= bestThreshold) {
            predictions[i] = leftPrediction;
        }
        else {
            predictions[i] = rightPrediction;
        }
    }

    return predictions;
}

size_t DecisionStump::split_feature() const {
    if (!fitted || !hasSplit) {
        throw ModelUsageError(
            "Модель не имеет найденного разбиения"
        );
    }

    return bestFeature;
}

double DecisionStump::split_threshold() const {
    if (!fitted || !hasSplit) {
        throw ModelUsageError(
            "Модель не имеет найденного разбиения"
        );
    }

    return bestThreshold;
}
