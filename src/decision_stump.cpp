
#include "decision_stump.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <limits>
#include <vector>
#include <utility>

using namespace std;

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

    for (size_t i = 0; i < n; ++i) {
        for (double value : features[i]) {
            if (!isfinite(value)) {
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

    // Константное предсказание на случай отсутствия разбиения.
    if (criterion == SplitCriterion::Variance) {
        long double sum = 0.0L;

        for (double value : target) {
            sum += value;
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

    // Массив индексов объектов.
    vector<size_t> indices(n);
    iota(indices.begin(), indices.end(), 0);

    size_t selectedFeature = 0;
    double selectedThreshold = 0.0;

    long double bestScore = -numeric_limits<long double>::infinity();

    // Для регрессии сохраняем сумму левой группы.
    long double bestLeftSum = 0.0L;
    size_t bestLeftCount = 0;

    // Подготовка классов для классификации.
    vector<size_t> targetClasses;
    vector<size_t> totalClassCounts;
    size_t classCount = 0;

    if (criterion == SplitCriterion::Gini) {
        map<double, size_t> classIndices;

        targetClasses.resize(n);

        for (size_t i = 0; i < n; ++i) {
            auto [it, inserted] = classIndices.emplace(
                target[i],
                classIndices.size()
            );

            targetClasses[i] = it->second;
        }

        classCount = classIndices.size();
        totalClassCounts.assign(classCount, 0);

        for (size_t classIndex : targetClasses) {
            ++totalClassCounts[classIndex];
        }
    }

    // Перебираем все признаки.
    for (size_t feature = 0; feature < d; ++feature) {

        // Сортируем индексы по текущему признаку.
        sort(
            indices.begin(),
            indices.end(),
            [&features, feature](size_t a, size_t b) {
                if (features[a][feature] == features[b][feature]) {
                    return a < b;
                }

                return features[a][feature] < features[b][feature];
            }
        );

        size_t leftCount = 0;
        long double leftSum = 0.0L;

        // Агрегаты для классификации.
        vector<size_t> leftClassCounts(classCount, 0);
        vector<size_t> rightClassCounts = totalClassCounts;

        long double leftSquares = 0.0L;
        long double rightSquares = 0.0L;

        if (criterion == SplitCriterion::Gini) {
            for (size_t count : totalClassCounts) {
                rightSquares +=
                    static_cast<long double>(count) * count;
            }
        }

        long double totalSum = 0.0L;

        if (criterion == SplitCriterion::Variance) {
            for (double value : target) {
                totalSum += value;
            }
        }

        // Перебираем возможные положения порога.
        for (size_t pos = 0; pos + 1 < n; ++pos) {
            const size_t objectIndex = indices[pos];

            // Переносим объект из правой группы в левую.
            ++leftCount;

            if (criterion == SplitCriterion::Variance) {
                leftSum += target[objectIndex];
            }
            else {
                const size_t classIndex =
                    targetClasses[objectIndex];

                const size_t leftBefore =
                    leftClassCounts[classIndex];

                const size_t rightBefore =
                    rightClassCounts[classIndex];

                leftSquares +=
                    static_cast<long double>(2 * leftBefore + 1);

                rightSquares +=
                    static_cast<long double>(1) -
                    static_cast<long double>(2 * rightBefore);

                ++leftClassCounts[classIndex];
                --rightClassCounts[classIndex];
            }

            // Между одинаковыми значениями порог ставить нельзя.
            const double currentValue =
                features[indices[pos]][feature];

            const double nextValue =
                features[indices[pos + 1]][feature];

            if (currentValue == nextValue) {
                continue;
            }

            const size_t rightCount = n - leftCount;

            // Проверяем минимальный размер обоих листьев.
            if (leftCount < minSamplesLeaf ||
                rightCount < minSamplesLeaf) {
                continue;
            }

            long double score = 0.0L;

            if (criterion == SplitCriterion::Variance) {
                const long double rightSum =
                    totalSum - leftSum;

                score =
                    leftSum * leftSum / leftCount +
                    rightSum * rightSum / rightCount;
            }
            else {
                score =
                    leftSquares / leftCount +
                    rightSquares / rightCount;
            }

            // Сохраняем лучшее разбиение.
            // При равенстве оставляем первое найденное:
            // сначала меньший индекс признака, затем меньший порог.
            if (score > bestScore) {
                bestScore = score;

                selectedFeature = feature;
                selectedThreshold = midpoint(
                    currentValue,
                    nextValue
                );

                if (criterion == SplitCriterion::Variance) {
                    bestLeftSum = leftSum;
                    bestLeftCount = leftCount;
                }
            }
        }
    }

    // Если допустимого разбиения нет, используем константное предсказание.
    if (bestScore == -numeric_limits<long double>::infinity()) {
        fitted = true;
        return;
    }

    bestFeature = selectedFeature;
    bestThreshold = selectedThreshold;
    hasSplit = true;

    // Вычисляем предсказания листьев по найденному разбиению.
    if (criterion == SplitCriterion::Variance) {
        const long double totalSum = accumulate(
            target.begin(),
            target.end(),
            0.0L
        );

        const size_t rightCount = n - bestLeftCount;
        const long double rightSum = totalSum - bestLeftSum;

        leftPrediction = static_cast<double>(
            bestLeftSum / bestLeftCount
        );

        rightPrediction = static_cast<double>(
            rightSum / rightCount
        );
    }
    else {
        map<double, size_t> leftFrequencies;
        map<double, size_t> rightFrequencies;

        for (size_t i = 0; i < n; ++i) {
            if (features[i][bestFeature] <= bestThreshold) {
                ++leftFrequencies[target[i]];
            }
            else {
                ++rightFrequencies[target[i]];
            }
        }

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
        for (double value : features[i]) {
            if (!isfinite(value)) {
                throw InvalidParameterError(
                    "Матрица для предсказания содержит некорректные значения"
                );
            }
        }

        if (!hasSplit) {
            predictions[i] = constantPrediction;
        }
        else if (features[i][bestFeature] <= bestThreshold) {
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
