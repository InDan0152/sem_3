
#include "split_finder.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

using namespace std;

optional<Split> SplitFinder::findBestSplit(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    SplitCriterion criterion,
    size_t minSamplesLeaf
) {
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

    if (features.rows() != target.size()) {
        throw DimensionMismatchError(
            "Количество объектов и целевых значений не совпадает"
        );
    }

    if (objectIds.size() < 2 || featureIds.empty()) {
        return nullopt;
    }

    for (size_t id : objectIds) {
        if (id >= features.rows()) {
            throw InvalidParameterError(
                "Индекс объекта выходит за границы матрицы"
            );
        }

        if (!isfinite(target[id])) {
            throw InvalidParameterError(
                "Целевые значения содержат некорректные значения"
            );
        }
    }

    for (size_t feature : featureIds) {
        if (feature >= features.columns()) {
            throw InvalidParameterError(
                "Индекс признака выходит за границы матрицы"
            );
        }
    }

    for (size_t id : objectIds) {
        for (size_t feature : featureIds) {
            if (!isfinite(features(id, feature))) {
                throw InvalidParameterError(
                    "Матрица содержит некорректные значения"
                );
            }
        }
    }

    const size_t n = objectIds.size();

    if (minSamplesLeaf > n / 2) {
        return nullopt;
    }

    // Кодируем классы целочисленными индексами.
    vector<size_t> objectClasses(features.rows(), 0);
    vector<size_t> totalClassCounts;

    size_t classCount = 0;

    if (criterion == SplitCriterion::Gini) {
        map<double, size_t> classIndices;

        for (size_t id : objectIds) {
            classIndices.emplace(target[id], 0);
        }

        size_t nextClassIndex = 0;

        for (auto& [label, index] : classIndices) {
            (void)label;
            index = nextClassIndex++;
        }

        classCount = classIndices.size();
        totalClassCounts.assign(classCount, 0);

        for (size_t id : objectIds) {
            size_t classIndex = classIndices.at(target[id]);

            objectClasses[id] = classIndex;
            ++totalClassCounts[classIndex];
        }
    }

    long double totalSum = 0.0L;

    if (criterion == SplitCriterion::Variance) {
        for (size_t id : objectIds) {
            totalSum += static_cast<long double>(target[id]);
        }
    }

    long double bestScore =
        -numeric_limits<long double>::infinity();

    optional<Split> bestSplit;

    vector<size_t> sortedIds = objectIds;

    // Перебираем выбранные признаки.
    for (size_t feature : featureIds) {
        sort(
            sortedIds.begin(),
            sortedIds.end(),
            [&features, feature](size_t a, size_t b) {
                double valueA = features(a, feature);
                double valueB = features(b, feature);

                if (valueA == valueB) {
                    return a < b;
                }

                return valueA < valueB;
            }
        );

        size_t leftCount = 0;
        long double leftSum = 0.0L;

        vector<size_t> leftClassCounts(classCount, 0);
        vector<size_t> rightClassCounts = totalClassCounts;

        long double leftSquares = 0.0L;
        long double rightSquares = 0.0L;

        if (criterion == SplitCriterion::Gini) {
            for (size_t count : totalClassCounts) {
                rightSquares +=
                    static_cast<long double>(count) *
                    static_cast<long double>(count);
            }
        }

        for (size_t pos = 0; pos + 1 < n; ++pos) {
            size_t objectIndex = sortedIds[pos];

            ++leftCount;

            if (criterion == SplitCriterion::Variance) {
                leftSum +=
                    static_cast<long double>(target[objectIndex]);
            }
            else {
                size_t classIndex = objectClasses[objectIndex];

                size_t leftBefore = leftClassCounts[classIndex];
                size_t rightBefore = rightClassCounts[classIndex];

                leftSquares +=
                    2.0L * static_cast<long double>(leftBefore) + 1.0L;

                rightSquares +=
                    1.0L -
                    2.0L * static_cast<long double>(rightBefore);

                ++leftClassCounts[classIndex];
                --rightClassCounts[classIndex];
            }

            double currentValue =
                features(sortedIds[pos], feature);

            double nextValue =
                features(sortedIds[pos + 1], feature);

            // Между одинаковыми значениями порог не ставим.
            if (currentValue == nextValue) {
                continue;
            }

            size_t rightCount = n - leftCount;

            if (leftCount < minSamplesLeaf ||
                rightCount < minSamplesLeaf) {
                continue;
            }

            long double score = 0.0L;

            if (criterion == SplitCriterion::Variance) {
                long double rightSum = totalSum - leftSum;

                score =
                    leftSum * leftSum /
                        static_cast<long double>(leftCount) +
                    rightSum * rightSum /
                        static_cast<long double>(rightCount);
            }
            else {
                score =
                    leftSquares / static_cast<long double>(leftCount) +
                    rightSquares / static_cast<long double>(rightCount);
            }

            if (!isfinite(score)) {
                throw InvalidParameterError(
                    "Оценка разбиения вышла за поддерживаемый численный диапазон"
                );
            }

            double threshold = midpoint(currentValue, nextValue);

            // midpoint может округлиться до правого значения. В таком случае
            // <= отправил бы оба значения влево; левое значение — безопасный
            // порог между соседними представимыми числами.
            if (threshold >= nextValue) {
                threshold = currentValue;
            }

            const bool betterScore = score > bestScore;
            const bool deterministicTie =
                bestSplit.has_value() && score == bestScore &&
                (feature < bestSplit->featureIndex ||
                 (feature == bestSplit->featureIndex &&
                  threshold < bestSplit->threshold));

            if (betterScore || deterministicTie) {
                bestScore = score;
                bestSplit = Split{feature, threshold, score};
            }
        }
    }

    return bestSplit;
}
