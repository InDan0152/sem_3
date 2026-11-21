
#include "split_finder.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

using namespace std;

namespace {

struct SortedObject {
    size_t objectId;
    size_t classIndex;
    size_t weight;
};

void validateCriterion(SplitCriterion criterion) {
    if (criterion != SplitCriterion::Variance &&
        criterion != SplitCriterion::Gini) {
        throw InvalidParameterError("Неизвестный критерий разбиения");
    }
}

void validateInput(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    size_t minSamplesLeaf,
    span<const size_t> counts
) {
    if (minSamplesLeaf == 0) {
        throw InvalidParameterError(
            "Минимальное количество объектов в листе должно быть больше нуля"
        );
    }

    if (features.rows() != target.size()) {
        throw DimensionMismatchError(
            "Количество объектов и целевых значений не совпадает"
        );
    }

    if (!counts.empty() && counts.size() != features.rows()) {
        throw DimensionMismatchError(
            "Количество кратностей не совпадает с числом строк матрицы"
        );
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

        if (!counts.empty() && counts[id] == 0) {
            throw InvalidParameterError(
                "Объект с нулевой кратностью не должен находиться в узле"
            );
        }
    }
}

size_t getWeight(
    size_t objectId,
    span<const size_t> counts
) {
    return counts.empty() ? 1 : counts[objectId];
}

FeatureSplitResult findForFeatureImpl(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    size_t feature,
    SplitCriterion criterion,
    size_t minSamplesLeaf,
    span<const size_t> counts
) {
    validateCriterion(criterion);
    validateInput(
        features, target, objectIds, minSamplesLeaf, counts
    );

    if (feature >= features.columns()) {
        throw InvalidParameterError(
            "Индекс признака выходит за границы матрицы"
        );
    }

    FeatureSplitResult result;

    const size_t n = objectIds.size();

    if (n < 2 || minSamplesLeaf > n / 2) {
        return result;
    }

    map<double, size_t> classIndices;
    vector<long double> totalClassCounts;

    if (criterion == SplitCriterion::Gini) {
        for (size_t id : objectIds) {
            classIndices.emplace(target[id], 0);
        }

        size_t nextIndex = 0;
        for (auto& [label, index] : classIndices) {
            (void)label;
            index = nextIndex++;
        }

        totalClassCounts.assign(classIndices.size(), 0.0L);
    }

    vector<SortedObject> sortedObjects;
    sortedObjects.reserve(n);

    long double totalSum = 0.0L;
    long double totalWeight = 0.0L;

    for (size_t id : objectIds) {
        const double value = features(id, feature);

        if (!isfinite(value)) {
            throw InvalidParameterError(
                "Матрица содержит некорректные значения"
            );
        }

        const size_t weight = getWeight(id, counts);
        const long double w = static_cast<long double>(weight);

        size_t classIndex = 0;

        if (criterion == SplitCriterion::Variance) {
            totalSum += w * static_cast<long double>(target[id]);
        } else {
            classIndex = classIndices.at(target[id]);
            totalClassCounts[classIndex] += w;
        }

        totalWeight += w;

        sortedObjects.push_back({id, classIndex, weight});
    }

    if (totalWeight <= 0.0L) {
        throw InvalidParameterError(
            "Суммарная кратность объектов должна быть положительной"
        );
    }

    sort(
        sortedObjects.begin(),
        sortedObjects.end(),
        [&features, feature](const SortedObject& a,
                             const SortedObject& b) {
            const double va = features(a.objectId, feature);
            const double vb = features(b.objectId, feature);

            if (va == vb) {
                return a.objectId < b.objectId;
            }

            return va < vb;
        }
    );

    const double firstValue =
        features(sortedObjects.front().objectId, feature);
    const double lastValue =
        features(sortedObjects.back().objectId, feature);

    result.isConstant = (firstValue == lastValue);

    if (result.isConstant) {
        return result;
    }

    size_t leftUniqueCount = 0;
    long double leftWeight = 0.0L;
    long double leftSum = 0.0L;

    vector<long double> leftClassCounts(
        totalClassCounts.size(), 0.0L
    );

    long double leftSquares = 0.0L;
    long double rightSquares = 0.0L;

    if (criterion == SplitCriterion::Gini) {
        for (long double count : totalClassCounts) {
            rightSquares += count * count;
        }
    }

    long double bestScore =
        -numeric_limits<long double>::infinity();

    optional<Split> bestSplit;

    for (size_t pos = 0; pos + 1 < n; ++pos) {
        const SortedObject& object = sortedObjects[pos];

        const size_t id = object.objectId;
        const long double w =
            static_cast<long double>(object.weight);

        ++leftUniqueCount;
        leftWeight += w;

        if (criterion == SplitCriterion::Variance) {
            leftSum += w * static_cast<long double>(target[id]);
        } else {
            const size_t c = object.classIndex;

            const long double a = leftClassCounts[c];
            const long double b = totalClassCounts[c] - a;

            leftSquares += 2.0L * a * w + w * w;
            rightSquares += -2.0L * b * w + w * w;

            leftClassCounts[c] += w;
        }

        const double currentValue =
            features(sortedObjects[pos].objectId, feature);
        const double nextValue =
            features(sortedObjects[pos + 1].objectId, feature);

        if (currentValue == nextValue) {
            continue;
        }

        const size_t rightUniqueCount = n - leftUniqueCount;

        if (leftUniqueCount < minSamplesLeaf ||
            rightUniqueCount < minSamplesLeaf) {
            continue;
        }

        const long double rightWeight = totalWeight - leftWeight;

        if (leftWeight <= 0.0L || rightWeight <= 0.0L) {
            continue;
        }

        long double score = 0.0L;

        if (criterion == SplitCriterion::Variance) {
            const long double rightSum = totalSum - leftSum;

            score =
                leftSum * leftSum / leftWeight +
                rightSum * rightSum / rightWeight;
        } else {
            score =
                leftSquares / leftWeight +
                rightSquares / rightWeight;
        }

        if (!isfinite(score)) {
            throw InvalidParameterError(
                "Оценка разбиения вышла за численный диапазон"
            );
        }

        double threshold = midpoint(currentValue, nextValue);

        if (threshold >= nextValue || threshold < currentValue) {
            threshold = currentValue;
        }

        const bool betterScore = score > bestScore;

        const bool deterministicTie =
            bestSplit.has_value() &&
            score == bestScore &&
            threshold < bestSplit->threshold;

        if (betterScore || deterministicTie) {
            bestScore = score;
            bestSplit = Split{feature, threshold, score};
        }
    }

    result.best = bestSplit;
    return result;
}

optional<Split> findBestSplitImpl(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    SplitCriterion criterion,
    size_t minSamplesLeaf,
    span<const size_t> counts
) {
    validateCriterion(criterion);
    validateInput(
        features, target, objectIds, minSamplesLeaf, counts
    );

    optional<Split> bestSplit;

    for (size_t feature : featureIds) {
        FeatureSplitResult candidate = findForFeatureImpl(
            features,
            target,
            objectIds,
            feature,
            criterion,
            minSamplesLeaf,
            counts
        );

        if (!candidate.best.has_value()) {
            continue;
        }

        const Split& split = *candidate.best;

        const bool betterScore =
            !bestSplit.has_value() ||
            split.score > bestSplit->score;

        const bool deterministicTie =
            bestSplit.has_value() &&
            split.score == bestSplit->score &&
            (split.featureIndex < bestSplit->featureIndex ||
             (split.featureIndex == bestSplit->featureIndex &&
              split.threshold < bestSplit->threshold));

        if (betterScore || deterministicTie) {
            bestSplit = split;
        }
    }

    return bestSplit;
}

} // namespace

optional<Split> SplitFinder::findBestSplit(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    SplitCriterion criterion,
    size_t minSamplesLeaf,
    span<const size_t> counts
) {
    return findBestSplitImpl(
        features,
        target,
        objectIds,
        featureIds,
        criterion,
        minSamplesLeaf,
        counts
    );
}

optional<Split> SplitFinder::findBestSplit(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    SplitCriterion criterion,
    size_t minSamplesLeaf,
    span<const size_t> counts
) {
    vector<size_t> featureIds(features.columns());
    iota(featureIds.begin(), featureIds.end(), size_t{0});

    return findBestSplitImpl(
        features,
        target,
        objectIds,
        featureIds,
        criterion,
        minSamplesLeaf,
        counts
    );
}

FeatureSplitResult SplitFinder::findForFeature(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    size_t feature,
    SplitCriterion criterion,
    size_t minSamplesLeaf,
    span<const size_t> counts
) {
    return findForFeatureImpl(
        features,
        target,
        objectIds,
        feature,
        criterion,
        minSamplesLeaf,
        counts
    );
}
