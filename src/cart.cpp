
#include "cart.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>

using namespace std;

namespace {

constexpr size_t maxDemonstrationDepth = 64;

// Проверка обучающих данных.
void validateTrainingData(
    const Matrix& features,
    span<const double> target
) {
    const size_t n = features.rows();
    const size_t d = features.columns();

    if (n == 0 || target.empty()) {
        throw InvalidParameterError(
            "Нельзя обучить дерево на пустой выборке"
        );
    }

    if (d == 0) {
        throw InvalidParameterError(
            "Матрица признаков не должна быть пустой"
        );
    }

    if (n != target.size()) {
        throw DimensionMismatchError(
            "Количество объектов и целевых значений не совпадает"
        );
    }

    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(target[i])) {
            throw InvalidParameterError(
                "Целевые значения содержат некорректные значения"
            );
        }

        for (size_t j = 0; j < d; ++j) {
            if (!isfinite(features(i, j))) {
                throw InvalidParameterError(
                    "Матрица признаков содержит некорректные значения"
                );
            }
        }
    }
}

// Проверка bootstrap-кратностей.
void validateCounts(
    size_t objectCount,
    span<const size_t> counts
) {
    if (counts.empty()) {
        return;
    }

    if (counts.size() != objectCount) {
        throw DimensionMismatchError(
            "Количество bootstrap-кратностей не совпадает "
            "с количеством объектов"
        );
    }

    const bool hasPositiveCount = any_of(
        counts.begin(),
        counts.end(),
        [](size_t count) {
            return count > 0;
        }
    );

    if (!hasPositiveCount) {
        throw InvalidParameterError(
            "Bootstrap-выборка не содержит объектов"
        );
    }
}

// Создание списка индексов объектов с положительной кратностью.
vector<size_t> makeObjectIds(
    size_t objectCount,
    span<const size_t> counts
) {
    vector<size_t> objectIds;
    objectIds.reserve(objectCount);

    for (size_t i = 0; i < objectCount; ++i) {
        if (counts.empty() || counts[i] > 0) {
            objectIds.push_back(i);
        }
    }

    return objectIds;
}

// Создание списка всех признаков.
vector<size_t> makeFeatureIds(size_t featureCount) {
    vector<size_t> featureIds(featureCount);
    iota(featureIds.begin(), featureIds.end(), 0);
    return featureIds;
}

// Возвращает суммарную кратность объектов узла.
long double calculateTotalWeight(
    const vector<size_t>& objectIds,
    span<const size_t> counts
) {
    long double totalWeight = 0.0L;

    for (size_t id : objectIds) {
        const size_t weight = counts.empty() ? 1 : counts[id];
        totalWeight += static_cast<long double>(weight);
    }

    return totalWeight;
}

// Вычисляет количество признаков, которые нужно рассматривать.
size_t getFeatureSubsetSize(
    const FeatureSearchParams& params,
    size_t featureCount
) {
    if (featureCount == 0) {
        return 0;
    }

    switch (params.mode) {
    case FeatureMode::All:
        return featureCount;

    case FeatureMode::Sqrt:
        return max(
            size_t{1},
            static_cast<size_t>(
                sqrt(static_cast<double>(featureCount))
            )
        );

    case FeatureMode::Count:
        return min(params.count, featureCount);
    }

    throw InvalidParameterError(
        "Неизвестный режим выбора признаков"
    );
}

// Ищет лучшее разбиение среди случайно выбранных признаков.
optional<Split> findRandomFeatureSplit(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    SplitCriterion criterion,
    size_t minSamplesLeaf,
    span<const size_t> counts,
    const FeatureSearchParams& featureSearch,
    mt19937& generator
) {
    if (featureIds.empty()) {
        return nullopt;
    }

    vector<size_t> shuffledFeatures = featureIds;

    shuffle(
        shuffledFeatures.begin(),
        shuffledFeatures.end(),
        generator
    );

    const size_t initialCount = getFeatureSubsetSize(
        featureSearch,
        shuffledFeatures.size()
    );

    vector<size_t> selectedFeatures;
    selectedFeatures.reserve(initialCount + 1);

    optional<Split> bestSplit;
    bool foundNonConstant = false;

    // Сначала рассматриваем первые q случайных признаков.
    for (size_t i = 0; i < initialCount; ++i) {
        const size_t feature = shuffledFeatures[i];

        FeatureSplitResult result = SplitFinder::findForFeature(
            features,
            target,
            objectIds,
            feature,
            criterion,
            minSamplesLeaf,
            counts
        );

        if (!result.isConstant) {
            foundNonConstant = true;
        }

        selectedFeatures.push_back(feature);

        if (result.best.has_value()) {
            if (!bestSplit.has_value() ||
                result.best->score > bestSplit->score ||
                (result.best->score == bestSplit->score &&
                 (result.best->featureIndex < bestSplit->featureIndex ||
                  (result.best->featureIndex == bestSplit->featureIndex &&
                   result.best->threshold < bestSplit->threshold)))) {
                bestSplit = result.best;
            }
        }
    }

    // Если все первые q признаков константны,
    // продолжаем до первого неконстантного признака.
    if (!foundNonConstant) {
        for (size_t i = initialCount; i < shuffledFeatures.size(); ++i) {
            const size_t feature = shuffledFeatures[i];

            FeatureSplitResult result = SplitFinder::findForFeature(
                features,
                target,
                objectIds,
                feature,
                criterion,
                minSamplesLeaf,
                counts
            );

            selectedFeatures.push_back(feature);

            if (!result.isConstant) {
                if (result.best.has_value()) {
                    if (!bestSplit.has_value() ||
                        result.best->score > bestSplit->score ||
                        (result.best->score == bestSplit->score &&
                         (result.best->featureIndex < bestSplit->featureIndex ||
                          (result.best->featureIndex == bestSplit->featureIndex &&
                           result.best->threshold < bestSplit->threshold)))) {
                        bestSplit = result.best;
                    }
                }

                // После первого неконстантного признака
                // больше признаки не рассматриваем.
                break;
            }
        }
    }

    return bestSplit;
}

// Формирует сводку построенного дерева.
CartTreeSummary summarizeTree(const CartNode* root) {
    if (root == nullptr) {
        throw ModelUsageError("Дерево ещё не обучено");
    }

    CartTreeSummary summary;
    vector<pair<const CartNode*, size_t>> pending{{root, 0}};

    while (!pending.empty()) {
        const auto [node, depth] = pending.back();
        pending.pop_back();

        ++summary.nodeCount;
        summary.depth = max(summary.depth, depth);

        if (node->isLeaf) {
            ++summary.leafCount;
            continue;
        }

        summary.splits.push_back({
            depth,
            node->featureIndex,
            node->threshold
        });

        if (node->right) {
            pending.emplace_back(node->right.get(), depth + 1);
        }

        if (node->left) {
            pending.emplace_back(node->left.get(), depth + 1);
        }
    }

    return summary;
}

} // namespace


// ============================================================
// CART REGRESSOR
// ============================================================

CartRegressor::CartRegressor(
    optional<size_t> max_depth,
    size_t min_samples_split,
    size_t min_samples_leaf,
    FeatureSearchParams feature_search,
    uint32_t random_state
)
    : maxDepth(max_depth),
      minSamplesSplit(min_samples_split),
      minSamplesLeaf(min_samples_leaf),
      featureSearch(feature_search),
      randomState(random_state)
{
    if (maxDepth.has_value() &&
        *maxDepth > maxDemonstrationDepth) {
        throw InvalidParameterError(
            "Максимальная глубина CART не должна превышать 64"
        );
    }

    if (minSamplesSplit < 2) {
        throw InvalidParameterError(
            "Минимальное количество объектов для разбиения "
            "должно быть не меньше двух"
        );
    }

    if (minSamplesLeaf < 1) {
        throw InvalidParameterError(
            "Минимальное количество объектов в листе "
            "должно быть не меньше одного"
        );
    }

    if (featureSearch.mode == FeatureMode::Count &&
        featureSearch.count == 0) {
        throw InvalidParameterError(
            "Количество случайно выбираемых признаков должно быть больше нуля"
        );
    }
}


// Построение узла регрессионного дерева.
unique_ptr<CartNode> CartRegressor::buildNode(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    span<const size_t> counts,
    size_t depth,
    mt19937& generator
) {
    auto node = make_unique<CartNode>();

    long double weightedSum = 0.0L;
    const long double totalWeight = calculateTotalWeight(
        objectIds,
        counts
    );

    for (size_t id : objectIds) {
        const size_t weight = counts.empty() ? 1 : counts[id];

        weightedSum +=
            static_cast<long double>(target[id]) *
            static_cast<long double>(weight);
    }

    node->prediction = static_cast<double>(
        weightedSum / totalWeight
    );

    if (!isfinite(node->prediction)) {
        throw InvalidParameterError(
            "Предсказание листа вышло за поддерживаемый численный диапазон"
        );
    }

    const double firstTarget = target[objectIds.front()];

    const bool identicalTargets = all_of(
        objectIds.begin(),
        objectIds.end(),
        [&target, firstTarget](size_t id) {
            return target[id] == firstTarget;
        }
    );

    // Условия остановки.
    if (identicalTargets ||
        (maxDepth.has_value() && depth >= *maxDepth) ||
        objectIds.size() < minSamplesSplit ||
        minSamplesLeaf > objectIds.size() / 2 ||
        featureIds.empty()) {
        return node;
    }

    optional<Split> split = findRandomFeatureSplit(
        features,
        target,
        objectIds,
        featureIds,
        SplitCriterion::Variance,
        minSamplesLeaf,
        counts,
        featureSearch,
        generator
    );

    if (!split.has_value()) {
        return node;
    }

    vector<size_t> leftIds;
    vector<size_t> rightIds;

    leftIds.reserve(objectIds.size());
    rightIds.reserve(objectIds.size());

    for (size_t id : objectIds) {
        if (features(id, split->featureIndex) <= split->threshold) {
            leftIds.push_back(id);
        }
        else {
            rightIds.push_back(id);
        }
    }

    if (leftIds.empty() || rightIds.empty()) {
        return node;
    }

    node->isLeaf = false;
    node->featureIndex = split->featureIndex;
    node->threshold = split->threshold;

    node->left = buildNode(
        features,
        target,
        leftIds,
        featureIds,
        counts,
        depth + 1,
        generator
    );

    node->right = buildNode(
        features,
        target,
        rightIds,
        featureIds,
        counts,
        depth + 1,
        generator
    );

    return node;
}


// Обучение регрессионного дерева без bootstrap-кратностей.
void CartRegressor::fit(
    const Matrix& features,
    span<const double> target
) {
    fit_with_counts(features, target, {});
}


// Обучение регрессионного дерева с bootstrap-кратностями.
void CartRegressor::fit_with_counts(
    const Matrix& features,
    span<const double> target,
    span<const size_t> counts
) {
    validateTrainingData(features, target);
    validateCounts(features.rows(), counts);

    vector<size_t> objectIds = makeObjectIds(
        features.rows(),
        counts
    );

    vector<size_t> featureIds = makeFeatureIds(
        features.columns()
    );

    mt19937 generator(randomState);

    unique_ptr<CartNode> newRoot = buildNode(
        features,
        target,
        objectIds,
        featureIds,
        counts,
        0,
        generator
    );

    root = move(newRoot);
    featureCount = features.columns();
    fitted = true;
}


// Предсказание регрессионного дерева.
vector<double> CartRegressor::predict(
    const Matrix& features
) const {
    if (!fitted || !root) {
        throw ModelUsageError(
            "Регрессионное дерево ещё не обучено"
        );
    }

    if (features.columns() != featureCount) {
        throw DimensionMismatchError(
            "Количество признаков не совпадает с обучающей выборкой"
        );
    }

    vector<double> predictions;
    predictions.reserve(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        for (size_t j = 0; j < features.columns(); ++j) {
            if (!isfinite(features(i, j))) {
                throw InvalidParameterError(
                    "Матрица для предсказания содержит некорректные значения"
                );
            }
        }

        const CartNode* current = root.get();

        while (!current->isLeaf) {
            if (features(i, current->featureIndex) <=
                current->threshold) {
                current = current->left.get();
            }
            else {
                current = current->right.get();
            }
        }

        predictions.push_back(current->prediction);
    }

    return predictions;
}


CartTreeSummary CartRegressor::summary() const {
    if (!fitted) {
        throw ModelUsageError(
            "Регрессионное дерево ещё не обучено"
        );
    }

    return summarizeTree(root.get());
}


// ============================================================
// CART CLASSIFIER
// ============================================================

CartClassifier::CartClassifier(
    optional<size_t> max_depth,
    size_t min_samples_split,
    size_t min_samples_leaf,
    FeatureSearchParams feature_search,
    uint32_t random_state
)
    : maxDepth(max_depth),
      minSamplesSplit(min_samples_split),
      minSamplesLeaf(min_samples_leaf),
      featureSearch(feature_search),
      randomState(random_state)
{
    if (maxDepth.has_value() &&
        *maxDepth > maxDemonstrationDepth) {
        throw InvalidParameterError(
            "Максимальная глубина CART не должна превышать 64"
        );
    }

    if (minSamplesSplit < 2) {
        throw InvalidParameterError(
            "Минимальное количество объектов для разбиения "
            "должно быть не меньше двух"
        );
    }

    if (minSamplesLeaf < 1) {
        throw InvalidParameterError(
            "Минимальное количество объектов в листе "
            "должно быть не меньше одного"
        );
    }

    if (featureSearch.mode == FeatureMode::Count &&
        featureSearch.count == 0) {
        throw InvalidParameterError(
            "Количество случайно выбираемых признаков должно быть больше нуля"
        );
    }
}


// Построение узла классификационного дерева.
unique_ptr<CartNode> CartClassifier::buildNode(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    span<const size_t> counts,
    size_t depth,
    mt19937& generator
) {
    auto node = make_unique<CartNode>();

    node->classProbabilities.assign(
        classLabels.size(),
        0.0
    );

    long double totalWeight = calculateTotalWeight(
        objectIds,
        counts
    );

    // Взвешенные частоты классов.
    for (size_t id : objectIds) {
        const size_t weight = counts.empty() ? 1 : counts[id];

        auto it = lower_bound(
            classLabels.begin(),
            classLabels.end(),
            target[id]
        );

        if (it == classLabels.end() || *it != target[id]) {
            throw InvalidParameterError(
                "Не удалось определить индекс класса"
            );
        }

        const size_t classIndex = static_cast<size_t>(
            it - classLabels.begin()
        );

        node->classProbabilities[classIndex] +=
            static_cast<double>(weight);
    }

    // Нормировка частот до вероятностей.
    for (double& probability : node->classProbabilities) {
        probability /= static_cast<double>(totalWeight);
    }

    const double firstTarget = target[objectIds.front()];

    const bool identicalTargets = all_of(
        objectIds.begin(),
        objectIds.end(),
        [&target, firstTarget](size_t id) {
            return target[id] == firstTarget;
        }
    );

    // Условия остановки.
    if (identicalTargets ||
        (maxDepth.has_value() && depth >= *maxDepth) ||
        objectIds.size() < minSamplesSplit ||
        minSamplesLeaf > objectIds.size() / 2 ||
        featureIds.empty()) {
        return node;
    }

    optional<Split> split = findRandomFeatureSplit(
        features,
        target,
        objectIds,
        featureIds,
        SplitCriterion::Gini,
        minSamplesLeaf,
        counts,
        featureSearch,
        generator
    );

    if (!split.has_value()) {
        return node;
    }

    vector<size_t> leftIds;
    vector<size_t> rightIds;

    leftIds.reserve(objectIds.size());
    rightIds.reserve(objectIds.size());

    for (size_t id : objectIds) {
        if (features(id, split->featureIndex) <= split->threshold) {
            leftIds.push_back(id);
        }
        else {
            rightIds.push_back(id);
        }
    }

    if (leftIds.empty() || rightIds.empty()) {
        return node;
    }

    node->isLeaf = false;
    node->featureIndex = split->featureIndex;
    node->threshold = split->threshold;

    node->left = buildNode(
        features,
        target,
        leftIds,
        featureIds,
        counts,
        depth + 1,
        generator
    );

    node->right = buildNode(
        features,
        target,
        rightIds,
        featureIds,
        counts,
        depth + 1,
        generator
    );

    return node;
}


// Обучение классификационного дерева без bootstrap-кратностей.
void CartClassifier::fit(
    const Matrix& features,
    span<const double> target
) {
    fit_with_counts(features, target, {});
}


// Обучение классификационного дерева с bootstrap-кратностями.
void CartClassifier::fit_with_counts(
    const Matrix& features,
    span<const double> target,
    span<const size_t> counts
) {
    validateTrainingData(features, target);
    validateCounts(features.rows(), counts);

    vector<double> newClasses(
        target.begin(),
        target.end()
    );

    sort(newClasses.begin(), newClasses.end());

    newClasses.erase(
        unique(newClasses.begin(), newClasses.end()),
        newClasses.end()
    );

    vector<size_t> objectIds = makeObjectIds(
        features.rows(),
        counts
    );

    vector<size_t> featureIds = makeFeatureIds(
        features.columns()
    );

    mt19937 generator(randomState);

    // Сохраняем старое состояние на случай ошибки обучения.
    vector<double> oldClasses = move(classLabels);
    classLabels = move(newClasses);

    try {
        unique_ptr<CartNode> newRoot = buildNode(
            features,
            target,
            objectIds,
            featureIds,
            counts,
            0,
            generator
        );

        root = move(newRoot);
        featureCount = features.columns();
        fitted = true;
    }
    catch (...) {
        classLabels = move(oldClasses);
        throw;
    }
}


// Вероятности классов.
Matrix CartClassifier::predict_proba(
    const Matrix& features
) const {
    if (!fitted || !root) {
        throw ModelUsageError(
            "Классификационное дерево ещё не обучено"
        );
    }

    if (features.columns() != featureCount) {
        throw DimensionMismatchError(
            "Количество признаков не совпадает с обучающей выборкой"
        );
    }

    Matrix probabilities(
        features.rows(),
        classLabels.size()
    );

    for (size_t i = 0; i < features.rows(); ++i) {
        for (size_t j = 0; j < features.columns(); ++j) {
            if (!isfinite(features(i, j))) {
                throw InvalidParameterError(
                    "Матрица для предсказания содержит некорректные значения"
                );
            }
        }

        const CartNode* current = root.get();

        while (!current->isLeaf) {
            if (features(i, current->featureIndex) <=
                current->threshold) {
                current = current->left.get();
            }
            else {
                current = current->right.get();
            }
        }

        for (size_t j = 0; j < classLabels.size(); ++j) {
            probabilities(i, j) =
                current->classProbabilities[j];
        }
    }

    return probabilities;
}


// Предсказание класса.
vector<double> CartClassifier::predict(
    const Matrix& features
) const {
    Matrix probabilities = predict_proba(features);

    vector<double> predictions;
    predictions.reserve(features.rows());

    for (size_t i = 0; i < features.rows(); ++i) {
        size_t bestClass = 0;

        for (size_t j = 1; j < classLabels.size(); ++j) {
            if (probabilities(i, j) >
                probabilities(i, bestClass)) {
                bestClass = j;
            }
        }

        predictions.push_back(classLabels[bestClass]);
    }

    return predictions;
}


// Получение списка классов.
span<const double> CartClassifier::classes() const {
    if (!fitted) {
        throw ModelUsageError(
            "Классификационное дерево ещё не обучено"
        );
    }

    return classLabels;
}


CartTreeSummary CartClassifier::summary() const {
    if (!fitted) {
        throw ModelUsageError(
            "Классификационное дерево ещё не обучено"
        );
    }

    return summarizeTree(root.get());
}
