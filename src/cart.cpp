
#include "cart.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <utility>

using namespace std;

namespace {

constexpr size_t maxDemonstrationDepth = 64;

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

// Конструктор регрессионного дерева.
CartRegressor::CartRegressor(
    size_t max_depth,
    size_t min_samples_split,
    size_t min_samples_leaf
)
    : maxDepth(max_depth),
      minSamplesSplit(min_samples_split),
      minSamplesLeaf(min_samples_leaf)
{
    if (maxDepth > maxDemonstrationDepth) {
        throw InvalidParameterError(
            "Максимальная глубина CART не должна превышать 64"
        );
    }

    if (minSamplesSplit < 2) {
        throw InvalidParameterError(
            "Минимальное количество объектов для разбиения должно быть не меньше двух"
        );
    }

    if (minSamplesLeaf == 0) {
        throw InvalidParameterError(
            "Минимальное количество объектов в листе должно быть больше нуля"
        );
    }
}

// Рекурсивное построение узла регрессионного дерева.
unique_ptr<CartNode> CartRegressor::buildNode(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    size_t depth
) {
    auto node = make_unique<CartNode>();

    // Предсказание листа — среднее целевых значений.
    long double sum = 0.0L;

    for (size_t id : objectIds) {
        sum += static_cast<long double>(target[id]);
    }

    node->prediction = static_cast<double>(
        sum / static_cast<long double>(objectIds.size())
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

    // Условия остановки рекурсии.
    if (identicalTargets ||
        depth >= maxDepth ||
        objectIds.size() < minSamplesSplit ||
        minSamplesLeaf > objectIds.size() / 2 ||
        featureIds.empty()) {
        return node;
    }

    // Ищем лучшее разбиение.
    optional<Split> split = SplitFinder::findBestSplit(
        features,
        target,
        objectIds,
        featureIds,
        SplitCriterion::Variance,
        minSamplesLeaf
    );

    if (!split.has_value()) {
        return node;
    }

    // Распределяем индексы объектов между потомками.
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

    // Защита от некорректного разбиения.
    if (leftIds.empty() || rightIds.empty()) {
        return node;
    }

    node->isLeaf = false;
    node->featureIndex = split->featureIndex;
    node->threshold = split->threshold;

    // Рекурсивно строим дочерние узлы.
    node->left = buildNode(
        features,
        target,
        leftIds,
        featureIds,
        depth + 1
    );

    node->right = buildNode(
        features,
        target,
        rightIds,
        featureIds,
        depth + 1
    );

    return node;
}

// Обучение регрессионного дерева.
void CartRegressor::fit(
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

    // Проверяем входные данные.
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

    vector<size_t> objectIds(n);
    iota(objectIds.begin(), objectIds.end(), 0);

    vector<size_t> featureIds(d);
    iota(featureIds.begin(), featureIds.end(), 0);

    // Строим новое дерево.
    unique_ptr<CartNode> newRoot = buildNode(
        features,
        target,
        objectIds,
        featureIds,
        0
    );

    root = move(newRoot);
    featureCount = d;
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
        throw ModelUsageError("Регрессионное дерево ещё не обучено");
    }
    return summarizeTree(root.get());
}

// ============================================================
// CART CLASSIFIER
// ============================================================

// Конструктор классификационного дерева.
CartClassifier::CartClassifier(
    size_t max_depth,
    size_t min_samples_split,
    size_t min_samples_leaf
)
    : maxDepth(max_depth),
      minSamplesSplit(min_samples_split),
      minSamplesLeaf(min_samples_leaf)
{
    if (maxDepth > maxDemonstrationDepth) {
        throw InvalidParameterError(
            "Максимальная глубина CART не должна превышать 64"
        );
    }

    if (minSamplesSplit < 2) {
        throw InvalidParameterError(
            "Минимальное количество объектов для разбиения должно быть не меньше двух"
        );
    }

    if (minSamplesLeaf == 0) {
        throw InvalidParameterError(
            "Минимальное количество объектов в листе должно быть больше нуля"
        );
    }
}

// Рекурсивное построение узла классификационного дерева.
unique_ptr<CartNode> CartClassifier::buildNode(
    const Matrix& features,
    span<const double> target,
    const vector<size_t>& objectIds,
    const vector<size_t>& featureIds,
    size_t depth
) {
    auto node = make_unique<CartNode>();

    // Вычисляем распределение классов в текущем узле.
    node->classProbabilities.assign(classLabels.size(), 0.0);

    for (size_t id : objectIds) {
        auto it = lower_bound(
            classLabels.begin(),
            classLabels.end(),
            target[id]
        );

        size_t classIndex =
            static_cast<size_t>(it - classLabels.begin());

        node->classProbabilities[classIndex] += 1.0;
    }

    // Нормируем частоты классов.
    for (double& probability : node->classProbabilities) {
        probability /= static_cast<double>(objectIds.size());
    }

    const double firstTarget = target[objectIds.front()];
    const bool identicalTargets = all_of(
        objectIds.begin(),
        objectIds.end(),
        [&target, firstTarget](size_t id) {
            return target[id] == firstTarget;
        }
    );

    // Условия остановки рекурсии.
    if (identicalTargets ||
        depth >= maxDepth ||
        objectIds.size() < minSamplesSplit ||
        minSamplesLeaf > objectIds.size() / 2 ||
        featureIds.empty()) {
        return node;
    }

    // Ищем лучшее разбиение по критерию Джини.
    optional<Split> split = SplitFinder::findBestSplit(
        features,
        target,
        objectIds,
        featureIds,
        SplitCriterion::Gini,
        minSamplesLeaf
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

    // Строим дочерние узлы.
    node->left = buildNode(
        features,
        target,
        leftIds,
        featureIds,
        depth + 1
    );

    node->right = buildNode(
        features,
        target,
        rightIds,
        featureIds,
        depth + 1
    );

    return node;
}

// Обучение классификационного дерева.
void CartClassifier::fit(
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
                "Метки классов содержат некорректные значения"
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

    // Формируем отсортированный список уникальных классов.
    vector<double> newClasses(target.begin(), target.end());

    sort(newClasses.begin(), newClasses.end());

    newClasses.erase(
        unique(newClasses.begin(), newClasses.end()),
        newClasses.end()
    );

    vector<size_t> objectIds(n);
    iota(objectIds.begin(), objectIds.end(), 0);

    vector<size_t> featureIds(d);
    iota(featureIds.begin(), featureIds.end(), 0);

    // Сохраняем классы для построения узлов.
    vector<double> oldClasses = move(classLabels);
    classLabels = move(newClasses);

    try {
        unique_ptr<CartNode> newRoot = buildNode(
            features,
            target,
            objectIds,
            featureIds,
            0
        );

        root = move(newRoot);
        featureCount = d;
        fitted = true;
    }
    catch (...) {
        classLabels = move(oldClasses);
        throw;
    }
}

// Получение вероятностей классов.
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
        throw ModelUsageError("Классификационное дерево ещё не обучено");
    }
    return summarizeTree(root.get());
}
