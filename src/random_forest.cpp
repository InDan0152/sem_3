#include "random_forest.hpp"
#include "exceptions.hpp"

#include <cmath>
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <random>
#include <thread>

using namespace std;

namespace {

// Выполняет независимые задачи в фиксированном пуле jthread.
// Исключения из рабочих потоков передаются вызывающему потоку.
template <typename Function>
void parallelFor(
    size_t taskCount,
    size_t requestedThreads,
    Function&& function
) {
    if (taskCount == 0) {
        return;
    }

    const size_t threadCount = min(taskCount, requestedThreads);

    if (threadCount <= 1) {
        for (size_t task = 0; task < taskCount; ++task) {
            function(task, 0);
        }
        return;
    }

    atomic<size_t> nextTask{0};
    atomic<bool> stop{false};
    exception_ptr workerException;
    mutex exceptionMutex;

    {
        vector<jthread> workers;
        workers.reserve(threadCount);

        for (size_t workerId = 0; workerId < threadCount; ++workerId) {
            workers.emplace_back([&, workerId] {
                while (!stop.load()) {
                    const size_t task = nextTask.fetch_add(1);
                    if (task >= taskCount) {
                        break;
                    }

                    try {
                        function(task, workerId);
                    } catch (...) {
                        {
                            lock_guard<mutex> lock(exceptionMutex);
                            if (!workerException) {
                                workerException = current_exception();
                            }
                        }
                        stop.store(true);
                        break;
                    }
                }
            });
        }
    } // Деструкторы jthread ожидают завершения всех рабочих потоков.

    if (workerException) {
        rethrow_exception(workerException);
    }
}

// Проверка режима и значения max_features относительно числа признаков.
void validateFeatureSearch(
    const FeatureSearchParams& params,
    size_t featureCount
) {
    switch (params.mode) {
        case FeatureMode::All:
        case FeatureMode::Sqrt:
            break;
        case FeatureMode::Count:
            if (params.count == 0 || params.count > featureCount) {
                throw InvalidParameterError(
                    "max_features Count должен быть от 1 до числа признаков"
                );
            }
            break;
        default:
            throw InvalidParameterError(
                "Указан неизвестный режим max_features"
            );
    }
}

// Проверка параметров Random Forest.
void validateParams(const RandomForestParams& params) {
    if (params.n_estimators == 0) {
        throw InvalidParameterError(
            "Количество деревьев должно быть больше нуля"
        );
    }

    if (params.n_jobs == 0) {
        throw InvalidParameterError(
            "Количество потоков должно быть больше нуля"
        );
    }

    if (params.max_samples.has_value() &&
        *params.max_samples == 0) {
        throw InvalidParameterError(
            "Размер bootstrap-выборки должен быть больше нуля"
        );
    }

    if (!params.bootstrap && params.max_samples.has_value()) {
        throw InvalidParameterError(
            "Параметр max_samples нельзя задавать при bootstrap=false"
        );
    }

    if (params.min_samples_split < 2) {
        throw InvalidParameterError(
            "min_samples_split должен быть не меньше 2"
        );
    }

    if (params.min_samples_leaf == 0) {
        throw InvalidParameterError(
            "min_samples_leaf должен быть больше нуля"
        );
    }

    if (params.max_features.has_value()) {
        switch (params.max_features->mode) {
            case FeatureMode::All:
            case FeatureMode::Sqrt:
                break;
            case FeatureMode::Count:
                if (params.max_features->count == 0) {
                    throw InvalidParameterError(
                        "Количество признаков должно быть больше нуля"
                    );
                }
                break;
            default:
                throw InvalidParameterError(
                    "Указан неизвестный режим max_features"
                );
        }
    }
}

// Проверка параметров относительно размера выборки.
void validateTrainingSize(
    const RandomForestParams& params,
    size_t sampleCount
) {
    if (sampleCount == 0) {
        throw InvalidParameterError(
            "Нельзя обучить лес на пустой выборке"
        );
    }

    if (params.max_samples.has_value() &&
        *params.max_samples > sampleCount) {
        throw InvalidParameterError(
            "max_samples не может превышать размер исходной выборки"
        );
    }
}

// Создание bootstrap-кратностей.
vector<size_t> makeBootstrapCounts(
    size_t sampleCount,
    size_t bootstrapSize,
    mt19937& generator
) {
    if (sampleCount == 0 || bootstrapSize == 0) {
        throw InvalidParameterError(
            "Размеры bootstrap-выборки должны быть больше нуля"
        );
    }

    vector<size_t> counts(sampleCount, 0);

    uniform_int_distribution<size_t> distribution(
        0,
        sampleCount - 1
    );

    for (size_t i = 0; i < bootstrapSize; ++i) {
        const size_t objectId = distribution(generator);
        ++counts[objectId];
    }

    return counts;
}

// Создание массива единичных кратностей без bootstrap.
vector<size_t> makeFullSampleCounts(size_t sampleCount) {
    return vector<size_t>(sampleCount, 1);
}

// Получение размера выборки для одного дерева.
size_t getBootstrapSize(
    const RandomForestParams& params,
    size_t sampleCount
) {
    if (!params.bootstrap) {
        return sampleCount;
    }

    return params.max_samples.value_or(sampleCount);
}

// Получение seed для отдельного дерева.
uint32_t makeTreeSeed(
    uint32_t randomState,
    size_t treeIndex
) {
    uint64_t value =
        static_cast<uint64_t>(randomState) +
        0x9E3779B97F4A7C15ULL *
        (static_cast<uint64_t>(treeIndex) + 1);

    value = (value ^ (value >> 30)) *
        0xBF58476D1CE4E5B9ULL;

    value = (value ^ (value >> 27)) *
        0x94D049BB133111EBULL;

    value ^= value >> 31;

    return static_cast<uint32_t>(value);
}

} // namespace

// RANDOM FOREST REGRESSOR

RandomForestRegressor::RandomForestRegressor(
    RandomForestParams params
)
    : params(move(params))
{
    validateParams(this->params);

    if (this->params.max_depth.has_value() &&
        *this->params.max_depth > 64) {
        throw InvalidParameterError(
            "Максимальная глубина дерева не должна превышать 64"
        );
    }
}

void RandomForestRegressor::fit(
    const Matrix& features,
    span<const double> target
) {
    const size_t sampleCount = features.rows();
    const size_t featureCountNew = features.columns();

    validateTrainingSize(params, sampleCount);

    if (featureCountNew == 0) {
        throw InvalidParameterError(
            "Матрица признаков не должна быть пустой"
        );
    }

    if (sampleCount != target.size()) {
        throw DimensionMismatchError(
            "Количество объектов и целевых значений не совпадает"
        );
    }

    const FeatureSearchParams featureSearch =
        params.max_features.value_or(
            FeatureSearchParams{FeatureMode::All, 1}
        );

    validateFeatureSearch(featureSearch, featureCountNew);

    const size_t bootstrapSize =
        getBootstrapSize(params, sampleCount);

    vector<unique_ptr<CartRegressor>> newTrees(params.n_estimators);

    parallelFor(params.n_estimators, params.n_jobs,
        [&](size_t i, size_t) {
        const uint32_t treeSeed =
            makeTreeSeed(params.random_state, i);

        mt19937 generator(treeSeed);

        vector<size_t> counts;

        if (params.bootstrap) {
            counts = makeBootstrapCounts(
                sampleCount,
                bootstrapSize,
                generator
            );
        } else {
            counts = makeFullSampleCounts(sampleCount);
        }

        auto tree = make_unique<CartRegressor>(
            params.max_depth,
            params.min_samples_split,
            params.min_samples_leaf,
            featureSearch,
            treeSeed
        );

        tree->fit_with_counts(features, target, counts);

        newTrees[i] = move(tree);
    });

    trees = move(newTrees);
    featureCount = featureCountNew;
    fitted = true;
}

vector<double> RandomForestRegressor::predict(
    const Matrix& features
) const {
    if (!fitted || trees.empty()) {
        throw ModelUsageError(
            "Случайный лес ещё не обучен"
        );
    }

    if (features.columns() != featureCount) {
        throw DimensionMismatchError(
            "Количество признаков не совпадает с обучающей выборкой"
        );
    }

    vector<double> result(features.rows());
    const size_t workerCount = min(params.n_jobs, features.rows());

    if (workerCount <= 1) {
        vector<long double> sums(features.rows(), 0.0L);
        for (const auto& tree : trees) {
            const vector<double> predictions = tree->predict(features);
            for (size_t i = 0; i < predictions.size(); ++i) {
                sums[i] += static_cast<long double>(predictions[i]);
            }
        }
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = static_cast<double>(
                sums[i] / static_cast<long double>(trees.size())
            );
        }
    } else {
        parallelFor(workerCount, workerCount,
            [&](size_t workerId, size_t) {
                const size_t begin = features.rows() * workerId / workerCount;
                const size_t end = features.rows() * (workerId + 1) / workerCount;
                Matrix batch(end - begin, featureCount, features.layout());

                for (size_t row = begin; row < end; ++row) {
                    for (size_t column = 0; column < featureCount; ++column) {
                        batch(row - begin, column) = features(row, column);
                    }
                }

                vector<long double> sums(end - begin, 0.0L);
                for (const auto& tree : trees) {
                    const vector<double> predictions = tree->predict(batch);
                    for (size_t i = 0; i < predictions.size(); ++i) {
                        sums[i] += static_cast<long double>(predictions[i]);
                    }
                }

                for (size_t row = begin; row < end; ++row) {
                    result[row] = static_cast<double>(
                        sums[row - begin] / static_cast<long double>(trees.size())
                    );
                }
            });
    }

    for (double prediction : result) {
        if (!isfinite(prediction)) {
            throw InvalidParameterError(
                "Предсказание случайного леса вышло за численный диапазон"
            );
        }
    }

    return result;
}

// RANDOM FOREST CLASSIFIER

RandomForestClassifier::RandomForestClassifier(
    RandomForestParams params
)
    : params(move(params))
{
    validateParams(this->params);

    if (this->params.max_depth.has_value() &&
        *this->params.max_depth > 64) {
        throw InvalidParameterError(
            "Максимальная глубина дерева не должна превышать 64"
        );
    }
}

void RandomForestClassifier::fit(
    const Matrix& features,
    span<const double> target
) {
    const size_t sampleCount = features.rows();
    const size_t featureCountNew = features.columns();

    validateTrainingSize(params, sampleCount);

    if (featureCountNew == 0) {
        throw InvalidParameterError(
            "Матрица признаков не должна быть пустой"
        );
    }

    if (sampleCount != target.size()) {
        throw DimensionMismatchError(
            "Количество объектов и целевых значений не совпадает"
        );
    }

    const FeatureSearchParams featureSearch =
        params.max_features.value_or(
            FeatureSearchParams{FeatureMode::Sqrt, 1}
        );

    validateFeatureSearch(featureSearch, featureCountNew);

    const size_t bootstrapSize =
        getBootstrapSize(params, sampleCount);

    vector<unique_ptr<CartClassifier>> newTrees(params.n_estimators);

    parallelFor(params.n_estimators, params.n_jobs,
        [&](size_t i, size_t) {
        const uint32_t treeSeed =
            makeTreeSeed(params.random_state, i);

        mt19937 generator(treeSeed);

        vector<size_t> counts;

        if (params.bootstrap) {
            counts = makeBootstrapCounts(
                sampleCount,
                bootstrapSize,
                generator
            );
        } else {
            counts = makeFullSampleCounts(sampleCount);
        }

        auto tree = make_unique<CartClassifier>(
            params.max_depth,
            params.min_samples_split,
            params.min_samples_leaf,
            featureSearch,
            treeSeed
        );

        tree->fit_with_counts(features, target, counts);

        newTrees[i] = move(tree);
    });

    // Получаем общий список классов.
    vector<double> newClassLabels(
        newTrees.front()->classes().begin(),
        newTrees.front()->classes().end()
    );

    trees = move(newTrees);
    classLabels = move(newClassLabels);
    featureCount = featureCountNew;
    fitted = true;
}

Matrix RandomForestClassifier::predict_proba(
    const Matrix& features
) const {
    if (!fitted || trees.empty()) {
        throw ModelUsageError(
            "Случайный лес ещё не обучен"
        );
    }

    if (features.columns() != featureCount) {
        throw DimensionMismatchError(
            "Количество признаков не совпадает с обучающей выборкой"
        );
    }

    Matrix result(features.rows(), classLabels.size(), features.layout());
    const size_t workerCount = min(params.n_jobs, features.rows());

    if (workerCount <= 1) {
        for (const auto& tree : trees) {
            const Matrix probabilities = tree->predict_proba(features);
            for (size_t i = 0; i < probabilities.rows(); ++i) {
                for (size_t j = 0; j < probabilities.columns(); ++j) {
                    result(i, j) += probabilities(i, j);
                }
            }
        }
        const double treeCount = static_cast<double>(trees.size());
        for (size_t i = 0; i < result.rows(); ++i) {
            for (size_t j = 0; j < result.columns(); ++j) {
                result(i, j) /= treeCount;
            }
        }
    } else {
        parallelFor(workerCount, workerCount,
            [&](size_t workerId, size_t) {
                const size_t begin = features.rows() * workerId / workerCount;
                const size_t end = features.rows() * (workerId + 1) / workerCount;
                Matrix batch(end - begin, featureCount, features.layout());

                for (size_t row = begin; row < end; ++row) {
                    for (size_t column = 0; column < featureCount; ++column) {
                        batch(row - begin, column) = features(row, column);
                    }
                }

                Matrix sums(end - begin, classLabels.size());
                for (const auto& tree : trees) {
                    const Matrix probabilities = tree->predict_proba(batch);
                    for (size_t i = 0; i < probabilities.rows(); ++i) {
                        for (size_t j = 0; j < probabilities.columns(); ++j) {
                            sums(i, j) += probabilities(i, j);
                        }
                    }
                }

                const double treeCount = static_cast<double>(trees.size());
                for (size_t row = begin; row < end; ++row) {
                    for (size_t column = 0; column < classLabels.size(); ++column) {
                        result(row, column) =
                            sums(row - begin, column) / treeCount;
                    }
                }
            });
    }

    return result;
}

vector<double> RandomForestClassifier::predict(
    const Matrix& features
) const {
    const Matrix probabilities = predict_proba(features);

    vector<double> predictions;
    predictions.reserve(probabilities.rows());

    for (size_t i = 0; i < probabilities.rows(); ++i) {
        size_t bestClass = 0;

        for (size_t j = 1; j < probabilities.columns(); ++j) {
            if (probabilities(i, j) >
                probabilities(i, bestClass)) {
                bestClass = j;
            }
        }

        predictions.push_back(classLabels[bestClass]);
    }

    return predictions;
}

span<const double> RandomForestClassifier::classes() const {
    if (!fitted) {
        throw ModelUsageError(
            "Случайный лес ещё не обучен"
        );
    }

    return classLabels;
}

vector<CartTreeSummary> RandomForestRegressor::tree_summaries() const {
    if (!fitted) {
        throw ModelUsageError("Случайный лес ещё не обучен");
    }

    vector<CartTreeSummary> summaries;
    summaries.reserve(trees.size());
    for (const auto& tree : trees) {
        summaries.push_back(tree->summary());
    }
    return summaries;
}

vector<CartTreeSummary> RandomForestClassifier::tree_summaries() const {
    if (!fitted) {
        throw ModelUsageError("Случайный лес ещё не обучен");
    }

    vector<CartTreeSummary> summaries;
    summaries.reserve(trees.size());
    for (const auto& tree : trees) {
        summaries.push_back(tree->summary());
    }
    return summaries;
}
