
#pragma once

#include "model.hpp"
#include "distance.hpp"
#include "neighbor_search.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

// Способ взвешивания соседей.
enum class Weighting {
    Uniform,
    Distance
};

// Общая часть алгоритма kNN.
// Не является моделью и не наследуется от IModel.
class KnnSearch {
private:
    Matrix trainFeatures{0, 0};
    std::vector<double> trainTargets;

    std::size_t k;
    std::unique_ptr<IDistance> distance;
    Weighting weighting;

    bool fitted = false;

public:
    KnnSearch(
        std::size_t neighborsCount,
        std::unique_ptr<IDistance> distanceMetric,
        Weighting weightingMethod
    );

    void fit(
        const Matrix& features,
        std::span<const double> target
    );

    std::vector<Neighbor> findNeighbors(
        std::span<const double> query
    ) const;

    std::vector<double> getWeights(
        const std::vector<Neighbor>& neighbors
    ) const;

    double getTarget(std::size_t index) const;

    std::span<const double> targets() const;

    std::size_t featureCount() const;

    bool isFitted() const noexcept;
};

// kNN для задачи регрессии.
class KnnRegressor : public IRegressor {
private:
    KnnSearch search;

public:
    KnnRegressor(
        std::size_t k,
        std::unique_ptr<IDistance> distance,
        Weighting weighting = Weighting::Uniform
    );

    void fit(
        const Matrix& features,
        std::span<const double> target
    ) override;

    std::vector<double> predict(
        const Matrix& features
    ) const override;
};

// kNN для задачи классификации.
class KnnClassifier : public IClassifier {
private:
    KnnSearch search;
    std::vector<double> classLabels;

public:
    KnnClassifier(
        std::size_t k,
        std::unique_ptr<IDistance> distance,
        Weighting weighting = Weighting::Uniform
    );

    void fit(
        const Matrix& features,
        std::span<const double> target
    ) override;

    std::vector<double> predict(
        const Matrix& features
    ) const override;

    Matrix predict_proba(
        const Matrix& features
    ) const override;

    std::span<const double> classes() const override;
};
