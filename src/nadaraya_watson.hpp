
#pragma once

#include "model.hpp"
#include "distance.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

// Общая часть алгоритма Nadaraya-Watson.
class NadarayaWatsonSearch {
private:
    Matrix trainFeatures{0, 0};
    std::vector<double> trainTargets;

    double bandwidth;
    std::unique_ptr<IDistance> distance;

    bool fitted = false;

public:
    NadarayaWatsonSearch(
        double bandwidth,
        std::unique_ptr<IDistance> distance
    );

    void fit(
        const Matrix& features,
        std::span<const double> target
    );

    std::vector<double> getWeights(
        std::span<const double> query
    ) const;

    double getTarget(std::size_t index) const;

    std::size_t featureCount() const;

    bool isFitted() const noexcept;
};

// Nadaraya-Watson для регрессии.
class NadarayaWatsonRegressor : public IRegressor {
private:
    NadarayaWatsonSearch search;

public:
    NadarayaWatsonRegressor(
        double bandwidth,
        std::unique_ptr<IDistance> distance
    );

    void fit(
        const Matrix& features,
        std::span<const double> target
    ) override;

    std::vector<double> predict(
        const Matrix& features
    ) const override;
};

// Nadaraya-Watson для классификации.
class NadarayaWatsonClassifier : public IClassifier {
private:
    NadarayaWatsonSearch search;
    std::vector<double> classLabels;

public:
    NadarayaWatsonClassifier(
        double bandwidth,
        std::unique_ptr<IDistance> distance
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
