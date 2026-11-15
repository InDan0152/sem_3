
#pragma once

#include "matrix.hpp"

#include <span>
#include <vector>

using namespace std;

// Базовый интерфейс модели.
class IModel {
public:
    virtual ~IModel() = default;

    virtual void fit(
        const Matrix& features,
        span<const double> target
    ) = 0;

    virtual vector<double> predict(
        const Matrix& features
    ) const = 0;
};

// Интерфейс регрессора.
class IRegressor : public IModel {
public:
    ~IRegressor() override = default;
};

// Интерфейс классификатора.
class IClassifier : public IModel {
public:
    ~IClassifier() override = default;

    // Вероятности принадлежности объектов классам.
    virtual Matrix predict_proba(
        const Matrix& features
    ) const = 0;

    // Список классов в порядке возрастания.
    virtual span<const double> classes() const = 0;
};
