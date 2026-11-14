#pragma once

#include "matrix.hpp"

#include <span>
#include <vector>

using namespace std;

// Общий интерфейс модели машинного обучения.
class IModel {
public:
    virtual ~IModel() = default;

    // Обучение модели.
    virtual void fit(
        const Matrix& features,
        span<const double> target
    ) = 0;

    // Получение предсказаний.
    virtual vector<double> predict(
        const Matrix& features
    ) const = 0;
};