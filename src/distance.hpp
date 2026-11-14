#pragma once

#include <span>

using namespace std;

// Базовый интерфейс метрики расстояния.
class IDistance {
public:
    virtual ~IDistance() = default;

    virtual double compute(
        span<const double> a,
        span<const double> b
    ) const = 0;
};

// Евклидова метрика.
class EuclideanDistance : public IDistance {
public:
    double compute(
        span<const double> a,
        span<const double> b
    ) const override;
};

// Манхэттенская метрика.
class ManhattanDistance : public IDistance {
public:
    double compute(
        span<const double> a,
        span<const double> b
    ) const override;
};

// Метрика Чебышёва.
class ChebyshevDistance : public IDistance {
public:
    double compute(
        span<const double> a,
        span<const double> b
    ) const override;
};