#pragma once

#include "cart.hpp"
#include "model.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

// Общие параметры случайного леса.
struct RandomForestParams {
    // Количество деревьев.
    size_t n_estimators = 100;

    // Использовать bootstrap-выборку.
    bool bootstrap = true;

    // Размер bootstrap-выборки.
    // Если не задан, используется размер исходной выборки.
    optional<size_t> max_samples = nullopt;

    // Количество признаков для поиска разделения.
    // Если не задано, выбирается автоматически:
    // Sqrt для классификации, All для регрессии.
    optional<FeatureSearchParams> max_features = nullopt;

    // Максимальная глубина деревьев.
    // Если не задана, ограничение отсутствует.
    optional<size_t> max_depth = nullopt;

    // Минимальное количество объектов для разделения.
    size_t min_samples_split = 2;

    // Минимальное количество объектов в листе.
    size_t min_samples_leaf = 1;

    // Начальное состояние генератора случайных чисел.
    uint32_t random_state = 0;

    // Количество потоков для обучения и предсказания.
    size_t n_jobs = 1;
};


// Случайный лес для задачи регрессии.
class RandomForestRegressor : public IRegressor {
private:
    RandomForestParams params;

    // Деревья, составляющие лес.
    vector<unique_ptr<CartRegressor>> trees;

    size_t featureCount = 0;
    bool fitted = false;

public:
    explicit RandomForestRegressor(
        RandomForestParams params = {}
    );

    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    vector<double> predict(
        const Matrix& features
    ) const override;

    // Сводная информация о каждом дереве после обучения.
    vector<CartTreeSummary> tree_summaries() const;
};


// Случайный лес для задачи классификации.
class RandomForestClassifier : public IClassifier {
private:
    RandomForestParams params;

    // Деревья, составляющие лес.
    vector<unique_ptr<CartClassifier>> trees;

    // Общий список классов.
    vector<double> classLabels;

    size_t featureCount = 0;
    bool fitted = false;

public:
    explicit RandomForestClassifier(
        RandomForestParams params = {}
    );

    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    vector<double> predict(
        const Matrix& features
    ) const override;

    Matrix predict_proba(
        const Matrix& features
    ) const override;

    span<const double> classes() const override;

    // Сводная информация о каждом дереве после обучения.
    vector<CartTreeSummary> tree_summaries() const;
};