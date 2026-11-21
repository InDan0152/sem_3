
#pragma once

#include "model.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <span>
#include <vector>

using namespace std;

/**
 * @brief Решающий пень для регрессии или классификации.
 *
 * Критерий Variance выбирает разбиение для регрессии, а Gini — для
 * классификации. Для разбиения выбирается один признак j и порог t:
 * объекты с features(i, j) <= t попадают в левый лист, остальные — в правый.
 * Регрессионные листья предсказывают среднее откликов, классификационные —
 * наиболее частую метку. Если допустимое разбиение не найдено, модель
 * предсказывает константное среднее либо наиболее частую метку всей выборки.
 */
class DecisionStump : public IModel {
private:
    SplitCriterion criterion;
    size_t minSamplesLeaf;

    bool fitted = false;
    bool hasSplit = false;

    size_t featureCount = 0;
    size_t bestFeature = 0;
    double bestThreshold = 0.0;

    double leftPrediction = 0.0;
    double rightPrediction = 0.0;
    double constantPrediction = 0.0;

public:
    /**
     * @brief Создаёт решающий пень с заданным критерием и минимальным размером
     *        каждого листа.
     * @param criterion Variance для регрессии или Gini для классификации.
     * @param min_samples_leaf Минимальное число обучающих объектов в каждом
     *        дочернем листе; должно быть больше нуля.
     * @throws InvalidParameterError Если min_samples_leaf равен нулю или
     *         передан неизвестный критерий.
     */
    DecisionStump(
        SplitCriterion criterion,
        size_t min_samples_leaf
    );

    /**
     * @brief Обучает пень и сохраняет предсказание для листа или листьев.
     * @param features Матрица n × d обучающих признаков; n и d должны быть
     *        ненулевыми, все значения признаков — конечными.
     * @param target Конечные отклики длины n.
     * @pre Строки features соответствуют элементам target с теми же индексами.
     * @post При успехе модель готова к predict(). Если SplitFinder не находит
     *       допустимого разбиения, модель остаётся обученной с константным
     *       предсказанием.
     * @throws InvalidParameterError Для пустой выборки, нуля признаков или
     *         не конечных значений признаков/откликов.
     * @throws DimensionMismatchError Если длина target не равна числу строк.
     */
    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    /**
     * @brief Предсказывает отклик для каждой строки матрицы.
     * @param features Матрица m × d; d должно совпадать с числом признаков
     *        обучающей матрицы. Все значения должны быть конечными.
     * @return m предсказаний в порядке строк features. При отсутствии
     *         допустимого разбиения каждому объекту соответствует общее
     *         константное предсказание.
     * @pre Модель должна успешно пройти fit().
     * @throws ModelUsageError Если модель ещё не обучена.
     * @throws DimensionMismatchError Если число признаков отличается от
     *         обучающего.
     * @throws InvalidParameterError Если features содержит не конечное значение.
     */
    vector<double> predict(
        const Matrix& features
    ) const override;

    /**
     * @brief Проверяет, найдено ли допустимое разбиение после обучения.
     * @return true, если модель обучена и содержит два листа.
     * @throws ModelUsageError Если модель ещё не обучена.
     */
    bool has_split() const;

    /**
     * @brief Возвращает предсказание левого листа.
     * @throws ModelUsageError Если модель не обучена или допустимое разбиение
     *         не найдено.
     */
    double left_prediction() const;

    /**
     * @brief Возвращает предсказание правого листа.
     * @throws ModelUsageError Если модель не обучена или допустимое разбиение
     *         не найдено.
     */
    double right_prediction() const;

    /**
     * @brief Возвращает константное предсказание модели без разбиения.
     * @throws ModelUsageError Если модель ещё не обучена.
     */
    double constant_prediction() const;

    /**
     * @brief Возвращает индекс признака выбранного разбиения.
     * @return Индекс столбца обучающей матрицы, по которому выполнено разбиение.
     * @pre Обучение завершилось с найденным допустимым разбиением.
     * @throws ModelUsageError Если модель не обучена или разбиение не найдено.
     */
    size_t split_feature() const;

    /**
     * @brief Возвращает порог выбранного разбиения.
     * @return Порог t; при предсказании используется условие feature <= t для
     *         левого листа.
     * @pre Обучение завершилось с найденным допустимым разбиением.
     * @throws ModelUsageError Если модель не обучена или разбиение не найдено.
     */
    double split_threshold() const;
};
