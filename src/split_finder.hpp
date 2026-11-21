
#pragma once

#include "matrix.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

/**
 * @brief Критерий оценки разбиения для регрессии или классификации.
 */
enum class SplitCriterion {
    Variance, /**< Регрессионный критерий: максимизируется сумма S_L²/n_L + S_R²/n_R. */
    Gini      /**< Классификационный критерий: максимизируется A_L/n_L + A_R/n_R. */
};

/**
 * @brief Найденное бинарное разбиение.
 */
struct Split {
    std::size_t featureIndex; /**< Индекс признака матрицы, по которому делится выборка. */
    double threshold;         /**< Порог: слева feature <= threshold, справа — больше. */
    long double score;        /**< Оценка критерия; SplitFinder выбирает максимальную. */
};

/**
 * @brief Общий поиск лучшего допустимого бинарного разбиения.
 *
 * При равной оценке выбирается меньший индекс признака, затем меньший порог.
 * Для каждого рассматриваемого признака значения объектов сортируются за
 * O(n log n), после чего границы проверяются за O(n). Для f признаков и n
 * индексов объектов поиск занимает O(f n log n) времени, не считая входной
 * проверки данных.
 */
class SplitFinder {
public:
    /**
     * @brief Ищет лучшее разбиение только по разрешённому подмножеству признаков.
     * @param features Матрица признаков; target должен содержать по одному
     *        значению на каждую её строку.
     * @param target Отклики, индексируемые глобальными индексами строк features.
     * @param objectIds Индексы объектов матрицы, составляющих текущую выборку
     *        (например, объекты узла CART); каждый индекс должен быть в границах
     *        строк матрицы.
     * @param featureIds Индексы признаков, среди которых разрешён поиск; каждый
     *        индекс должен быть в границах столбцов матрицы.
     * @param criterion Критерий Variance или Gini.
     * @param minSamplesLeaf Минимальное число объектов в каждой стороне
     *        допустимого разбиения; должно быть больше нуля.
     * @return Лучшее допустимое разбиение или std::nullopt, если объектных
     *         индексов меньше двух, список разрешённых признаков пуст либо ни
     *         одна граница не удовлетворяет ограничению листа.
     * @pre Длина target равна числу строк features; отклики выбранных объектов
     *      и значения выбранных признаков конечны.
     * @throws InvalidParameterError При нулевом minSamplesLeaf, неизвестном
     *         критерии, индексах вне границ, не конечных входных значениях или
     *         не конечной оценке разбиения.
     * @throws DimensionMismatchError Если размер target не совпадает с числом
     *         строк features.
     */
    static std::optional<Split> findBestSplit(
        const Matrix& features,
        std::span<const double> target,
        const std::vector<std::size_t>& objectIds,
        const std::vector<std::size_t>& featureIds,
        SplitCriterion criterion,
        std::size_t minSamplesLeaf
    );

    /**
     * @brief Ищет лучшее допустимое разбиение по всем столбцам features.
     * @param features Матрица признаков; target должен содержать по одному
     *        значению на каждую её строку.
     * @param target Отклики, индексируемые глобальными индексами строк features.
     * @param objectIds Индексы объектов матрицы для поиска; каждый индекс должен
     *        быть в границах строк матрицы.
     * @param criterion Критерий Variance или Gini.
     * @param minSamplesLeaf Минимальное число объектов в каждой стороне
     *        допустимого разбиения; должно быть больше нуля.
     * @return Лучшее допустимое разбиение или std::nullopt, если объектов меньше
     *         двух, в матрице нет признаков либо нет допустимой границы.
     * @pre Длина target равна числу строк features; отклики выбранных объектов
     *      и значения всех признаков выбранных объектов конечны.
     * @throws InvalidParameterError При нулевом minSamplesLeaf, неизвестном
     *         критерии, недопустимом индексе объекта, не конечных входных
     *         значениях или не конечной оценке разбиения.
     * @throws DimensionMismatchError Если размер target не совпадает с числом
     *         строк features.
     */
    static std::optional<Split> findBestSplit(
        const Matrix& features,
        std::span<const double> target,
        const std::vector<std::size_t>& objectIds,
        SplitCriterion criterion,
        std::size_t minSamplesLeaf
    );
};
