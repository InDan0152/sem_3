
#pragma once

#include "model.hpp"
#include "split_finder.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

using namespace std;

/**
 * @brief Узел дерева CART.
 *
 * Лист имеет isLeaf == true и не имеет потомков. Внутренний узел имеет
 * isLeaf == false, задаёт разбиение по featureIndex и threshold и владеет
 * левым и правым потомками через unique_ptr. Для регрессии в prediction листа
 * хранится средний отклик. Для классификации classProbabilities хранит
 * распределение меток в узле в порядке CartClassifier::classes(); в листьях
 * это распределение используется для вероятностного предсказания.
 */
struct CartNode {
    bool isLeaf = true; /**< true для листа, false для узла с разбиением. */
    size_t featureIndex = 0; /**< Нулевой индекс признака внутреннего узла. */
    double threshold = 0.0; /**< Порог признака; левая ветвь принимает значение <= порога. */
    double prediction = 0.0; /**< Числовой прогноз регрессионного листа — среднее откликов. */

    vector<double> classProbabilities; /**< Частоты классов, нормированные до вероятностей; порядок совпадает с classes(). */

    unique_ptr<CartNode> left; /**< Владение левым дочерним узлом; у листа указатель пуст. */
    unique_ptr<CartNode> right; /**< Владение правым дочерним узлом; у листа указатель пуст. */
};

/**
 * @brief Описание одного внутреннего разбиения в сводке дерева.
 */
struct CartSplitDescription {
    size_t depth = 0; /**< Глубина узла с разбиением: корень имеет глубину 0. */
    size_t featureIndex = 0; /**< Нулевой индекс признака матрицы. */
    double threshold = 0.0; /**< Порог в шкале соответствующего признака. */
};

/**
 * @brief Сводные размеры и разбиения обученного дерева.
 *
 * Значения описывают фактически построенное дерево, в том числе если рост
 * остановился раньше заданной максимальной глубины.
 */
struct CartTreeSummary {
    size_t nodeCount = 0; /**< Общее число внутренних узлов и листьев. */
    size_t leafCount = 0; /**< Число листьев. */
    size_t depth = 0; /**< Максимальная глубина узла; у дерева из одного корня равна 0. */
    vector<CartSplitDescription> splits; /**< Описания всех внутренних узлов, по одному на разбиение. */
};

/**
 * @brief Дерево CART для регрессии.
 *
 * В каждом узле выбирается разбиение, максимизирующее критерий регрессии
 * SplitCriterion::Variance через общий SplitFinder. В листе возвращается
 * средний целевой отклик объектов этого листа. Рост прекращается при
 * достижении max_depth, недостаточном числе объектов, одинаковых откликах,
 * невозможности удовлетворить min_samples_leaf или отсутствии разбиения.
 */
class CartRegressor : public IRegressor {
private:
    unique_ptr<CartNode> root;

    size_t maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;
    size_t featureCount = 0;

    bool fitted = false;

    /**
     * @brief Рекурсивно строит регрессионное поддерево для заданного набора строк.
     * @param features Матрица обучающих признаков.
     * @param target Отклики, индексируемые индексами строк features.
     * @param objectIds Индексы объектов текущего узла.
     * @param featureIds Индексы признаков, разрешённых для разбиения узла.
     * @param depth Глубина строящегося узла, корень имеет глубину 0.
     * @return Владение построенным узлом с его поддеревом.
     */
    unique_ptr<CartNode> buildNode(
        const Matrix& features,
        span<const double> target,
        const vector<size_t>& objectIds,
        const vector<size_t>& featureIds,
        size_t depth
    );

public:
    /**
     * @brief Создаёт регрессионное дерево CART.
     * @param max_depth Максимальная глубина дерева; корень имеет глубину 0,
     *        значение 0 оставляет корень листом. Значение ограничено сверху 64.
     * @param min_samples_split Узел может делиться, если содержит не меньше
     *        этого числа объектов; значение должно быть не меньше 2.
     * @param min_samples_leaf Минимальное число объектов в каждой дочерней
     *        части разбиения; значение должно быть не меньше 1.
     * @throws InvalidParameterError Если max_depth больше 64,
     *         min_samples_split меньше 2 или min_samples_leaf равен 0.
     */
    CartRegressor(
        size_t max_depth = 5,
        size_t min_samples_split = 2,
        size_t min_samples_leaf = 1
    );

    /**
     * @brief Обучает регрессионное дерево по откликам.
     * @param features Матрица n × d; n и d должны быть ненулевыми, значения
     *        признаков должны быть конечными.
     * @param target Конечные числовые отклики длины n.
     * @pre Отклик с индексом i относится к строке features с индексом i.
     * @post После успешного вызова дерево доступно для predict() и summary().
     * @throws InvalidParameterError Для пустой выборки, нуля признаков,
     *         не конечных входных данных, непредставимого прогноза листа или
     *         не конечной оценки разбиения.
     * @throws DimensionMismatchError Если длина target не совпадает с числом
     *         строк features.
     */
    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    /**
     * @brief Предсказывает числовой отклик обходом дерева до листа.
     * @param features Матрица m × d; число столбцов должно совпадать с числом
     *        признаков при обучении, значения должны быть конечными.
     * @return Вектор из m средних откликов соответствующих листьев, в порядке
     *         строк features.
     * @pre Дерево должно быть обучено.
     * @throws ModelUsageError Если дерево ещё не обучено.
     * @throws DimensionMismatchError Если число признаков отличается от
     *         обучающего.
     * @throws InvalidParameterError Если features содержит не конечное значение.
     */
    vector<double> predict(
        const Matrix& features
    ) const override;

    /**
     * @brief Возвращает сводку фактически построенного дерева.
     * @return Число узлов, листьев, максимальную глубину и описания разбиений.
     * @pre Дерево должно быть обучено.
     * @throws ModelUsageError Если дерево ещё не обучено.
     */
    CartTreeSummary summary() const;
};

/**
 * @brief Дерево CART для классификации.
 *
 * Исходные метки классов сохраняются без перенумерации, а уникальные классы
 * сортируются по возрастанию. В узлах частоты классов хранятся как вероятности
 * в этом порядке. Для разбиения используется критерий SplitCriterion::Gini.
 * Предсказанный класс — метка с наибольшей вероятностью; при равенстве
 * вероятностей выбирается меньшая метка. Рост останавливается по тем же
 * ограничениям глубины, размера узла и размера листа, что и у регрессора,
 * а также при одинаковых метках или отсутствии допустимого разбиения.
 */
class CartClassifier : public IClassifier {
private:
    unique_ptr<CartNode> root;

    size_t maxDepth;
    size_t minSamplesSplit;
    size_t minSamplesLeaf;
    size_t featureCount = 0;

    vector<double> classLabels;

    bool fitted = false;

    /**
     * @brief Рекурсивно строит классификационное поддерево для заданных строк.
     * @param features Матрица обучающих признаков.
     * @param target Метки классов, индексируемые индексами строк features.
     * @param objectIds Индексы объектов текущего узла.
     * @param featureIds Индексы признаков, разрешённых для разбиения узла.
     * @param depth Глубина строящегося узла, корень имеет глубину 0.
     * @return Владение построенным узлом с его поддеревом.
     */
    unique_ptr<CartNode> buildNode(
        const Matrix& features,
        span<const double> target,
        const vector<size_t>& objectIds,
        const vector<size_t>& featureIds,
        size_t depth
    );

public:
    /**
     * @brief Создаёт классификационное дерево CART.
     * @param max_depth Максимальная глубина дерева; корень имеет глубину 0,
     *        значение 0 оставляет корень листом. Значение ограничено сверху 64.
     * @param min_samples_split Узел может делиться, если содержит не меньше
     *        этого числа объектов; значение должно быть не меньше 2.
     * @param min_samples_leaf Минимальное число объектов в каждой дочерней
     *        части разбиения; значение должно быть не меньше 1.
     * @throws InvalidParameterError Если max_depth больше 64,
     *         min_samples_split меньше 2 или min_samples_leaf равен 0.
     */
    CartClassifier(
        size_t max_depth = 5,
        size_t min_samples_split = 2,
        size_t min_samples_leaf = 1
    );

    /**
     * @brief Обучает классификационное дерево по меткам классов.
     * @param features Матрица n × d; n и d должны быть ненулевыми, значения
     *        признаков должны быть конечными.
     * @param target Конечные метки классов длины n.
     * @pre Метка с индексом i относится к строке features с индексом i.
     * @post После успешного вызова дерево и отсортированный список классов
     *       доступны для предсказаний.
     * @throws InvalidParameterError Для пустой выборки, нуля признаков,
     *         не конечных входных данных или не конечной оценки разбиения.
     * @throws DimensionMismatchError Если длина target не совпадает с числом
     *         строк features.
     */
    void fit(
        const Matrix& features,
        span<const double> target
    ) override;

    /**
     * @brief Предсказывает метку класса с максимальной вероятностью.
     * @param features Матрица m × d; число столбцов должно совпадать с числом
     *        признаков при обучении, значения должны быть конечными.
     * @return Вектор из m исходных меток классов, в порядке строк features.
     *         При равных вероятностях выбирается меньшая метка.
     * @pre Дерево должно быть обучено.
     * @throws ModelUsageError Если дерево ещё не обучено.
     * @throws DimensionMismatchError Если число признаков отличается от
     *         обучающего.
     * @throws InvalidParameterError Если features содержит не конечное значение.
     */
    vector<double> predict(
        const Matrix& features
    ) const override;

    /**
     * @brief Вычисляет распределение классов в листе для каждой строки.
     * @param features Матрица m × d; число столбцов должно совпадать с числом
     *        признаков при обучении, значения должны быть конечными.
     * @return Матрица m × K вероятностей: столбец j соответствует classes()[j],
     *         строки следуют порядку объектов features. Вероятности в каждой
     *         строке суммируются до 1.
     * @pre Дерево должно быть обучено.
     * @throws ModelUsageError Если дерево ещё не обучено.
     * @throws DimensionMismatchError Если число признаков отличается от
     *         обучающего.
     * @throws InvalidParameterError Если features содержит не конечное значение.
     */
    Matrix predict_proba(
        const Matrix& features
    ) const override;

    /**
     * @brief Возвращает уникальные метки классов по возрастанию.
     * @return Невладеющий span исходных меток; элемент j соответствует столбцу
     *         j матрицы, возвращаемой predict_proba(). Повторное обучение может
     *         изменить содержимое и инвалидировать ранее полученный span.
     * @pre Дерево должно быть обучено.
     * @throws ModelUsageError Если дерево ещё не обучено.
     */
    span<const double> classes() const override;

    /**
     * @brief Возвращает сводку фактически построенного дерева.
     * @return Число узлов, листьев, максимальную глубину и описания разбиений.
     * @pre Дерево должно быть обучено.
     * @throws ModelUsageError Если дерево ещё не обучено.
     */
    CartTreeSummary summary() const;
};
