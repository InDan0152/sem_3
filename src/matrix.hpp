
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "matrix_slice.hpp"

using namespace std;

/** @brief Порядок размещения элементов Matrix в памяти. */
enum class MatrixLayout {
    RowMajor,    /**< Элементы каждой строки размещены подряд. */
    ColumnMajor /**< Элементы каждого столбца размещены подряд. */
};

/**
 * @brief Владеющая матрица числовых значений с порядком хранения RowMajor или
 *        ColumnMajor.
 *
 * Строки и столбцы можно получать как невладеющие MatrixSlice при любом
 * порядке хранения. Неплотный срез нельзя преобразовать в span без копирования.
 */
class Matrix {
private:
    size_t rowCount;
    size_t columnCount;
    vector<double> data;
    MatrixLayout storageLayout;

    // Вычисление позиции элемента в одномерном массиве.
    size_t index(size_t row, size_t column) const noexcept;

public:
    /** @brief Создаёт матрицу заданного размера, заполненную нулями. */
    Matrix(
        size_t rows,
        size_t columns,
        MatrixLayout layout = MatrixLayout::RowMajor
    );

    /** @brief Создаёт матрицу-копию прямоугольного двумерного вектора. */
    explicit Matrix(
        const vector<vector<double>>& values,
        MatrixLayout layout = MatrixLayout::RowMajor
    );

    /** @brief Возвращает число строк. */
    size_t rows() const noexcept;
    /** @brief Возвращает число столбцов. */
    size_t columns() const noexcept;

    /** @brief Возвращает порядок хранения элементов матрицы. */
    MatrixLayout layout() const noexcept;

    /** @brief Возвращает элемент без проверки индексов; матрица должна быть lvalue. */
    double& operator()(size_t row, size_t column) & noexcept;
    /** @brief Возвращает константную ссылку на элемент без проверки индексов. */
    const double& operator()(size_t row, size_t column) const & noexcept;

    /** @brief Создаёт изменяемый невладеющий срез строки при любом layout. */
    MatrixSlice<double> operator()(
        size_t row,
        nullopt_t
    ) &;

    /** @brief Создаёт константный невладеющий срез строки при любом layout. */
    MatrixSlice<const double> operator()(
        size_t row,
        nullopt_t
    ) const &;

    /** @brief Создаёт изменяемый невладеющий срез столбца при любом layout. */
    MatrixSlice<double> operator()(
        nullopt_t,
        size_t column
    ) &;

    /** @brief Создаёт константный невладеющий срез столбца при любом layout. */
    MatrixSlice<const double> operator()(
        nullopt_t,
        size_t column
    ) const &;

    /** @brief Срезы из временной матрицы запрещены, чтобы не допустить висячий срез. */
    MatrixSlice<double> operator()(
        size_t row,
        nullopt_t
    ) && = delete;

    MatrixSlice<const double> operator()(
        size_t row,
        nullopt_t
    ) const && = delete;

    MatrixSlice<double> operator()(
        nullopt_t,
        size_t column
    ) && = delete;

    MatrixSlice<const double> operator()(
        nullopt_t,
        size_t column
    ) const && = delete;

    /** @brief Доступ к элементу временной матрицы запрещён. */
    double& operator()(size_t row, size_t column) && = delete;
    const double& operator()(size_t row, size_t column) const && = delete;

    /**
     * @brief Возвращает span строки без проверки индекса.
     * @throws NonContiguousSliceError Если матрица хранится в ColumnMajor.
     */
    span<double> operator[](size_t row);
    /**
     * @brief Возвращает константный span строки без проверки индекса.
     * @throws NonContiguousSliceError Если матрица хранится в ColumnMajor.
     */
    span<const double> operator[](size_t row) const;

    /**
     * @brief Возвращает проверяемый span строки.
     * @throws out_of_range Если индекс строки вне диапазона.
     * @throws NonContiguousSliceError Если матрица хранится в ColumnMajor.
     */
    span<double> at(size_t row);
    /**
     * @brief Константный вариант проверяемого доступа к строке.
     * @throws out_of_range Если индекс строки вне диапазона.
     * @throws NonContiguousSliceError Если матрица хранится в ColumnMajor.
     */
    span<const double> at(size_t row) const;

    /**
     * @brief Возвращает проверяемую ссылку на элемент матрицы.
     * @throws out_of_range Если индекс строки или столбца вне диапазона.
     */
    double& at(size_t row, size_t column);
    /**
     * @brief Возвращает проверяемую константную ссылку на элемент матрицы.
     * @throws out_of_range Если индекс строки или столбца вне диапазона.
     */
    const double& at(size_t row, size_t column) const;
};
