
#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "matrix_slice.hpp"

using namespace std;

// Порядок хранения элементов матрицы.
enum class MatrixLayout {
    RowMajor,
    ColumnMajor
};

// Матрица числовых признаков.
class Matrix {
private:
    size_t rowCount;
    size_t columnCount;
    vector<double> data;
    MatrixLayout storageLayout;

    // Вычисление позиции элемента в одномерном массиве.
    size_t index(size_t row, size_t column) const noexcept;

public:
    // Создание матрицы заданного размера.
    Matrix(
        size_t rows,
        size_t columns,
        MatrixLayout layout = MatrixLayout::RowMajor
    );

    // Создание матрицы из двумерного вектора.
    explicit Matrix(
        const vector<vector<double>>& values,
        MatrixLayout layout = MatrixLayout::RowMajor
    );

    size_t rows() const noexcept;
    size_t columns() const noexcept;

    MatrixLayout layout() const noexcept;

    // Доступ к отдельному элементу.
    double& operator()(size_t row, size_t column) & noexcept;
    const double& operator()(size_t row, size_t column) const & noexcept;

    // Получение строки.
    MatrixSlice<double> operator()(
        size_t row,
        nullopt_t
    ) &;

    MatrixSlice<const double> operator()(
        size_t row,
        nullopt_t
    ) const &;

    // Получение столбца.
    MatrixSlice<double> operator()(
        nullopt_t,
        size_t column
    ) &;

    MatrixSlice<const double> operator()(
        nullopt_t,
        size_t column
    ) const &;

    // Запрещаем получение срезов из временной матрицы.
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

    // Доступ к отдельному элементу.
    double& operator()(size_t row, size_t column) && = delete;
    const double& operator()(size_t row, size_t column) const && = delete;

    // Старый интерфейс доступа к строкам.
    span<double> operator[](size_t row);
    span<const double> operator[](size_t row) const;

    span<double> at(size_t row);
    span<const double> at(size_t row) const;

    double& at(size_t row, size_t column);
    const double& at(size_t row, size_t column) const;
};
