#pragma once

#include <cstddef>
#include <span>
#include <vector>

using namespace std;

// Матрица числовых признаков с хранением в одномерном массиве.
class Matrix {
private:
    size_t rowCount;
    size_t columnCount;

    // Все элементы хранятся последовательно по строкам.
    vector<double> data;

public:
    // Создание матрицы заданного размера.
    Matrix(size_t rows, size_t columns);

    // Создание матрицы из двумерного вектора.
    explicit Matrix(const vector<vector<double>>& values);

    // Получение количества строк и столбцов.
    size_t rows() const noexcept;
    size_t columns() const noexcept;

    // Неконстантный доступ к строке без копирования.
    span<double> operator[](size_t row);

    // Константный доступ к строке без копирования.
    span<const double> operator[](size_t row) const;

    // Проверяемый доступ к строке.
    span<double> at(size_t row);
    span<const double> at(size_t row) const;
};