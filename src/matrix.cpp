#include "matrix.hpp"

#include <stdexcept>

using namespace std;

// Создание матрицы заданного размера.
Matrix::Matrix(size_t rows, size_t columns)
    : rowCount(rows),
      columnCount(columns) {

    // Проверяем переполнение до выделения памяти.
    if (columns != 0 &&
        rows > data.max_size() / columns) {
        throw length_error("Размер матрицы слишком велик");
    }

    data.assign(rows * columns, 0.0);
}

// Создание матрицы из двумерного вектора.
Matrix::Matrix(const vector<vector<double>>& values)
    : rowCount(values.size()),
      columnCount(values.empty() ? 0 : values[0].size()) {

    // Проверяем одинаковое количество элементов во всех строках.
    for (const auto& row : values) {
        if (row.size() != columnCount) {
            throw invalid_argument(
                "Все строки матрицы должны иметь одинаковую длину"
            );
        }
    }

   // Проверяем переполнение перед выделением памяти.
    if (columnCount != 0 &&
        rowCount > data.max_size() / columnCount) {
        throw length_error("Размер матрицы слишком велик");
    }

    // Переносим данные в плоский массив.
    data.reserve(rowCount * columnCount);

    for (const auto& row : values) {
        data.insert(data.end(), row.begin(), row.end());
    }
}

// Получение количества строк.
size_t Matrix::rows() const noexcept {
    return rowCount;
}

// Получение количества столбцов.
size_t Matrix::columns() const noexcept {
    return columnCount;
}

// Неконстантный доступ к строке.
span<double> Matrix::operator[](size_t row) {
    if (columnCount == 0) {return {};}
    return span<double>(
        data.data() + row * columnCount,
        columnCount
    );
}

// Константный доступ к строке.
span<const double> Matrix::operator[](size_t row) const {
    if (columnCount == 0) {return {};}
    return span<const double>(
        data.data() + row * columnCount,
        columnCount
    );
}

// Проверяемый доступ к строке.
span<double> Matrix::at(size_t row) {
    if (row >= rowCount) {
        throw out_of_range("Индекс строки матрицы вне диапазона");
    }

    return (*this)[row];
}

// Проверяемый константный доступ к строке.
span<const double> Matrix::at(size_t row) const {
    if (row >= rowCount) {
        throw out_of_range("Индекс строки матрицы вне диапазона");
    }

    return (*this)[row];
}