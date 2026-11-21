
#include "matrix.hpp"

#include <stdexcept>

using namespace std;

// Создание матрицы заданного размера.
Matrix::Matrix(
    size_t rows,
    size_t columns,
    MatrixLayout layout
)
    : rowCount(rows),
      columnCount(columns),
      storageLayout(layout) {

    if (columns != 0 && rows > data.max_size() / columns) {
        throw length_error("Размер матрицы слишком велик");
    }

    data.assign(rows * columns, 0.0);
}

// Создание матрицы из двумерного вектора.
Matrix::Matrix(
    const vector<vector<double>>& values,
    MatrixLayout layout
)
    : rowCount(values.size()),
      columnCount(values.empty() ? 0 : values[0].size()),
      storageLayout(layout) {

    for (const auto& row : values) {
        if (row.size() != columnCount) {
            throw invalid_argument(
                "Все строки матрицы должны иметь одинаковую длину"
            );
        }
    }

    if (columnCount != 0 &&
        rowCount > data.max_size() / columnCount) {
        throw length_error("Размер матрицы слишком велик");
    }

    data.resize(rowCount * columnCount);

    for (size_t row = 0; row < rowCount; ++row) {
        for (size_t column = 0; column < columnCount; ++column) {
            (*this)(row, column) = values[row][column];
        }
    }
}

// Вычисление позиции элемента.
size_t Matrix::index(size_t row, size_t column) const noexcept {
    if (storageLayout == MatrixLayout::RowMajor) {
        return row * columnCount + column;
    }

    return column * rowCount + row;
}

size_t Matrix::rows() const noexcept {
    return rowCount;
}

size_t Matrix::columns() const noexcept {
    return columnCount;
}

MatrixLayout Matrix::layout() const noexcept {
    return storageLayout;
}

// Неконстантный доступ к элементу.
double& Matrix::operator()(size_t row, size_t column) & noexcept {
    return data[index(row, column)];
}

// Константный доступ к элементу.
const double& Matrix::operator()(
    size_t row,
    size_t column
) const & noexcept {
    return data[index(row, column)];
}

// Получение строки неконстантной матрицы.
MatrixSlice<double> Matrix::operator()(
    size_t row,
    nullopt_t
) & {
    if (row >= rowCount) {
        throw out_of_range("Индекс строки матрицы вне диапазона");
    }

    if (storageLayout == MatrixLayout::RowMajor) {
        double* pointer = columnCount == 0
            ? nullptr
            : data.data() + row * columnCount;

        return MatrixSlice<double>(
            pointer,
            columnCount,
            1,
            SliceAxis::Row
        );
    }

    double* pointer = rowCount == 0 || columnCount == 0
        ? nullptr
        : data.data() + row;

    return MatrixSlice<double>(
        pointer,
        columnCount,
        rowCount,
        SliceAxis::Row
    );
}

// Получение строки константной матрицы.
MatrixSlice<const double> Matrix::operator()(
    size_t row,
    nullopt_t
) const & {
    if (row >= rowCount) {
        throw out_of_range("Индекс строки матрицы вне диапазона");
    }

    const double* pointer = columnCount == 0
        ? nullptr
        : data.data() + (
            storageLayout == MatrixLayout::RowMajor
                ? row * columnCount
                : row
        );

    return MatrixSlice<const double>(
        pointer,
        columnCount,
        storageLayout == MatrixLayout::RowMajor ? 1 : rowCount,
        SliceAxis::Row
    );
}

// Получение столбца неконстантной матрицы.
MatrixSlice<double> Matrix::operator()(
    nullopt_t,
    size_t column
) & {
    if (column >= columnCount) {
        throw out_of_range("Индекс столбца матрицы вне диапазона");
    }

    double* pointer = rowCount == 0
        ? nullptr
        : data.data() + (
            storageLayout == MatrixLayout::RowMajor
                ? column
                : column * rowCount
        );

    return MatrixSlice<double>(
        pointer,
        rowCount,
        storageLayout == MatrixLayout::RowMajor ? columnCount : 1,
        SliceAxis::Column
    );
}

// Получение столбца константной матрицы.
MatrixSlice<const double> Matrix::operator()(
    nullopt_t,
    size_t column
) const & {
    if (column >= columnCount) {
        throw out_of_range("Индекс столбца матрицы вне диапазона");
    }

    const double* pointer = rowCount == 0
        ? nullptr
        : data.data() + (
            storageLayout == MatrixLayout::RowMajor
                ? column
                : column * rowCount
        );

    return MatrixSlice<const double>(
        pointer,
        rowCount,
        storageLayout == MatrixLayout::RowMajor ? columnCount : 1,
        SliceAxis::Column
    );
}

// Старый интерфейс доступа к строке.
span<double> Matrix::operator[](size_t row) {
    if (storageLayout != MatrixLayout::RowMajor) {
        throw NonContiguousSliceError(
            "Строка матрицы ColumnMajor не является непрерывной"
        );
    }

    if (columnCount == 0) {
        return {};
    }

    return span<double>(data.data() + row * columnCount, columnCount);
}

span<const double> Matrix::operator[](size_t row) const {
    if (storageLayout != MatrixLayout::RowMajor) {
        throw NonContiguousSliceError(
            "Строка матрицы ColumnMajor не является непрерывной"
        );
    }

    if (columnCount == 0) {
        return {};
    }

    return span<const double>(data.data() + row * columnCount, columnCount);
}

// Проверяемый доступ к строке.
span<double> Matrix::at(size_t row) {
    if (row >= rowCount) {
        throw out_of_range("Индекс строки матрицы вне диапазона");
    }

    return (*this)[row];
}

span<const double> Matrix::at(size_t row) const {
    if (row >= rowCount) {
        throw out_of_range("Индекс строки матрицы вне диапазона");
    }

    return (*this)[row];
}

double& Matrix::at(size_t row, size_t column) {
    if (row >= rowCount || column >= columnCount) {
        throw out_of_range("Индекс элемента матрицы вне диапазона");
    }
    return (*this)(row, column);
}

const double& Matrix::at(size_t row, size_t column) const {
    if (row >= rowCount || column >= columnCount) {
        throw out_of_range("Индекс элемента матрицы вне диапазона");
    }
    return (*this)(row, column);
}
