
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "matrix.hpp"

using namespace std;

int main() {
    // Проверка обычного среза.
    double values[] = {10.0, 20.0, 30.0, 40.0};

    MatrixSlice<double> slice(
        values,
        4,
        1,
        SliceAxis::Row
    );

    assert(slice.size() == 4);
    assert(!slice.empty());
    assert(slice.stride() == 1);
    assert(slice.axis() == SliceAxis::Row);
    assert(slice.is_contiguous());
    assert(slice.data() == values);

    // Проверка чтения и изменения элементов.
    assert(slice(0) == 10.0);
    slice(1) = 25.0;
    assert(values[1] == 25.0);

    // Проверка получения span.
    span<double> valuesSpan = slice.as_span();
    assert(valuesSpan.size() == 4);
    assert(valuesSpan[2] == 30.0);

    // Проверка среза с шагом 2.
    MatrixSlice<double> steppedSlice(
        values,
        2,
        2,
        SliceAxis::Column
    );

    assert(steppedSlice.size() == 2);
    assert(steppedSlice.stride() == 2);
    assert(!steppedSlice.is_contiguous());
    assert(steppedSlice(0) == 10.0);
    assert(steppedSlice(1) == 30.0);

    // Проверка запрета получения span для непрерывного среза.
    bool nonContiguousError = false;

    try {
        steppedSlice.as_span();
    } catch (const NonContiguousSliceError&) {
        nonContiguousError = true;
    }

    assert(nonContiguousError);

    // Проверка выхода за границы.
    bool outOfRangeError = false;

    try {
        slice.at(4);
    } catch (const out_of_range&) {
        outOfRangeError = true;
    }

    assert(outOfRangeError);

    // Проверка константного представления.
    MatrixSlice<const double> constSlice = slice.as_const();

    assert(constSlice.size() == 4);
    assert(constSlice(1) == 25.0);

    // Проверка пустого среза.
    MatrixSlice<double> emptySlice(
        nullptr,
        0,
        1,
        SliceAxis::Row
    );

    assert(emptySlice.empty());
    assert(emptySlice.size() == 0);
    assert(emptySlice.data() == nullptr);
    assert(emptySlice.is_contiguous());
    assert(emptySlice.as_span().empty());

    // Проверка запрета нулевого шага.
    bool invalidStrideError = false;

    try {
        MatrixSlice<double> invalidSlice(
            values,
            2,
            0,
            SliceAxis::Row
        );
    } catch (const invalid_argument&) {
        invalidStrideError = true;
    }

    assert(invalidStrideError);


    // Проверка матрицы с построчным хранением.
    Matrix rowMajor(
        vector<vector<double>>{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        },
        MatrixLayout::RowMajor
    );

    assert(rowMajor.rows() == 2);
    assert(rowMajor.columns() == 3);
    assert(rowMajor.layout() == MatrixLayout::RowMajor);

    // Проверка доступа к отдельным элементам.
    assert(rowMajor(0, 0) == 1.0);
    assert(rowMajor(1, 2) == 6.0);

    // Проверка строки.
    auto row = rowMajor(0, nullopt);

    assert(row.size() == 3);
    assert(row.is_contiguous());
    assert(row(0) == 1.0);
    assert(row(2) == 3.0);

    // Проверка изменения исходной матрицы через срез.
    row(1) = 20.0;
    assert(rowMajor(0, 1) == 20.0);

    // Проверка столбца.
    auto column = rowMajor(nullopt, 1);

    assert(column.size() == 2);
    assert(!column.is_contiguous());
    assert(column(0) == 20.0);
    assert(column(1) == 5.0);

    // Проверка запрета получения span из столбца.
    bool columnSpanError = false;

    try {
        column.as_span();
    } catch (const NonContiguousSliceError&) {
        columnSpanError = true;
    }

    assert(columnSpanError);

    // Проверка константной матрицы.
    const Matrix& constRowMajor = rowMajor;
    auto constRow = constRowMajor(0, nullopt);
    auto constColumn = constRowMajor(nullopt, 1);

    assert(constRow(1) == 20.0);
    assert(constColumn(1) == 5.0);

    // Проверка старого интерфейса доступа к строкам.
    assert(rowMajor[0][0] == 1.0);
    assert(rowMajor.at(1)[2] == 6.0);

    // Проверка матрицы с постолбцовым хранением.
    Matrix columnMajor(
        vector<vector<double>>{
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        },
        MatrixLayout::ColumnMajor
    );

    assert(columnMajor.rows() == 2);
    assert(columnMajor.columns() == 3);
    assert(columnMajor.layout() == MatrixLayout::ColumnMajor);

    // Проверка строки в ColumnMajor.
    auto columnMajorRow = columnMajor(0, nullopt);

    assert(columnMajorRow.size() == 3);
    assert(!columnMajorRow.is_contiguous());
    assert(columnMajorRow(0) == 1.0);
    assert(columnMajorRow(1) == 2.0);
    assert(columnMajorRow(2) == 3.0);

    // Проверка столбца в ColumnMajor.
    auto columnMajorColumn = columnMajor(nullopt, 1);

    assert(columnMajorColumn.size() == 2);
    assert(columnMajorColumn.is_contiguous());
    assert(columnMajorColumn(0) == 2.0);
    assert(columnMajorColumn(1) == 5.0);

    // Проверка изменения матрицы через столбцовый срез.
    columnMajorColumn(0) = 50.0;
    assert(columnMajor(0, 1) == 50.0);

    // Проверка эквивалентности значений двух вариантов хранения.
    for (size_t i = 0; i < rowMajor.rows(); ++i) {
        for (size_t j = 0; j < rowMajor.columns(); ++j) {
            if (i == 0 && j == 1) {
                assert(rowMajor(i, j) == 20.0);
                assert(columnMajor(i, j) == 50.0);
            } else {
                assert(rowMajor(i, j) == columnMajor(i, j));
            }
        }
    }

    // Проверка запрета span для строки ColumnMajor.
    bool rowSpanError = false;

    try {
        columnMajor.at(0);
    } catch (const NonContiguousSliceError&) {
        rowSpanError = true;
    }

    assert(rowSpanError);

    // Проверка выхода за границы матрицы.
    bool matrixOutOfRange = false;

    try {
        rowMajor.at(2, 0);
    } catch (const out_of_range&) {
        matrixOutOfRange = true;
    }

    assert(matrixOutOfRange);

    cout << "Все тесты MatrixSlice пройдены!" << endl;

    return 0;
}
