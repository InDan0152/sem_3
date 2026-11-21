
#include <cassert>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "matrix.hpp"

using namespace std;

template <typename MatrixType>
concept CanGetRowSlice = requires(MatrixType&& matrix) {
    forward<MatrixType>(matrix)(size_t{}, nullopt);
};

template <typename MatrixType>
concept CanGetColumnSlice = requires(MatrixType&& matrix) {
    forward<MatrixType>(matrix)(nullopt, size_t{});
};

static_assert(CanGetRowSlice<Matrix&>);
static_assert(CanGetRowSlice<const Matrix&>);
static_assert(!CanGetRowSlice<Matrix>);
static_assert(!CanGetRowSlice<const Matrix>);
static_assert(CanGetColumnSlice<Matrix&>);
static_assert(CanGetColumnSlice<const Matrix&>);
static_assert(!CanGetColumnSlice<Matrix>);
static_assert(!CanGetColumnSlice<const Matrix>);

static_assert(is_assignable_v<
    decltype(declval<const MatrixSlice<double>&>()(size_t{})),
    double
>);
static_assert(!is_assignable_v<
    decltype(declval<const MatrixSlice<const double>&>()(size_t{})),
    double
>);
static_assert(noexcept(declval<const MatrixSlice<double>&>().as_const()));
static_assert(noexcept(declval<const MatrixSlice<const double>&>().as_const()));

void testSingleElementSlices() {
    Matrix rowMajor(
        vector<vector<double>>{{11.0, 12.0}},
        MatrixLayout::RowMajor
    );
    const auto rowMajorColumn = rowMajor(nullopt, 1);

    assert(rowMajorColumn.size() == 1);
    assert(rowMajorColumn.axis() == SliceAxis::Column);
    assert(rowMajorColumn.stride() == 2);
    assert(rowMajorColumn.is_contiguous());
    assert(rowMajorColumn(0) == 12.0);
    assert(rowMajorColumn.data() == &rowMajor(0, 1));
    assert((*rowMajorColumn).data() == rowMajorColumn.data());

    const auto constRowMajorColumn = rowMajorColumn.as_const();
    static_assert(is_same_v<decltype(constRowMajorColumn),
                            const MatrixSlice<const double>>);
    assert(constRowMajorColumn(0) == 12.0);
    assert(constRowMajorColumn.as_span().data() == rowMajorColumn.data());

    Matrix columnMajor(
        vector<vector<double>>{{21.0}, {22.0}},
        MatrixLayout::ColumnMajor
    );
    const auto columnMajorRow = columnMajor(0, nullopt);

    assert(columnMajorRow.size() == 1);
    assert(columnMajorRow.axis() == SliceAxis::Row);
    assert(columnMajorRow.stride() == 2);
    assert(columnMajorRow.is_contiguous());
    assert(columnMajorRow(0) == 21.0);
    assert(columnMajorRow.data() == &columnMajor(0, 0));
    assert(columnMajorRow.as_span().data() == columnMajorRow.data());

    const auto constColumnMajorRow = columnMajorRow.as_const();
    assert(constColumnMajorRow(0) == 21.0);
    assert((*constColumnMajorRow).data() == columnMajorRow.data());
}

int main() {
    testSingleElementSlices();

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
    span<double> dereferencedValues = *slice;
    assert(dereferencedValues.data() == values);
    assert(dereferencedValues.size() == 4);

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

    // Оба способа получения span отклоняют неплотный срез.
    bool nonContiguousError = false;

    try {
        steppedSlice.as_span();
    } catch (const NonContiguousSliceError&) {
        nonContiguousError = true;
    }

    assert(nonContiguousError);

    nonContiguousError = false;
    try {
        (void)*steppedSlice;
    } catch (const NonContiguousSliceError&) {
        nonContiguousError = true;
    }
    assert(nonContiguousError);

    const MatrixSlice<const double> constSteppedSlice =
        steppedSlice.as_const();
    assert(constSteppedSlice.data() == steppedSlice.data());
    assert(constSteppedSlice.stride() == steppedSlice.stride());
    nonContiguousError = false;
    try {
        (void)*constSteppedSlice;
    } catch (const NonContiguousSliceError&) {
        nonContiguousError = true;
    }
    assert(nonContiguousError);
    nonContiguousError = false;
    try {
        (void)constSteppedSlice.as_span();
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
    assert(constSlice.data() == slice.data());
    assert((*constSlice).data() == values);
    assert(constSlice.as_span().data() == values);

    // Константность оболочки не запрещает менять элементы изменяемого среза.
    const MatrixSlice<double> constWrapper = slice;
    constWrapper(2) = 31.0;
    assert(values[2] == 31.0);

    // Пустой срез нормализует указатель в nullptr даже при ненулевом входе.
    MatrixSlice<double> emptySlice(
        values,
        0,
        3,
        SliceAxis::Row
    );

    assert(emptySlice.empty());
    assert(emptySlice.size() == 0);
    assert(emptySlice.data() == nullptr);
    assert(emptySlice.stride() == 3);
    assert(emptySlice.is_contiguous());
    assert(emptySlice.as_span().empty());
    assert((*emptySlice).empty());
    const MatrixSlice<const double> constEmptySlice = emptySlice.as_const();
    assert(constEmptySlice.data() == nullptr);
    assert(constEmptySlice.as_span().empty());

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
    assert(row.axis() == SliceAxis::Row);
    assert(row.stride() == 1);
    assert(row.is_contiguous());
    assert(row(0) == 1.0);
    assert(row(2) == 3.0);

    // Проверка изменения исходной матрицы через срез.
    row(1) = 20.0;
    assert(rowMajor(0, 1) == 20.0);
    assert(row.data() == &rowMajor(0, 0));

    const MatrixSlice<double> constRowWrapper = row;
    constRowWrapper(1) = 21.0;
    assert(rowMajor(0, 1) == 21.0);
    row(1) = 20.0;

    // Проверка столбца.
    auto column = rowMajor(nullopt, 1);

    assert(column.size() == 2);
    assert(column.axis() == SliceAxis::Column);
    assert(column.stride() == 3);
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

    static_assert(is_same_v<decltype(constRow), MatrixSlice<const double>>);
    static_assert(is_same_v<decltype(constColumn), MatrixSlice<const double>>);
    assert(constRow(1) == 20.0);
    assert(constColumn(1) == 5.0);
    assert(constRow.data() == row.data());
    assert(constRow.as_span().data() == &rowMajor(0, 0));
    bool constColumnSpanError = false;
    try {
        (void)*constColumn;
    } catch (const NonContiguousSliceError&) {
        constColumnSpanError = true;
    }
    assert(constColumnSpanError);

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
    assert(columnMajorRow.axis() == SliceAxis::Row);
    assert(columnMajorRow.stride() == 2);
    assert(!columnMajorRow.is_contiguous());
    assert(columnMajorRow(0) == 1.0);
    assert(columnMajorRow(1) == 2.0);
    assert(columnMajorRow(2) == 3.0);

    // Проверка столбца в ColumnMajor.
    auto columnMajorColumn = columnMajor(nullopt, 1);

    assert(columnMajorColumn.size() == 2);
    assert(columnMajorColumn.axis() == SliceAxis::Column);
    assert(columnMajorColumn.stride() == 1);
    assert(columnMajorColumn.is_contiguous());
    assert(columnMajorColumn(0) == 2.0);
    assert(columnMajorColumn(1) == 5.0);

    // Проверка изменения матрицы через столбцовый срез.
    columnMajorColumn(0) = 50.0;
    assert(columnMajor(0, 1) == 50.0);
    assert(columnMajorColumn.data() == &columnMajor(0, 1));

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
