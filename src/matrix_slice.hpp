
#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "exceptions.hpp"

using namespace std;

// Ось матричного среза.
enum class SliceAxis {
    Row,
    Column
};

// Невладеющий срез элементов матрицы.
template <typename T>
class MatrixSlice {
private:
    T* elements = nullptr;
    size_t elementCount = 0;
    size_t elementStride = 1;
    SliceAxis sliceAxis = SliceAxis::Row;

public:
    MatrixSlice() noexcept = default;

    // Создание среза.
    MatrixSlice(
        T* data,
        size_t size,
        size_t stride,
        SliceAxis axis
    )
        : elements(size == 0 ? nullptr : data),
          elementCount(size),
          elementStride(stride),
          sliceAxis(axis) {

        if (stride == 0) {
            throw invalid_argument("Шаг среза не может быть равен нулю");
        }

        if (size > 0 && data == nullptr) {
            throw invalid_argument(
                "Непустой срез не может иметь нулевой указатель"
            );
        }
    }

    // Количество элементов.
    size_t size() const noexcept {
        return elementCount;
    }

    // Проверка пустоты.
    bool empty() const noexcept {
        return elementCount == 0;
    }

    // Шаг между элементами.
    size_t stride() const noexcept {
        return elementStride;
    }

    // Ось среза.
    SliceAxis axis() const noexcept {
        return sliceAxis;
    }

    // Проверка непрерывного расположения элементов.
    bool is_contiguous() const noexcept {
        return elementCount <= 1 || elementStride == 1;
    }

    // Указатель на первый элемент.
    T* data() const noexcept {
        return elements;
    }

    // Доступ к элементу по индексу.
    T& operator()(size_t index) const noexcept {
        return elements[index * elementStride];
    }

    // Проверяемый доступ к элементу.
    T& at(size_t index) const {
        if (index >= elementCount) {
            throw out_of_range("Индекс среза вне диапазона");
        }

        return (*this)(index);
    }

    // Получение span.
    span<T> operator*() const {
        if (!is_contiguous()) {
            throw NonContiguousSliceError(
                "Невозможно получить span из непрерывного среза"
            );
        }

        if (empty()) {
            return {};
        }

        return span<T>(elements, elementCount);
    }

    // Явное получение span.
    span<T> as_span() const {
        return **this;
    }

    // Получение константного представления среза.
    MatrixSlice<const remove_const_t<T>> as_const() const {
        return MatrixSlice<const remove_const_t<T>>(
            elements,
            elementCount,
            elementStride,
            sliceAxis
        );
    }
};
