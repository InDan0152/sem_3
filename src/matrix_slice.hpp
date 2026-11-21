
#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <type_traits>

#include "exceptions.hpp"

using namespace std;

/** @brief Ось матрицы, вдоль которой построен срез. */
enum class SliceAxis {
    Row,    /**< Срез строки. */
    Column /**< Срез столбца. */
};

/**
 * @brief Невладеющее представление элементов строки или столбца матрицы.
 *
 * Срез хранит указатель, длину и шаг, но не владеет данными. Его срок жизни не
 * должен превышать срок жизни исходного буфера. При RowMajor строки, а при ColumnMajor
 * столбцы хранятся непрерывно. Непрерывность конкретного среза сообщает
 * is_contiguous(); неплотный срез можно читать по индексу, но нельзя
 * представить span без копирования.
 *
 * @tparam T Тип элемента; const T задаёт срез только для чтения.
 */
template <typename T>
class MatrixSlice {
private:
    T* elements = nullptr;
    size_t elementCount = 0;
    size_t elementStride = 1;
    SliceAxis sliceAxis = SliceAxis::Row;

public:
    /** @brief Создаёт пустой срез. */
    MatrixSlice() noexcept = default;

    /** @brief Создаёт срез с указанными данными, размером, шагом и осью. */
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

    /** @brief Возвращает число элементов среза. */
    size_t size() const noexcept {
        return elementCount;
    }

    /** @brief Проверяет, не содержит ли срез элементов. */
    bool empty() const noexcept {
        return elementCount == 0;
    }

    /** @brief Возвращает шаг между соседними элементами в исходной матрице. */
    size_t stride() const noexcept {
        return elementStride;
    }

    /** @brief Возвращает ось среза. */
    SliceAxis axis() const noexcept {
        return sliceAxis;
    }

    /** @brief Проверяет, можно ли представить элементы среза непрерывным span. */
    bool is_contiguous() const noexcept {
        return elementCount <= 1 || elementStride == 1;
    }

    /** @brief Возвращает указатель на первый элемент или nullptr для пустого среза. */
    T* data() const noexcept {
        return elements;
    }

    /** @brief Возвращает элемент без проверки индекса с учётом шага среза. */
    T& operator()(size_t index) const noexcept {
        return elements[index * elementStride];
    }

    /**
     * @brief Возвращает элемент с проверкой индекса.
     * @throws out_of_range Если индекс не меньше размера среза.
     */
    T& at(size_t index) const {
        if (index >= elementCount) {
            throw out_of_range("Индекс среза вне диапазона");
        }

        return (*this)(index);
    }

    /**
     * @brief Получает span для непрерывного среза.
     * @throws NonContiguousSliceError Если срез неплотный.
     */
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

    /** @brief Получает span; условия непрерывности и исключение те же, что у operator*. */
    span<T> as_span() const {
        return **this;
    }

    /** @brief Возвращает константное представление того же невладеющего среза. */
    MatrixSlice<const remove_const_t<T>> as_const() const noexcept {
        return MatrixSlice<const remove_const_t<T>>(
            elements,
            elementCount,
            elementStride,
            sliceAxis
        );
    }
};
