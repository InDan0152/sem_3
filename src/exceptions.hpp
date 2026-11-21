#pragma once

#include <stdexcept>
#include <string>

using namespace std;

/**
 * @brief Общее исключение проекта на основе std::runtime_error.
 *
 * Специализированные исключения ниже напрямую наследуют стандартные типы
 * исключений и не являются производными от ProjectException.
 */
class ProjectException : public runtime_error {
public:
    /** @brief Создаёт исключение с поясняющим сообщением. */
    explicit ProjectException(const string& message)
        : runtime_error(message) {}
};

/**
 * @brief Исключение для несовпадения размеров связанных данных.
 *
 * Используется, например, когда число откликов не совпадает с числом строк
 * матрицы или число признаков при предсказании отличается от обучающего.
 */
class DimensionMismatchError : public invalid_argument {
public:
    /** @brief Создаёт исключение с поясняющим сообщением. */
    explicit DimensionMismatchError(const string& message)
        : invalid_argument(message) {}
};

/**
 * @brief Исключение для недопустимого вызова модели в её текущем состоянии.
 *
 * Используется, например, при предсказании или запросе сводки до обучения.
 */
class ModelUsageError : public runtime_error {
public:
    /** @brief Создаёт исключение с поясняющим сообщением. */
    explicit ModelUsageError(const string& message)
        : runtime_error(message) {}
};

/**
 * @brief Исключение для недопустимых параметров или входных числовых данных.
 *
 * Используется при некорректных настройках моделей, пустой обучающей выборке
 * и не конечных значениях признаков или откликов.
 */
class InvalidParameterError : public invalid_argument {
public:
    /** @brief Создаёт исключение с поясняющим сообщением. */
    explicit InvalidParameterError(const string& message)
        : invalid_argument(message) {}
};

/**
 * @brief Исключение при запросе непрерывного span для неплотных данных.
 *
 * Используется MatrixSlice при получении span для неплотного среза и Matrix
 * при запросе span строки в ColumnMajor.
 */
class NonContiguousSliceError : public logic_error {
public:
    /** @brief Создаёт исключение с поясняющим сообщением. */
    explicit NonContiguousSliceError(const string& message)
        : logic_error(message) {}
};
