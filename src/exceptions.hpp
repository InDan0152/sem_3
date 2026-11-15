#pragma once

#include <stdexcept>
#include <string>

using namespace std;

// Базовое исключение проекта.
class ProjectException : public runtime_error {
public:
    explicit ProjectException(const string& message)
        : runtime_error(message) {}
};

// Ошибка несовпадения размерностей.
class DimensionMismatchError : public invalid_argument {
public:
    explicit DimensionMismatchError(const string& message)
        : invalid_argument(message) {}
};

// Ошибка некорректного использования модели.
class ModelUsageError : public runtime_error {
public:
    explicit ModelUsageError(const string& message)
        : runtime_error(message) {}
};

// Ошибка некорректных параметров.
class InvalidParameterError : public invalid_argument {
public:
    explicit InvalidParameterError(const string& message)
        : invalid_argument(message) {}
};

// Ошибка попытки получить span из непрерывно не расположенных элементов.
class NonContiguousSliceError : public logic_error {
public:
    explicit NonContiguousSliceError(const string& message)
        : logic_error(message) {}
};