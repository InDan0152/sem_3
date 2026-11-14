#include "distance.hpp"
#include "exceptions.hpp"

#include <cmath>
#include <stdexcept>

using namespace std;

// Евклидово расстояние
// d(a, b) = sqrt(sum((a[i] - b[i])^2))
double EuclideanDistance::compute(
    span<const double> a,
    span<const double> b
) const {
    if (a.size() != b.size()) {
        throw DimensionMismatchError("Размерности векторов не совпадают");
    }

    double sum = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        double difference = a[i] - b[i];
        sum += difference * difference;
    }

    return sqrt(sum);
}

// Манхэттенское расстояние
// d(a, b) = sum(|a[i] - b[i]|)
double ManhattanDistance::compute(
    span<const double> a,
    span<const double> b
) const {
    if (a.size() != b.size()) {
        throw DimensionMismatchError("Размерности векторов не совпадают");
    }

    double sum = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        sum += abs(a[i] - b[i]);
    }

    return sum;
}

// Расстояние Чебышёва
// d(a, b) = max(|a[i] - b[i]|)
double ChebyshevDistance::compute(
    span<const double> a,
    span<const double> b
) const {
    if (a.size() != b.size()) {
        throw DimensionMismatchError("Размерности векторов не совпадают");
    }

    double maxDistance = 0.0;

    for (size_t i = 0; i < a.size(); ++i) {
        double difference = abs(a[i] - b[i]);

        if (difference > maxDistance) {
            maxDistance = difference;
        }
    }

    return maxDistance;
}