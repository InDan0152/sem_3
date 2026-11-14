#pragma once

#include "matrix.hpp"
#include "distance.hpp"

#include <cstddef>
#include <span>
#include <vector>

using namespace std;

// Информация о найденном соседе.
struct Neighbor {
    double distance;
    size_t index;
};

// Поиск k ближайших соседей.
vector<Neighbor> findKNearestNeighbors(
    const Matrix& trainFeatures,
    span<const double> query,
    size_t k,
    const IDistance& distance
);