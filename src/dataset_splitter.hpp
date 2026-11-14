#pragma once

#include "dataset.hpp"

#include <cstddef>
#include <cstdint>

using namespace std;

DatasetSplit splitDataset(
    const SupervisedDataset& dataset,
    double testRatio = 0.2,
    uint32_t seed = 42
);