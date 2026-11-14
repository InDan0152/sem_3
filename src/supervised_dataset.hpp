#pragma once

#include "dataset.hpp"
#include <map>
#include <string>
using namespace std;

SupervisedDataset prepareSupervisedDataset(
    const Dataset& dataset,
    const string& targetName,
    const map<string, double>& categoricalTargetMapping = {}
);