#pragma once

#include "dataset.hpp"

#include <string>
#include <vector>

using namespace std;

vector<string> parseCsvLine(
    const string& line,
    char delimiter
);

void loadCsv(
    const string& filename,
    Dataset& dataset,
    const CsvConfig& config
);
