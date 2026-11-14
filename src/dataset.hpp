#pragma once

#include <optional>
#include <string>
#include <vector>

using namespace std;

struct CsvConfig {
    char delimiter = ';';
    string missingValue = "unknown";
    bool hasHeader = true;
};

enum class FeatureType {
    Numerical,
    Categorical
};


struct Column {
    string name;
    FeatureType type;

    vector<optional<double>> numericalValues;
    vector<optional<string>> categoricalValues;
};

struct Dataset {
    vector<Column> columns;
};

struct SupervisedDataset {
    Dataset features;
    vector<double> targets;
};

struct DatasetSplit {
    SupervisedDataset train;
    SupervisedDataset test;
};