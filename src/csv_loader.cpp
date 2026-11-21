#include "csv_loader.hpp"

#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <cstdlib>
#include <stdexcept>

using namespace std;


vector<string> parseCsvLine(
    const string& line,
    char delimiter
) {

    vector<string> fields;
    string field;
    bool insideQuotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];

        if (c == '"') {
            if (insideQuotes &&
                i + 1 < line.size() &&
                line[i + 1] == '"') {
                field += '"';
                ++i;
            } else {
                insideQuotes = !insideQuotes;
            }
        } else if (c == delimiter && !insideQuotes) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }

    if (insideQuotes) {
        throw invalid_argument("CSV содержит незакрытую кавычку");
    }

    fields.push_back(field);
    return fields;
}


bool isNumber(const string& value) {
    if (value.empty()) {
        return false;
    }

    char* end = nullptr;
    strtod(value.c_str(), &end);

    return end != value.c_str() && *end == '\0';
}


void loadCsv(
    const string& filename,
    Dataset& dataset,
    const CsvConfig& config
) {

    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Не удалось открыть CSV-файл: " + filename);
    }

    dataset.columns.clear();

    string line;
    size_t lineNumber = 0;


    if (!getline(file, line)) {
        throw invalid_argument("CSV-файл пуст");
    }

    ++lineNumber;

    vector<string> firstRow;
    try {
        firstRow = parseCsvLine(line, config.delimiter);
    } catch (const invalid_argument& error) {
        throw invalid_argument(
            "Ошибка в строке " + to_string(lineNumber) + ": " + error.what()
        );
    }


    vector<string> featureNames;

    if (config.hasHeader) {
        featureNames = move(firstRow);
    } else {
        for (size_t i = 0; i < firstRow.size(); ++i) {
            featureNames.push_back("feature_" + to_string(i + 1));
        }
    }

    const size_t expectedFields = featureNames.size();


    vector<vector<string>> rawRows;


    if (!config.hasHeader) {
        rawRows.push_back(move(firstRow));
    }


    while (getline(file, line)) {
        ++lineNumber;

        if (line.empty()) {
            continue;
        }

        vector<string> fields;
        try {
            fields = parseCsvLine(line, config.delimiter);
        } catch (const invalid_argument& error) {
            throw invalid_argument(
                "Ошибка в строке " + to_string(lineNumber) + ": " + error.what()
            );
        }

        if (fields.size() != expectedFields) {
            throw invalid_argument(
                "Ошибка в строке " + to_string(lineNumber) +
                ": ожидалось полей " + to_string(expectedFields) +
                ", получено " + to_string(fields.size())
            );
        }

        rawRows.push_back(move(fields));
    }

    if (file.bad()) {
        throw runtime_error("Ошибка при чтении CSV-файла");
    }




    for (size_t col = 0; col < expectedFields; ++col) {
        bool numerical = true;
        bool hasValues = false;


        for (const auto& row : rawRows) {
            const string& value = row[col];

            if (value == config.missingValue) {
                continue;
            }

            hasValues = true;

            if (!isNumber(value)) {
                numerical = false;
                break;
            }
        }

// пустой столбец - категориальный
        if (!hasValues) {
            numerical = false;
        }

        Column column;
        column.name = featureNames[col];

        if (numerical) {
            column.type = FeatureType::Numerical;

            for (const auto& row : rawRows) {
                const string& value = row[col];

                if (value == config.missingValue) {
                    column.numericalValues.push_back(nullopt);
                } else {
                    column.numericalValues.push_back(stod(value));
                }
            }
        } else {
            column.type = FeatureType::Categorical;

            for (const auto& row : rawRows) {
                const string& value = row[col];

                if (value == config.missingValue) {
                    column.categoricalValues.push_back(nullopt);
                } else {
                    column.categoricalValues.push_back(value);
                }
            }
        }

        dataset.columns.push_back(move(column));
    }
    // проверка длинны столбцов
    for (const auto& column : dataset.columns) {
        size_t columnSize;

        if (column.type == FeatureType::Numerical) {columnSize = column.numericalValues.size();}
        else {columnSize = column.categoricalValues.size();}

        if (columnSize != rawRows.size()) {
            throw runtime_error("Неправильная длина столбца: " + column.name);
        }
    }
}
