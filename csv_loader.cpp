#include "csv_loader.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cstdlib>

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
        cerr << "Ошибка: не закрыта кавычка в строке CSV\n";
        return {};
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


bool loadCsv(
    const string& filename,
    Dataset& dataset,
    const CsvConfig& config
) {
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Ошибка: не удалось открыть CSV-файл: "
             << filename << '\n';
        return false;
    }

    dataset.columns.clear();

    string line;
    size_t lineNumber = 0;


    if (!getline(file, line)) {
        cerr << "Ошибка: CSV-файл пуст\n";
        return false;
    }

    ++lineNumber;

    auto firstRow = parseCsvLine(line, config.delimiter);

    if (firstRow.empty()) {
        cerr << "Ошибка: некорректная первая строка CSV\n";
        return false;
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

        auto fields = parseCsvLine(line, config.delimiter);

        if (fields.empty()) {
            cerr << "Ошибка в строке " << lineNumber
                 << ": некорректный формат CSV\n";
            return false;
        }

        if (fields.size() != expectedFields) {
            cerr << "Ошибка в строке " << lineNumber
                 << ": ожидалось полей: " << expectedFields
                 << ", получено: " << fields.size() << '\n';
            return false;
        }

        rawRows.push_back(move(fields));
    }

    if (file.bad()) {
        cerr << "Ошибка при чтении CSV-файла\n";
        return false;
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
            cerr << "Ошибка: неправильная длина столбца "<< column.name << endl;
        return false;
    }
}
    return true;
}