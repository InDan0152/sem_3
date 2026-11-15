
#include "nadaraya_watson.hpp"
#include "exceptions.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;

bool approximatelyEqual(
    double a,
    double b,
    double tolerance = 1e-6
) {
    return abs(a - b) < tolerance;
}

// Проверка NW-регрессии на постоянной функции.
void testConstantRegression() {
    Matrix features(3, 1);

    features(0, 0) = 1.0;
    features(1, 0) = 2.0;
    features(2, 0) = 3.0;

    vector<double> target = {5.0, 5.0, 5.0};

    NadarayaWatsonRegressor model(
        1.0,
        make_unique<EuclideanDistance>()
    );

    model.fit(features, target);

    Matrix testFeatures(2, 1);
    testFeatures(0, 0) = 1.5;
    testFeatures(1, 0) = 2.5;

    vector<double> predictions = model.predict(testFeatures);

    assert(predictions.size() == 2);
    assert(approximatelyEqual(predictions[0], 5.0));
    assert(approximatelyEqual(predictions[1], 5.0));
}

// Проверка NW-регрессии: результат должен быть
// взвешенным средним обучающих целевых значений.
void testRegressionRange() {
    Matrix features(3, 1);

    features(0, 0) = 0.0;
    features(1, 0) = 1.0;
    features(2, 0) = 2.0;

    vector<double> target = {10.0, 20.0, 30.0};

    NadarayaWatsonRegressor model(
        0.5,
        make_unique<EuclideanDistance>()
    );

    model.fit(features, target);

    Matrix testFeatures(1, 1);
    testFeatures(0, 0) = 1.0;

    vector<double> predictions = model.predict(testFeatures);

    assert(predictions.size() == 1);
    assert(predictions[0] >= 10.0);
    assert(predictions[0] <= 30.0);
    assert(approximatelyEqual(predictions[0], 20.0));
}

// Проверка классификации и списка классов.
void testClassification() {
    Matrix features(4, 1);

    features(0, 0) = 0.0;
    features(1, 0) = 0.1;
    features(2, 0) = 5.0;
    features(3, 0) = 5.1;

    vector<double> target = {10.0, 10.0, 42.0, 42.0};

    NadarayaWatsonClassifier model(
        0.5,
        make_unique<EuclideanDistance>()
    );

    model.fit(features, target);

    auto classes = model.classes();

    assert(classes.size() == 2);
    assert(classes[0] == 10.0);
    assert(classes[1] == 42.0);

    Matrix testFeatures(2, 1);
    testFeatures(0, 0) = 0.05;
    testFeatures(1, 0) = 5.05;

    vector<double> predictions = model.predict(testFeatures);

    assert(predictions.size() == 2);
    assert(predictions[0] == 10.0);
    assert(predictions[1] == 42.0);
}

// Проверка вероятностей классов.
void testProbabilities() {
    Matrix features(4, 1);

    features(0, 0) = 0.0;
    features(1, 0) = 0.1;
    features(2, 0) = 5.0;
    features(3, 0) = 5.1;

    vector<double> target = {10.0, 10.0, 42.0, 42.0};

    NadarayaWatsonClassifier model(
        0.5,
        make_unique<EuclideanDistance>()
    );

    model.fit(features, target);

    Matrix testFeatures(2, 1);
    testFeatures(0, 0) = 0.05;
    testFeatures(1, 0) = 5.05;

    Matrix probabilities = model.predict_proba(testFeatures);

    assert(probabilities.rows() == 2);
    assert(probabilities.columns() == 2);

    for (size_t i = 0; i < probabilities.rows(); ++i) {
        double sum = 0.0;

        for (size_t j = 0; j < probabilities.columns(); ++j) {
            double probability = probabilities(i, j);

            assert(probability >= 0.0);
            assert(probability <= 1.0);

            sum += probability;
        }

        assert(approximatelyEqual(sum, 1.0));
    }
}

// Проверка работы с ColumnMajor.
void testColumnMajor() {
    Matrix features(3, 2, MatrixLayout::ColumnMajor);

    features(0, 0) = 0.0;
    features(0, 1) = 0.0;

    features(1, 0) = 1.0;
    features(1, 1) = 1.0;

    features(2, 0) = 2.0;
    features(2, 1) = 2.0;

    vector<double> target = {10.0, 20.0, 30.0};

    NadarayaWatsonRegressor model(
        1.0,
        make_unique<EuclideanDistance>()
    );

    model.fit(features, target);

    Matrix testFeatures(1, 2, MatrixLayout::ColumnMajor);

    testFeatures(0, 0) = 1.0;
    testFeatures(0, 1) = 1.0;

    vector<double> predictions = model.predict(testFeatures);

    assert(predictions.size() == 1);
    assert(approximatelyEqual(predictions[0], 20.0));
}

// Проверка повторного обучения.
void testRepeatedFit() {
    Matrix features(2, 1);

    features(0, 0) = 0.0;
    features(1, 0) = 1.0;

    vector<double> firstTarget = {10.0, 20.0};

    NadarayaWatsonRegressor model(
        1.0,
        make_unique<EuclideanDistance>()
    );

    model.fit(features, firstTarget);

    vector<double> secondTarget = {100.0, 200.0};

    model.fit(features, secondTarget);

    Matrix testFeatures(1, 1);
    testFeatures(0, 0) = 0.5;

    vector<double> predictions = model.predict(testFeatures);

    assert(approximatelyEqual(predictions[0], 150.0));
}

// Проверка некорректной ширины окна.
void testInvalidBandwidth() {
    bool exceptionThrown = false;

    try {
        NadarayaWatsonRegressor model(
            0.0,
            make_unique<EuclideanDistance>()
        );
    }
    catch (const InvalidParameterError&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

int main() {
    testConstantRegression();
    testRegressionRange();
    testClassification();
    testProbabilities();
    testColumnMajor();
    testRepeatedFit();
    testInvalidBandwidth();

    cout << "Все тесты Nadaraya-Watson пройдены!" << endl;

    return 0;
}
