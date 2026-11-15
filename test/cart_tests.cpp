
#include "cart.hpp"
#include "decision_stump.hpp"
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

// Регрессия: дерево глубины 0 должно возвращать среднее.
void testRegressionDepthZero() {
    Matrix features(4, 1);

    for (size_t i = 0; i < 4; ++i) {
        features(i, 0) = static_cast<double>(i);
    }

    vector<double> target = {10.0, 20.0, 30.0, 40.0};

    CartRegressor model(0);
    model.fit(features, target);

    Matrix testFeatures(2, 1);
    testFeatures(0, 0) = 0.0;
    testFeatures(1, 0) = 3.0;

    vector<double> predictions = model.predict(testFeatures);

    assert(approximatelyEqual(predictions[0], 25.0));
    assert(approximatelyEqual(predictions[1], 25.0));
}

// Регрессия: дерево должно разделять данные.
void testRegressionSplit() {
    Matrix features(4, 1);

    features(0, 0) = 0.0;
    features(1, 0) = 1.0;
    features(2, 0) = 2.0;
    features(3, 0) = 3.0;

    vector<double> target = {10.0, 10.0, 30.0, 30.0};

    CartRegressor model(1);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    assert(approximatelyEqual(predictions[0], 10.0));
    assert(approximatelyEqual(predictions[1], 10.0));
    assert(approximatelyEqual(predictions[2], 30.0));
    assert(approximatelyEqual(predictions[3], 30.0));
}

// Глубина 1 CART должна соответствовать DecisionStump.
void testStumpEquivalence() {
    Matrix features(6, 2);

    vector<double> target = {
        10.0, 10.0, 10.0,
        40.0, 40.0, 40.0
    };

    for (size_t i = 0; i < 6; ++i) {
        features(i, 0) = static_cast<double>(i);
        features(i, 1) = static_cast<double>(5 - i);
    }

    CartRegressor cart(1, 2, 1);
    DecisionStump stump(SplitCriterion::Variance, 1);

    cart.fit(features, target);
    stump.fit(features, target);

    vector<double> cartPredictions = cart.predict(features);
    vector<double> stumpPredictions = stump.predict(features);

    assert(cartPredictions.size() == stumpPredictions.size());

    for (size_t i = 0; i < cartPredictions.size(); ++i) {
        assert(
            approximatelyEqual(
                cartPredictions[i],
                stumpPredictions[i]
            )
        );
    }
}

// Классификация с метками 10 и 42.
void testClassification() {
    Matrix features(4, 1);

    features(0, 0) = 0.0;
    features(1, 0) = 1.0;
    features(2, 0) = 2.0;
    features(3, 0) = 3.0;

    vector<double> target = {10.0, 10.0, 42.0, 42.0};

    CartClassifier model(2);
    model.fit(features, target);

    auto classes = model.classes();

    assert(classes.size() == 2);
    assert(classes[0] == 10.0);
    assert(classes[1] == 42.0);

    vector<double> predictions = model.predict(features);

    assert(predictions[0] == 10.0);
    assert(predictions[1] == 10.0);
    assert(predictions[2] == 42.0);
    assert(predictions[3] == 42.0);
}

// Вероятности классов должны суммироваться до 1.
void testClassificationProbabilities() {
    Matrix features(4, 1);

    features(0, 0) = 0.0;
    features(1, 0) = 1.0;
    features(2, 0) = 2.0;
    features(3, 0) = 3.0;

    vector<double> target = {10.0, 10.0, 42.0, 42.0};

    CartClassifier model(2);
    model.fit(features, target);

    Matrix probabilities = model.predict_proba(features);

    assert(probabilities.rows() == 4);
    assert(probabilities.columns() == 2);

    for (size_t i = 0; i < probabilities.rows(); ++i) {
        double sum = 0.0;

        for (size_t j = 0; j < probabilities.columns(); ++j) {
            double value = probabilities(i, j);

            assert(value >= 0.0);
            assert(value <= 1.0);

            sum += value;
        }

        assert(approximatelyEqual(sum, 1.0));
    }
}

// Проверка ColumnMajor.
void testColumnMajor() {
    Matrix features(4, 2, MatrixLayout::ColumnMajor);

    for (size_t i = 0; i < 4; ++i) {
        features(i, 0) = static_cast<double>(i);
        features(i, 1) = static_cast<double>(i * 2);
    }

    vector<double> target = {10.0, 10.0, 30.0, 30.0};

    CartRegressor model(1);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    assert(approximatelyEqual(predictions[0], 10.0));
    assert(approximatelyEqual(predictions[3], 30.0));
}

// Проверка константных признаков.
void testConstantFeatures() {
    Matrix features(4, 1);

    for (size_t i = 0; i < 4; ++i) {
        features(i, 0) = 5.0;
    }

    vector<double> target = {10.0, 20.0, 30.0, 40.0};

    CartRegressor model(5);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    for (double prediction : predictions) {
        assert(approximatelyEqual(prediction, 25.0));
    }
}

// Проверка невозможного разбиения.
void testImpossibleSplit() {
    Matrix features(2, 1);

    features(0, 0) = 1.0;
    features(1, 0) = 1.0;

    vector<double> target = {10.0, 20.0};

    CartRegressor model(5);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    assert(approximatelyEqual(predictions[0], 15.0));
    assert(approximatelyEqual(predictions[1], 15.0));
}

// Проверка некорректных параметров.
void testInvalidParameters() {
    bool exceptionThrown = false;

    try {
        CartRegressor model(5, 1, 1);
    }
    catch (const InvalidParameterError&) {
        exceptionThrown = true;
    }

    assert(exceptionThrown);
}

int main() {
    testRegressionDepthZero();
    testRegressionSplit();
    testStumpEquivalence();
    testClassification();
    testClassificationProbabilities();
    testColumnMajor();
    testConstantFeatures();
    testImpossibleSplit();
    testInvalidParameters();

    cout << "Все тесты CART пройдены!" << endl;

    return 0;
}
