#include "random_forest.hpp"
#include "exceptions.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace std;

// Проверки работают и в Release: в отличие от assert, не отключаются через NDEBUG.
#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            throw runtime_error( \
                string("Проверка не пройдена: ") + #condition + \
                " (" + __FILE__ + ":" + to_string(__LINE__) + ")" \
            ); \
        } \
    } while (false)

bool approximatelyEqual(
    double a,
    double b,
    double tolerance = 1e-6
) {
    return abs(a - b) < tolerance;
}

// Регрессия: без bootstrap каждое дерево получает все объекты.
// При глубине 0 лес должен возвращать среднее значение target.
void testRegressionWithoutBootstrap() {
    Matrix features(4, 1);

    for (size_t i = 0; i < 4; ++i) {features(i, 0) = 0.0;}

    vector<double> target = {10.0, 20.0, 30.0, 40.0};

    RandomForestParams params;
    params.n_estimators = 5;
    params.bootstrap = false;
    params.random_state = 42;

    RandomForestRegressor model(params);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    CHECK(predictions.size() == 4);

    for (double prediction : predictions) {
        CHECK(approximatelyEqual(prediction, 25.0));
    }
}

// Классификация: проверяем метки классов и предсказания.
void testClassificationWithoutBootstrap() {
    Matrix features(4, 1);

    for (size_t i = 0; i < 4; ++i) {
        features(i, 0) = static_cast<double>(i);
    }

    vector<double> target = {10.0, 10.0, 42.0, 42.0};

    RandomForestParams params;
    params.n_estimators = 5;
    params.bootstrap = false;
    params.max_depth = 2;
    params.random_state = 42;

    RandomForestClassifier model(params);
    model.fit(features, target);

    auto classes = model.classes();

    CHECK(classes.size() == 2);
    CHECK(classes[0] == 10.0);
    CHECK(classes[1] == 42.0);

    vector<double> predictions = model.predict(features);

    CHECK(predictions[0] == 10.0);
    CHECK(predictions[1] == 10.0);
    CHECK(predictions[2] == 42.0);
    CHECK(predictions[3] == 42.0);
}

// Вероятности классов должны находиться в [0, 1]
// и в каждой строке суммироваться до 1.
void testClassificationProbabilities() {
    Matrix features(4, 1);

    for (size_t i = 0; i < 4; ++i) {
        features(i, 0) = static_cast<double>(i);
    }

    vector<double> target = {10.0, 10.0, 42.0, 42.0};

    RandomForestParams params;
    params.n_estimators = 10;
    params.bootstrap = true;
    params.random_state = 123;

    RandomForestClassifier model(params);
    model.fit(features, target);

    Matrix probabilities = model.predict_proba(features);

    CHECK(probabilities.rows() == 4);
    CHECK(probabilities.columns() == 2);

    for (size_t i = 0; i < probabilities.rows(); ++i) {
        double sum = 0.0;

        for (size_t j = 0; j < probabilities.columns(); ++j) {
            double value = probabilities(i, j);

            CHECK(value >= 0.0);
            CHECK(value <= 1.0);

            sum += value;
        }

        CHECK(approximatelyEqual(sum, 1.0));
    }
}

// Одинаковый random_state должен давать одинаковые предсказания.
void testReproducibility() {
    Matrix features(6, 2);

    vector<double> target = {
        10.0, 12.0, 15.0,
        30.0, 35.0, 40.0
    };

    for (size_t i = 0; i < 6; ++i) {
        features(i, 0) = static_cast<double>(i);
        features(i, 1) = static_cast<double>(i * 2);
    }

    RandomForestParams params;
    params.n_estimators = 20;
    params.bootstrap = true;
    params.max_depth = 3;
    params.random_state = 777;

    RandomForestRegressor first(params);
    RandomForestRegressor second(params);

    first.fit(features, target);
    second.fit(features, target);

    vector<double> firstPredictions = first.predict(features);
    vector<double> secondPredictions = second.predict(features);

    CHECK(firstPredictions.size() == secondPredictions.size());

    for (size_t i = 0; i < firstPredictions.size(); ++i) {
        CHECK(firstPredictions[i] == secondPredictions[i]);
    }
}

// max_samples не должен задаваться при bootstrap=false.
void testMaxSamplesWithoutBootstrap() {
    RandomForestParams params;
    params.bootstrap = false;
    params.max_samples = 3;

    bool exceptionThrown = false;

    try {
        RandomForestRegressor model(params);
    }
    catch (const InvalidParameterError&) {
        exceptionThrown = true;
    }

    CHECK(exceptionThrown);
}

// Число bootstrap-объектов не может превышать размер выборки.
void testMaxSamplesTooLarge() {
    Matrix features(4, 1);

    vector<double> target = {1.0, 2.0, 3.0, 4.0};

    RandomForestParams params;
    params.n_estimators = 3;
    params.max_samples = 5;

    RandomForestRegressor model(params);

    bool exceptionThrown = false;

    try {
        model.fit(features, target);
    }
    catch (const InvalidParameterError&) {
        exceptionThrown = true;
    }

    CHECK(exceptionThrown);
}

// Проверка некорректного количества деревьев.
void testInvalidEstimatorCount() {
    RandomForestParams params;
    params.n_estimators = 0;

    bool exceptionThrown = false;

    try {
        RandomForestRegressor model(params);
    }
    catch (const InvalidParameterError&) {
        exceptionThrown = true;
    }

    CHECK(exceptionThrown);
}

// Нельзя предсказывать до обучения модели.
void testPredictionBeforeFit() {
    RandomForestParams params;

    RandomForestRegressor model(params);

    Matrix features(1, 1);

    bool exceptionThrown = false;

    try {
        model.predict(features);
    }
    catch (const ModelUsageError&) {
        exceptionThrown = true;
    }

    CHECK(exceptionThrown);
}

// Count должен находиться в диапазоне от 1 до числа признаков.
void testMaxFeaturesCountValidation() {
    Matrix features(4, 2);
    vector<double> target = {1.0, 2.0, 3.0, 4.0};

    RandomForestParams params;
    params.max_features = FeatureSearchParams{FeatureMode::Count, 3};

    RandomForestRegressor model(params);
    bool exceptionThrown = false;

    try {
        model.fit(features, target);
    } catch (const InvalidParameterError&) {
        exceptionThrown = true;
    }

    CHECK(exceptionThrown);
}

// Нулевое число потоков недопустимо.
void testInvalidJobCount() {
    RandomForestParams params;
    params.n_jobs = 0;

    bool exceptionThrown = false;
    try {
        RandomForestRegressor model(params);
    } catch (const InvalidParameterError&) {
        exceptionThrown = true;
    }

    CHECK(exceptionThrown);
}

// Параллельное обучение должно давать те же предсказания, что и последовательное.
void testParallelRegressionMatchesSequential() {
    Matrix features(12, 2);
    vector<double> target(12);

    for (size_t i = 0; i < 12; ++i) {
        features(i, 0) = static_cast<double>(i);
        features(i, 1) = static_cast<double>(i % 3);
        target[i] = static_cast<double>(i * i - 2 * i);
    }

    RandomForestParams sequentialParams;
    sequentialParams.n_estimators = 17;
    sequentialParams.random_state = 2026;
    sequentialParams.max_depth = 4;

    RandomForestParams parallelParams = sequentialParams;
    parallelParams.n_jobs = 3;

    RandomForestRegressor sequential(sequentialParams);
    RandomForestRegressor parallel(parallelParams);
    sequential.fit(features, target);
    parallel.fit(features, target);

    const auto expected = sequential.predict(features);
    const auto actual = parallel.predict(features);
    CHECK(expected.size() == actual.size());

    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(approximatelyEqual(expected[i], actual[i]));
    }
}

// Параллельное обучение и predict_proba классификатора сохраняют результат.
void testParallelClassificationMatchesSequential() {
    Matrix features(12, 2);
    vector<double> target(12);

    for (size_t i = 0; i < 12; ++i) {
        features(i, 0) = static_cast<double>(i);
        features(i, 1) = static_cast<double>(i % 4);
        target[i] = i < 4 ? 2.0 : (i < 8 ? 5.0 : 9.0);
    }

    RandomForestParams sequentialParams;
    sequentialParams.n_estimators = 19;
    sequentialParams.random_state = 91;
    sequentialParams.max_depth = 3;

    RandomForestParams parallelParams = sequentialParams;
    parallelParams.n_jobs = 4;

    RandomForestClassifier sequential(sequentialParams);
    RandomForestClassifier parallel(parallelParams);
    sequential.fit(features, target);
    parallel.fit(features, target);

    const Matrix expected = sequential.predict_proba(features);
    const Matrix actual = parallel.predict_proba(features);
    CHECK(expected.rows() == actual.rows());
    CHECK(expected.columns() == actual.columns());

    for (size_t i = 0; i < expected.rows(); ++i) {
        for (size_t j = 0; j < expected.columns(); ++j) {
            CHECK(approximatelyEqual(expected(i, j), actual(i, j)));
        }
    }

    const auto expectedLabels = sequential.predict(features);
    const auto actualLabels = parallel.predict(features);
    CHECK(expectedLabels == actualLabels);
}

int main() {
    testRegressionWithoutBootstrap();
    testClassificationWithoutBootstrap();
    testClassificationProbabilities();
    testReproducibility();
    testMaxSamplesWithoutBootstrap();
    testMaxSamplesTooLarge();
    testInvalidEstimatorCount();
    testPredictionBeforeFit();
    testMaxFeaturesCountValidation();
    testInvalidJobCount();
    testParallelRegressionMatchesSequential();
    testParallelClassificationMatchesSequential();

    cout << "Все тесты Random Forest пройдены!" << endl;

    return 0;
}