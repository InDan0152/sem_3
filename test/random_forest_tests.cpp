#include "random_forest.hpp"
#include "exceptions.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
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


// Один лес из одного дерева без bootstrap должен совпадать с обычным CART.
void testSingleEstimatorWithoutBootstrapMatchesCart() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}, {4.0}});
    const vector<double> regressionTarget{0.0, 1.0, 8.0, 9.0, 10.0};

    RandomForestParams regressionParams;
    regressionParams.n_estimators = 1;
    regressionParams.bootstrap = false;
    regressionParams.max_features = FeatureSearchParams{FeatureMode::All, 1};
    regressionParams.max_depth = 3;
    regressionParams.random_state = 123;

    RandomForestRegressor forestRegressor(regressionParams);
    CartRegressor cartRegressor(3, 2, 1, {FeatureMode::All, 1}, 123);
    forestRegressor.fit(features, regressionTarget);
    cartRegressor.fit(features, regressionTarget);
    CHECK(forestRegressor.predict(features) == cartRegressor.predict(features));

    const vector<double> classTarget{2.0, 2.0, 5.0, 5.0, 5.0};
    RandomForestParams classificationParams = regressionParams;
    classificationParams.max_features = FeatureSearchParams{FeatureMode::All, 1};

    RandomForestClassifier forestClassifier(classificationParams);
    CartClassifier cartClassifier(3, 2, 1, {FeatureMode::All, 1}, 123);
    forestClassifier.fit(features, classTarget);
    cartClassifier.fit(features, classTarget);

    const Matrix forestProbabilities = forestClassifier.predict_proba(features);
    const Matrix cartProbabilities = cartClassifier.predict_proba(features);
    CHECK(forestProbabilities.rows() == cartProbabilities.rows());
    CHECK(forestProbabilities.columns() == cartProbabilities.columns());
    for (size_t i = 0; i < forestProbabilities.rows(); ++i) {
        for (size_t j = 0; j < forestProbabilities.columns(); ++j) {
            CHECK(approximatelyEqual(forestProbabilities(i, j), cartProbabilities(i, j)));
        }
    }
}

// Проверяем один класс, равные вероятности и порядок разрешения ничьей.
void testOneClassAndProbabilityTie() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}});

    RandomForestParams params;
    params.n_estimators = 5;
    params.bootstrap = false;
    params.max_depth = 0;
    params.n_jobs = 3;

    const vector<double> oneClassTarget{7.0, 7.0, 7.0, 7.0};
    RandomForestClassifier oneClassModel(params);
    oneClassModel.fit(features, oneClassTarget);
    CHECK(oneClassModel.classes().size() == 1);
    const Matrix oneClassProbabilities = oneClassModel.predict_proba(features);
    for (size_t i = 0; i < oneClassProbabilities.rows(); ++i) {
        CHECK(oneClassProbabilities(i, 0) == 1.0);
    }

    const vector<double> tiedTarget{1.0, 0.0, 1.0, 0.0};
    RandomForestClassifier tiedModel(params);
    tiedModel.fit(features, tiedTarget);
    const auto labels = tiedModel.classes();
    CHECK(labels.size() == 2);
    CHECK(labels[0] == 0.0);
    CHECK(labels[1] == 1.0);
    const Matrix tiedProbabilities = tiedModel.predict_proba(features);
    const vector<double> tiedPredictions = tiedModel.predict(features);
    for (size_t i = 0; i < tiedProbabilities.rows(); ++i) {
        CHECK(approximatelyEqual(tiedProbabilities(i, 0), 0.5));
        CHECK(approximatelyEqual(tiedProbabilities(i, 1), 0.5));
        CHECK(tiedPredictions[i] == 0.0);
    }
}

// Bootstrap может не включить редкий класс в отдельные деревья.
// Класс всё равно остаётся в общей матрице вероятностей леса.
void testBootstrapMayOmitRareClassFromTrees() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> target{0.0, 0.0, 0.0, 1.0};

    RandomForestParams params;
    params.n_estimators = 64;
    params.bootstrap = true;
    params.max_samples = 1;
    params.max_depth = 0;
    params.random_state = 31415;
    params.n_jobs = 4;

    RandomForestClassifier model(params);
    model.fit(features, target);
    CHECK(model.classes().size() == 2);
    const Matrix probabilities = model.predict_proba(features);
    CHECK(probabilities.columns() == 2);

    for (size_t i = 0; i < probabilities.rows(); ++i) {
        CHECK(probabilities(i, 0) > 0.0);
        CHECK(probabilities(i, 1) > 0.0);
        CHECK(approximatelyEqual(probabilities(i, 0) + probabilities(i, 1), 1.0));
    }
}

// Повторное обучение должно обновлять состояние модели и список классов.
void testRepeatedFitAndEmptyPredictions() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}});
    RandomForestParams params;
    params.n_estimators = 7;
    params.bootstrap = false;
    params.max_depth = 0;
    params.n_jobs = 3;

    RandomForestRegressor regressor(params);
    const vector<double> firstTarget{1.0, 2.0, 3.0, 4.0};
    const vector<double> secondTarget{10.0, 20.0, 30.0, 40.0};
    regressor.fit(features, firstTarget);
    const auto firstPredictions = regressor.predict(features);
    regressor.fit(features, secondTarget);
    const auto regressionPredictions = regressor.predict(features);
    for (double value : regressionPredictions) {
        CHECK(approximatelyEqual(value, 25.0));
    }
    regressor.fit(features, firstTarget);
    const auto repeatedFirstPredictions = regressor.predict(features);
    CHECK(firstPredictions == repeatedFirstPredictions);

    RandomForestClassifier classifier(params);
    const vector<double> firstClasses{1.0, 1.0, 2.0, 2.0};
    const vector<double> secondClasses{5.0, 6.0, 7.0, 8.0};
    classifier.fit(features, firstClasses);
    classifier.fit(features, secondClasses);
    const auto labels = classifier.classes();
    CHECK(labels.size() == 4);
    CHECK(labels[0] == 5.0 && labels[3] == 8.0);

    Matrix emptyFeatures(0, 1);
    CHECK(regressor.predict(emptyFeatures).empty());
    CHECK(classifier.predict(emptyFeatures).empty());
    const Matrix emptyProbabilities = classifier.predict_proba(emptyFeatures);
    CHECK(emptyProbabilities.rows() == 0);
    CHECK(emptyProbabilities.columns() == 4);
}

// Обе схемы хранения матрицы должны давать одинаковые результаты.
void testBothMatrixLayouts() {
    const vector<vector<double>> values{
        {0.0, 1.0}, {1.0, 0.0}, {2.0, 1.0},
        {3.0, 0.0}, {4.0, 1.0}, {5.0, 0.0}
    };
    const vector<double> target{0.0, 1.0, 4.0, 9.0, 16.0, 25.0};
    Matrix rowMajor(values, MatrixLayout::RowMajor);
    Matrix columnMajor(values, MatrixLayout::ColumnMajor);

    RandomForestParams params;
    params.n_estimators = 13;
    params.random_state = 808;
    params.max_depth = 3;
    params.n_jobs = 4;

    RandomForestRegressor rowModel(params);
    RandomForestRegressor columnModel(params);
    rowModel.fit(rowMajor, target);
    columnModel.fit(columnMajor, target);
    const auto rowPredictions = rowModel.predict(rowMajor);
    const auto columnPredictions = columnModel.predict(columnMajor);
    CHECK(rowPredictions.size() == columnPredictions.size());
    for (size_t i = 0; i < rowPredictions.size(); ++i) {
        CHECK(approximatelyEqual(rowPredictions[i], columnPredictions[i]));
    }

    const vector<double> classTarget{0.0, 0.0, 1.0, 1.0, 2.0, 2.0};
    RandomForestClassifier rowClassifier(params);
    RandomForestClassifier columnClassifier(params);
    rowClassifier.fit(rowMajor, classTarget);
    columnClassifier.fit(columnMajor, classTarget);
    const Matrix rowProbabilities = rowClassifier.predict_proba(rowMajor);
    const Matrix columnProbabilities = columnClassifier.predict_proba(columnMajor);
    CHECK(rowProbabilities.rows() == columnProbabilities.rows());
    CHECK(rowProbabilities.columns() == columnProbabilities.columns());
    for (size_t i = 0; i < rowProbabilities.rows(); ++i) {
        for (size_t j = 0; j < rowProbabilities.columns(); ++j) {
            CHECK(approximatelyEqual(rowProbabilities(i, j), columnProbabilities(i, j)));
        }
    }
}

// Проверяем входные ошибки и передачу исключений из рабочих потоков.
void testTrainingAndWorkerExceptions() {
    const double infinity = numeric_limits<double>::infinity();
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> target{0.0, 1.0, 2.0, 3.0};

    RandomForestParams params;
    params.n_estimators = 12;
    params.n_jobs = 4;
    params.max_depth = 2;

    RandomForestRegressor model(params);
    model.fit(features, target);
    const auto previousPredictions = model.predict(features);

    bool emptyTrainingRejected = false;
    try {
        Matrix emptyFeatures(0, 1);
        const vector<double> emptyTarget;
        model.fit(emptyFeatures, emptyTarget);
    } catch (const InvalidParameterError&) {
        emptyTrainingRejected = true;
    }
    CHECK(emptyTrainingRejected);

    bool noFeaturesRejected = false;
    try {
        Matrix noFeatures(4, 0);
        model.fit(noFeatures, target);
    } catch (const InvalidParameterError&) {
        noFeaturesRejected = true;
    }
    CHECK(noFeaturesRejected);

    bool sizeMismatchRejected = false;
    try {
        const vector<double> wrongTarget{1.0, 2.0};
        model.fit(features, wrongTarget);
    } catch (const DimensionMismatchError&) {
        sizeMismatchRejected = true;
    }
    CHECK(sizeMismatchRejected);

    bool workerExceptionRethrown = false;
    try {
        const vector<double> invalidTarget{0.0, infinity, 2.0, 3.0};
        model.fit(features, invalidTarget);
    } catch (const InvalidParameterError&) {
        workerExceptionRethrown = true;
    }
    CHECK(workerExceptionRethrown);
    CHECK(model.predict(features) == previousPredictions);

    bool predictionWorkerExceptionRethrown = false;
    try {
        Matrix invalidPrediction({{0.0}, {infinity}, {2.0}, {3.0}});
        (void)model.predict(invalidPrediction);
    } catch (const InvalidParameterError&) {
        predictionWorkerExceptionRethrown = true;
    }
    CHECK(predictionWorkerExceptionRethrown);

    const double nan = numeric_limits<double>::quiet_NaN();

    bool nanTargetRejected = false;
    try {
        const vector<double> invalidTarget{0.0, nan, 2.0, 3.0};
        model.fit(features, invalidTarget);
    } catch (const InvalidParameterError&) {
        nanTargetRejected = true;
    }
    CHECK(nanTargetRejected);
    CHECK(model.predict(features) == previousPredictions);

    bool nanFeatureRejected = false;
    try {
        Matrix invalidFeatures({{0.0}, {nan}, {2.0}, {3.0}});
        model.fit(invalidFeatures, target);
    } catch (const InvalidParameterError&) {
        nanFeatureRejected = true;
    }
    CHECK(nanFeatureRejected);

    bool nanPredictionRejected = false;
    try {
        Matrix invalidPrediction({{0.0}, {nan}, {2.0}, {3.0}});
        (void)model.predict(invalidPrediction);
    } catch (const InvalidParameterError&) {
        nanPredictionRejected = true;
    }
    CHECK(nanPredictionRejected);

    bool wrongPredictionWidthRejected = false;
    try {
        Matrix wrongWidth(2, 2);
        (void)model.predict(wrongWidth);
    } catch (const DimensionMismatchError&) {
        wrongPredictionWidthRejected = true;
    }
    CHECK(wrongPredictionWidthRejected);
}

// Число потоков больше числа задач не должно менять результат.
void testMoreWorkersThanTreesOrRows() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> target{0.0, 1.0, 4.0, 9.0};

    RandomForestParams sequentialParams;
    sequentialParams.n_estimators = 2;
    sequentialParams.max_depth = 2;
    sequentialParams.random_state = 99;

    RandomForestParams parallelParams = sequentialParams;
    parallelParams.n_jobs = 16;

    RandomForestRegressor sequential(sequentialParams);
    RandomForestRegressor parallel(parallelParams);
    sequential.fit(features, target);
    parallel.fit(features, target);

    Matrix oneRow({{1.5}});
    const auto expected = sequential.predict(oneRow);
    const auto actual = parallel.predict(oneRow);
    CHECK(expected.size() == 1 && actual.size() == 1);
    CHECK(approximatelyEqual(expected[0], actual[0]));
}

// Проверяем нулевой max_samples и неизвестный режим max_features.
void testAdditionalInvalidParameters() {
    bool zeroSamplesRejected = false;
    try {
        RandomForestParams params;
        params.max_samples = 0;
        RandomForestRegressor model(params);
    } catch (const InvalidParameterError&) {
        zeroSamplesRejected = true;
    }
    CHECK(zeroSamplesRejected);

    bool unknownFeatureModeRejected = false;
    try {
        RandomForestParams params;
        params.max_features = FeatureSearchParams{static_cast<FeatureMode>(99), 1};
        RandomForestRegressor model(params);
    } catch (const InvalidParameterError&) {
        unknownFeatureModeRejected = true;
    }
    CHECK(unknownFeatureModeRejected);
}


// Усредняются вероятности деревьев, а не их готовые метки классов.
void testClassifierAveragesProbabilitiesInsteadOfVoting() {
    // Повторяем только формирование bootstrap-планов, чтобы подобрать
    // воспроизводимый пример, где усреднение вероятностей отличается
    // от голосования готовыми метками деревьев.
    const auto treeSeed = [](uint32_t randomState, size_t treeIndex) {
        uint64_t value = static_cast<uint64_t>(randomState) +
            0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(treeIndex) + 1);
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        value ^= value >> 31;
        return static_cast<uint32_t>(value);
    };

    constexpr size_t estimatorCount = 3;
    constexpr size_t bootstrapSize = 2;
    uint32_t selectedSeed = 0;
    double expectedClassOneProbability = 0.0;
    size_t classOneVotes = 0;
    bool scenarioFound = false;

    for (uint32_t candidateSeed = 0; candidateSeed < 10000; ++candidateSeed) {
        double probabilitySum = 0.0;
        size_t votesForOne = 0;

        for (size_t treeIndex = 0; treeIndex < estimatorCount; ++treeIndex) {
            mt19937 generator(treeSeed(candidateSeed, treeIndex));
            uniform_int_distribution<size_t> distribution(0, 1);
            size_t classOneCount = 0;
            for (size_t draw = 0; draw < bootstrapSize; ++draw) {
                classOneCount += distribution(generator);
            }

            const double probability =
                static_cast<double>(classOneCount) / bootstrapSize;
            probabilitySum += probability;
            if (probability > 0.5) {
                ++votesForOne;
            }
        }

        const double averageProbability = probabilitySum / estimatorCount;
        if (averageProbability > 0.5 && votesForOne <= 1) {
            selectedSeed = candidateSeed;
            expectedClassOneProbability = averageProbability;
            classOneVotes = votesForOne;
            scenarioFound = true;
            break;
        }
    }

    CHECK(scenarioFound);
    CHECK(classOneVotes <= 1); // Голосование деревьев выбрало бы класс 0.

    Matrix features(vector<vector<double>>{{0.0}, {1.0}});
    const vector<double> target{0.0, 1.0};

    RandomForestParams params;
    params.n_estimators = estimatorCount;
    params.bootstrap = true;
    params.max_samples = bootstrapSize;
    params.max_depth = 0;
    params.random_state = selectedSeed;
    params.n_jobs = 1;

    RandomForestClassifier model(params);
    model.fit(features, target);

    const Matrix probabilities = model.predict_proba(features);
    const vector<double> predictions = model.predict(features);
    CHECK(probabilities.columns() == 2);
    for (size_t i = 0; i < probabilities.rows(); ++i) {
        CHECK(approximatelyEqual(probabilities(i, 1), expectedClassOneProbability));
        CHECK(approximatelyEqual(probabilities(i, 0), 1.0 - expectedClassOneProbability));
        CHECK(predictions[i] == 1.0); // Решение по усреднённым вероятностям.
    }
}

// Проверяем обучение на одном объекте и на матрице константных признаков.
void testSingleObjectAndConstantFeatures() {
    Matrix oneObject(vector<vector<double>>{{3.0}});
    const vector<double> oneTarget{17.0};

    RandomForestParams params;
    params.n_estimators = 5;
    params.bootstrap = false;
    params.n_jobs = 3;

    RandomForestRegressor oneObjectRegressor(params);
    oneObjectRegressor.fit(oneObject, oneTarget);
    CHECK(oneObjectRegressor.predict(oneObject).size() == 1);
    CHECK(oneObjectRegressor.predict(oneObject)[0] == 17.0);

    RandomForestClassifier oneObjectClassifier(params);
    oneObjectClassifier.fit(oneObject, oneTarget);
    CHECK(oneObjectClassifier.predict(oneObject)[0] == 17.0);
    const Matrix oneObjectProbabilities = oneObjectClassifier.predict_proba(oneObject);
    CHECK(oneObjectProbabilities(0, 0) == 1.0);

    Matrix constantFeatures({{2.0, 5.0}, {2.0, 5.0}, {2.0, 5.0}, {2.0, 5.0}});
    const vector<double> constantTarget{1.0, 2.0, 3.0, 4.0};
    RandomForestRegressor constantModel(params);
    constantModel.fit(constantFeatures, constantTarget);
    for (double prediction : constantModel.predict(constantFeatures)) {
        CHECK(approximatelyEqual(prediction, 2.5));
    }
}

// Bootstrap-планы должны быть воспроизводимы независимо от числа потоков.
void testBootstrapPlansIndependentOfWorkerScheduling() {
    Matrix features({{0.0}, {0.0}, {0.0}, {0.0}, {0.0}, {0.0}});
    const vector<double> target{0.0, 2.0, 5.0, 8.0, 13.0, 21.0};

    RandomForestParams sequentialParams;
    sequentialParams.n_estimators = 31;
    sequentialParams.max_depth = 0;
    sequentialParams.random_state = 20261122;
    sequentialParams.n_jobs = 1;

    RandomForestParams parallelParams = sequentialParams;
    parallelParams.n_jobs = 5;

    RandomForestRegressor sequential(sequentialParams);
    RandomForestRegressor parallel(parallelParams);
    sequential.fit(features, target);
    parallel.fit(features, target);

    const auto expected = sequential.predict(features);
    const auto actual = parallel.predict(features);
    CHECK(expected.size() == actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(expected[i] == actual[i]);
    }
}

// Публичная сводка леса должна возвращать размер каждого обученного дерева.
void testTreeSummaries() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}});
    const vector<double> regressionTarget{0.0, 0.0, 1.0, 1.0, 2.0, 2.0};
    const vector<double> classificationTarget{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};

    RandomForestParams params;
    params.n_estimators = 4;
    params.max_depth = 3;
    params.n_jobs = 2;

    RandomForestRegressor regressor(params);
    bool regressorRejectedBeforeFit = false;
    try {
        (void)regressor.tree_summaries();
    } catch (const ModelUsageError&) {
        regressorRejectedBeforeFit = true;
    }
    CHECK(regressorRejectedBeforeFit);

    regressor.fit(features, regressionTarget);
    const auto regressionSummaries = regressor.tree_summaries();
    CHECK(regressionSummaries.size() == params.n_estimators);
    for (const auto& summary : regressionSummaries) {
        CHECK(summary.nodeCount >= 1);
        CHECK(summary.leafCount >= 1);
        CHECK(summary.nodeCount >= summary.leafCount);
    }

    RandomForestClassifier classifier(params);
    classifier.fit(features, classificationTarget);
    const auto classificationSummaries = classifier.tree_summaries();
    CHECK(classificationSummaries.size() == params.n_estimators);
    for (const auto& summary : classificationSummaries) {
        CHECK(summary.nodeCount >= 1);
        CHECK(summary.leafCount >= 1);
        CHECK(summary.nodeCount >= summary.leafCount);
    }
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
    testSingleEstimatorWithoutBootstrapMatchesCart();
    testOneClassAndProbabilityTie();
    testBootstrapMayOmitRareClassFromTrees();
    testRepeatedFitAndEmptyPredictions();
    testBothMatrixLayouts();
    testTrainingAndWorkerExceptions();
    testMoreWorkersThanTreesOrRows();
    testAdditionalInvalidParameters();
    testClassifierAveragesProbabilitiesInsteadOfVoting();
    testSingleObjectAndConstantFeatures();
    testBootstrapPlansIndependentOfWorkerScheduling();
    testTreeSummaries();

    cout << "Все тесты Random Forest пройдены!" << endl;

    return 0;
}