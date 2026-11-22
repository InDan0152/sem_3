#include "cart.hpp"
#include "decision_stump.hpp"
#include "exceptions.hpp"

#include <iostream>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;

#define CHECK(condition) \
    do { \
        if (!(condition)) { \
            throw runtime_error( \
                string("Проверка не пройдена: ") + #condition + \
                " (" + __FILE__ + ":" + to_string(__LINE__) + ")" \
            ); \
        } \
    } while (false)

static_assert(noexcept(declval<MatrixSlice<double>>()(size_t{})));
static_assert(noexcept(declval<Matrix&>()(size_t{}, size_t{})));

namespace {

bool close(double a, double b, double tolerance = 1e-12) {
    return abs(a - b) <= tolerance;
}

void testSplitTieBreaksByFeatureAndThreshold() {
    Matrix features({
        {0.0, 0.0},
        {1.0, 1.0},
        {2.0, 2.0},
        {3.0, 3.0}
    });
    const vector<double> target{0.0, 1.0, 1.0, 0.0};
    const vector<size_t> objects{0, 1, 2, 3};
    const vector<size_t> reversedFeatures{1, 0};

    const auto split = SplitFinder::findBestSplit(
        features,
        target,
        objects,
        reversedFeatures,
        SplitCriterion::Variance,
        1
    );

    CHECK(split.has_value());
    CHECK(split->featureIndex == 0);
    CHECK(close(split->threshold, 0.5));
}

void testAdjacentRepresentableThresholdRoutesCorrectly() {
    const double lower = 1.0;
    const double upper = nextafter(lower, numeric_limits<double>::infinity());
    Matrix features(vector<vector<double>>{{lower}, {upper}});
    const vector<double> target{0.0, 10.0};

    CartRegressor model(1, 2, 1);
    model.fit(features, target);
    const CartTreeSummary summary = model.summary();

    CHECK(summary.splits.size() == 1);
    CHECK(summary.splits[0].threshold == lower);
    const vector<double> predictions = model.predict(features);
    CHECK(predictions[0] == 0.0);
    CHECK(predictions[1] == 10.0);
}

void testSplitFinderHonorsFeatureSubset() {
    Matrix features({
        {7.0, 0.0},
        {7.0, 1.0},
        {7.0, 2.0},
        {7.0, 3.0}
    });
    const vector<double> target{0.0, 0.0, 10.0, 10.0};
    const vector<size_t> objects{0, 1, 2, 3};
    const vector<size_t> allowedFeatures{0};

    const auto split = SplitFinder::findBestSplit(
        features,
        target,
        objects,
        allowedFeatures,
        SplitCriterion::Variance,
        1
    );

    CHECK(!split.has_value());

    Matrix tiedFeatures({
        {0.0, 0.0},
        {1.0, 1.0},
        {2.0, 2.0},
        {3.0, 3.0}
    });
    const vector<double> classTarget{10.0, 10.0, 42.0, 42.0};
    const vector<size_t> reversedFeatureTie{1, 0};
    const auto giniSplit = SplitFinder::findBestSplit(
        tiedFeatures,
        classTarget,
        objects,
        reversedFeatureTie,
        SplitCriterion::Gini,
        1
    );
    CHECK(giniSplit.has_value());
    CHECK(giniSplit->featureIndex == 0);
}

void testIdenticalTargetsStopBothTrees() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> regressionTarget{4.0, 4.0, 4.0, 4.0};
    const vector<double> classificationTarget{42.0, 42.0, 42.0, 42.0};

    CartRegressor regressor(10);
    regressor.fit(features, regressionTarget);
    const CartTreeSummary regressionSummary = regressor.summary();
    CHECK(regressionSummary.nodeCount == 1);
    CHECK(regressionSummary.leafCount == 1);
    CHECK(regressionSummary.depth == 0);

    CartClassifier classifier(10);
    classifier.fit(features, classificationTarget);
    const CartTreeSummary classificationSummary = classifier.summary();
    CHECK(classificationSummary.nodeCount == 1);
    CHECK(classificationSummary.leafCount == 1);
}

void testClassificationStumpEquivalence() {
    Matrix features({
        {0.0, 4.0},
        {1.0, 3.0},
        {2.0, 2.0},
        {3.0, 1.0},
        {4.0, 0.0}
    });
    const vector<double> target{10.0, 10.0, 42.0, 10.0, 42.0};

    CartClassifier tree(1, 2, 1);
    DecisionStump stump(SplitCriterion::Gini, 1);
    tree.fit(features, target);
    stump.fit(features, target);

    CHECK(tree.predict(features) == stump.predict(features));
}

void testProbabilitiesUseGlobalSortedClassesAndTieBreak() {
    Matrix features({{0.0}, {1.0}, {2.0}});
    const vector<double> target{42.0, 10.0, 42.0};

    CartClassifier classifier(0);
    classifier.fit(features, target);

    const span<const double> classes = classifier.classes();
    CHECK(classes.size() == 2);
    CHECK(classes[0] == 10.0);
    CHECK(classes[1] == 42.0);

    Matrix probabilities = classifier.predict_proba(features);
    CHECK(close(probabilities(0, 0), 1.0 / 3.0));
    CHECK(close(probabilities(0, 1), 2.0 / 3.0));

    Matrix separatedFeatures({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> separatedTarget{10.0, 10.0, 42.0, 42.0};
    CartClassifier separated(1);
    separated.fit(separatedFeatures, separatedTarget);
    const Matrix leafProbabilities =
        separated.predict_proba(separatedFeatures);
    CHECK(leafProbabilities(0, 0) == 1.0);
    CHECK(leafProbabilities(0, 1) == 0.0);
    CHECK(leafProbabilities(3, 0) == 0.0);
    CHECK(leafProbabilities(3, 1) == 1.0);

    Matrix tieFeatures(vector<vector<double>>{{0.0}, {1.0}});
    const vector<double> tieTarget{10.0, 42.0};
    CartClassifier tied(0);
    tied.fit(tieFeatures, tieTarget);
    CHECK(tied.predict(tieFeatures)[0] == 10.0);
}

void testLocalClassEncodingWithClassesAbsentFromLeaves() {
    Matrix features({
        {0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}
    });
    const vector<double> target{10.0, 10.0, 42.0, 42.0, 100.0, 100.0};

    CartClassifier classifier(2);
    classifier.fit(features, target);

    const span<const double> classes = classifier.classes();
    CHECK(classes.size() == 3);
    CHECK(classes[0] == 10.0);
    CHECK(classes[1] == 42.0);
    CHECK(classes[2] == 100.0);

    const CartTreeSummary summary = classifier.summary();
    CHECK(summary.splits.size() == 2);
    CHECK(summary.splits[0].threshold == 1.5);
    CHECK(summary.splits[1].threshold == 3.5);

    const Matrix probabilities = classifier.predict_proba(features);
    CHECK(probabilities(0, 0) == 1.0);
    CHECK(probabilities(0, 1) == 0.0);
    CHECK(probabilities(0, 2) == 0.0);
    CHECK(probabilities(2, 0) == 0.0);
    CHECK(probabilities(2, 1) == 1.0);
    CHECK(probabilities(2, 2) == 0.0);
    CHECK(probabilities(5, 0) == 0.0);
    CHECK(probabilities(5, 1) == 0.0);
    CHECK(probabilities(5, 2) == 1.0);
}

void testRefitEmptyPredictionAndClassReplacement() {
    Matrix firstFeatures(vector<vector<double>>{{0.0}, {1.0}});
    const vector<double> firstTarget{10.0, 10.0};
    CartClassifier classifier(2);
    classifier.fit(firstFeatures, firstTarget);

    Matrix secondFeatures({{0.0}, {1.0}, {2.0}});
    const vector<double> secondTarget{7.0, 9.0, 11.0};
    classifier.fit(secondFeatures, secondTarget);

    const span<const double> classes = classifier.classes();
    CHECK(classes.size() == 3);
    CHECK(classes[0] == 7.0 && classes[1] == 9.0 && classes[2] == 11.0);

    Matrix emptyTest(0, 1);
    CHECK(classifier.predict(emptyTest).empty());
    Matrix emptyProbabilities = classifier.predict_proba(emptyTest);
    CHECK(emptyProbabilities.rows() == 0);
    CHECK(emptyProbabilities.columns() == 3);
}

void testRegressorRepeatedFit() {
    Matrix features({{0.0}, {1.0}, {2.0}});
    const vector<double> firstTarget{1.0, 2.0, 3.0};
    const vector<double> secondTarget{10.0, 20.0, 30.0};
    CartRegressor regressor(0);

    regressor.fit(features, firstTarget);
    regressor.fit(features, secondTarget);
    const vector<double> predictions = regressor.predict(features);
    CHECK(predictions.size() == 3);
    CHECK(predictions[0] == 20.0);
    CHECK(predictions[2] == 20.0);

    Matrix emptyTest(0, 1);
    CHECK(regressor.predict(emptyTest).empty());
}

void testParametersAndImpossibleMinimumLeafSize() {
    bool rejectedSplit = false;
    try {
        CartRegressor invalid(1, 1, 1);
    } catch (const InvalidParameterError&) {
        rejectedSplit = true;
    }
    CHECK(rejectedSplit);

    bool rejectedLeaf = false;
    try {
        CartClassifier invalid(1, 2, 0);
    } catch (const InvalidParameterError&) {
        rejectedLeaf = true;
    }
    CHECK(rejectedLeaf);

    bool rejectedDepth = false;
    try {
        CartRegressor invalid(65);
    } catch (const InvalidParameterError&) {
        rejectedDepth = true;
    }
    CHECK(rejectedDepth);

    Matrix features({{0.0}, {1.0}, {2.0}});
    const vector<double> target{0.0, 1.0, 2.0};
    CartRegressor noSplit(5, 2, numeric_limits<size_t>::max());
    noSplit.fit(features, target);
    CHECK(noSplit.summary().nodeCount == 1);

    CartRegressor tooFewObjects(5, 4, 1);
    tooFewObjects.fit(features, target);
    CHECK(tooFewObjects.summary().nodeCount == 1);

    Matrix leafLimitedFeatures({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> leafLimitedTarget{0.0, 0.0, 0.0, 100.0};
    CartRegressor leafLimited(1, 2, 2);
    leafLimited.fit(leafLimitedFeatures, leafLimitedTarget);
    const vector<double> leafLimitedPredictions =
        leafLimited.predict(leafLimitedFeatures);
    CHECK(leafLimitedPredictions[0] == 0.0);
    CHECK(leafLimitedPredictions[3] == 50.0);
}

void testSingleObjectTraining() {
    Matrix features(vector<vector<double>>{{3.0}});
    const vector<double> regressionTarget{17.0};
    CartRegressor regressor(8);
    regressor.fit(features, regressionTarget);
    CHECK(regressor.summary().nodeCount == 1);
    CHECK(regressor.predict(features)[0] == 17.0);

    const vector<double> classificationTarget{42.0};
    CartClassifier classifier(8);
    classifier.fit(features, classificationTarget);
    CHECK(classifier.summary().nodeCount == 1);
    CHECK(classifier.predict(features)[0] == 42.0);
    const Matrix probabilities = classifier.predict_proba(features);
    CHECK(probabilities(0, 0) == 1.0);
}

void testFitAndPredictContractErrors() {
    const double infinity = numeric_limits<double>::infinity();
    Matrix validFeatures(vector<vector<double>>{{1.0}});
    const vector<double> validTarget{2.0};

    CartRegressor regressor(1);
    bool predictBeforeFitRejected = false;
    try {
        (void)regressor.predict(validFeatures);
    } catch (const ModelUsageError&) {
        predictBeforeFitRejected = true;
    }
    CHECK(predictBeforeFitRejected);

    bool emptyTrainingRejected = false;
    try {
        Matrix emptyFeatures(0, 1);
        const vector<double> emptyTarget;
        regressor.fit(emptyFeatures, emptyTarget);
    } catch (const InvalidParameterError&) {
        emptyTrainingRejected = true;
    }
    CHECK(emptyTrainingRejected);

    bool zeroColumnsRejected = false;
    try {
        Matrix noFeatures(1, 0);
        regressor.fit(noFeatures, validTarget);
    } catch (const InvalidParameterError&) {
        zeroColumnsRejected = true;
    }
    CHECK(zeroColumnsRejected);

    bool sizeMismatchRejected = false;
    try {
        const vector<double> wrongSizeTarget{2.0, 3.0};
        regressor.fit(validFeatures, wrongSizeTarget);
    } catch (const DimensionMismatchError&) {
        sizeMismatchRejected = true;
    }
    CHECK(sizeMismatchRejected);

    bool nonFiniteTargetRejected = false;
    try {
        const vector<double> badTarget{infinity};
        regressor.fit(validFeatures, badTarget);
    } catch (const InvalidParameterError&) {
        nonFiniteTargetRejected = true;
    }
    CHECK(nonFiniteTargetRejected);

    Matrix nonFiniteFeatures(vector<vector<double>>{{infinity}});
    bool nonFiniteFeaturesRejected = false;
    try {
        regressor.fit(nonFiniteFeatures, validTarget);
    } catch (const InvalidParameterError&) {
        nonFiniteFeaturesRejected = true;
    }
    CHECK(nonFiniteFeaturesRejected);

    regressor.fit(validFeatures, validTarget);
    bool wrongPredictWidthRejected = false;
    try {
        Matrix wrongWidth(1, 2);
        (void)regressor.predict(wrongWidth);
    } catch (const DimensionMismatchError&) {
        wrongPredictWidthRejected = true;
    }
    CHECK(wrongPredictWidthRejected);

    bool nonFinitePredictionRejected = false;
    try {
        (void)regressor.predict(nonFiniteFeatures);
    } catch (const InvalidParameterError&) {
        nonFinitePredictionRejected = true;
    }
    CHECK(nonFinitePredictionRejected);

    CartClassifier classifier(1);
    bool classesBeforeFitRejected = false;
    try {
        (void)classifier.classes();
    } catch (const ModelUsageError&) {
        classesBeforeFitRejected = true;
    }
    CHECK(classesBeforeFitRejected);
}

void testDefaultAndCheckedMatrixSliceAccess() {
    MatrixSlice<double> empty;
    CHECK(empty.empty());
    CHECK(empty.data() == nullptr);
    CHECK(empty.stride() == 1);

    bool rejectedIndex = false;
    try {
        (void)empty.at(0);
    } catch (const out_of_range&) {
        rejectedIndex = true;
    }
    CHECK(rejectedIndex);

    Matrix matrix(1, 1);
    bool rejectedMatrixIndex = false;
    try {
        (void)matrix.at(1, 0);
    } catch (const out_of_range&) {
        rejectedMatrixIndex = true;
    }
    CHECK(rejectedMatrixIndex);
}

void testMultilevelTreeSummaryAndLayouts() {
    const vector<vector<double>> values{
        {0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0},
        {4.0, 4.0}, {5.0, 5.0}, {6.0, 6.0}, {7.0, 7.0}
    };
    const vector<double> target{0.0, 1.0, 0.0, 1.0, 10.0, 11.0, 10.0, 11.0};

    Matrix rowMajor(values, MatrixLayout::RowMajor);
    Matrix columnMajor(values, MatrixLayout::ColumnMajor);
    CartRegressor rowTree(2);
    CartRegressor columnTree(2);
    rowTree.fit(rowMajor, target);
    columnTree.fit(columnMajor, target);

    const CartTreeSummary summary = rowTree.summary();
    CHECK(summary.depth == 2);
    CHECK(summary.nodeCount == 7);
    CHECK(summary.leafCount == 4);
    CHECK(summary.splits.size() == 3);
    CHECK(rowTree.predict(rowMajor) == columnTree.predict(columnMajor));

    CartRegressor stumpDepth(1);
    stumpDepth.fit(rowMajor, target);
    CHECK(stumpDepth.summary().depth == 1);
    CHECK(stumpDepth.predict(rowMajor) != rowTree.predict(rowMajor));

    const CartTreeSummary columnSummary = columnTree.summary();
    CHECK(summary.nodeCount == columnSummary.nodeCount);
    CHECK(summary.leafCount == columnSummary.leafCount);
    CHECK(summary.depth == columnSummary.depth);
    for (size_t i = 0; i < summary.splits.size(); ++i) {
        CHECK(summary.splits[i].featureIndex == columnSummary.splits[i].featureIndex);
        CHECK(summary.splits[i].threshold == columnSummary.splits[i].threshold);
    }

    CartClassifier rowClassifier(2);
    CartClassifier columnClassifier(2);
    rowClassifier.fit(rowMajor, target);
    columnClassifier.fit(columnMajor, target);
    CHECK(rowClassifier.predict(rowMajor) == columnClassifier.predict(columnMajor));

    const Matrix rowProbabilities = rowClassifier.predict_proba(rowMajor);
    const Matrix columnProbabilities = columnClassifier.predict_proba(columnMajor);
    CHECK(rowProbabilities.rows() == columnProbabilities.rows());
    CHECK(rowProbabilities.columns() == columnProbabilities.columns());
    for (size_t i = 0; i < rowProbabilities.rows(); ++i) {
        for (size_t j = 0; j < rowProbabilities.columns(); ++j) {
            CHECK(rowProbabilities(i, j) == columnProbabilities(i, j));
        }
    }
}

void testWeightedRegressionWithBootstrapCounts() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0}
    });

    const vector<double> target{0.0, 10.0, 100.0};
    const vector<size_t> counts{3, 1, 0};

    CartRegressor model(0);
    model.fit_with_counts(features, target, counts);

    const vector<double> predictions = model.predict(features);

    // (0 * 3 + 10 * 1) / (3 + 1) = 2.5.
    // Третий объект имеет нулевую кратность и не влияет на результат.
    for (double prediction : predictions) {
        CHECK(close(prediction, 2.5));
    }
}

void testWeightedClassificationWithBootstrapCounts() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0}
    });

    const vector<double> target{10.0, 42.0, 100.0};
    const vector<size_t> counts{3, 1, 0};

    CartClassifier model(0);
    model.fit_with_counts(features, target, counts);

    const span<const double> classes = model.classes();

    // Все классы исходной выборки должны сохраняться,
    // даже если кратность объекта соответствующего класса равна нулю.
    CHECK(classes.size() == 3);
    CHECK(classes[0] == 10.0);
    CHECK(classes[1] == 42.0);
    CHECK(classes[2] == 100.0);

    const Matrix probabilities = model.predict_proba(features);

    for (size_t i = 0; i < probabilities.rows(); ++i) {
        CHECK(close(probabilities(i, 0), 0.75));
        CHECK(close(probabilities(i, 1), 0.25));
        CHECK(close(probabilities(i, 2), 0.0));
    }

    const vector<double> predictions = model.predict(features);

    for (double prediction : predictions) {
        CHECK(prediction == 10.0);
    }
}

void testZeroBootstrapCountsAreIgnored() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    });

    const vector<double> target{5.0, 5.0, 1000.0, 2000.0};
    const vector<size_t> counts{1, 1, 0, 0};

    CartRegressor model(0);
    model.fit_with_counts(features, target, counts);

    const vector<double> predictions = model.predict(features);

    for (double prediction : predictions) {
        CHECK(close(prediction, 5.0));
    }
}

void testRandomStateReproducibility() {
    Matrix features({
        {0.0, 1.0, 4.0, 2.0},
        {1.0, 3.0, 2.0, 0.0},
        {2.0, 0.0, 1.0, 4.0},
        {3.0, 4.0, 3.0, 1.0},
        {4.0, 2.0, 0.0, 3.0},
        {5.0, 5.0, 5.0, 5.0}
    });

    const vector<double> target{
        0.0, 10.0, 0.0, 10.0, 20.0, 20.0
    };

    const FeatureSearchParams searchParams{
        FeatureMode::Sqrt, 1
    };

    CartRegressor first(
        3, 2, 1, searchParams, 12345
    );

    CartRegressor second(
        3, 2, 1, searchParams, 12345
    );

    first.fit(features, target);
    second.fit(features, target);

    const CartTreeSummary firstSummary = first.summary();
    const CartTreeSummary secondSummary = second.summary();

    CHECK(firstSummary.nodeCount == secondSummary.nodeCount);
    CHECK(firstSummary.leafCount == secondSummary.leafCount);
    CHECK(firstSummary.depth == secondSummary.depth);
    CHECK(firstSummary.splits.size() == secondSummary.splits.size());

    for (size_t i = 0; i < firstSummary.splits.size(); ++i) {
        CHECK(
            firstSummary.splits[i].featureIndex ==
            secondSummary.splits[i].featureIndex
        );

        CHECK(
            firstSummary.splits[i].threshold ==
            secondSummary.splits[i].threshold
        );
    }

    CHECK(first.predict(features) == second.predict(features));
}

void testRandomFeatureSearchModes() {
    Matrix features({
        {0.0, 5.0, 2.0, 8.0},
        {1.0, 4.0, 3.0, 7.0},
        {2.0, 3.0, 4.0, 6.0},
        {3.0, 2.0, 5.0, 5.0},
        {4.0, 1.0, 6.0, 4.0},
        {5.0, 0.0, 7.0, 3.0}
    });

    const vector<double> target{
        0.0, 0.0, 10.0, 10.0, 20.0, 20.0
    };

    CartRegressor sqrtModel(
        3,
        2,
        1,
        FeatureSearchParams{FeatureMode::Sqrt, 1},
        42
    );

    CartRegressor countModel(
        3,
        2,
        1,
        FeatureSearchParams{FeatureMode::Count, 2},
        42
    );

    sqrtModel.fit(features, target);
    countModel.fit(features, target);

    const vector<double> sqrtPredictions =
        sqrtModel.predict(features);

    const vector<double> countPredictions =
        countModel.predict(features);

    CHECK(sqrtPredictions.size() == target.size());
    CHECK(countPredictions.size() == target.size());

    for (double prediction : sqrtPredictions) {
        CHECK(isfinite(prediction));
    }

    for (double prediction : countPredictions) {
        CHECK(isfinite(prediction));
    }
}

void testUnboundedTreeDepth() {
    const vector<vector<double>> values{
        {0.0, 0.0},
        {1.0, 1.0},
        {2.0, 2.0},
        {3.0, 3.0},
        {4.0, 4.0},
        {5.0, 5.0},
        {6.0, 6.0},
        {7.0, 7.0}
    };

    const vector<double> target{
        0.0, 1.0, 0.0, 1.0,
        10.0, 11.0, 10.0, 11.0
    };

    Matrix features(values);

    CartRegressor model(
        nullopt,
        2,
        1,
        FeatureSearchParams{FeatureMode::All, 1},
        123
    );

    model.fit(features, target);

    const CartTreeSummary summary = model.summary();

    // Без ограничения глубины дерево должно иметь
    // возможность построить больше одного уровня.
    CHECK(summary.depth >= 2);
    CHECK(summary.nodeCount > 3);

    const vector<double> predictions = model.predict(features);

    CHECK(predictions.size() == target.size());

    for (double prediction : predictions) {
        CHECK(isfinite(prediction));
    }
}


void testWeightedSplitScoresAgainstManualReference() {
    // Регрессия: вручную считаем S_L^2/W_L + S_R^2/W_R.
    Matrix regressionFeatures({{0.0}, {1.0}, {2.0}});
    const vector<double> regressionTarget{0.0, 10.0, 0.0};
    const vector<size_t> regressionObjects{0, 1, 2};
    const vector<size_t> regressionFeature{0};
    const vector<size_t> regressionCounts{2, 1, 3};

    const auto regressionSplit = SplitFinder::findBestSplit(
        regressionFeatures,
        regressionTarget,
        regressionObjects,
        regressionFeature,
        SplitCriterion::Variance,
        1,
        regressionCounts
    );

    CHECK(regressionSplit.has_value());
    CHECK(regressionSplit->threshold == 1.5);
    // Для выбранного разбиения: (10^2 / 3) + (0^2 / 3) = 100/3.
    CHECK(close(static_cast<double>(regressionSplit->score), 100.0 / 3.0));

    // Классификация: вручную считаем sum(n_c^2)/W для обеих частей.
    Matrix classificationFeatures({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> classificationTarget{0.0, 0.0, 1.0, 1.0};
    const vector<size_t> classificationObjects{0, 1, 2, 3};
    const vector<size_t> classificationFeature{0};
    const vector<size_t> classificationCounts{3, 1, 2, 2};

    const auto classificationSplit = SplitFinder::findBestSplit(
        classificationFeatures,
        classificationTarget,
        classificationObjects,
        classificationFeature,
        SplitCriterion::Gini,
        1,
        classificationCounts
    );

    CHECK(classificationSplit.has_value());
    CHECK(classificationSplit->threshold == 1.5);
    // Слева четыре объекта класса 0, справа четыре класса 1: 16/4 + 16/4.
    CHECK(close(static_cast<double>(classificationSplit->score), 8.0));
}

void testMinimumLeafAndSplitUseUniqueSourceObjects() {
    Matrix features(vector<vector<double>>{{0.0}, {1.0}});
    const vector<double> target{0.0, 10.0};
    const vector<size_t> counts{10, 1};

    // Сумма кратностей равна 11, но различных исходных индексов только 2.
    CartRegressor splitLimited(nullopt, 3, 1);
    splitLimited.fit_with_counts(features, target, counts);
    CHECK(splitLimited.summary().nodeCount == 1);

    CartRegressor leafLimited(nullopt, 2, 2);
    leafLimited.fit_with_counts(features, target, counts);
    CHECK(leafLimited.summary().nodeCount == 1);
    const auto predictions = leafLimited.predict(features);
    CHECK(close(predictions[0], 10.0 / 11.0));
    CHECK(close(predictions[1], 10.0 / 11.0));
}

void testZeroImpurityImprovementDoesNotBlockSplit() {
    // Оба потомка имеют ту же дисперсию, что и корень, но target не константен.
    Matrix features({{0.0}, {0.0}, {1.0}, {1.0}});
    const vector<double> target{0.0, 1.0, 0.0, 1.0};

    CartRegressor model(1);
    model.fit(features, target);
    const CartTreeSummary summary = model.summary();

    CHECK(summary.depth == 1);
    CHECK(summary.nodeCount == 3);
    CHECK(summary.splits.size() == 1);
    CHECK(summary.splits[0].threshold == 0.5);
}

void testFallbackAfterConstantFirstFeatureCandidate() {
    Matrix features({
        {0.0, 0.0},
        {0.0, 1.0},
        {0.0, 2.0},
        {0.0, 3.0}
    });
    const vector<double> target{0.0, 0.0, 10.0, 10.0};

    // Для seed 0 первым просматривается признак 0, который константен.
    // Алгоритм должен продолжить поиск и найти разбиение по признаку 1.
    CartRegressor model(
        1,
        2,
        1,
        FeatureSearchParams{FeatureMode::Count, 1},
        0
    );
    model.fit(features, target);
    const CartTreeSummary summary = model.summary();

    CHECK(summary.splits.size() == 1);
    CHECK(summary.splits[0].featureIndex == 1);
}



void testFeatureSelectionIsRepeatedAtEachNode() {
    Matrix features({
        {0.0, 0.0}, {0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0},
        {1.0, 0.0}, {1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}
    });

    const vector<double> target{
        0.0, 0.0, 10.0, 10.0,
        100.0, 100.0, 110.0, 110.0
    };

    CartRegressor model(
        2,
        2,
        1,
        FeatureSearchParams{FeatureMode::Count, 1},
        0
    );

    model.fit(features, target);
    const CartTreeSummary summary = model.summary();

    CHECK(summary.depth == 2);
    CHECK(summary.splits.size() == 3);

    CHECK(summary.splits[0].depth == 0);
    CHECK(summary.splits[1].depth == 1);
    CHECK(summary.splits[2].depth == 1);

    bool selectedFeature0 = false;
    bool selectedFeature1 = false;

    for (const auto& split : summary.splits) {
        CHECK(split.featureIndex < features.columns());

        if (split.featureIndex == 0) {
            selectedFeature0 = true;
        } else if (split.featureIndex == 1) {
            selectedFeature1 = true;
        }
    }

    // При max_features=1 разные узлы могут выбирать разные признаки.
    CHECK(selectedFeature0);
    CHECK(selectedFeature1);
}

} // namespace

int main() {
    try {
        testSplitTieBreaksByFeatureAndThreshold();
        testAdjacentRepresentableThresholdRoutesCorrectly();
        testSplitFinderHonorsFeatureSubset();
        testIdenticalTargetsStopBothTrees();
        testClassificationStumpEquivalence();
        testProbabilitiesUseGlobalSortedClassesAndTieBreak();
        testLocalClassEncodingWithClassesAbsentFromLeaves();
        testRefitEmptyPredictionAndClassReplacement();
        testRegressorRepeatedFit();
        testParametersAndImpossibleMinimumLeafSize();
        testSingleObjectTraining();
        testFitAndPredictContractErrors();
        testDefaultAndCheckedMatrixSliceAccess();
        testMultilevelTreeSummaryAndLayouts();
        testWeightedRegressionWithBootstrapCounts();
        testWeightedClassificationWithBootstrapCounts();
        testZeroBootstrapCountsAreIgnored();
        testRandomStateReproducibility();
        testRandomFeatureSearchModes();
        testUnboundedTreeDepth();
        testWeightedSplitScoresAgainstManualReference();
        testMinimumLeafAndSplitUseUniqueSourceObjects();
        testZeroImpurityImprovementDoesNotBlockSplit();
        testFallbackAfterConstantFirstFeatureCandidate();
        testFeatureSelectionIsRepeatedAtEachNode();

        std::cout << "All CART audit tests passed!\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "CART audit test failed: " << e.what() << '\n';
        return 1;
    }
}