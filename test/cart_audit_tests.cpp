#include "cart.hpp"
#include "decision_stump.hpp"
#include "exceptions.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace std;

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

    assert(split.has_value());
    assert(split->featureIndex == 0);
    assert(close(split->threshold, 0.5));
}

void testAdjacentRepresentableThresholdRoutesCorrectly() {
    const double lower = 1.0;
    const double upper = nextafter(lower, numeric_limits<double>::infinity());
    Matrix features(vector<vector<double>>{{lower}, {upper}});
    const vector<double> target{0.0, 10.0};

    CartRegressor model(1, 2, 1);
    model.fit(features, target);
    const CartTreeSummary summary = model.summary();

    assert(summary.splits.size() == 1);
    assert(summary.splits[0].threshold == lower);
    const vector<double> predictions = model.predict(features);
    assert(predictions[0] == 0.0);
    assert(predictions[1] == 10.0);
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

    assert(!split.has_value());

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
    assert(giniSplit.has_value());
    assert(giniSplit->featureIndex == 0);
}

void testIdenticalTargetsStopBothTrees() {
    Matrix features({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> regressionTarget{4.0, 4.0, 4.0, 4.0};
    const vector<double> classificationTarget{42.0, 42.0, 42.0, 42.0};

    CartRegressor regressor(10);
    regressor.fit(features, regressionTarget);
    const CartTreeSummary regressionSummary = regressor.summary();
    assert(regressionSummary.nodeCount == 1);
    assert(regressionSummary.leafCount == 1);
    assert(regressionSummary.depth == 0);

    CartClassifier classifier(10);
    classifier.fit(features, classificationTarget);
    const CartTreeSummary classificationSummary = classifier.summary();
    assert(classificationSummary.nodeCount == 1);
    assert(classificationSummary.leafCount == 1);
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

    assert(tree.predict(features) == stump.predict(features));
}

void testProbabilitiesUseGlobalSortedClassesAndTieBreak() {
    Matrix features({{0.0}, {1.0}, {2.0}});
    const vector<double> target{42.0, 10.0, 42.0};

    CartClassifier classifier(0);
    classifier.fit(features, target);

    const span<const double> classes = classifier.classes();
    assert(classes.size() == 2);
    assert(classes[0] == 10.0);
    assert(classes[1] == 42.0);

    Matrix probabilities = classifier.predict_proba(features);
    assert(close(probabilities(0, 0), 1.0 / 3.0));
    assert(close(probabilities(0, 1), 2.0 / 3.0));

    Matrix separatedFeatures({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> separatedTarget{10.0, 10.0, 42.0, 42.0};
    CartClassifier separated(1);
    separated.fit(separatedFeatures, separatedTarget);
    const Matrix leafProbabilities =
        separated.predict_proba(separatedFeatures);
    assert(leafProbabilities(0, 0) == 1.0);
    assert(leafProbabilities(0, 1) == 0.0);
    assert(leafProbabilities(3, 0) == 0.0);
    assert(leafProbabilities(3, 1) == 1.0);

    Matrix tieFeatures(vector<vector<double>>{{0.0}, {1.0}});
    const vector<double> tieTarget{10.0, 42.0};
    CartClassifier tied(0);
    tied.fit(tieFeatures, tieTarget);
    assert(tied.predict(tieFeatures)[0] == 10.0);
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
    assert(classes.size() == 3);
    assert(classes[0] == 7.0 && classes[1] == 9.0 && classes[2] == 11.0);

    Matrix emptyTest(0, 1);
    assert(classifier.predict(emptyTest).empty());
    Matrix emptyProbabilities = classifier.predict_proba(emptyTest);
    assert(emptyProbabilities.rows() == 0);
    assert(emptyProbabilities.columns() == 3);
}

void testRegressorRepeatedFit() {
    Matrix features({{0.0}, {1.0}, {2.0}});
    const vector<double> firstTarget{1.0, 2.0, 3.0};
    const vector<double> secondTarget{10.0, 20.0, 30.0};
    CartRegressor regressor(0);

    regressor.fit(features, firstTarget);
    regressor.fit(features, secondTarget);
    const vector<double> predictions = regressor.predict(features);
    assert(predictions.size() == 3);
    assert(predictions[0] == 20.0);
    assert(predictions[2] == 20.0);

    Matrix emptyTest(0, 1);
    assert(regressor.predict(emptyTest).empty());
}

void testParametersAndImpossibleMinimumLeafSize() {
    bool rejectedSplit = false;
    try {
        CartRegressor invalid(1, 1, 1);
    } catch (const InvalidParameterError&) {
        rejectedSplit = true;
    }
    assert(rejectedSplit);

    bool rejectedLeaf = false;
    try {
        CartClassifier invalid(1, 2, 0);
    } catch (const InvalidParameterError&) {
        rejectedLeaf = true;
    }
    assert(rejectedLeaf);

    bool rejectedDepth = false;
    try {
        CartRegressor invalid(65);
    } catch (const InvalidParameterError&) {
        rejectedDepth = true;
    }
    assert(rejectedDepth);

    Matrix features({{0.0}, {1.0}, {2.0}});
    const vector<double> target{0.0, 1.0, 2.0};
    CartRegressor noSplit(5, 2, numeric_limits<size_t>::max());
    noSplit.fit(features, target);
    assert(noSplit.summary().nodeCount == 1);

    CartRegressor tooFewObjects(5, 4, 1);
    tooFewObjects.fit(features, target);
    assert(tooFewObjects.summary().nodeCount == 1);

    Matrix leafLimitedFeatures({{0.0}, {1.0}, {2.0}, {3.0}});
    const vector<double> leafLimitedTarget{0.0, 0.0, 0.0, 100.0};
    CartRegressor leafLimited(1, 2, 2);
    leafLimited.fit(leafLimitedFeatures, leafLimitedTarget);
    const vector<double> leafLimitedPredictions =
        leafLimited.predict(leafLimitedFeatures);
    assert(leafLimitedPredictions[0] == 0.0);
    assert(leafLimitedPredictions[3] == 50.0);
}

void testSingleObjectTraining() {
    Matrix features(vector<vector<double>>{{3.0}});
    const vector<double> regressionTarget{17.0};
    CartRegressor regressor(8);
    regressor.fit(features, regressionTarget);
    assert(regressor.summary().nodeCount == 1);
    assert(regressor.predict(features)[0] == 17.0);

    const vector<double> classificationTarget{42.0};
    CartClassifier classifier(8);
    classifier.fit(features, classificationTarget);
    assert(classifier.summary().nodeCount == 1);
    assert(classifier.predict(features)[0] == 42.0);
    const Matrix probabilities = classifier.predict_proba(features);
    assert(probabilities(0, 0) == 1.0);
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
    assert(predictBeforeFitRejected);

    bool emptyTrainingRejected = false;
    try {
        Matrix emptyFeatures(0, 1);
        const vector<double> emptyTarget;
        regressor.fit(emptyFeatures, emptyTarget);
    } catch (const InvalidParameterError&) {
        emptyTrainingRejected = true;
    }
    assert(emptyTrainingRejected);

    bool zeroColumnsRejected = false;
    try {
        Matrix noFeatures(1, 0);
        regressor.fit(noFeatures, validTarget);
    } catch (const InvalidParameterError&) {
        zeroColumnsRejected = true;
    }
    assert(zeroColumnsRejected);

    bool sizeMismatchRejected = false;
    try {
        const vector<double> wrongSizeTarget{2.0, 3.0};
        regressor.fit(validFeatures, wrongSizeTarget);
    } catch (const DimensionMismatchError&) {
        sizeMismatchRejected = true;
    }
    assert(sizeMismatchRejected);

    bool nonFiniteTargetRejected = false;
    try {
        const vector<double> badTarget{infinity};
        regressor.fit(validFeatures, badTarget);
    } catch (const InvalidParameterError&) {
        nonFiniteTargetRejected = true;
    }
    assert(nonFiniteTargetRejected);

    Matrix nonFiniteFeatures(vector<vector<double>>{{infinity}});
    bool nonFiniteFeaturesRejected = false;
    try {
        regressor.fit(nonFiniteFeatures, validTarget);
    } catch (const InvalidParameterError&) {
        nonFiniteFeaturesRejected = true;
    }
    assert(nonFiniteFeaturesRejected);

    regressor.fit(validFeatures, validTarget);
    bool wrongPredictWidthRejected = false;
    try {
        Matrix wrongWidth(1, 2);
        (void)regressor.predict(wrongWidth);
    } catch (const DimensionMismatchError&) {
        wrongPredictWidthRejected = true;
    }
    assert(wrongPredictWidthRejected);

    bool nonFinitePredictionRejected = false;
    try {
        (void)regressor.predict(nonFiniteFeatures);
    } catch (const InvalidParameterError&) {
        nonFinitePredictionRejected = true;
    }
    assert(nonFinitePredictionRejected);

    CartClassifier classifier(1);
    bool classesBeforeFitRejected = false;
    try {
        (void)classifier.classes();
    } catch (const ModelUsageError&) {
        classesBeforeFitRejected = true;
    }
    assert(classesBeforeFitRejected);
}

void testDefaultAndCheckedMatrixSliceAccess() {
    MatrixSlice<double> empty;
    assert(empty.empty());
    assert(empty.data() == nullptr);
    assert(empty.stride() == 1);

    bool rejectedIndex = false;
    try {
        (void)empty.at(0);
    } catch (const out_of_range&) {
        rejectedIndex = true;
    }
    assert(rejectedIndex);

    Matrix matrix(1, 1);
    bool rejectedMatrixIndex = false;
    try {
        (void)matrix.at(1, 0);
    } catch (const out_of_range&) {
        rejectedMatrixIndex = true;
    }
    assert(rejectedMatrixIndex);
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
    assert(summary.depth == 2);
    assert(summary.nodeCount == 7);
    assert(summary.leafCount == 4);
    assert(summary.splits.size() == 3);
    assert(rowTree.predict(rowMajor) == columnTree.predict(columnMajor));

    CartRegressor stumpDepth(1);
    stumpDepth.fit(rowMajor, target);
    assert(stumpDepth.summary().depth == 1);
    assert(stumpDepth.predict(rowMajor) != rowTree.predict(rowMajor));

    const CartTreeSummary columnSummary = columnTree.summary();
    assert(summary.nodeCount == columnSummary.nodeCount);
    assert(summary.leafCount == columnSummary.leafCount);
    assert(summary.depth == columnSummary.depth);
    for (size_t i = 0; i < summary.splits.size(); ++i) {
        assert(summary.splits[i].featureIndex == columnSummary.splits[i].featureIndex);
        assert(summary.splits[i].threshold == columnSummary.splits[i].threshold);
    }

    CartClassifier rowClassifier(2);
    CartClassifier columnClassifier(2);
    rowClassifier.fit(rowMajor, target);
    columnClassifier.fit(columnMajor, target);
    assert(rowClassifier.predict(rowMajor) == columnClassifier.predict(columnMajor));

    const Matrix rowProbabilities = rowClassifier.predict_proba(rowMajor);
    const Matrix columnProbabilities = columnClassifier.predict_proba(columnMajor);
    assert(rowProbabilities.rows() == columnProbabilities.rows());
    assert(rowProbabilities.columns() == columnProbabilities.columns());
    for (size_t i = 0; i < rowProbabilities.rows(); ++i) {
        for (size_t j = 0; j < rowProbabilities.columns(); ++j) {
            assert(rowProbabilities(i, j) == columnProbabilities(i, j));
        }
    }
}

} // namespace

int main() {
    testSplitTieBreaksByFeatureAndThreshold();
    testAdjacentRepresentableThresholdRoutesCorrectly();
    testSplitFinderHonorsFeatureSubset();
    testIdenticalTargetsStopBothTrees();
    testClassificationStumpEquivalence();
    testProbabilitiesUseGlobalSortedClassesAndTieBreak();
    testRefitEmptyPredictionAndClassReplacement();
    testRegressorRepeatedFit();
    testParametersAndImpossibleMinimumLeafSize();
    testSingleObjectTraining();
    testFitAndPredictContractErrors();
    testDefaultAndCheckedMatrixSliceAccess();
    testMultilevelTreeSummaryAndLayouts();
}
