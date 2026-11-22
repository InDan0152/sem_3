#include <sstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <cxxopts.hpp>
#include <memory>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <functional>
#include <chrono>

#include "csv_loader.hpp"
#include "statistics.hpp"
#include "preprocessing.hpp"
#include "exceptions.hpp"
#include "supervised_dataset.hpp"
#include "dataset_splitter.hpp"
#include "matrix.hpp"
#include "decision_stump.hpp"
#include "cart.hpp"
#include "random_forest.hpp"
#include "model.hpp"

using namespace std;

bool parseFillStrategy(
    const string& value,
    NumericalMissingStrategy& strategy
) {
    if (value == "mean") {
        strategy = NumericalMissingStrategy::Mean;
    }
    else if (value == "median") {
        strategy = NumericalMissingStrategy::Median;
    }
    else if (value == "zero") {
        strategy = NumericalMissingStrategy::Zero;
    }
    else {
        return false;
    }

    return true;
}
bool parseScalingStrategy(
    const string& value,
    ScalingStrategy& strategy
) {
    if (value == "minmax") {
        strategy = ScalingStrategy::MinMax;
    }
    else if (value == "standard") {
        strategy = ScalingStrategy::Standard;
    }
    else if (value == "robust") {
        strategy = ScalingStrategy::Robust;
    }
    else {
        return false;
    }

    return true;
}

bool parseEncodingStrategy(const string& value, EncodingStrategy& strategy) {
    if (value == "onehot") {
        strategy = EncodingStrategy::OneHot;
    }
    else if (value == "ordinal") {
        strategy = EncodingStrategy::Ordinal;
    }
    else {
        return false;
    }

    return true;
}

void printCartSummary(
    const CartTreeSummary& summary,
    const vector<string>& featureNames
) {
    cout << "\nСтруктура дерева CART:\n"
         << "Узлов: " << summary.nodeCount << '\n'
         << "Листьев: " << summary.leafCount << '\n'
         << "Фактическая глубина: " << summary.depth << '\n';

    const size_t displayedSplits = min(size_t{5}, summary.splits.size());
    cout << "Разбиения (первые " << displayedSplits << "):\n";

    for (size_t i = 0; i < displayedSplits; ++i) {
        const CartSplitDescription& split = summary.splits[i];
        const string featureName = split.featureIndex < featureNames.size()
            ? featureNames[split.featureIndex]
            : "feature_" + to_string(split.featureIndex);

        cout << "  depth " << split.depth << ": "
             << featureName << " <= "
             << setprecision(8) << split.threshold << '\n';
    }
}

void printForestSummary(
    const vector<CartTreeSummary>& summaries,
    const vector<string>& featureNames
) {
    size_t totalNodes = 0;
    size_t totalLeaves = 0;
    size_t maximumDepth = 0;

    for (const auto& summary : summaries) {
        totalNodes += summary.nodeCount;
        totalLeaves += summary.leafCount;
        maximumDepth = max(maximumDepth, summary.depth);
    }

    cout << "\nСтруктура случайного леса:\n"
         << "Деревьев: " << summaries.size() << '\n'
         << "Всего узлов: " << totalNodes << '\n'
         << "Всего листьев: " << totalLeaves << '\n'
         << "Максимальная глубина дерева: " << maximumDepth << '\n';

    const size_t displayedTrees = min(size_t{5}, summaries.size());
    cout << "Размеры первых " << displayedTrees << " деревьев:\n";
    for (size_t i = 0; i < displayedTrees; ++i) {
        cout << "  Дерево " << i + 1
             << ": узлов " << summaries[i].nodeCount
             << ", листьев " << summaries[i].leafCount
             << ", глубина " << summaries[i].depth << '\n';
    }

    if (!summaries.empty()) {
        cout << "\nРазбиения первого дерева:";
        printCartSummary(summaries.front(), featureNames);
    }
}

void printDecisionStumpSummary(
    const DecisionStump& stump,
    const vector<string>& featureNames
) {
    const streamsize previousPrecision = cout.precision();
    const ios::fmtflags previousFlags = cout.flags();

    cout << "\nРешающий пень:\n";
    if (!stump.has_split()) {
        cout << "Допустимое разбиение не найдено.\n"
             << "Константное предсказание: "
             << setprecision(8) << stump.constant_prediction() << '\n';
    }
    else {
        const size_t featureIndex = stump.split_feature();
        const string featureName = featureIndex < featureNames.size()
            ? featureNames[featureIndex]
            : "feature_" + to_string(featureIndex);

        cout << "Индекс признака: " << featureIndex
             << " (" << featureName << ")\n"
             << "Порог: " << setprecision(8) << stump.split_threshold() << '\n'
             << "Предсказание левого листа: "
             << stump.left_prediction() << '\n'
             << "Предсказание правого листа: "
             << stump.right_prediction() << '\n';
    }

    cout.flags(previousFlags);
    cout.precision(previousPrecision);
}


// Универсальный запуск моделей машинного обучения.
int runMachineLearning(
    const Dataset& dataset,
    const string& targetName,
    const PreprocessingConfig& preprocessingConfig,
    const string& modelName,
    const string& task,
    size_t maxDepth,
    size_t minSamplesSplit,
    size_t minSamplesLeaf,
    size_t nEstimators,
    size_t nJobs
) {
    // Подготовка целевого признака.
    SupervisedDataset supervisedDataset;
    map<double, string> classDisplayNames;

    if (task == "classification") {
        const map<string, double> targetMapping = {
            {"no", 0.0},
            {"yes", 1.0}
        };

        for (const auto& [name, value] : targetMapping) {
            classDisplayNames.emplace(value, name);
        }

        supervisedDataset = prepareSupervisedDataset(
            dataset,
            targetName,
            targetMapping
        );
    }
    else if (task == "regression") {
        supervisedDataset = prepareSupervisedDataset(
            dataset,
            targetName
        );
    }
    else {
        throw invalid_argument(
            "Неизвестная задача: " + task
        );
    }

    // Разделение данных.
    DatasetSplit split = splitDataset(
        supervisedDataset,
        0.2,
        42
    );

    cout << "\nОбучающих объектов: "
         << split.train.targets.size() << '\n';

    cout << "Тестовых объектов: "
         << split.test.targets.size() << '\n';

    // Предобработка обучающей и тестовой выборок.
    DatasetPreprocessor preprocessor(preprocessingConfig);

    preprocessor.fit(split.train.features);

    ProcessedDataset trainProcessed =
        preprocessor.transform(split.train.features);

    ProcessedDataset testProcessed =
        preprocessor.transform(split.test.features);

    Matrix trainMatrix(trainProcessed.data);
    Matrix testMatrix(testProcessed.data);

    cout << "\nПредобработка завершена.\n";
    cout << "Количество признаков: "
         << trainMatrix.columns() << '\n';

    cout << "\nАлгоритм: " << modelName << '\n';
    cout << "Задача: "
         << (task == "classification" ? "классификация" : "регрессия")
         << '\n';

    // ========================================================
    // СРАВНЕНИЕ CART И RANDOM FOREST
    // ========================================================

    if (modelName == "random_forest") {
        RandomForestParams forestParams;
        forestParams.n_estimators = nEstimators;
        forestParams.n_jobs = nJobs;
        forestParams.max_depth = maxDepth;
        forestParams.min_samples_split = minSamplesSplit;
        forestParams.min_samples_leaf = minSamplesLeaf;
        forestParams.random_state = 42;
        forestParams.max_features = FeatureSearchParams{
            task == "classification" ? FeatureMode::Sqrt : FeatureMode::All,
            1
        };

        cout << "\nСравнение CART и Random Forest на одной выборке\n"
             << "Максимальная глубина: " << maxDepth << '\n'
             << "Минимум для разбиения: " << minSamplesSplit << '\n'
             << "Минимум в листе: " << minSamplesLeaf << '\n'
             << "Количество деревьев: " << nEstimators << '\n'
             << "Количество потоков: " << nJobs << '\n'
             << "Bootstrap: включён\n"
             << "Seed: " << forestParams.random_state << '\n'
             << "max_features: "
             << (task == "classification" ? "Sqrt" : "All") << '\n';

        using Clock = chrono::steady_clock;

        if (task == "classification") {
            CartClassifier cart(maxDepth, minSamplesSplit, minSamplesLeaf);
            RandomForestClassifier forest(forestParams);

            auto start = Clock::now();
            cart.fit(trainMatrix, span<const double>(split.train.targets));
            const double cartFitMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            forest.fit(trainMatrix, span<const double>(split.train.targets));
            const double forestFitMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            const vector<double> cartPredictions = cart.predict(testMatrix);
            const double cartPredictMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            const vector<double> forestPredictions = forest.predict(testMatrix);
            const double forestPredictMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            const Matrix cartProbabilities = cart.predict_proba(testMatrix);
            const double cartProbabilityMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            const Matrix forestProbabilities = forest.predict_proba(testMatrix);
            const double forestProbabilityMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            const vector<double> classes(cart.classes().begin(), cart.classes().end());

            size_t cartCorrect = 0;
            size_t forestCorrect = 0;
            for (size_t i = 0; i < split.test.targets.size(); ++i) {
                cartCorrect += cartPredictions[i] == split.test.targets[i] ? 1U : 0U;
                forestCorrect += forestPredictions[i] == split.test.targets[i] ? 1U : 0U;
            }

            cout << fixed << setprecision(3);
            cout << "\nВремя работы (мс):\n"
                 << "CART: fit " << cartFitMs
                 << ", predict " << cartPredictMs
                 << ", predict_proba " << cartProbabilityMs << '\n'
                 << "Random Forest: fit " << forestFitMs
                 << ", predict " << forestPredictMs
                 << ", predict_proba " << forestProbabilityMs << '\n';

            cout << "\nAccuracy:\n"
                 << "CART: " << static_cast<double>(cartCorrect) / split.test.targets.size() << '\n'
                 << "Random Forest: " << static_cast<double>(forestCorrect) / split.test.targets.size() << '\n';

            printCartSummary(cart.summary(), trainProcessed.featureNames);
            printForestSummary(forest.tree_summaries(), trainProcessed.featureNames);

            const size_t outputCount = min(size_t{10}, forestPredictions.size());
            cout << "\nПервые " << outputCount << " предсказаний и вероятности классов:\n";
            for (size_t i = 0; i < outputCount; ++i) {
                cout << "\nОбъект " << i + 1
                     << ": факт " << split.test.targets[i]
                     << ", CART " << cartPredictions[i]
                     << ", Random Forest " << forestPredictions[i] << '\n';
                for (size_t j = 0; j < classes.size(); ++j) {
                    const auto name = classDisplayNames.find(classes[j]);
                    const string label = name == classDisplayNames.end()
                        ? to_string(classes[j]) : name->second;
                    cout << "  Класс " << label
                         << ": CART " << cartProbabilities(i, j) * 100.0
                         << "%, Random Forest " << forestProbabilities(i, j) * 100.0
                         << "%\n";
                }
            }
        } else {
            CartRegressor cart(maxDepth, minSamplesSplit, minSamplesLeaf);
            RandomForestRegressor forest(forestParams);

            auto start = Clock::now();
            cart.fit(trainMatrix, span<const double>(split.train.targets));
            const double cartFitMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            forest.fit(trainMatrix, span<const double>(split.train.targets));
            const double forestFitMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            const vector<double> cartPredictions = cart.predict(testMatrix);
            const double cartPredictMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            start = Clock::now();
            const vector<double> forestPredictions = forest.predict(testMatrix);
            const double forestPredictMs = chrono::duration<double, milli>(
                Clock::now() - start).count();

            auto calculateMae = [&](const vector<double>& predictions) {
                double errorSum = 0.0;
                for (size_t i = 0; i < predictions.size(); ++i) {
                    errorSum += abs(predictions[i] - split.test.targets[i]);
                }
                return errorSum / static_cast<double>(predictions.size());
            };

            cout << fixed << setprecision(3);
            cout << "\nВремя работы (мс):\n"
                 << "CART: fit " << cartFitMs
                 << ", predict " << cartPredictMs << '\n'
                 << "Random Forest: fit " << forestFitMs
                 << ", predict " << forestPredictMs << '\n';

            cout << "\nMAE:\n"
                 << "CART: " << calculateMae(cartPredictions) << '\n'
                 << "Random Forest: " << calculateMae(forestPredictions) << '\n';

            printCartSummary(cart.summary(), trainProcessed.featureNames);
            printForestSummary(forest.tree_summaries(), trainProcessed.featureNames);

            const size_t outputCount = min(size_t{10}, forestPredictions.size());
            cout << "\nПервые " << outputCount << " предсказаний:\n";
            cout << "№ | Факт | CART | Random Forest\n";
            for (size_t i = 0; i < outputCount; ++i) {
                cout << i + 1 << " | " << split.test.targets[i]
                     << " | " << cartPredictions[i]
                     << " | " << forestPredictions[i] << '\n';
            }
        }

        return 0;
    }

    // ========================================================
    // КЛАССИФИКАЦИЯ
    // ========================================================

    if (task == "classification") {
        unique_ptr<IModel> model;
        function<void(const vector<string>&)> printModelSummary;
        function<Matrix(const Matrix&)> predictProbabilities;
        bool printSummaryBeforeTrainedMessage = false;

        if (modelName == "decision_stump") {
            auto stump = make_unique<DecisionStump>(SplitCriterion::Gini, 1);
            DecisionStump* stumpView = stump.get();
            model = move(stump);
            printModelSummary = [stumpView](const vector<string>& featureNames) {
                printDecisionStumpSummary(*stumpView, featureNames);
            };
        }
        else if (modelName == "cart") {
            auto cart = make_unique<CartClassifier>(
                maxDepth,
                minSamplesSplit,
                minSamplesLeaf
            );
            CartClassifier* cartView = cart.get();
            model = move(cart);
            printModelSummary = [cartView](const vector<string>& featureNames) {
                printCartSummary(cartView->summary(), featureNames);
            };
            predictProbabilities = [cartView](const Matrix& features) {
                return cartView->predict_proba(features);
            };
            printSummaryBeforeTrainedMessage = true;

            cout << "Максимальная глубина: " << maxDepth << '\n';
            cout << "Минимум для разбиения: "
                 << minSamplesSplit << '\n';
            cout << "Минимум в листе: "
                 << minSamplesLeaf << '\n';
        }
        else {
            throw invalid_argument(
                "Неизвестная модель: " + modelName
            );
        }

        model->fit(
            trainMatrix,
            span<const double>(split.train.targets)
        );

        if (printSummaryBeforeTrainedMessage) {
            printModelSummary(trainProcessed.featureNames);
        }

        cout << "\nМодель обучена.\n";

        if (printModelSummary && !printSummaryBeforeTrainedMessage) {
            printModelSummary(trainProcessed.featureNames);
        }

        vector<double> predictions = model->predict(testMatrix);
        Matrix probabilities(0, 0);
        if (predictProbabilities) {
            probabilities = predictProbabilities(testMatrix);
        }
        vector<double> classes = split.train.targets;
        sort(classes.begin(), classes.end());
        classes.erase(unique(classes.begin(), classes.end()), classes.end());

         const size_t classCount = classes.size();

        vector<vector<size_t>> confusion(
            classCount,
            vector<size_t>(classCount, 0)
        );

        size_t correctPredictions = 0;

        for (size_t i = 0; i < predictions.size(); ++i) {
            auto actualIt = find(
                classes.begin(),
                classes.end(),
                split.test.targets[i]
            );

            auto predictedIt = find(
                classes.begin(),
                classes.end(),
                predictions[i]
            );

            if (actualIt == classes.end() ||
                predictedIt == classes.end()) {
                throw runtime_error(
                    "Обнаружен неизвестный класс в предсказаниях"
                );
            }

            size_t actualIndex =
                static_cast<size_t>(actualIt - classes.begin());

            size_t predictedIndex =
                static_cast<size_t>(predictedIt - classes.begin());

            ++confusion[actualIndex][predictedIndex];

            if (actualIndex == predictedIndex) {
                ++correctPredictions;
            }
        }

        // Accuracy.
        double accuracy =
            static_cast<double>(correctPredictions) /
            static_cast<double>(predictions.size());

        // Macro Precision, Recall и F1.
        double precisionSum = 0.0;
        double recallSum = 0.0;
        double f1Sum = 0.0;

        for (size_t c = 0; c < classCount; ++c) {
            size_t truePositive = confusion[c][c];

            size_t falsePositive = 0;
            size_t falseNegative = 0;

            for (size_t j = 0; j < classCount; ++j) {
                if (j != c) {
                    falsePositive += confusion[j][c];
                    falseNegative += confusion[c][j];
                }
            }

            double precision = 0.0;
            double recall = 0.0;
            double f1 = 0.0;

            size_t predictedCount =
                truePositive + falsePositive;

            size_t actualCount =
                truePositive + falseNegative;

            if (predictedCount > 0) {
                precision =
                    static_cast<double>(truePositive) /
                    static_cast<double>(predictedCount);
            }

            if (actualCount > 0) {
                recall =
                    static_cast<double>(truePositive) /
                    static_cast<double>(actualCount);
            }

            if (precision + recall > 0.0) {
                f1 = 2.0 * precision * recall /
                     (precision + recall);
            }

            precisionSum += precision;
            recallSum += recall;
            f1Sum += f1;
        }

        double macroPrecision =
            precisionSum / static_cast<double>(classCount);

        double macroRecall =
            recallSum / static_cast<double>(classCount);

        double macroF1 =
            f1Sum / static_cast<double>(classCount);

        cout << fixed << setprecision(4);

        cout << "\nМетрики классификации:\n";
        cout << "Accuracy:  " << accuracy << '\n';
        cout << "Precision: " << macroPrecision << '\n';
        cout << "Recall:    " << macroRecall << '\n';
        cout << "F1-score:  " << macroF1 << '\n';

        cout << "\nМатрица ошибок:\n";

        cout << "Реальный / Предсказанный";

        for (double label : classes) {
            const auto name = classDisplayNames.find(label);
            cout << "\t" << (name == classDisplayNames.end()
                ? to_string(label)
                : name->second);
        }

        cout << '\n';

        for (size_t i = 0; i < classCount; ++i) {
            const auto name = classDisplayNames.find(classes[i]);
            cout << (name == classDisplayNames.end()
                ? to_string(classes[i])
                : name->second);

            for (size_t j = 0; j < classCount; ++j) {
                cout << "\t" << confusion[i][j];
            }

            cout << '\n';
        }

        const size_t outputCount = min(
            size_t{10},
            predictions.size()
        );

        cout << "\nРезультаты классификации:\n";
        cout << fixed << setprecision(3);

        for (size_t i = 0; i < outputCount; ++i) {
            cout << "\nОбъект " << i + 1 << '\n';
            const auto actualName =
                classDisplayNames.find(split.test.targets[i]);
            const auto predictedName =
                classDisplayNames.find(predictions[i]);
            cout << "Реальный класс: "
                 << (actualName == classDisplayNames.end()
                     ? to_string(split.test.targets[i])
                     : actualName->second)
                 << '\n';
            cout << "Предсказанный класс: "
                 << (predictedName == classDisplayNames.end()
                     ? to_string(predictions[i])
                     : predictedName->second)
                 << '\n';

            if (predictProbabilities) {
                cout << "Вероятности:\n";
                for (size_t j = 0; j < classes.size(); ++j) {
                    const auto name = classDisplayNames.find(classes[j]);
                    cout << "  Класс "
                         << (name == classDisplayNames.end()
                             ? to_string(classes[j])
                             : name->second)
                         << ": "
                         << probabilities(i, j) * 100.0 << "%\n";
                }
            }

        }

        return 0;
    }

    // ========================================================
    // РЕГРЕССИЯ
    // ========================================================

    unique_ptr<IModel> model;
    function<void(const vector<string>&)> printModelSummary;
    bool printSummaryBeforeTrainedMessage = false;

    if (modelName == "decision_stump") {
        auto stump = make_unique<DecisionStump>(SplitCriterion::Variance, 1);
        DecisionStump* stumpView = stump.get();
        model = move(stump);
        printModelSummary = [stumpView](const vector<string>& featureNames) {
            printDecisionStumpSummary(*stumpView, featureNames);
        };
    }
    else if (modelName == "cart") {
        auto cart = make_unique<CartRegressor>(
            maxDepth,
            minSamplesSplit,
            minSamplesLeaf
        );
        CartRegressor* cartView = cart.get();
        model = move(cart);
        printModelSummary = [cartView](const vector<string>& featureNames) {
            printCartSummary(cartView->summary(), featureNames);
        };
        printSummaryBeforeTrainedMessage = true;

        cout << "Максимальная глубина: " << maxDepth << '\n';
        cout << "Минимум для разбиения: "
             << minSamplesSplit << '\n';
        cout << "Минимум в листе: "
             << minSamplesLeaf << '\n';
    }
    else {
        throw invalid_argument(
            "Неизвестная модель: " + modelName
        );
    }

    model->fit(
        trainMatrix,
        span<const double>(split.train.targets)
    );

    if (printSummaryBeforeTrainedMessage) {
        printModelSummary(trainProcessed.featureNames);
    }

    cout << "\nМодель обучена.\n";

    if (printModelSummary && !printSummaryBeforeTrainedMessage) {
        printModelSummary(trainProcessed.featureNames);
    }

    vector<double> predictions = model->predict(testMatrix);

    if (predictions.empty()) {
        throw runtime_error(
            "Тестовая выборка пуста"
        );
    }

    // Рассчитываем MAE и RMSE.
    double absoluteErrorSum = 0.0;
    double squaredErrorSum = 0.0;

    for (size_t i = 0; i < predictions.size(); ++i) {
        double error =
            predictions[i] - split.test.targets[i];

        absoluteErrorSum += abs(error);
        squaredErrorSum += error * error;
    }

    double mae =
        absoluteErrorSum / predictions.size();

    double rmse =
        sqrt(squaredErrorSum / predictions.size());

    cout << fixed << setprecision(3);

    cout << "\nОценка качества регрессии:\n";
    cout << "MAE:  " << mae << '\n';
    cout << "RMSE: " << rmse << '\n';

    const size_t outputCount = min(
        size_t{10},
        predictions.size()
    );

    cout << "\nРезультаты регрессии:\n";
    cout << "№ | Реальное значение | Прогноз\n";
    cout << "--------------------------------\n";

    for (size_t i = 0; i < outputCount; ++i) {
        cout << i + 1 << " | "
             << split.test.targets[i] << " | "
             << predictions[i] << '\n';
    }

    return 0;
}


int main(int argc, char* argv[]) {
#ifdef _WIN32
    (void)system("chcp 65001 > nul");
#endif
     try {

const string defaultInput =
    (filesystem::path(__FILE__).parent_path().parent_path() /
     "bank-additional/bank-additional/bank-additional.csv").string();


const string defaultOutput =
    (filesystem::path(__FILE__).parent_path().parent_path() /
     "bank-additional-processed.csv").string();


cxxopts::Options options("lab1", "Анализ и предобработка датасета");

options.add_options()
    ("h,help", "Показать справку")
    ("i,input", "Путь к входному CSV-файлу",
        cxxopts::value<string>()->default_value(defaultInput))
    ("o,output", "Путь для сохранения обработанного CSV-файла",
        cxxopts::value<string>()->default_value(defaultOutput))
    ("b,bins", "Количество интервалов гистограммы",
        cxxopts::value<int>()->default_value("10"))
    ("f,fill", "Способ заполнения пропусков: mean, median, zero",
        cxxopts::value<string>()->default_value("median"))
    ("s,scale", "Способ масштабирования: minmax, standard, robust",
        cxxopts::value<string>()->default_value("minmax"))
    ("e,encode", "Способ кодирования: onehot, ordinal",
        cxxopts::value<string>()->default_value("onehot"))
    ("model", "Модель: decision_stump, cart, random_forest (сравнение CART и леса)",
        cxxopts::value<string>()->default_value(""))
    ("target", "Название целевого признака",
        cxxopts::value<string>()->default_value("y"))
    ("task", "Задача: classification или regression",
        cxxopts::value<string>()->default_value("classification"))
    ("max-depth", "Максимальная глубина CART",
        cxxopts::value<size_t>()->default_value("5"))
    ("min-samples-split", "Минимум объектов для разбиения CART",
        cxxopts::value<size_t>()->default_value("2"))
    ("min-samples-leaf", "Минимум объектов в листе CART и Random Forest",
        cxxopts::value<size_t>()->default_value("1"))
    ("n-estimators", "Количество деревьев Random Forest",
        cxxopts::value<size_t>()->default_value("100"))
    ("n-jobs", "Количество потоков Random Forest",
        cxxopts::value<size_t>()->default_value("1"));

cxxopts::ParseResult arguments;

try {
    arguments = options.parse(argc, argv);
}
catch (const cxxopts::exceptions::exception& error) {
    cerr << "Ошибка параметров: " << error.what() << '\n';
    cerr << "Используйте --help для просмотра справки.\n";
    return 1;
}

if (arguments.count("help")) {
    cout << options.help() << '\n';
    return 0;
}

const string filename = arguments["input"].as<string>();
const filesystem::path outputPath = arguments["output"].as<string>();

PreprocessingConfig preprocessingConfig;
const string fillValue = arguments["fill"].as<string>();
if (!parseFillStrategy(fillValue, preprocessingConfig.numericalMissing)) {
    cerr << "Ошибка: неизвестная стратегия заполнения: " << fillValue << '\n';
    cerr << "Допустимые значения: mean, median, zero\n";
    return 1;
}

const string scaleValue = arguments["scale"].as<string>();
if (!parseScalingStrategy(scaleValue, preprocessingConfig.scaling)) {
    cerr << "Ошибка: неизвестный способ масштабирования: " << scaleValue << '\n';
    cerr << "Допустимые значения: minmax, standard, robust\n";
    return 1;
}

const string encodeValue = arguments["encode"].as<string>();
if (!parseEncodingStrategy(encodeValue, preprocessingConfig.encoding)) {
    cerr << "Ошибка: неизвестный способ кодирования: " << encodeValue << '\n';
    cerr << "Допустимые значения: onehot, ordinal\n";
    return 1;
}

const int binsValue = arguments["bins"].as<int>();
if (binsValue <= 0) {
    cerr << "Ошибка: количество интервалов должно быть больше нуля.\n";
    return 1;
}
const size_t binCount = static_cast<size_t>(binsValue);

const CsvConfig config{ ';', "unknown", true };
Dataset dataset;
loadCsv(filename, dataset, config);

const string modelName = arguments["model"].as<string>();
if (!modelName.empty()) {
    return runMachineLearning(
        dataset,
        arguments["target"].as<string>(),
        preprocessingConfig,
        modelName,
        arguments["task"].as<string>(),
        arguments["max-depth"].as<size_t>(),
        arguments["min-samples-split"].as<size_t>(),
        arguments["min-samples-leaf"].as<size_t>(),
        arguments["n-estimators"].as<size_t>(),
        arguments["n-jobs"].as<size_t>()
    );
}


    size_t recordCount = 0;

    if (!dataset.columns.empty()) {
        const auto& firstColumn = dataset.columns[0];

        if (firstColumn.type == FeatureType::Numerical) {
            recordCount = firstColumn.numericalValues.size();
        }
        else {
            recordCount = firstColumn.categoricalValues.size();
        }
    }

    cout << "CSV-файл успешно загружен\n\n";

    cout << "Объектов: " << recordCount << endl;
    cout << "Признаков: " << dataset.columns.size() << "\n\n";

    cout << "# | Признак | Тип | Пропуски" << endl;
    cout << "-----------------------------------------------" << endl;

    size_t totalMissing = 0;
    size_t featuresWithMissing = 0;

    for (size_t i = 0; i < dataset.columns.size(); ++i) {
        const auto& column = dataset.columns[i];

        size_t missingCount = 0;

        if (column.type == FeatureType::Numerical) {
            for (const auto& value : column.numericalValues) {
                if (!value.has_value()) {
                    ++missingCount;
                }
            }
        }
        else {
            for (const auto& value : column.categoricalValues) {
                if (!value.has_value()) {
                    ++missingCount;
                }
            }
        }

        totalMissing += missingCount;

        if (missingCount > 0) {
            ++featuresWithMissing;
        }

        double missingPercentage = 0.0;

        if (recordCount > 0) {
            missingPercentage =
                static_cast<double>(missingCount) / recordCount * 100.0;
        }

        cout << i << " | " << column.name << " | "
             << (column.type == FeatureType::Numerical
                 ? "числовой" : "категориальный")
             << " | " << missingCount;

        if (missingCount > 0) {
            cout << " (" << fixed << setprecision(2)
                 << missingPercentage << "%)";
        }

        cout << '\n';
    }

    cout << "\nВсего пропусков: " << totalMissing
         << " в " << featuresWithMissing << " признаках" << endl;


 //цикл выбора признака
    while (true) {
        cout << "\nВыберите признак для анализа (0-"
             << dataset.columns.size() - 1 << ").\n";
        cout << "Введите -1 для выхода.\n\n";

        cout << "Ваш выбор: ";

        string input;
        getline(cin, input);

        stringstream inputStream(input);
        int choice;
        char extra;

        if (!(inputStream >> choice) || (inputStream >> extra)) {
            cout << "Ошибка: введите целое число.\n";
            continue;
        }


        if (input == "-1") {
            string answer;

            cout << "\nСохранить преобразованный датасет? (y/n): ";
            getline(cin, answer);

            if (answer == "y" || answer == "Y" ||
                answer == "д" || answer == "Д") {

                ProcessedDataset processedDataset;

                preprocessDataset(
                    dataset,
                    processedDataset,
                    preprocessingConfig
                );
                saveProcessedDataset(
                    processedDataset,
                    outputPath.string()
                );

                cout << "Датасет успешно сохранён: "
                     << outputPath << '\n';

                cout << "Количество объектов: "
                     << processedDataset.data.size() << '\n';

                cout << "Количество признаков после обработки: "
                     << processedDataset.featureNames.size() << '\n';

                filesystem::path binaryPath = outputPath;
                binaryPath.replace_extension(".bin");

                saveProcessedDatasetBinary(
                    processedDataset,
                    binaryPath.string()
                );

                cout << "\nБинарный файл сохранён: "
                     << binaryPath << '\n';

                ProcessedDataset loadedDataset;
                loadProcessedDatasetBinary(
                    binaryPath.string(),
                    loadedDataset
                );

                if (compareProcessedDatasets(processedDataset, loadedDataset)) {
                    cout << "Проверка бинарного файла пройдена успешно";
                } else {
                    cout << "Ошибка: данные после загрузки отличаются" << endl;
                }

                cout << "\nРазмеры файлов:" << endl;
                cout << "CSV: "
                     << filesystem::file_size(outputPath)
                     << " байт" << endl;
                cout << "Binary: "
                     << filesystem::file_size(binaryPath)
                     << " байт" << endl;
            }

            break;
        }

        if (choice < 0 ||
            choice >= static_cast<int>(dataset.columns.size())) {
            cout << "Ошибка: такого признака нет." << endl;
            continue;
        }

        const auto& column = dataset.columns[choice];

//анализ числового признака
        if (column.type == FeatureType::Numerical) {
            NumericalStatistics stats;

            calculateNumericalStatistics(column, stats, binCount);

            cout << "\nСтатистика признака " << choice
                 << ": " << column.name << " (числовой)\n\n";

            cout << "Пропусков: " << stats.missingCount << endl;

            if (stats.hasValues) {
                cout << fixed << setprecision(3);

                cout << "Min: " << stats.min
                     << "    Max: " << stats.max << endl;

                cout << "Mean: " << stats.mean
                     << "    Var: " << stats.variance << endl;

                cout << "\n";

                cout << "q05: " << stats.q05
                     << "\nq25: " << stats.q25
                     << "\nq50: " << stats.q50
                     << "\nq75: " << stats.q75
                     << "\nq95: " << stats.q95 << endl;

//числовая гистограмма
                cout << "\nГистограмма:\n";

                size_t maxFrequency = 0;

                for (const auto& bin : stats.histogram) {
                    if (bin.frequency > maxFrequency) {
                        maxFrequency = bin.frequency;
                    }
                }

                for (size_t i = 0; i < stats.histogram.size(); ++i) {
                    const auto& bin = stats.histogram[i];

                    cout << "[" << bin.lowerBound << ", "
                         << bin.upperBound
                         << (i + 1 == stats.histogram.size() ? "] " : ") ");

                    size_t barLength = 0;

                    if (maxFrequency > 0) {
                        barLength = static_cast<size_t>(
                            30.0 * bin.frequency / maxFrequency
                        );
                    }

                    for (size_t j = 0; j < barLength; ++j) {
                        cout << '#';
                    }

                    cout << " " << bin.frequency << endl;
                }
            }
            else {
                cout << "Нет данных для расчёта статистики.\n";
            }
        }
//анализ категориального признака
        else {
            CategoricalStatistics categoryStats;

            calculateCategoricalStatistics(column, categoryStats);

            cout << "\nСтатистика признака " << choice
                 << ": " << column.name << " (категориальный)\n\n";

            cout << "Пропусков: "
                 << categoryStats.missingCount << '\n';

            cout << "Количество категорий: "
                 << categoryStats.categories.size() << "\n\n";

            cout << "Категории:"<< endl;

            for (const auto& category : categoryStats.categories) {
                cout << category.category
                     << " | " << category.frequency << endl;
            }
//категориальная гистограмма
            cout << "\nГистограмма:"<< endl;

            size_t maxFrequency = 0;

            for (const auto& category : categoryStats.categories) {
                if (category.frequency > maxFrequency) {
                    maxFrequency = category.frequency;
                }
            }

            for (const auto& category : categoryStats.categories) {
                cout << left << setw(25) << category.category;

                size_t barLength = 0;

                if (maxFrequency > 0) {
                    barLength = static_cast<size_t>(
                        30.0 * category.frequency / maxFrequency
                    );
                }

                for (size_t i = 0; i < barLength; ++i) {
                    cout << '#';
                }

                cout << " " << category.frequency << endl;
            }
        }
    }

        return 0;
    }
    catch (const std::exception& error) {
        cerr << "\nОшибка выполнения программы: "
             << error.what() << '\n';

        return 1;
    }
    catch (...) {
        cerr << "\nПроизошла неизвестная ошибка.\n";
        return 1;
    }
}
