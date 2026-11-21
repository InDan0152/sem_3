
#include "decision_stump.hpp"
#include "exceptions.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <memory>

using namespace std;

// Проверка условия.
void check(bool condition, const string& message) {
    if (!condition) {
        throw runtime_error(message);
    }
}

// Проверка приблизительного равенства чисел.
void check_near(double actual, double expected, const string& message) {
    if (abs(actual - expected) > 1e-9) {
        throw runtime_error(message);
    }
}

// Проверка, что функция выбрасывает исключение.
template <typename Function>
void check_throws(Function function, const string& message) {
    bool exceptionThrown = false;

    try {
        function();
    }
    catch (const exception&) {
        exceptionThrown = true;
    }

    check(exceptionThrown, message);
}

// Проверка, что функция выбрасывает исключение нужного типа.
template <typename ExpectedException, typename Function>
void check_throws_type(Function function, const string& message) {
    bool correctExceptionThrown = false;

    try {
        function();
    }
    catch (const ExpectedException&) {
        correctExceptionThrown = true;
    }
    catch (...) {
        throw runtime_error(
            message + ": выброшено исключение неправильного типа"
        );
    }

    check(correctExceptionThrown, message);
}

// Регрессия: проверяем выбор разбиения и предсказания.
void test_regression() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    });

    vector<double> target = {0.0, 0.0, 10.0, 10.0};

    DecisionStump model(SplitCriterion::Variance, 1);
    model.fit(features, target);

    check(model.split_feature() == 0,
          "Регрессия: выбран неправильный признак");

    check_near(model.split_threshold(), 1.5,
               "Регрессия: неправильный порог");

    vector<double> predictions = model.predict(features);

    check_near(predictions[0], 0.0, "Регрессия: неправильный прогноз слева");
    check_near(predictions[1], 0.0, "Регрессия: неправильный прогноз слева");
    check_near(predictions[2], 10.0, "Регрессия: неправильный прогноз справа");
    check_near(predictions[3], 10.0, "Регрессия: неправильный прогноз справа");
}

// Классификация: проверяем определение классов в листьях.
void test_classification() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    });

    vector<double> target = {0.0, 0.0, 1.0, 1.0};

    DecisionStump model(SplitCriterion::Gini, 1);
    model.fit(features, target);

    check(model.split_feature() == 0,
          "Классификация: выбран неправильный признак");

    check_near(model.split_threshold(), 1.5,
               "Классификация: неправильный порог");

    vector<double> predictions = model.predict(features);

    check_near(predictions[0], 0.0, "Классификация: неправильный класс слева");
    check_near(predictions[1], 0.0, "Классификация: неправильный класс слева");
    check_near(predictions[2], 1.0, "Классификация: неправильный класс справа");
    check_near(predictions[3], 1.0, "Классификация: неправильный класс справа");
}

// Константный признак: разбиение невозможно.
void test_constant_feature() {
    Matrix features({
        {5.0},
        {5.0},
        {5.0}
    });

    vector<double> target = {1.0, 2.0, 3.0};

    DecisionStump model(SplitCriterion::Variance, 1);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    for (double prediction : predictions) {
        check_near(prediction, 2.0,
                   "Константный признак: неправильное среднее");
    }
}

// Проверяем, что min_samples_leaf запрещает слишком маленькие листья.
void test_min_samples_leaf() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    });

    vector<double> target = {1.0, 2.0, 10.0, 11.0};

    DecisionStump model(SplitCriterion::Variance, 3);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    double expectedMean = 6.0;

    for (double prediction : predictions) {
        check_near(prediction, expectedMean,
                   "min_samples_leaf: ожидалось постоянное предсказание");
    }
}

// Проверяем, что порог не устанавливается между одинаковыми значениями.
void test_duplicate_values() {
    Matrix features({
        {1.0},
        {1.0},
        {2.0},
        {2.0}
    });

    vector<double> target = {0.0, 0.0, 10.0, 10.0};

    DecisionStump model(SplitCriterion::Variance, 1);
    model.fit(features, target);

    check_near(model.split_threshold(), 1.5,
               "Повторяющиеся значения: неправильный порог");
}

// Проверяем predict до обучения.
void test_predict_before_fit() {
    Matrix features(vector<vector<double>>{
        {0.0},
        {1.0}
    });

    DecisionStump model(SplitCriterion::Variance, 1);

    check_throws_type<ModelUsageError>(
        [&]() {
            model.predict(features);
        },
        "predict до fit: ожидался ModelUsageError"
    );
}

// Проверяем несовпадение количества признаков.
void test_wrong_feature_count() {
    Matrix trainFeatures({
        {0.0},
        {1.0},
        {2.0}
    });

    vector<double> target = {0.0, 1.0, 2.0};

    DecisionStump model(SplitCriterion::Variance, 1);
    model.fit(trainFeatures, target);

    Matrix testFeatures({
        {0.0, 1.0},
        {1.0, 2.0}
    });

    check_throws_type<DimensionMismatchError>(
        [&]() {
            model.predict(testFeatures);
        },
        "predict: ожидался DimensionMismatchError при неверном числе признаков"
    );
}


// Результат работы наивного алгоритма.
struct NaiveResult {
    bool hasSplit = false;
    size_t feature = 0;
    double threshold = 0.0;
    double leftPrediction = 0.0;
    double rightPrediction = 0.0;
    double constantPrediction = 0.0;
    vector<double> predictions;
};

// Наивный алгоритм: перебирает все возможные разбиения.
NaiveResult naive_decision_stump(
    const Matrix& features,
    const vector<double>& target,
    SplitCriterion criterion,
    size_t minSamplesLeaf
) {
    NaiveResult result;

    size_t n = features.rows();
    size_t d = features.columns();

    // Предсказание константой на случай отсутствия разбиения.
    if (criterion == SplitCriterion::Variance) {
        result.constantPrediction =
            accumulate(target.begin(), target.end(), 0.0) / n;
    }
    else {
        map<double, size_t> frequencies;

        for (double value : target) {
            frequencies[value]++;
        }

        size_t bestCount = 0;

        for (const auto& [value, count] : frequencies) {
            if (count > bestCount) {
                bestCount = count;
                result.constantPrediction = value;
            }
        }
    }

    double bestLoss = numeric_limits<double>::infinity();

    // Перебираем признаки.
    for (size_t feature = 0; feature < d; ++feature) {
        vector<double> values;

        for (size_t i = 0; i < n; ++i) {
            values.push_back(features[i][feature]);
        }

        sort(values.begin(), values.end());

        values.erase(
            unique(values.begin(), values.end()),
            values.end()
        );

        // Перебираем все пороги между различными значениями.
        for (size_t k = 0; k + 1 < values.size(); ++k) {
            double threshold = midpoint(values[k], values[k + 1]);

            vector<double> left;
            vector<double> right;

            for (size_t i = 0; i < n; ++i) {
                if (features[i][feature] <= threshold) {
                    left.push_back(target[i]);
                }
                else {
                    right.push_back(target[i]);
                }
            }

            if (left.size() < minSamplesLeaf ||
                right.size() < minSamplesLeaf) {
                continue;
            }

            double loss = 0.0;
            double leftPrediction = 0.0;
            double rightPrediction = 0.0;

            if (criterion == SplitCriterion::Variance) {
                leftPrediction =
                    accumulate(left.begin(), left.end(), 0.0) / left.size();

                rightPrediction =
                    accumulate(right.begin(), right.end(), 0.0) / right.size();

                for (double value : left) {
                    loss += (value - leftPrediction) *
                            (value - leftPrediction);
                }

                for (double value : right) {
                    loss += (value - rightPrediction) *
                            (value - rightPrediction);
                }
            }
            else {
                map<double, size_t> leftCounts;
                map<double, size_t> rightCounts;

                for (double value : left) {
                    leftCounts[value]++;
                }

                for (double value : right) {
                    rightCounts[value]++;
                }

                size_t leftBestCount = 0;
                size_t rightBestCount = 0;

                for (const auto& [value, count] : leftCounts) {
                    if (count > leftBestCount) {
                        leftBestCount = count;
                        leftPrediction = value;
                    }
                }

                for (const auto& [value, count] : rightCounts) {
                    if (count > rightBestCount) {
                        rightBestCount = count;
                        rightPrediction = value;
                    }
                }

                double leftSquares = 0.0;
                double rightSquares = 0.0;

                for (const auto& [value, count] : leftCounts) {
                    leftSquares +=
                        static_cast<double>(count) * count;
                }

                for (const auto& [value, count] : rightCounts) {
                    rightSquares +=
                        static_cast<double>(count) * count;
                }

                loss =
                    static_cast<double>(left.size()) -
                    leftSquares / left.size() +
                    static_cast<double>(right.size()) -
                    rightSquares / right.size();
            }

            // При равенстве оставляем первое разбиение:
            // сначала меньший признак, затем меньший порог.
            if (!result.hasSplit || loss < bestLoss - 1e-10) {
                result.hasSplit = true;
                bestLoss = loss;
                result.feature = feature;
                result.threshold = threshold;
                result.leftPrediction = leftPrediction;
                result.rightPrediction = rightPrediction;
            }
        }
    }

    result.predictions.resize(n);

    for (size_t i = 0; i < n; ++i) {
        if (!result.hasSplit) {
            result.predictions[i] = result.constantPrediction;
        }
        else if (features[i][result.feature] <= result.threshold) {
            result.predictions[i] = result.leftPrediction;
        }
        else {
            result.predictions[i] = result.rightPrediction;
        }
    }

    return result;
}

// Сравнение быстрого алгоритма с наивным на случайных данных.
void test_random_against_naive() {
    mt19937 generator(42);

    uniform_int_distribution<int> objectDistribution(6, 20);
    uniform_int_distribution<int> featureDistribution(1, 5);
    uniform_int_distribution<int> valueDistribution(0, 5);
    uniform_int_distribution<int> targetDistribution(0, 2);

    for (int test = 0; test < 200; ++test) {
        size_t n = static_cast<size_t>(objectDistribution(generator));
        size_t d = static_cast<size_t>(featureDistribution(generator));

        // Создаём случайную матрицу признаков.
        vector<vector<double>> data(n, vector<double>(d));

        for (size_t i = 0; i < n; ++i) {
            // Первый признак гарантированно допускает разбиение.
            data[i][0] = static_cast<double>(i % 2);

            for (size_t j = 1; j < d; ++j) {
                data[i][j] =
                    static_cast<double>(valueDistribution(generator));
            }
        }

        Matrix features(data);

        // Проверяем оба критерия.
        for (SplitCriterion criterion :
             {SplitCriterion::Variance, SplitCriterion::Gini}) {

            vector<double> target(n);

            for (size_t i = 0; i < n; ++i) {
                if (criterion == SplitCriterion::Variance) {
                    target[i] =
                        static_cast<double>(targetDistribution(generator));
                }
                else {
                    target[i] =
                        static_cast<double>(targetDistribution(generator));
                }
            }

            size_t minSamplesLeaf =
                1 + static_cast<size_t>(generator() % (n / 3));

            // Эталонный результат.
            NaiveResult expected = naive_decision_stump(
                features,
                target,
                criterion,
                minSamplesLeaf
            );

            // Результат быстрого алгоритма.
            DecisionStump model(criterion, minSamplesLeaf);
            model.fit(features, target);

            check(expected.hasSplit,
                  "Наивный алгоритм не нашёл допустимое разбиение");

            check(model.split_feature() == expected.feature,
                  "Случайный тест: отличается выбранный признак");

            check_near(
                model.split_threshold(),
                expected.threshold,
                "Случайный тест: отличается порог"
            );

            vector<double> actual = model.predict(features);

            check(actual.size() == expected.predictions.size(),
                  "Случайный тест: отличается количество прогнозов");

            for (size_t i = 0; i < n; ++i) {
                check_near(
                    actual[i],
                    expected.predictions[i],
                    "Случайный тест: отличаются предсказания"
                );
            }
        }
    }
}

// Проверяем несовпадение количества объектов и целевых значений.
void test_fit_dimension_mismatch() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0}
    });

    vector<double> target = {0.0, 1.0};

    DecisionStump model(SplitCriterion::Variance, 1);

    check_throws_type<DimensionMismatchError>(
        [&]() {
            model.fit(features, target);
        },
        "fit: ожидалось DimensionMismatchError"
    );
}

// Проверяем обучение на пустой выборке.
void test_empty_training_data() {
    Matrix features(0, 1);
    vector<double> target;

    DecisionStump model(SplitCriterion::Variance, 1);

    check_throws_type<InvalidParameterError>(
        [&]() {
            model.fit(features, target);
        },
        "Пустая выборка: ожидалось InvalidParameterError"
    );
}

// Проверяем недопустимое значение min_samples_leaf.
void test_invalid_min_samples_leaf() {
    check_throws_type<InvalidParameterError>(
        []() {
            DecisionStump model(SplitCriterion::Variance, 0);
        },
        "min_samples_leaf = 0: ожидалось InvalidParameterError"
    );
}

// Проверяем доступ к строке матрицы за границами.
void test_matrix_out_of_range() {
    Matrix features(2, 2);

    check_throws_type<out_of_range>(
        [&]() {
            features.at(2);
        },
        "Matrix::at: ожидалось out_of_range"
    );
}

// Проверяем классификацию, если все объекты принадлежат одному классу.
void test_single_class_classification() {
    Matrix features({
        {1.0},
        {2.0},
        {3.0}
    });

    vector<double> target = {5.0, 5.0, 5.0};

    DecisionStump model(SplitCriterion::Gini, 1);
    model.fit(features, target);

    vector<double> predictions = model.predict(features);

    for (double prediction : predictions) {
        check_near(
            prediction,
            5.0,
            "Один класс: ожидалось постоянное предсказание"
        );
    }
}

void test_single_object_training() {
    Matrix features(vector<vector<double>>{{2.5}});

    for (const auto [criterion, targetValue] : {
             pair{SplitCriterion::Variance, 17.0},
             pair{SplitCriterion::Gini, 42.0}
         }) {
        const vector<double> target{targetValue};
        DecisionStump model(criterion, 1);
        model.fit(features, target);

        check(!model.has_split(),
              "Один объект: разбиение не должно быть найдено");
        check_near(model.constant_prediction(), targetValue,
                   "Один объект: неверный константный прогноз");

        const vector<double> predictions = model.predict(features);
        check(predictions.size() == 1,
              "Один объект: неверный размер результата predict");
        check_near(predictions[0], targetValue,
                   "Один объект: неверное предсказание");
    }
}

void test_predict_non_finite_feature() {
    Matrix trainFeatures(vector<vector<double>>{{0.0}, {1.0}});
    const vector<double> target{0.0, 1.0};
    DecisionStump model(SplitCriterion::Variance, 1);
    model.fit(trainFeatures, target);

    Matrix invalidFeatures(vector<vector<double>>{{
        numeric_limits<double>::quiet_NaN()
    }});
    check_throws_type<InvalidParameterError>(
        [&]() {
            (void)model.predict(invalidFeatures);
        },
        "predict с NaN: ожидался InvalidParameterError"
    );
}

// Проверяем работу модели через базовый интерфейс IModel.
void test_i_model_integration() {
    Matrix features({
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    });

    vector<double> target = {0.0, 0.0, 10.0, 10.0};

    unique_ptr<IModel> model = make_unique<DecisionStump>(
        SplitCriterion::Variance,
        1
    );

    model->fit(features, target);

    vector<double> predictions = model->predict(features);

    check(predictions.size() == 4,
          "IModel: неправильное количество предсказаний");

    check_near(predictions[0], 0.0,
               "IModel: неправильное предсказание слева");

    check_near(predictions[3], 10.0,
               "IModel: неправильное предсказание справа");
}
int main() {
    try {
        test_regression();
        cout << "[OK] Регрессия" << endl;

        test_classification();
        cout << "[OK] Классификация" << endl;

        test_constant_feature();
        cout << "[OK] Константный признак" << endl;

        test_min_samples_leaf();
        cout << "[OK] min_samples_leaf" << endl;

        test_duplicate_values();
        cout << "[OK] Повторяющиеся значения" << endl;

        test_predict_before_fit();
        cout << "[OK] predict до fit" << endl;

        test_wrong_feature_count();
        cout << "[OK] Несовпадение размерностей" << endl;

        test_fit_dimension_mismatch();
        cout << "[OK] Несовпадение размерностей при fit" << endl;

        test_empty_training_data();
        cout << "[OK] Пустая обучающая выборка" << endl;

        test_invalid_min_samples_leaf();
        cout << "[OK] Некорректный min_samples_leaf" << endl;

        test_matrix_out_of_range();
        cout << "[OK] Выход за границы Matrix::at()" << endl;

        test_single_class_classification();
        cout << "[OK] Классификация с одним классом" << endl;

        test_single_object_training();
        cout << "[OK] Обучение на одном объекте" << endl;

        test_predict_non_finite_feature();
        cout << "[OK] Неконечный признак predict" << endl;

        test_i_model_integration();
        cout << "[OK] Интеграция через IModel" << endl;

        test_random_against_naive();
        cout << "[OK] Сравнение с наивным алгоритмом (200 наборов)" << endl;

        cout << "\nВсе тесты DecisionStump пройдены!" << endl;
        return 0;
    }
    catch (const exception& error) {
        cerr << "\n[ОШИБКА] " << error.what() << endl;
        return 1;
    }
}
