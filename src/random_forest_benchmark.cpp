#include "random_forest.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kTrainRows = 6000;
constexpr std::size_t kTestRows = 1500;
constexpr std::size_t kFeatureCount = 16;
constexpr std::size_t kTreeCount = 50;
constexpr std::size_t kRepeats = 5;
constexpr std::size_t kMaxDepth = 12;
constexpr std::uint32_t kRandomState = 42;

struct Dataset {
    Matrix train;
    Matrix test;
    std::vector<double> regressionTrain;
    std::vector<double> classificationTrain;
};

Dataset makeDataset() {
    std::mt19937 generator(20261123);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    Dataset data{
        Matrix(kTrainRows, kFeatureCount),
        Matrix(kTestRows, kFeatureCount),
        std::vector<double>(kTrainRows),
        std::vector<double>(kTrainRows)
    };

    auto fill = [&](Matrix& features,
                    std::vector<double>& regressionTarget,
                    std::vector<double>& classificationTarget) {
        for (std::size_t row = 0; row < features.rows(); ++row) {
            double score = 0.0;
            for (std::size_t column = 0; column < features.columns(); ++column) {
                const double value = distribution(generator);
                features(row, column) = value;
                if (column < 6) {
                    score += value * (1.0 / static_cast<double>(column + 1));
                }
            }
            regressionTarget[row] = score;
            classificationTarget[row] = score >= 0.0 ? 1.0 : 0.0;
        }
    };

    std::vector<double> ignoredRegression(kTestRows);
    std::vector<double> ignoredClassification(kTestRows);
    fill(data.train, data.regressionTrain, data.classificationTrain);
    fill(data.test, ignoredRegression, ignoredClassification);
    return data;
}

double milliseconds(Clock::time_point start, Clock::time_point finish) {
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }
    return values[middle];
}

#ifdef __clang__
const char* compilerName() { return "Clang " __clang_version__; }
#elif defined(__GNUC__)
const char* compilerName() { return "GCC " __VERSION__; }
#elif defined(_MSC_VER)
const char* compilerName() { return "MSVC"; }
#else
const char* compilerName() { return "Unknown compiler"; }
#endif

std::string processorName() {
#ifdef _WIN32
    char* value = nullptr;
    std::size_t size = 0;

    if (_dupenv_s(&value, &size, "PROCESSOR_IDENTIFIER") == 0 &&
        value != nullptr) {
        std::string result(value);
        std::free(value);
        return result;
    }

    return "not available (PROCESSOR_IDENTIFIER)";
#elif defined(__linux__)
    std::ifstream cpuInfo("/proc/cpuinfo");
    std::string line;

    while (std::getline(cpuInfo, line)) {
        const std::string key = "model name";
        if (line.rfind(key, 0) == 0) {
            const auto separator = line.find(':');
            if (separator != std::string::npos) {
                return line.substr(separator + 2);
            }
        }
    }

    return "not available (/proc/cpuinfo)";
#else
    return "not available";
#endif
}

struct Measurements {
    std::vector<double> fit;
    std::vector<double> predict;
    std::vector<double> predictProba;
    double checksum = 0.0;
};

void runClassificationBenchmark(const Dataset& data, std::size_t jobs) {
    RandomForestParams params;
    params.n_estimators = kTreeCount;
    params.max_depth = kMaxDepth;
    params.random_state = kRandomState;
    params.n_jobs = jobs;
    params.max_features = FeatureSearchParams{FeatureMode::Sqrt, 1};

    Measurements result;
    for (std::size_t repeat = 0; repeat < kRepeats; ++repeat) {
        RandomForestClassifier model(params);

        auto start = Clock::now();
        model.fit(data.train, data.classificationTrain);
        auto finish = Clock::now();
        result.fit.push_back(milliseconds(start, finish));

        start = Clock::now();
        const auto predictions = model.predict(data.test);
        finish = Clock::now();
        result.predict.push_back(milliseconds(start, finish));

        start = Clock::now();
        const Matrix probabilities = model.predict_proba(data.test);
        finish = Clock::now();
        result.predictProba.push_back(milliseconds(start, finish));

        result.checksum += std::accumulate(predictions.begin(), predictions.end(), 0.0);
        for (std::size_t row = 0; row < probabilities.rows(); ++row) {
            result.checksum += probabilities(row, 0) + probabilities(row, 1);
        }
    }

    std::cout << "Classification, n_jobs=" << jobs << '\n'
              << "  fit median:          " << median(result.fit) << " ms\n"
              << "  predict median:      " << median(result.predict) << " ms\n"
              << "  predict_proba median:" << median(result.predictProba) << " ms\n"
              << "  checksum:             " << result.checksum << "\n";
}

void runRegressionBenchmark(const Dataset& data, std::size_t jobs) {
    RandomForestParams params;
    params.n_estimators = kTreeCount;
    params.max_depth = kMaxDepth;
    params.random_state = kRandomState;
    params.n_jobs = jobs;

    std::vector<double> fitTimes;
    std::vector<double> predictTimes;
    double checksum = 0.0;

    for (std::size_t repeat = 0; repeat < kRepeats; ++repeat) {
        RandomForestRegressor model(params);

        auto start = Clock::now();
        model.fit(data.train, data.regressionTrain);
        auto finish = Clock::now();
        fitTimes.push_back(milliseconds(start, finish));

        start = Clock::now();
        const auto predictions = model.predict(data.test);
        finish = Clock::now();
        predictTimes.push_back(milliseconds(start, finish));
        checksum += std::accumulate(predictions.begin(), predictions.end(), 0.0);
    }

    std::cout << "Regression, n_jobs=" << jobs << '\n'
              << "  fit median:     " << median(fitTimes) << " ms\n"
              << "  predict median: " << median(predictTimes) << " ms\n"
              << "  checksum:        " << checksum << "\n";
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(3);
    const Dataset data = makeDataset();

    std::cout << "Random Forest benchmark\n"
              << "Compiler: " << compilerName() << '\n'
              << "Processor: " << processorName() << '\n'
              << "Hardware threads reported: "
              << std::thread::hardware_concurrency() << '\n'
              << "Build mode: "
#ifdef NDEBUG
              << "Release\n"
#else
              << "Debug (not suitable for performance comparison)\n"
#endif
              << "Train matrix: " << kTrainRows << " x " << kFeatureCount << '\n'
              << "Test matrix:  " << kTestRows << " x " << kFeatureCount << '\n'
              << "Trees: " << kTreeCount << ", max_depth: " << kMaxDepth
              << ", random_state: " << kRandomState
              << ", repeats: " << kRepeats << "\n\n";

    for (const std::size_t jobs : {1U, 4U}) {
        runRegressionBenchmark(data, jobs);
        runClassificationBenchmark(data, jobs);
        std::cout << '\n';
    }
}
