
#include "neighbor_search.hpp"
#include "exceptions.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>

using namespace std;

vector<Neighbor> findKNearestNeighbors(
    const Matrix& trainFeatures,
    span<const double> query,
    size_t k,
    const IDistance& distance
) {
    // Проверяем корректность количества соседей.
    if (k == 0 || k > trainFeatures.rows()) {
        throw InvalidParameterError("Некорректное количество соседей");
    }

    // Проверяем размерность входного вектора.
    if (query.size() != trainFeatures.columns()) {
        throw DimensionMismatchError("Размерность входного вектора не совпадает с размерностью признаков");
    }

    // Компаратор для max heap.
    // На вершине кучи находится самый далёкий сосед.
    // При одинаковых расстояниях выше находится сосед с большим индексом.
    auto comparator = [](const Neighbor& a, const Neighbor& b) {
        if (a.distance != b.distance) {
            return a.distance < b.distance;
        }

        return a.index < b.index;
    };

    priority_queue<
        Neighbor,
        vector<Neighbor>,
        decltype(comparator)
    > neighbors(comparator);

    // Перебираем все объекты обучающей выборки.
    for (size_t i = 0; i < trainFeatures.rows(); ++i) {
        // Получаем строку матрицы с признаками текущего объекта.
        span<const double> currentRow = trainFeatures[i];

        // Вычисляем расстояние до входного объекта.
        double currentDistance = distance.compute(currentRow, query);

        // Создаём информацию о найденном соседе.
        Neighbor currentNeighbor{
            currentDistance,
            i
        };

        // Если куча ещё не заполнена, добавляем соседа.
        if (neighbors.size() < k) {
            neighbors.push(currentNeighbor);
        }
        // Если новый сосед лучше самого далёкого в куче,
        // заменяем вершину.
        else {
            const Neighbor& worstNeighbor = neighbors.top();

            bool isBetter =
                currentNeighbor.distance < worstNeighbor.distance ||
                (currentNeighbor.distance == worstNeighbor.distance &&
                 currentNeighbor.index < worstNeighbor.index);

            if (isBetter) {
                neighbors.pop();
                neighbors.push(currentNeighbor);
            }
        }
    }

    // Переносим найденных соседей в результирующий вектор.
    vector<Neighbor> result;
    result.reserve(neighbors.size());

    while (!neighbors.empty()) {
        result.push_back(neighbors.top());
        neighbors.pop();
    }

    // Сортируем соседей по возрастанию расстояния.
    // При одинаковых расстояниях сортируем по индексу.
    sort(
        result.begin(),
        result.end(),
        [](const Neighbor& a, const Neighbor& b) {
            if (a.distance != b.distance) {
                return a.distance < b.distance;
            }

            return a.index < b.index;
        }
    );

    return result;
}
