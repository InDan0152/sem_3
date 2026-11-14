cmake_minimum_required(VERSION 3.24)

project(lab_1_data_processing LANGUAGES CXX)

# Подключение библиотеки cxxopts
include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.1
)

FetchContent_MakeAvailable(cxxopts)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Библиотека с логикой загрузки, анализа и предобработки
add_library(lab1_core
    src/csv_loader.cpp
    src/statistics.cpp
    src/preprocessing.cpp
    src/supervised_dataset.cpp
    src/dataset_splitter.cpp
)

target_include_directories(lab1_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Предупреждения компилятора
if(MSVC)
    target_compile_options(lab1_core PRIVATE /W4)
else()
    target_compile_options(lab1_core PRIVATE -Wall -Wextra)
endif()

# Консольное приложение
add_executable(lab1 src/lab1.cpp)
target_link_libraries(lab1 PRIVATE cxxopts::cxxopts)
target_link_libraries(lab1 PRIVATE lab1_core)

if(MSVC)
    target_compile_options(lab1 PRIVATE /W4)
else()
    target_compile_options(lab1 PRIVATE -Wall -Wextra)
endif()

# Автоматические тесты
include(CTest)

if(BUILD_TESTING)
    add_executable(lab1_tests src/lab1_tests.cpp)
    target_link_libraries(lab1_tests PRIVATE lab1_core)

    if(MSVC)
        target_compile_options(lab1_tests PRIVATE /W4)
    else()
        target_compile_options(lab1_tests PRIVATE -Wall -Wextra)
    endif()

    add_test(NAME lab1_tests COMMAND lab1_tests)
endif()
