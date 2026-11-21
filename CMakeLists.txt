
cmake_minimum_required(VERSION 3.24)

project(lab_1_data_processing LANGUAGES CXX)

# Генерация документации является необязательной и не входит в обычную сборку.
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )

    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating HTML documentation with Doxygen"
        VERBATIM
    )
endif()

# Подключение библиотеки cxxopts
include(FetchContent)

FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.2.1
)

FetchContent_MakeAvailable(cxxopts)

# Настройки языка
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Общие настройки компилятора
add_library(project_options INTERFACE)

# Предупреждения компилятора
if(MSVC)
    target_compile_options(project_options INTERFACE /W4)
else()
    target_compile_options(
        project_options
        INTERFACE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Санитайзеры включаются только там, где доступны runtime-библиотеки
if(MSVC)
    # AddressSanitizer для MSVC
    target_compile_options(
        project_options
        INTERFACE
        "$<$<CONFIG:Debug>:/fsanitize=address>"
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # AddressSanitizer и UndefinedBehaviorSanitizer
    target_compile_options(
        project_options
        INTERFACE
        "$<$<CONFIG:Debug>:-fsanitize=address,undefined>"
        "$<$<CONFIG:Debug>:-fno-omit-frame-pointer>"
    )

    target_link_options(
        project_options
        INTERFACE
        "$<$<CONFIG:Debug>:-fsanitize=address,undefined>"
    )
else()
    message(STATUS "Санитайзеры отключены для текущей конфигурации компилятора")
endif()

# Библиотека с логикой загрузки, анализа и моделей
add_library(lab1_core
    src/csv_loader.cpp
    src/statistics.cpp
    src/preprocessing.cpp
    src/supervised_dataset.cpp
    src/dataset_splitter.cpp
    src/matrix.cpp
    src/decision_stump.cpp
    src/split_finder.cpp
    src/cart.cpp
)

target_include_directories(
    lab1_core
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(lab1_core PUBLIC project_options)

# Консольное приложение
add_executable(lab1 src/lab1.cpp)

target_link_libraries(
    lab1
    PRIVATE
    lab1_core
    cxxopts::cxxopts
)

# Автоматические тесты
include(CTest)

if(BUILD_TESTING)

 # Тесты ЛР1
add_executable(lab1_tests
    test/lab1_tests.cpp
)

target_link_libraries(lab1_tests
    PRIVATE lab1_core
)

add_test(
    NAME lab1_tests
    COMMAND lab1_tests
)

# Тесты DecisionStump
add_executable(decision_stump_tests
    test/decision_stump_tests.cpp
)

target_link_libraries(decision_stump_tests
    PRIVATE lab1_core
)

add_test(
    NAME decision_stump_tests
    COMMAND decision_stump_tests
)

# Тесты MatrixSlice
add_executable(matrix_slice_tests
    test/matrix_slice_tests.cpp
)

target_link_libraries(matrix_slice_tests
    PRIVATE lab1_core
)

add_test(
    NAME matrix_slice_tests
    COMMAND matrix_slice_tests
)

add_executable(cart_tests
    test/cart_tests.cpp
)

target_link_libraries(cart_tests
    PRIVATE lab1_core
)

add_test(
    NAME cart_tests
    COMMAND cart_tests
)

add_executable(cart_audit_tests
    test/cart_audit_tests.cpp
)

target_link_libraries(
    cart_audit_tests
    PRIVATE lab1_core
)

add_test(
    NAME cart_audit_tests
    COMMAND cart_audit_tests
)

add_test(
    NAME decision_stump_cli_tests
    COMMAND ${CMAKE_COMMAND}
        -DAPP=$<TARGET_FILE:lab1>
        -DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/test/decision_stump_cli_tests.cmake
)
endif()
