cmake_minimum_required(VERSION 3.24)

project(lab_1_data_processing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lab1_core
    csv_loader.cpp
)
target_include_directories(lab1_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lab1
    lab1.cpp
)
target_link_libraries(lab1 PRIVATE lab1_core)
