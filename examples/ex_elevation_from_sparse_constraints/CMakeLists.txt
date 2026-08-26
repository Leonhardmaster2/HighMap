cmake_minimum_required(VERSION 3.22)

project(ex_elevation_from_sparse_constraints)

add_executable(${PROJECT_NAME} ex_elevation_from_sparse_constraints.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE highmap)
