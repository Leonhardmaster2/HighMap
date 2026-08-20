project(mixbox)

set(MIXBOX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mixbox/cpp)

add_library(${PROJECT_NAME} STATIC ${MIXBOX_DIR}/mixbox.cpp)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${MIXBOX_DIR})
