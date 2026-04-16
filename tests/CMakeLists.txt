project(highmap_tests)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(googletest)

file(GLOB TEST_SOURCES src/*.cpp)

add_executable(highmap_tests ${TEST_SOURCES})

target_link_libraries(highmap_tests
  PRIVATE
    GTest::gtest
    GTest::gtest_main
    highmap
)