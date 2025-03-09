cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Create a static library for CppUnitTest
add_library(CppUnitTest STATIC
    Failure.cpp
    Main.cpp
    Test.h
    TestRegistry.cpp
    TestResult.cpp
    Failure.h
    Test.cpp
    TestHarness.h
    TestRegistry.h
    TestResult.h
)

# Include directories for the library
target_include_directories(CppUnitTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
