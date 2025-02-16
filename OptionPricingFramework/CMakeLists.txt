cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(PyTorchCpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set the build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Ensure debug symbols are included
set(CMAKE_CXX_FLAGS_DEBUG "-g")

# Suppress specific warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

# Define source files for CppUnitTest
set(SOURCE_FILES_CPPUNITTEST
    CppUnitTest/Failure.cpp
    CppUnitTest/Main.cpp
    CppUnitTest/Test.h
    CppUnitTest/TestRegistry.cpp
    CppUnitTest/TestResult.cpp
    CppUnitTest/Failure.h
    CppUnitTest/Test.cpp
    CppUnitTest/TestHarness.h
    CppUnitTest/TestRegistry.h
    CppUnitTest/TestResult.h
)

set(SOURCE_FILES_EXAMPLE_AAD
    tests/testTorchCppExample.cpp
)

set(SOURCE_FILES_OPTION_PRICING_BS
    tests/testBlackScholesClosedForm.cpp
)

set(SOURCE_FILES_OPTION_PRICING_MC
    include/methods/MonteCarlo.hpp
    src/methods/MonteCarlo.cpp
    tests/testMonteCarloEuropeanOption.cpp
)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torchCppExample ${SOURCE_FILES_EXAMPLE_AAD})
target_link_libraries(torchCppExample "${TORCH_LIBRARIES}")

add_executable(blackScholesClosedForm ${SOURCE_FILES_OPTION_PRICING_BS})
target_link_libraries(blackScholesClosedForm "${TORCH_LIBRARIES}")

add_executable(monteCarloEuropeanOption ${SOURCE_FILES_OPTION_PRICING_MC})
target_link_libraries(monteCarloEuropeanOption "${TORCH_LIBRARIES}")


#set_property(TARGET torchCppExample PROPERTY CXX_STANDARD 17)
