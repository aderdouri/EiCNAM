cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(addCpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-Wall -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O2")

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

# Define source files for AAD (basic Instrumentation)
set(SOURCE_FILES_COMPFINANCE_AAD
    CompFinance/AAD.cpp
    CompFinance/AAD.h
    CompFinance/AADNode.h
    CompFinance/AADNumber.h
    CompFinance/AADTape.h
    CompFinance/blocklist.h
    Tutorials/basicInstrumentation.cpp
)

# Define source files for AADExpr (advanced Instrumentation)
set(SOURCE_FILES_COMPFINANCE_AAD_EXPR
    CompFinance/AADExpr.h
    CompFinance/AADNode.h
    CompFinance/AAD.cpp
    CompFinance/AAD.h
    CompFinance/AADTape.h
    CompFinance/blocklist.h
    CompFinance/gaussians.h
    Tutorials/basicInstrumentationExpr.cpp
)

set(SOURCE_FILES_CRTP
    Tutorials/curiouslyRecursiveTemplatePattern.cpp
)

set(SOURCE_FILES_DEDUCING_THIS
    Tutorials/incrementable.cpp
)

set(SOURCE_FILES_MONTE_CARLO
    Tutorials/monteCarloSimulation.cpp
)

add_executable(toyCodeChp9sections2to4 Tutorials/toyCodeChp9sections2to4.cpp)
add_executable(toyCodeChp9section5 Tutorials/toyCodeChp9section5.cpp)
add_executable(toyCodeChp15section2 Tutorials/toyCodeChp15section2.cpp)
add_executable(monteCarloSimulation ${SOURCE_FILES_MONTE_CARLO})

add_executable(basicInstrumentation ${SOURCE_FILES_COMPFINANCE_AAD})
target_compile_definitions(basicInstrumentation PRIVATE AADET=0)

add_executable(basicInstrumentationExpr ${SOURCE_FILES_COMPFINANCE_AAD_EXPR})
target_compile_definitions(basicInstrumentationExpr PRIVATE AADET=1)

add_executable(tangent_book01 Tutorials/tangent_book01.cpp)
add_executable(adjoint_book01 Tutorials/adjoint_book01.cpp)
add_executable(black_scholes_tangent Tutorials/black_scholes_tangent.cpp)
