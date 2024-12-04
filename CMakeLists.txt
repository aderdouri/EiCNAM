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
)

# Examples
set(SOURCE_FILES_CRTP
    Tutorials/src/curiouslyRecursiveTemplatePattern.cpp
)

set(SOURCE_FILES_DEDUCING_THIS
    Tutorials/src/incrementable.cpp
)

set(SOURCE_FILES_MONTE_CARLO
    Tutorials/src/monteCarloSimulation.cpp
)

set(SOURCE_FILES_AD_STARTER
    Tutorials/src/Utils.h
    Tutorials/src/Utils.cpp
    Tutorials/src/AdStarter.h
    Tutorials/src/AdStarter.cpp
    Tutorials/tests/AdStarterTest.cpp
)
add_executable(AdStarter ${SOURCE_FILES_CPPUNITTEST} ${SOURCE_FILES_AD_STARTER})


set(SOURCE_FILES_BLACK_FORMULA
    Tutorials/src/Utils.h
    Tutorials/src/Utils.cpp
    Tutorials/src/BlackFormula.h
    Tutorials/src/BlackFormula.cpp
    Tutorials/src/BlackFormulaTest.cpp    
)
add_executable(BlackFormula ${SOURCE_FILES_BLACK_FORMULA})

set(SOURCE_FILES_SABR_VOLATILITY_FORMULA
    Tutorials/src/Utils.h
    Tutorials/src/Utils.cpp
    Tutorials/src/SabrVolatilityFormula.h
    Tutorials/src/SabrVolatilityFormula.cpp
    Tutorials/src/SabrVolatilityFormulaTest.cpp
)
add_executable(SabrVolatilityFormula ${SOURCE_FILES_SABR_VOLATILITY_FORMULA})

set(SOURCE_FILES_SABR_PRICE_FORMULA
    Tutorials/src/Utils.h
    Tutorials/src/Utils.cpp
    Tutorials/src/BlackFormula.h
    Tutorials/src/BlackFormula.cpp
    Tutorials/src/SabrVolatilityFormula.h
    Tutorials/src/SabrVolatilityFormula.cpp
    Tutorials/src/SabrPriceFormula.cpp
)
add_executable(SabrPriceFormula ${SOURCE_FILES_SABR_PRICE_FORMULA})


add_executable(toyCodeChp9sections2to4 Tutorials/src/toyCodeChp9sections2to4.cpp)
add_executable(toyCodeChp9section5 Tutorials/src/toyCodeChp9section5.cpp)
add_executable(toyCodeChp15section2 Tutorials/src/toyCodeChp15section2.cpp)
add_executable(monteCarloSimulation ${SOURCE_FILES_MONTE_CARLO})


set(SOURCE_FILES_BASIC_INSTRUMENTATION 
    Tutorials/tests/BasicInstrumentationTest.cpp
)

set(SOURCE_FILES_BASIC_INSTRUMENTATION_EXPR 
    Tutorials/tests/BasicInstrumentationExprTest.cpp
)

add_executable (autoDiff ${SOURCE_FILES_CPPUNITTEST} ${SOURCE_FILES_COMPFINANCE_AAD} ${SOURCE_FILES_BASIC_INSTRUMENTATION})
target_compile_definitions(autoDiff PRIVATE AADET=0)

add_executable (autoDiffExpr ${SOURCE_FILES_CPPUNITTEST} ${SOURCE_FILES_COMPFINANCE_AAD} ${SOURCE_FILES_BASIC_INSTRUMENTATION_EXPR})
target_compile_definitions(autoDiffExpr PRIVATE AADET=1)