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
#add_executable(AdStarter ${SOURCE_FILES_CPPUNITTEST} ${SOURCE_FILES_AD_STARTER})


set(SOURCE_FILES_BLACK_FORMULA
    Tutorials/src/Utils.h
    Tutorials/src/Utils.cpp
    Tutorials/src/BlackFormula.h
    Tutorials/src/BlackFormula.cpp
    Tutorials/src/BlackFormulaTest.cpp    
)
#add_executable(BlackFormula ${SOURCE_FILES_BLACK_FORMULA})

set(SOURCE_FILES_SABR_VOLATILITY_FORMULA
    Tutorials/src/Utils.h
    Tutorials/src/Utils.cpp
    Tutorials/src/SabrVolatilityFormula.h
    Tutorials/src/SabrVolatilityFormula.cpp
    Tutorials/src/SabrVolatilityFormulaTest.cpp
)
#add_executable(SabrVolatilityFormula ${SOURCE_FILES_SABR_VOLATILITY_FORMULA})

set(SOURCE_FILES_SABR_PRICE_FORMULA
    Tutorials/src/Utils.h
    Tutorials/src/Utils.cpp
    Tutorials/src/BlackFormula.h
    Tutorials/src/BlackFormula.cpp
    Tutorials/src/SabrVolatilityFormula.h
    Tutorials/src/SabrVolatilityFormula.cpp
    Tutorials/src/SabrPriceFormula.cpp
)
#add_executable(SabrPriceFormula ${SOURCE_FILES_SABR_PRICE_FORMULA})


#add_executable(toyCodeChp9sections2to4 Tutorials/src/toyCodeChp9sections2to4.cpp)
#add_executable(toyCodeChp9section5 Tutorials/src/toyCodeChp9section5.cpp)
#add_executable(toyCodeChp15section2 Tutorials/src/toyCodeChp15section2.cpp)
#add_executable(monteCarloSimulation ${SOURCE_FILES_MONTE_CARLO})


set(SOURCE_FILES_BASIC_INSTRUMENTATION 
    Tutorials/tests/BasicInstrumentationTest.cpp
)

set(SOURCE_FILES_BASIC_INSTRUMENTATION_EXPR 
    Tutorials/tests/BasicInstrumentationExprTest.cpp
)

#add_executable (autoDiff ${SOURCE_FILES_CPPUNITTEST} ${SOURCE_FILES_COMPFINANCE_AAD} ${SOURCE_FILES_BASIC_INSTRUMENTATION})
#target_compile_definitions(autoDiff PRIVATE AADET=0)

#add_executable (autoDiffExpr ${SOURCE_FILES_CPPUNITTEST} ${SOURCE_FILES_COMPFINANCE_AAD} ${SOURCE_FILES_BASIC_INSTRUMENTATION_EXPR})
#target_compile_definitions(autoDiffExpr PRIVATE AADET=1)


set(SOURCE_FILES_DOC_UWE 
    dcoUwe/dco_a1s/dco_a1s_type.hpp
    dcoUwe/dco_a1s/dco_a1s_type.cpp
    dcoUwe/dco_a1s/main.cpp
)

#add_executable (dcoUwe ${SOURCE_FILES_DOC_UWE})

# Add executable for matrix exponential example
#add_executable(matrixExponential Tutorials/src/matrixExponontial.cpp)

# Link only Eigen3 library
#target_link_libraries(matrixExponential Eigen3::Eigen)


add_executable(mdspan Tutorials/src/mdspan.cpp)


set(SOURCE_FILES_COMPFINANCE_ALL
    CompFinance/AAD.cpp
    CompFinance/AAD.cpp
    CompFinance/AAD.h
    CompFinance/AADExpr.h
    CompFinance/AADNode.h
    CompFinance/AADNumber.h
    CompFinance/AADTape.h
    CompFinance/ConcurrentQueue.h
    CompFinance/ThreadPool.cpp
    CompFinance/analytics.h
    CompFinance/blocklist.h
    CompFinance/choldc.h
    CompFinance/gaussians.h
    CompFinance/interp.h
    CompFinance/ivs.h
    CompFinance/main.h
    CompFinance/matrix.h
    CompFinance/mcBase.cpp
    CompFinance/mcBase.h
    CompFinance/mcMdl.h
    CompFinance/mcMdlBS.h
    CompFinance/mcMdlDupire.h
    CompFinance/mcMdlMultiDisplaced.h
    CompFinance/mcPrd.h
    CompFinance/mcPrdMulti.h
    CompFinance/mrg32k3a.h
    CompFinance/sobol.cpp
    CompFinance/sobol.h
    CompFinance/store.h
    CompFinance/store.cpp
    CompFinance/ThreadPool.h
    CompFinance/toyCode.h
    CompFinance/utility.h
    #CompFinanceTests/BlackScholesTest.cpp
    #CompFinanceTests/mertonTest.cpp
    #CompFinanceTests/AADriskPutEuropeanTest.cpp
    CompFinanceTests/AADriskPutBarrierTest.cpp
    CompFinanceTests/DupireCalibrationTest.cpp
    CompFinanceTests/DupireBarrierMcRisksTest.cpp
    CompFinanceTests/DupireMonteCarlo.cpp
    CompFinanceTests/DupireMonteCarloFDM.cpp
)

add_executable (compFinanceTest ${SOURCE_FILES_CPPUNITTEST} ${SOURCE_FILES_COMPFINANCE_ALL})


set(SOURCE_FILES_EXAMPLE_AAD
    OptionPricingFramework/tests/testTorchCppExample.cpp
)

set(SOURCE_FILES_OPTION_PRICING_BS
    OptionPricingFramework/tests/testBlackScholesClosedForm.cpp
)

set(SOURCE_FILES_OPTION_PRICING_MC
    OptionPricingFramework/include/methods/MonteCarlo.hpp
    OptionPricingFramework/src/methods/MonteCarlo.cpp
    OptionPricingFramework/tests/testMonteCarloEuropeanOption.cpp
)

set(SOURCE_FILES_EIGEN_CPP
    OptionPricingFramework/tests/testEigenCppExample.cpp
)

# Set the paths to the third-party libraries
set(THIRD_PARTY_LIBS_DIR $ENV{HOME}/Downloads/ThirdPartyLibs)

# Include Eigen3
set(EIGEN3_INCLUDE_DIR ${THIRD_PARTY_LIBS_DIR}/eigen-3.4.0)
include_directories(${EIGEN3_INCLUDE_DIR})

# Include libtorch
set(Torch_DIR ${THIRD_PARTY_LIBS_DIR}/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# Exclude libkineto.a from linking
list(REMOVE_ITEM TORCH_LIBRARIES "${THIRD_PARTY_LIBS_DIR}/libtorch/lib/libkineto.a")

# Print the paths of Eigen3 and Torch
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Torch include directory: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch library directory: ${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torchCppExample ${SOURCE_FILES_EXAMPLE_AAD})
target_link_libraries(torchCppExample "${TORCH_LIBRARIES}")

add_executable(blackScholesClosedForm ${SOURCE_FILES_OPTION_PRICING_BS})
target_link_libraries(blackScholesClosedForm "${TORCH_LIBRARIES}")

add_executable(monteCarloEuropeanOption ${SOURCE_FILES_OPTION_PRICING_MC})
target_link_libraries(monteCarloEuropeanOption "${TORCH_LIBRARIES}")

add_executable(eigenCppExample ${SOURCE_FILES_EIGEN_CPP})
target_include_directories(eigenCppExample PRIVATE ${EIGEN3_INCLUDE_DIR})



