cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Define source files for OptionPricingFramework
set(SOURCE_FILES_OPTION_PRICING_BS
    include/Methods/BlackScholesClosedForm.hpp
    src/Methods/BlackScholesClosedForm.cpp
    tests/testBlackScholesClosedForm.cpp
)

set(SOURCE_FILES_OPTION_PRICING_BS_GRADIENTS
    include/Methods/BlackScholesClosedForm.hpp
    src/Methods/BlackScholesClosedForm.cpp
    tests/testBSGradients.cpp
)

set(SOURCE_FILES_OPTION_STANDARD_BS
    include/Methods/BS.hpp
    src/Methods/BS.cpp
    tests/testBS.cpp
)

set(SOURCE_FILES_OPTION_STANDARD_BSWrapper
    include/Methods/BSWrapper.hpp
    src/Methods/BSWrapper.cpp
)

set(SOURCE_FILES_OPTION_PRICING_MC
    include/Methods/MonteCarlo.hpp
    src/Methods/MonteCarlo.cpp
    tests/testMonteCarloEuropeanOption.cpp
)

set(SOURCE_FILES_EXAMPLE_AAD
    tests/testTorchCppExample.cpp
)

set(SOURCE_FILES_EIGEN_CPP
    tests/testEigenCppExample.cpp
)

set(SOURCE_FILES_LOG_NORMAL_PROCESS_TEST
    include/Processes/StochasticProcess.hpp
    include/Processes/LogNormalProcess.hpp
    src/Processes/StochasticProcess.cpp
    src/Processes/LogNormalProcess.cpp
    tests/testLogNormalProcess.cpp
)

set(SOURCE_FILES_BINOMIAL_TREE
    include/Methods/BinomialTree.hpp
    src/Methods/BinomialTree.cpp
    tests/testBinomialTree.cpp
)

# Define executables
add_executable(blackScholesClosedForm ${SOURCE_FILES_OPTION_PRICING_BS})
target_link_libraries(blackScholesClosedForm "${TORCH_LIBRARIES}")

add_executable(bsGradients ${SOURCE_FILES_OPTION_PRICING_BS_GRADIENTS})
target_link_libraries(bsGradients "${TORCH_LIBRARIES}")

add_executable(testBS ${SOURCE_FILES_OPTION_STANDARD_BS})
target_link_libraries(testBS "${TORCH_LIBRARIES}")

add_executable(monteCarloEuropeanOption ${SOURCE_FILES_OPTION_PRICING_MC})
target_link_libraries(monteCarloEuropeanOption "${TORCH_LIBRARIES}")

add_executable(torchCppExample ${SOURCE_FILES_EXAMPLE_AAD})
target_link_libraries(torchCppExample "${TORCH_LIBRARIES}")

add_executable(eigenCppExample ${SOURCE_FILES_EIGEN_CPP})
target_include_directories(eigenCppExample PRIVATE ${EIGEN3_INCLUDE_DIR})

add_executable(testLogNormalProcess ${SOURCE_FILES_LOG_NORMAL_PROCESS_TEST})
target_link_libraries(testLogNormalProcess "${TORCH_LIBRARIES}")

add_executable(binomialTree ${SOURCE_FILES_BINOMIAL_TREE})
target_link_libraries(binomialTree "${TORCH_LIBRARIES}")

add_executable(testLinAL tests/testLinAL.cpp)
target_link_libraries(testLinAL "${TORCH_LIBRARIES}")

add_executable(singleTest tests/singleTest.cpp)
target_link_libraries(singleTest "${TORCH_LIBRARIES}")

add_executable(testCheckGPUs tests/testCheckGPUs.cpp)
target_link_libraries(testCheckGPUs "${TORCH_LIBRARIES}")
