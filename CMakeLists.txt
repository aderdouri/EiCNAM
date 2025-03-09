cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
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

# Include CppUnitTest as a static library
add_subdirectory(CppUnitTest)

# Include CompFinance as a shared library
add_subdirectory(CompFinance)

# Include CompFinanceTests as a separate project
add_subdirectory(CompFinanceTests)

# Include OptionPricingFramework as a separate project
add_subdirectory(OptionPricingFramework)

set(SOURCE_FILES_BASIC_INSTRUMENTATION 
    Tutorials/tests/BasicInstrumentationTest.cpp
)

set(SOURCE_FILES_BASIC_INSTRUMENTATION_EXPR 
     Tutorials/tests/BasicInstrumentationExprTest.cpp
)

add_executable(autoDiff
    ${SOURCE_FILES_BASIC_INSTRUMENTATION}
)
target_compile_definitions(autoDiff PRIVATE AADET=0)
target_link_libraries(autoDiff PRIVATE CppUnitTest CompFinance_AAD)

add_executable(autoDiffExpr
    ${SOURCE_FILES_BASIC_INSTRUMENTATION_EXPR}
)
target_compile_definitions(autoDiffExpr PRIVATE AADET=1)
target_link_libraries(autoDiffExpr PRIVATE CppUnitTest CompFinance_AADExpr)
