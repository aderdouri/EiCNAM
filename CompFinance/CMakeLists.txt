cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Define the CompFinance library as a shared library
add_library(CompFinance SHARED
    AAD.cpp
    AAD.h
    AADNode.h
    AADNumber.h
    AADTape.h
    blocklist.h
    ConcurrentQueue.h
    ThreadPool.cpp
    analytics.h
    choldc.h
    gaussians.h
    interp.h
    ivs.h
    main.h
    matrix.h
    mcBase.cpp
    mcBase.h
    mcMdl.h
    mcMdlBS.h
    mcMdlDupire.h
    mcMdlMultiDisplaced.h
    mcPrd.h
    mcPrdMulti.h
    mrg32k3a.h
    sobol.cpp
    sobol.h
    store.h
    store.cpp
    ThreadPool.h
    toyCode.h
    utility.h
)

# Define source files for AAD (basic Instrumentation)
add_library(CompFinance_AAD SHARED
    AAD.cpp
    AAD.h
    AADNode.h
    AADNumber.h
    AADTape.h
    blocklist.h
)

# Define source files for AADExpr (advanced Instrumentation)
add_library(CompFinance_AADExpr SHARED
    AADExpr.h
    AADNode.h
    AAD.cpp
    AAD.h
    AADTape.h
    blocklist.h
    gaussians.h
)

# Include directories
target_include_directories(CompFinance PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CompFinance_AAD PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CompFinance_AADExpr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})