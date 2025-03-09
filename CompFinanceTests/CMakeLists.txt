cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

# Define source files for CompFinanceTests
set(SOURCE_FILES_COMPFINANCE_TESTS
    AADriskPutBarrierTest.cpp
    DupireCalibrationTest.cpp
    DupireBarrierMcRisksTest.cpp
    DupireMonteCarlo.cpp
    DupireMonteCarloFDM.cpp
)

# Define executables
add_executable(AADriskPutBarrierTest AADriskPutBarrierTest.cpp)
target_link_libraries(AADriskPutBarrierTest PRIVATE CompFinance CppUnitTest)

add_executable(DupireCalibrationTest DupireCalibrationTest.cpp)
target_link_libraries(DupireCalibrationTest PRIVATE CompFinance CppUnitTest)

add_executable(DupireBarrierMcRisksTest DupireBarrierMcRisksTest.cpp)
target_link_libraries(DupireBarrierMcRisksTest PRIVATE CompFinance CppUnitTest)

add_executable(DupireMonteCarlo DupireMonteCarlo.cpp)
target_link_libraries(DupireMonteCarlo PRIVATE CompFinance CppUnitTest)

add_executable(DupireMonteCarloFDM DupireMonteCarloFDM.cpp)
target_link_libraries(DupireMonteCarloFDM PRIVATE CompFinance CppUnitTest)

add_executable(toyCodeChp9section5 toyCodeChp9section5.cpp)
add_executable(toyCodeChp9sections2to4 toyCodeChp9sections2to4.cpp)
add_executable(toyCodeChp15section2 toyCodeChp15section2.cpp)

add_executable(aadTape aadTape.cpp)
add_executable(optimizedAADTape optimizedAADTape.cpp)
add_executable(memoryPool memoryPool.cpp)