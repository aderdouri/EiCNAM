cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(addCpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CXX_FLAGS "-Wall -O0")
set(CMAKE_CXX_FLAGS "${CXX_FLAGS}")

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

set(SOURCE_FILES_TUTORIALS_01
    Tutorials/toyCodeChp9section5.cpp
)

set(SOURCE_FILES_TUTORIALS_02
    Tutorials/toyCodeChp9sections2to4.cpp
)

set(SOURCE_FILES_TUTORIALS_03
    Tutorials/toyCodeChp15section2.cpp
)

add_executable(toyCodeChp9section5 ${SOURCE_FILES_TUTORIALS_01})
#add_executable(toyCodeChp9sections2to4 ${SOURCE_FILES_TUTORIALS_02})
#add_executable(toyCodeChp15section2 ${SOURCE_FILES_TUTORIALS_03})