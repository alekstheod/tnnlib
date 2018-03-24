project(tnnlib)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z").
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

add_subdirectory(Utilities)
add_subdirectory(NeuralNetwork)
add_subdirectory(Test)
add_subdirectory(doc)

IF(NOT DEFINED ENV{DISABLE_OCR})
add_subdirectory(ocr)
ENDIF()


