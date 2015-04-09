project(tnnlib)
cmake_minimum_required(VERSION 2.8)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif()

add_subdirectory(Utilities)
add_subdirectory(NeuralNetwork)
add_subdirectory(Test)
add_subdirectory(gmock-1.7.0)
add_subdirectory(NeuralNetwork_ut)

IF(NOT DEFINED ENV{DISABLE_OCR})
add_subdirectory(ocr)
ENDIF()

enable_testing()
add_test(NeuralNetwork_ut NeuralNetwork_ut/NeuralNetwork_ut)


find_program(asciidoc_cmd NAMES asciidoc)
if(NOT ${asciidoc_cmd} EQUAL "asciidoc_cmd-NOTFOUND")
add_custom_target(design ALL #COMMAND ${asciidoc_cmd} --filter install ${CMAKE_CURRENT_LIST_DIR}/doc/plantuml.zip
			     COMMAND ${asciidoc_cmd} --verbose -dbook ${CMAKE_CURRENT_LIST_DIR}/doc/structural/documentation.txt
			     )
endif()
