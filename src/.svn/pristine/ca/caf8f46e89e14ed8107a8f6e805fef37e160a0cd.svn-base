project(ocr)

cmake_minimum_required(VERSION 2.8)

if(NOT MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set( CMAKE_CXX_COMPILER clang++ )
endif()

find_package(PNG REQUIRED)
message(STATUS "LibPNG package was found with include dir: '${PNG_INCLUDE_DIRS}'")

set( BOOST_COMPONENTS_NEEDED filesystem system serialization)
set(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS ${BOOST_COMPONENTS_NEEDED} REQUIRED)

set(NeuralNetworkDir ${PROJECT_SOURCE_DIR}/../NeuralNetwork)

add_executable(ocr src/main.cpp )

include_directories(
	${Boost_INCLUDE_DIRS}
	${PNG_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})

target_link_libraries(ocr  
	NeuralNetwork
	${Boost_LIBRARIES}
	${PNG_LIBRARIES}
	)

		      
		      