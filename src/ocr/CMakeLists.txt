project(ocr)

cmake_minimum_required(VERSION 2.8)

find_package(PNG REQUIRED)
message(STATUS "LibPNG package was found with include dir: '${PNG_INCLUDE_DIRS}'")

set( BOOST_COMPONENTS_NEEDED filesystem system serialization)
set(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS ${BOOST_COMPONENTS_NEEDED} REQUIRED)

set(NeuralNetworkDir ${PROJECT_SOURCE_DIR}/../NeuralNetwork)

link_directories(${Boost_LIBRARY_DIRS})

add_executable(ocr src/main.cpp )

add_definitions(-D_SCL_SECURE_NO_WARNINGS)

include_directories(
	${Boost_INCLUDE_DIRS}
	${PNG_INCLUDE_DIRS})

target_link_libraries(ocr  
	NeuralNetwork
	${Boost_LIBRARIES}
	${PNG_LIBRARIES}
	)

		      
		      