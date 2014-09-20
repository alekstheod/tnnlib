project(NeuralNetwork_ut)
cmake_minimum_required(VERSION 2.8)
#SET( CMAKE_CXX_COMPILER clang++ )

set( BOOST_COMPONENTS_NEEDED serialization filesystem system )
set(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS ${BOOST_COMPONENTS_NEEDED} REQUIRED)
if(Boost_FOUND)

if(NOT MSVC)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if(MSVC)
	SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /DWIN32 /D_WINDOWS /EHsc" CACHE STRING "Debug compiler flags" FORCE)
	SET(CMAKE_CXX_FLAGS_RELEASE "/MT /DWIN32 /D_WINDOWS /EHsc" CACHE STRING "Debug compiler flags" FORCE)
endif()

add_executable(NeuralNetwork_ut
	main.cpp
	NeuralNetwork/Neuron/NeuronTest.cpp
	NeuralNetwork/Perceptron/PerceptronTest.cpp)

include_directories(.
	../gmock/include
	../gmock/gtest/include
	${Boost_INCLUDE_DIRS}
)

	target_link_libraries(NeuralNetwork_ut
		NeuralNetwork
		gtest
		gmock
		Utilities
		${Boost_LIBRARIES}
)

endif() 
