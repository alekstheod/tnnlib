project(Test)

cmake_minimum_required(VERSION 2.8)

#SET( CMAKE_CXX_COMPILER clang++ )

SET(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS system filesystem serialization) 

link_directories(${Boost_LIBRARY_DIRS})

add_executable(Test src/Test.cpp)

include_directories(${Boost_INCLUDE_DIRS})


link_directories(${Boost_LIBRARY_DIRS})

target_link_libraries(Test  
	NeuralNetwork
	Utilities
	${Boost_LIBRARIES}
	)
		      
