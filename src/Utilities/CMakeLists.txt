project(Utilities)
cmake_minimum_required(VERSION 2.8)
#SET( CMAKE_CXX_COMPILER clang++ )

if(NOT MSVC)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

SET(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED) 
if(Boost_FOUND)

add_library(Utilities
	    Design/Factory.h
	    Design/Singleton.h
	    Design/StateMachine.h
	    MPL/Tuple.h
	    MPL/TypeList.h
	    MPL/TypeTraits.h
	    Memory/LRUCache.h
	    StrUtil/StrUtil.h
	    System/Time.h
	    System/Time.cpp)

target_include_directories(Utilities
		    PUBLIC
			  ${CMAKE_SOURCE_DIR}
		    PRIVATE
			  ${Boost_INCLUDE_DIRS})
endif()
