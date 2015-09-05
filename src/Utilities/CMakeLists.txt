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
	    Utilities/Design/Factory.h
	    Utilities/Design/Singleton.h
	    Utilities/Design/StateMachine.h
	    Utilities/MPL/Tuple.h
	    Utilities/MPL/TypeList.h
	    Utilities/MPL/TypeTraits.h
	    Utilities/Math/Math.cpp
	    Utilities/Math/Math.h
	    Utilities/Memory/LRUCache.h
	    Utilities/StrUtil/StrUtil.h
	    Utilities/System/Time.h
	    Utilities/System/Time.cpp)

target_include_directories(Utilities
		      PUBLIC
			.
		      PRIVATE
			${Boost_INCLUDE_DIRS})
endif()
