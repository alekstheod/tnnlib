
add_library(range-v3 INTERFACE)

if (${MSVC})
	target_include_directories(range-v3 INTERFACE
		${CMAKE_CURRENT_SOURCE_DIR}/range-v3/win)
else ()
	target_include_directories(range-v3 INTERFACE
		${CMAKE_CURRENT_SOURCE_DIR}/range-v3/nix)
endif()

# Deprecated std::result_of<> usage (MSVC 17)
target_compile_definitions(range-v3 INTERFACE
	_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
