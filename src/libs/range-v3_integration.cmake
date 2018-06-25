
add_library(range-v3 INTERFACE)

if (${MSVC})
	target_include_directories(range-v3 INTERFACE
		${CMAKE_CURRENT_SOURCE_DIR}/range-v3/win)
	target_compile_options(range-v3 INTERFACE
		# '+=': conversion from '__int64' to 'int', possible loss of data
		# (fixed in the original library)
		"/wd4244")
else ()
	target_include_directories(range-v3 INTERFACE
		${CMAKE_CURRENT_SOURCE_DIR}/range-v3/nix)
endif()

# Deprecated std::result_of<> usage (MSVC 17)
target_compile_definitions(range-v3 INTERFACE
	_SILENCE_CXX17_RESULT_OF_DEPRECATION_WARNING)
