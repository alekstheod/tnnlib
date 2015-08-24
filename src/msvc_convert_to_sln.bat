@echo off

if "%1" == "" (
	echo Help:
	echo   %~n0 path-to-source
	echo.

	echo Example:
	echo   cd tnnlib\src
	echo   mkdir build
	echo   cd build
	echo   ..\build_msvc ..

	exit /B 1
	)

:: Path to CMake
set cmake=cmake
:: CMake generator
set msvc_version=Visual Studio 12 2013
:: Set to 0 if you want to disable OpenCL (e.g. OpenCL is not installed)
set enable_opencl=0

:: ZLIB configuration
set zlib_include_dir=C:\libs\zlib\zlib-1.2.8
set zlib_debug_lib=C:\libs\libpng\lpng1618\projects\vstudio\Debug Library\zlib.lib
set zlib_release_lib=C:\libs\libpng\lpng1618\projects\vstudio\Release Library\zlib.lib

:: PNG configuration
set libpng_include_dir=C:\libs\libpng\lpng1618
set libpng_debug_lib=C:\libs\libpng\lpng1618\projects\vstudio\Debug Library\libpng16.lib
set libpng_release_lib=C:\libs\libpng\lpng1618\projects\vstudio\Release Library\libpng16.lib

:: PYTHON configuration
set python_exe=C:\Programs\Python27\python.exe

:: BOOST configuration
set boost_root=C:\libs\boost\boost_1_57_0
set boost_libraries_dir=C:\libs\boost\boost_1_57_0\build\msvc-12.0\32\lib







:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set opencl_cmake_flag=
if %enable_opencl% EQU 0 (
	set opencl_cmake_flag=-DCMAKE_DISABLE_FIND_PACKAGE_OpenCL:bool=TRUE
	)

%cmake% -G "%msvc_version%" -Wno-dev ^
	%opencl_cmake_flag% ^
	"-DZLIB_INCLUDE_DIR=%zlib_include_dir%" ^
	"-DZLIB_LIBRARY:PATH=debug;%zlib_debug_lib%;optimized;%zlib_release_lib%" ^
	"-DPNG_PNG_INCLUDE_DIR=%libpng_include_dir%" ^
	"-DPNG_LIBRARY:PATH=debug;%libpng_debug_lib%;optimized;%libpng_release_lib%" ^
	"-DPYTHON_EXECUTABLE=%python_exe%" ^
	"-DBOOST_ROOT:PATH=%boost_root%" ^
	"-DBOOST_LIBRARYDIR:PATH=%boost_libraries_dir%" ^
	"%1"
