project(NeuralNetwork_ut)

set(BOOST_COMPONENTS_NEEDED filesystem system)
set(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS ${BOOST_COMPONENTS_NEEDED} REQUIRED)
if(Boost_FOUND)
  link_directories(${Boost_LIBRARY_DIRS})

  add_executable(NeuralNetwork_ut
    main.cpp)

  add_definitions(-DETC_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/etc")
  target_include_directories(NeuralNetwork_ut
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Boost_INCLUDE_DIRS}
    )

  target_link_libraries(NeuralNetwork_ut
    NeuralNetwork
    Utilities
    ${Boost_LIBRARIES}
    )

  set(UNIT_TEST NeuralNetwork_ut)
  add_test(${UNIT_TEST} ${UNIT_TEST})
  add_custom_command(
    TARGET ${UNIT_TEST}
    COMMENT "Run tests"
    POST_BUILD 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIGURATION> -R "${UNIT_TEST}" --output-on-failures --verbose
    )

  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)
    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
    setup_target_for_coverage(NAME ${UNIT_TEST}_coverage ${UNIT_TEST} coverage)
    target_link_libraries(${UNIT_TEST} gcov)
  endif()

endif()
