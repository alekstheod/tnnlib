project(Test)

SET(BOOST_MIN_VERSION "1.67.0")
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS system filesystem serialization)

link_directories(${Boost_LIBRARY_DIRS})

add_executable(Test src/Test.cpp)

target_include_directories(Test
  PRIVATE
  ${Boost_INCLUDE_DIRS})

set_target_properties(Test PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
)

link_directories(${Boost_LIBRARY_DIRS})

target_link_libraries(Test
  NeuralNetwork
  Utilities
  ${Boost_LIBRARIES}
  )
