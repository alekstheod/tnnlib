project(NeuralNetwork)

SET(BOOST_MIN_VERSION "1.68.0")
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED)
find_package(OpenCL)

add_library(NeuralNetwork
  NeuralNetwork.cpp
  INeuralLayer.h
  Config.h
  LearningAlgorithm/BackPropagation/BepAlgorithm.h
  LearningAlgorithm/BackPropagation/BPNeuralLayer.h
  LearningAlgorithm/BackPropagation/BPNeuron.h
  LearningAlgorithm/BackPropagation/ErrorFunction.h
  Neuron/INeuron.h
  Neuron/Neuron.h
  Neuron/ActivationFunction/BiopolarSigmoidFunction.h
  Neuron/ActivationFunction/IActivationFunction.h
  Neuron/ActivationFunction/LogScaleSoftmaxFunction.h
  Neuron/ActivationFunction/SoftmaxFunction.h
  Neuron/ActivationFunction/SigmoidFunction.h
  Neuron/ActivationFunction/TanhFunction.h
  Perceptron/Perceptron.h
  NeuralLayer/NeuralLayer.h
  NeuralLayer/OpenCLNeuralLayer.h
  SOM/INode.h
  SOM/INeighbourhood.h
  SOM/IPosition.h
  SOM/K2DPosition.h
  SOM/K2DNeighbourhood.h
  SOM/KNode.h
  SOM/KohonenMap.h
  Serialization/NeuralLayerMemento.h
  Serialization/NeuronMemento.h
  Serialization/PerceptronMemento.h
  )

set_target_properties(NeuralNetwork PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(NeuralNetwork
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIR})

set_target_properties(NeuralNetwork PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
)

target_link_libraries(NeuralNetwork
  Utilities
  range-v3
  cereal
  ${Boost_LIBRARIES}
  ${OpenCL_LIBRARY})

enable_testing()
add_subdirectory(tests)