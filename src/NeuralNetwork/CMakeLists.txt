project(NeuralNetwork)
cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/NeuralNetwork/cmake")
set( BOOST_COMPONENTS_NEEDED serialization)
SET(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS ${BOOST_COMPONENTS_NEEDED} REQUIRED) 
find_package(OpenCL)

add_library(NeuralNetwork
	  NeuralNetwork.cpp
	  INeuralLayer.h
	  NNException.h
	  Config.h
	  LearningAlgorithm/BackPropagation/BepAlgorithm.h
	  LearningAlgorithm/BackPropagation/BPNeuralLayer.h
	  LearningAlgorithm/BackPropagation/BPNeuron.h
	  LearningAlgorithm/BackPropagation/ErrorFunction.h
	  LearningAlgorithm/BackPropagation/OutputNeuralLayer.h
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
			.
			../Utilities
			${Boost_INCLUDE_DIRS}
			${OPENCL_INCLUDE_DIR})
	
target_link_libraries(NeuralNetwork 
		      Utilities
		      ${Boost_LIBRARIES}
		      ${OPENCL_LIBRARIES})
		      
		      
add_subdirectory(NeuralNetwork_ut)