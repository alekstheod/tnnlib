project(NeuralNetwork)
cmake_minimum_required(VERSION 2.8)

set( BOOST_COMPONENTS_NEEDED serialization)
SET(BOOST_MIN_VERSION "1.46.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS ${BOOST_COMPONENTS_NEEDED} REQUIRED) 
if(Boost_FOUND)
#SET( CMAKE_CXX_COMPILER clang++ )

if(NOT MSVC)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

add_library(NeuralNetwork 
	  ./NeuralNetwork/NeuralNetwork.cpp
	  ./NeuralNetwork/INeuralLayer.h
	  ./NeuralNetwork/NNException.h
	  ./NeuralNetwork/Config.h
	  ./NeuralNetwork/LearningAlgorithm/BackPropagation/BepAlgorithm.h
	  ./NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuralLayer.h
	  ./NeuralNetwork/LearningAlgorithm/BackPropagation/BPNeuron.h
	  ./NeuralNetwork/LearningAlgorithm/BackPropagation/ErrorFunction.h
	  ./NeuralNetwork/LearningAlgorithm/BackPropagation/OutputNeuralLayer.h
	  ./NeuralNetwork/Neuron/INeuron.h
	  ./NeuralNetwork/Neuron/Neuron.h
	  ./NeuralNetwork/Neuron/ActivationFunction/BiopolarSigmoidFunction.h
	  ./NeuralNetwork/Neuron/ActivationFunction/IActivationFunction.h
	  ./NeuralNetwork/Neuron/ActivationFunction/LogScaleSoftmaxFunction.h
	  ./NeuralNetwork/Neuron/ActivationFunction/SoftmaxFunction.h
	  ./NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h
	  ./NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h
	  ./NeuralNetwork/Perceptron/Perceptron.h
	  ./NeuralNetwork/Perceptron/NeuralLayer/NeuralLayer.h
	  ./NeuralNetwork/SOM/INode.h
	  ./NeuralNetwork/SOM/INeighbourhood.h
	  ./NeuralNetwork/SOM/IPosition.h
	  ./NeuralNetwork/SOM/K2DPosition.h
	  ./NeuralNetwork/SOM/K2DNeighbourhood.h
	  ./NeuralNetwork/SOM/KNode.h
	  ./NeuralNetwork/SOM/KohonenMap.h
	  ./NeuralNetwork/Serialization/NeuralLayerMemento.h
	  ./NeuralNetwork/Serialization/NeuronMemento.h
	  ./NeuralNetwork/Serialization/PerceptronMemento.h
	  )

set_target_properties(NeuralNetwork PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(NeuralNetwork
		  PUBLIC
			.
			..Utilities
			${Boost_INCLUDE_DIRS})
	
target_link_libraries(NeuralNetwork
		      Utilities
		      ${Boost_LIBRARIES})

endif()
