project(tnnlib)
cmake_minimum_required(VERSION 2.8)

add_subdirectory(Utilities)

add_subdirectory(NeuralNetwork)

add_subdirectory(Test)

add_subdirectory(gmock)
add_subdirectory(NeuralNetwork_ut)


add_subdirectory(ocr) 
