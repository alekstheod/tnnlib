tnnlib
======

template neural network library v3

Flexible interface which allows to define a neural network with different types of layers, Neurons and Activation functions:

```cpp                        
    typedef nn::Perceptron<float, 
                           nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 2>, 
                           nn::NeuralLayer<nn::Neuron, nn::TanhFunction, 20>, 
                           nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 1>
                           > Perceptron;
                           
    typedef nn::bp::BepAlgorithm< Perceptron, nn::bp::CrossEntropyError> Algo;
```




Calculating perceptron by using BEP algorithm:

```cpp
    typedef BepAlgorithm< Perceptron > Algo;
    Algo algorithm (0.09f );

    std::array< Algo::Prototype, 4> prototypes= { Algo::Prototype{{0.f, 1.f}, {1.f}} ,
        Algo::Prototype{{1.f, 0.f}, {1.f}} ,
        Algo::Prototype{{1.f, 1.f}, {0.f}} ,
        Algo::Prototype{{0.f, 0.f}, {0.f}}
    };

    unsigned int numOfEpochs = std::numeric_limits< unsigned int >::max();
    if( argc == 2 ){
      numOfEpochs = utils::lexical_cast< unsigned int >(argv[1]);
    }
    
    Perceptron perceptron = algorithm.calculate ( prototypes.begin(), prototypes.end(),
                                                  [] ( unsigned int epoch, float error, ) {
                                                        std::cout << error << std::endl;
                                                        return error >= 0.01.f
                                                  },
                                                  numOfEpochs);
```                                     

Building on Windows
===================

Library and tests depend on:
- Boost (serialization, system, filesystem)
- ZLIB
- LibPNG
- OpenCL
- range-v3
- cereal

`cereal` and `range-v3` libraries are supplied with this sources itself
(see src/libs folder).

Rest of libraries should be installed in the system.

The easiest way to do it on Windows is by using [vcpkg].
See [Quick Start] for installing instructions. Assume you have it in `C:\vcpkg`.

Install dependencies with next commands (by default, you will have 32 bit versions):

```
vcpkg install boost
vcpkg install zlib
vcpkg install libpng
vcpkg install opencl
```

For range-v3 you need to replace original sources with [Microsoft's fork for ranges]
in order to be able to compile it with VS. Replace src/libs/meta and src/libs/range
folders with corresponding folders from Range-V3-VS2015/include.

Now, run CMake as usual, but with specifying vcpkg tool-set (see [vcpkg] page for details):

```
mkdir build
cd build
cmake -G "Visual Studio 15 2017" -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..
```

[vcpkg]: https://github.com/Microsoft/vcpkg
[Quick Start]: https://github.com/Microsoft/vcpkg#quick-start
[Microsoft's fork for ranges]: https://github.com/Microsoft/Range-V3-VS2015

note
====
If your system does not have installed OpenCL you can disable it using the following flag:
-DCMAKE_DISABLE_FIND_PACKAGE_OpenCL:bool=TRUE

====
The convolution layer is not yet ready
