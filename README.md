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
    
    Perceptron perceptron = algorithm.calculatePerceptron ( prototypes.begin(), prototypes.end(),
                                                            [] ( unsigned int epoch, float error, ) {
                                                                std::cout << error << std::endl;
                                                                return error >= 0.01.f
                                                            },
                                                            numOfEpochs);
```                                     
In order to generate a VS solution please use a CMake with the following options:

```bash
cmake -DBoost_USE_STATIC_LIBS=ON -DZLIB_INCLUDE_DIR="C:\Program Files (x86)\GnuWin32\include" -DPNG_LIBRARY_DEBUG="C:/Program Files (x86)/GnuWin32/lib/libpng.lib" -DPNG_LIBRARY_RELEASE="C:/Program Files (x86)/GnuWin32/lib/libpng.lib" -DPNG_PNG_INCLUDE_DIR="C:\Program Files (x86)\GnuWin32\include" -DCMAKE_MODULE_LINKER_FLAGS="/machine:X86 /LIBPATH:C:\local\boost_1_57_0\lib32-msvc-12.0/" -DBoost_SERIALIZATION_LIBRARY_DEBUG="C:\local\boost_1_57_0\lib32-msvc-12.0/libboost_serialization-vc120-mt-sgd-1_57.lib" -DBoost_SERIALIZATION_LIBRARY_RELEASE="C:\local\boost_1_57_0\lib32-msvc-12.0/libboost_serialization-vc120-mt-s-1_57.lib" -DBoost_SYSTEM_LIBRARY_DEBUG="C:\local\boost_1_57_0\lib32-msvc-12.0/libboost_system-vc120-mt-sgd-1_57.lib" -DBoost_SYSTEM_LIBRARY_RELEASE="C:\local\boost_1_57_0\lib32-msvc-12.0/libboost_system-vc120-mt-s-1_57.lib" -DBoost_FILESYSTEM_LIBRARY_RELEASE="C:\local\boost_1_57_0\lib32-msvc-12.0/libboost_filesystem-vc120-mt-s-1_57.lib" -DBoost_FILESYSTEM_LIBRARY_DEBUG="C:\local\boost_1_57_0\lib32-msvc-12.0\libboost_filesystem-vc120-mt-sgd-1_57.lib" -DCMAKE_SHARED_LINKER_FLAGS="/machine:X86 /LIBPATH:C:\local\boost_1_57_0\lib32-msvc-12.0" -DCMAKE_EXE_LINKER_FLAGS="/machine:X86 /LIBPATH:C:\local\boost_1_57_0\lib32-msvc-12.0" -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /W3 /GR /EHsc" -DCMAKE_CXX_FLAGS_DEBUG="/D_DEBUG /MTd /Zi /Ob0 /Od /RTC1" .
```

You have to define your boost libraries, libpng and zlib. In this case we have boost_1_57 and libpng + zlib as a binary distributions.

note
====
If your system does not have installed OpenCL you can disable it using the following flag:
-DCMAKE_DISABLE_FIND_PACKAGE_OpenCL:bool=TRUE
