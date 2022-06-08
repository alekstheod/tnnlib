```
 _               _ _ _     
| |_ _ __  _ __ | (_) |__  
| __| '_ \| '_ \| | | '_ \ 
| |_| | | | | | | | | |_) |
 \__|_| |_|_| |_|_|_|_.__/ 
```

[![Build Status (Windows)](https://ci.appveyor.com/api/projects/status/gb8229o85xap8c9i?svg=true
)](https://ci.appveyor.com/project/grishavanika/tnnlib)
[![Build Status (Linux)](https://travis-ci.org/alekstheod/tnnlib.svg)](https://travis-ci.org/alekstheod/tnnlib)

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

The convolution layer is not yet ready
