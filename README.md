![Build result](https://app.travis-ci.com/alekstheod/tnnlib.svg?branch=master)

```
 _               _ _ _
| |_ _ __  _ __ | (_) |__
| __| '_ \| '_ \| | | '_ \
| |_| | | | | | | | | |_) |
 \__|_| |_|_| |_|_|_|_.__/
```

template neural network library v3

Flexible compile time interface which allows to define a neural network with different types of layers, Neurons and Activation functions:

```cpp
    typedef nn::Perceptron<float,
                           nn::InputLayer<nn::Neuron, nn::SigmoidFunction, 2>,
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
        Algo::Prototype{{Input{1.f}, Input{0.f}}, {1.f}} ,
        Algo::Prototype{{Input{1.f}, Input{1.f}}, {0.f}} ,
        Algo::Prototype{{Input{0.f}, Input{0.f}}, {0.f}}
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

- [Key features](#Features)
- [How to build](#Build)
- [Example ocr](#OCR)
- [Work in progress](#WIP)

### Features
The main idea of tnnlib is to provide a simple and intuitive DSL (lego like) which can
be used to define a neural network (Perceptron) in a modern c++ envirnoment.

### Define simple perceptron

```cpp
    typedef nn::Perceptron<float,
                           nn::InputLayer<nn::Neuron, nn::SigmoidFunction, 2>,
                           nn::NeuralLayer<nn::Neuron, nn::TanhFunction, 20>,
                           nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 1>
                           > Perceptron;
```

### Activation functions

There is a set of activation functions implemented that can be used in perceptron

Some of them and the most frequently used are:

- SigmoidFunction
- TanhFunction
- SoftmaxFunction

As defined in the #Define simple perceptron we can pass an activation
function type as an argument to the NeuralLayer interface or to the
Neuron interface directly when constructing a complex NeuralLayer.

### NeuralLayer

A simplest NeuralLayer instarface. This interface accepts
a type of the neuron used in a layer the activation function
which suppose to be the same for a whole layer and the number
of neurons available in that layer.

```cpp
nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 2>
```

### ComplexNeuralLayer

With complex layer we can define a layer which consists of heterogenic neuron types. That means
that we even can construct a layer in which a neuron can be anyting which fulfills the interface
of the neuron. Consider having a full perceptron acting as a neuron or a NeuralLayer which acts as
an individual neuron.

```cpp
nn::ComplexNeuralInputLayer< 2U, float, nn::Neuron< nn::SigmoidFunction, float >, nn::Neuron< nn::SoftmaxFunction, float > > layer;
```

### OpenCLNeuralLayer

OpenCL neural layer is meant to speedup a back propagation algorithm by calculating the dot products of
neurons inputs in parallel through the GPU or CPU OpenCL layer. Keep in mind that it only works if your system
contains a proper OpenCL installation and all modules and include files are located in a correct directory.
See the local_repository definition in the projects WORKSPACE file for more details.

### Build

The tnnlib library is using bazels (bazelisk) as a its build system.
Please use starndard bazelisk (bazel) commands to build the library.

```bash
bazel build --config=asan //... --test_tag_filters=-openCL
```

OpenCL targets are marked with the tag openCL and you have to exclude
these targets from building if openCL is not available on your system.
`--config=asan` stays for address sanetizer configuration.

### OCR

There is a small subproject which implements a simple ocr application which can recognize
handwritten digits by using the tnnlib code. Running this project is as simple as executing
the following command in your terminal:

```bash
bazel run //ocr:ocr -- external/OcrSamples/samples
```

This command will start the learning procedure through the back error propagation
algorithm for the set of samples in the ocr/samples directory. The result
of the process (when converged) will be stored in the file.

```bash
bazel-bin/ocr/ocr.runfiles/__main__/perceptron.json
```

This file describes a perceptron with the calculated weights which can be used to recognize the digits.
Trying it out is as simple as going to the directory where the json file is stored and executing the following
command:

```bash
./ocr/ocr perceptron.json external/OcrSamples/samples/0_4.png
```

This command will calculate the probabilities of all the digits (0-9) in the image 2.png.

### WIP

- Pooling layer
- Build on windows
