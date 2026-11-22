![CI](https://github.com/alekstheod/tnnlib/actions/workflows/ci.yml/badge.svg?branch=master)

# tnnlib

`tnnlib` is a modern C++23, template-based neural network library focused on compile-time composition. It provides a type-safe API for building perceptrons from layers, neurons, activation functions, and training components with minimal runtime overhead.

## Core Features

- **Compile-time network composition**: Define perceptrons with template-based layers and neurons.
- **Fluent builder API**: Construct networks with `nn::build<VarType>()` using `input`, `dense`, `conv`, `pool`, `async`, and `with_neuron`.
- **Layer support**: Input, dense/neural, complex/heterogeneous, convolution, pooling, async, and OpenCL-accelerated layers.
- **Neuron and activation options**: Includes `Neuron`, `PoolingNeuron`, `RecurrentNeuron`, and activation functions such as Sigmoid, Tanh, Softmax, LogScaleSoftmax, BiopolarSigmoid, ReLU, Binary, and Threshold.
- **Backpropagation training**: `BepAlgorithm` supports configurable error functions, momentum, batch training, and evaluation.
- **Connection matrix support**: Sparse/custom connections can be defined for non-standard wiring, including recurrent or cross-layer connections.
- **Serialization**: Cereal-based serialization for saving and loading trained contexts/models.
- **Bazel/Bazelisk build**: Reproducible builds and tests with Bazel, including AddressSanitizer configuration.

## Quick Start

### Define a network with templates

```cpp
#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"

using Perceptron = nn::Perceptron<
    float,
    nn::InputLayer<nn::Neuron, nn::SigmoidFunction, 2>,
    nn::NeuralLayer<nn::Neuron, nn::TanhFunction, 20>,
    nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 1>
>;
```

### Define a network with the builder API

```cpp
#include "NeuralNetwork/Perceptron/PerceptronBuilder.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"

using Perceptron = decltype(
    nn::build<float>()
        .input<2>()
        .dense<20>().with_neuron<nn::Neuron<nn::TanhFunction>>()
        .dense<1>()
)::type;
```

### Train with backpropagation

```cpp
#include "NeuralNetwork/BackPropagation/BepAlgorithm.h"
#include "NeuralNetwork/BackPropagation/ErrorFunction.h"

using Algo = nn::bp::BepAlgorithm<Perceptron, nn::bp::CrossEntropyError>;
using Input = typename Perceptron::Input;

Algo algorithm(0.09f);
std::array<Algo::Prototype, 4> prototypes = {
    Algo::Prototype{{Input{0.f}, Input{1.f}}, {1.f}},
    Algo::Prototype{{Input{1.f}, Input{0.f}}, {1.f}},
    Algo::Prototype{{Input{1.f}, Input{1.f}}, {0.f}},
    Algo::Prototype{{Input{0.f}, Input{0.f}}, {0.f}}
};

algorithm.calculate(
    prototypes.begin(),
    prototypes.end(),
    [](unsigned int /*epoch*/, float error) { return error >= 0.01f; }
);
```

## Build and Test

The project uses [Bazelisk](https://github.com/bazelbuild/bazelisk) as the Bazel launcher.

```bash
# Run tests (ASan, as used in CI)
bazelisk --host_jvm_args=-Xmx3g --host_jvm_args=-Xms512m test --jobs=2 --config=asan //...
```

OpenCL targets are tagged with `openCL`. On systems without OpenCL, exclude them:

```bash
bazelisk test --config=asan //... --test_tag_filters=-openCL
```

## OCR Example

The repository includes a small OCR example that trains a CNN-style perceptron on handwritten digit samples.

Train a model:

```bash
bazel run //ocr:ocr -- external/OcrSamples/samples
```

This produces `perceptron.json` in the working directory. Run recognition against an image:

```bash
bazel run //ocr:ocr -- perceptron.json external/OcrSamples/samples/0_4.png
```

## Documentation

- [Detailed documentation](doc/documentation.md)

## Work in Progress

- Windows build support
- Expanded documentation and API examples
- Additional performance and OpenCL workflow improvements

## License

This project is licensed under the BSD 3-Clause License. See [license.txt](license.txt).
