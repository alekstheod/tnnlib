/**
*  Copyright (c) 2011, Alex Theodoridis
*  All rights reserved.

*  Redistribution and use in source and binary forms, with
*  or without modification, are permitted provided that the
*  following conditions are met:
*  Redistributions of source code must retain the above
*  copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above
*  copyright notice, this list of conditions and the following
*  disclaimer in the documentation and/or other materials
*  provided with the distribution.

*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
*  AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
*  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
*  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
*  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
*  OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
*  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
*  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE,
*  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BepAlgorithm.h>
#include <NeuralNetwork/NeuralLayer/ConvolutionLayer.h>
#include <NeuralNetwork/NeuralLayer/NeuralLayer.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SigmoidFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/SoftmaxFunction.h>
#include <NeuralNetwork/Neuron/ActivationFunction/TanhFunction.h>
#include <NeuralNetwork/Neuron/Neuron.h>
#include <NeuralNetwork/Perceptron/ComplexLayer.h>
#include <NeuralNetwork/Perceptron/Perceptron.h>
#include <NeuralNetwork/SOM/K2DNeighbourhood.h>
#include <NeuralNetwork/SOM/K2DPosition.h>
#include <NeuralNetwork/SOM/KNode.h>
#include <NeuralNetwork/SOM/KohonenMap.h>

#include <Utilities/Design/Factory.h>
#include <Utilities/Design/Singleton.h>
#include <Utilities/StrUtil/StrUtil.h>
#include <Utilities/StrUtil/StrUtil.h>
#include <Utilities/System/Time.h>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <array>
#include <iostream>
#include <sstream>
#include <time.h>

using namespace nn;
using namespace bp;
using namespace std;
using namespace utils;

/*
 *
 */
int main(int argc, char** argv) {
    using ConvLayer =
     ConvolutionLayer< nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 2, 2 >,
                       Connection< InputRange< 0, 1 >, 0 >,
                       Connection< InputRange< 0, 1 >, 1 > >;

    using Perceptron = nn::Perceptron<
     float,
     ConvLayer,
     // nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 2, 2 >,
     nn::NeuralLayer< nn::Neuron, nn::TanhFunction, 20 >,
     nn::NeuralLayer< nn::Neuron, nn::SigmoidFunction, 1 > >;

    typedef BepAlgorithm< Perceptron > Algo;
    Algo algorithm(0.09f);
    std::array< Algo::Prototype, 4 > prototypes = {Algo::Prototype{{0.f, 1.f}, {1.f}},
                                                   Algo::Prototype{{1.f, 0.f}, {1.f}},
                                                   Algo::Prototype{{1.f, 1.f}, {0.f}},
                                                   Algo::Prototype{{0.f, 0.f}, {0.f}}};

    unsigned int numOfEpochs =
     argc < 2 ? std::numeric_limits< unsigned int >::max() :
                numOfEpochs = utils::lexical_cast< unsigned int >(argv[1]);

    Perceptron perceptron =
     algorithm.calculate(prototypes.begin(),
                         prototypes.end(),
                         [numOfEpochs](unsigned int epoch, float error) {
                             std::cout << "Epoch: " << epoch
                                       << " error: " << error << std::endl;
                             return error > 0.01f && epoch < numOfEpochs;
                         });

    using Memento = Perceptron::Memento;
    Memento memento = perceptron.getMemento();
    Perceptron perceptron2;
    {
        std::stringstream strm;
        {
            // Archive has to be destroyed before using the stream
            boost::archive::xml_oarchive oa(strm);
            // write class instance to archive
            oa << BOOST_SERIALIZATION_NVP(memento);
        }

        Memento memento2;
        boost::archive::xml_iarchive ia(strm);
        ia >> BOOST_SERIALIZATION_NVP(memento2);
        perceptron2.setMemento(memento);
    }

    std::array< float, 2 > outputs{0};
    std::array< float, 2 > input1{0, 0};
    perceptron2.calculate(input1.begin(), input1.end(), outputs.begin());
    std::cout << "0 0 " << outputs[0] << std::endl;

    std::array< float, 2 > input2{1, 0};
    perceptron2.calculate(input2.begin(), input2.end(), outputs.begin());
    std::cout << "1 0 " << outputs[0] << std::endl;

    std::array< float, 2 > input3{1, 1};
    perceptron2.calculate(input3.begin(), input3.end(), outputs.begin());
    std::cout << "1 1 " << outputs[0] << std::endl;

    std::array< float, 2 > input4{0, 1};
    perceptron2.calculate(input4.begin(), input4.end(), outputs.begin());
    std::cout << "0 1 " << outputs[0] << std::endl;

    /// Kohonen map implementation
    typedef kohonen::K2DPosition< float, 5 > Position;
    typedef kohonen::K2DNeighbourhood< kohonen::KNode< Position > > Neighbourhood;
    typedef nn::kohonen::KohonenMap< Neighbourhood, 25, 3 > KohonenMap;

    KohonenMap kohMap;

    typedef KohonenMap::InputType InputType;
    std::vector< InputType > inputsData;
    inputsData.push_back({{0.f, 255.f, 0.f}});
    inputsData.push_back({{255.f, 0.f, 0.f}});
    inputsData.push_back({{0.f, 0.f, 255.f}});

    kohMap.calculateWeights(inputsData.begin(), inputsData.end(), 10000, 0.4f, 8.0f);
    return 0;
}
