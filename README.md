tnnlib
======

template neural network library v3

Flexible interface which allows to define a neural network with a different types of layers, Neurons and Activation functions:

                        
    typedef nn::Perceptron<float, 
                           nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 2>, 
                           nn::NeuralLayer<nn::Neuron, nn::TanhFunction, 20>, 
                           nn::NeuralLayer<nn::Neuron, nn::SigmoidFunction, 1>
                           > Perceptron;
                           
    typedef nn::bp::BepAlgorithm< Perceptron, nn::bp::CrossEntropyError> Algo;





Calculating perceptrong by using BEP algorithm:

    typedef BepAlgorithm< Perceptron > Algo;
    Algo algorithm (0.09f, 0.01f );

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
                                                            [] ( float error ) {
                                                                std::cout << error << std::endl;
                                                            },
                                                            numOfEpochs);
