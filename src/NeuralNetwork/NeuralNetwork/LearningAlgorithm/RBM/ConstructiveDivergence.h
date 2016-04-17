#ifndef CONSTRUCTIVE_DIVERGENCE_H
#define CONSTRUCTIVE_DIVERGENCE_H

#include <NeuralNetwork/Perceptron/Perceptron.h>
#include <NeuralNetwork/LearningAlgorithm/BackPropagation/BepAlgorithm.h>

namespace nn{
  
namespace rbm
{
  
template<
	 typename PerceptronType,
         template <class> class ErrorCalculator = SquaredError
        >
class ConstructiveDivergence {
private:
    static_assert(PerceptronType::CONST_LAYERS_NUMBER == 2, "Ivalid number of layers in a given perceptron.");
    using BepAlgo = bp::BepAlgorithm<typename PerceptronType::reverse::type, ErrorCalculator>;
    BepAlgo m_bepAlgo;
    using Var = PerceptronType::Var;

private:
    using Perceptron = BepAlgo::Perceptron;
    
public:
    using Prototype = std::tuple< std::array<Var, inputsNumber>, std::array<Var, outputsNumber> >;
    using Memento = typename Perceptron::Memento;

private:

public:
    /// @brief constructor will initialize the object with a learning rate and maximum error limit.
    /// @param varP the learning rate.
    /// @param maxError the limit for the error. Algorithm will stop when we reach the limit.
    ConstructiveDivergence ( Var learningRate, Var maxError):m_bepAlgo ( learningRate, maxError ){}

    /// @brief execution of the single learning step in this algorithm.
    /// @param prototype a prototype used for this step.
    /// @param momentum a callback which will calculate a new delta, used in order to introduce momentum.
    /// @return error on this step.
    template<typename MomentumFunc>
    Var executeTrainingStep ( Prototype& prototype,  MomentumFunc momentum) {
    }

    template<typename Iterator, typename ErrorFunc>
    PerceptronType calculate ( Iterator begin, 
			       Iterator end,
                               ErrorFunc func,
                               unsigned int maxNumberOfEpochs = std::numeric_limits< unsigned int >::max() ) {
        return calculate(begin, end, func, maxNumberOfEpochs, DummyMomentum() );
    }
    
    void setMemento( Memento memento ){
      m_bepAlgo.setMemento(memento);
    }

    /// @brief will calculate a perceptron with appropriate weights.
    /// @param begin iterator which points to the first input.
    /// @param end iterator which points to the last input.
    /// @param ReportFunc error report function (callback).
    /// @param MomentumFunc function which will calculate a momentum.
    /// @return a calculated perceptron.
    template<typename Iterator, typename ErrorFunc, typename MomentumFunc>
    PerceptronType calculate ( Iterator begin, 
			       Iterator end,
                               ErrorFunc func, 
                               unsigned int limit = std::numeric_limits< unsigned int >::max(),
                               MomentumFunc momentum = DummyMomentum() ) {
    }
};

}
}

#endif