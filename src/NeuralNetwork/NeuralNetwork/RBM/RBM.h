#ifndef RBM_H
#define RBM_H
#include <algorithm>
#include <boost/numeric/conversion/cast.hpp>

namespace nn{

namespace detail{
  
/// Under construction.  
/// @brief implementation of Restricted boltzmann machine.
/// @arg VarType the type of the internal variable.
/// @arg InputLayerType the type of the input layer.
/// @arg OutputLayerType the type of the output - hidden layer.
template<typename VarType, 
	 typename InputLayerType,
	 typename OutputLayerType>
class RBM{
private:
  using Var = VarType;
  using InputTmp = typename InputLayerType::template rebindVar<VarType>::type;
  using OutputTmp = typename OutputLayerType::template rebindVar<VarType>::type;
  using InputLayer = INeuralLayer<typename InputTmp::template rebindInputs<OutputTmp::CONST_NEURONS_NUMBER>::type>;
  using OutputLayer = INeuralLayer<typename OutputTmp::template rebindInputs<InputTmp::CONST_NEURONS_NUMBER>::type>;

  static const std::size_t INPUTS_NUMBER = InputLayer::CONST_NEURONS_NUMBER; 
  InputLayer m_input;
  OutputLayer m_output;
  
public:
  RBM(){}
  
  /// @brief will calculate the outputs by the given input.
  /// @param [in]begin iterator to a first element of the input.
  /// @param [in]end iterator of the end of input elements.
  /// @param [out]out iterator of the output where the result will be stored.
  template<typename It,typename  OutIt>
  void calculate(It begin, It end, OutIt out){
        unsigned int inputId = 0;
        std::for_each(begin, end, [&](const Var& val){m_output.setInput(inputId++, val);});

        VarType biasInput = boost::numeric_cast<VarType>(1.f);
	m_input.setInput(m_output.size()-1, biasInput);
	m_input.calculateOutputs(m_output);
	m_output.calculateOutputs();
	std::transform( m_output.begin(),
			m_output.end(),
			out,
			std::bind(&OutputLayer::Neuron::getOutput, std::placeholders::_1));
  }
  
  ~RBM(){}
};

}


}


#endif