#ifndef RBM_H
#define RBM_H

namespace nn{

namespace detail{
  
/// @brief implementation of Restricted boltzmann machine.
/// @arg VarType the type of the internal variable.
/// @arg InputLayerType the type of the input layer.
/// @arg OutputLayerType the type of the output - hidden layer.
template<typename VarType, 
	 typename InputLayerType,
	 typename OutputLayerType>
class RBM{
private:
  using InputTmp = typename InputLayerType::template rebindVar<VarType>::type;
  using OutputTmp = typename OutputLayerType::template rebindVar<VarType>::type;
  using InputLayer = INeuralLayer<typename InputTmp::template rebindInputs<OutputTmp::CONST_NEURONS_NUMBER>::type>;
  using OutputLayer = INeuralLayer<typename OutputTmp::template rebindInputs<InputTmp::CONST_NEURONS_NUMBER>::type>;

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
        while( begin != end ) {
            m_output.setInput(inputId, *begin);
            begin++;
            inputId++;
        }

        VarType biasInput = boost::numeric_cast<VarType>(1.f);
        m_output.setInput(m_input.size()-1, biasInput);
	m_input.setInput(m_output.size()-1, biasInput);
        m_output.calculate(m_input);
	m_input.calculate(m_output);
	m_output.calculate();
	std::transform( m_output.begin(),
			m_output.end(),
			out,
			std::bind(&OutputLayer::Neuron::getOutput, std::placeholders::_1));
  }
  
  ~RBM(){}
};

}

template<typename VarType, 
	 typename InputLayerType, 
	 typename OutputLayerType,
	 typename It>
detail::RBM<VarType, InputLayerType, OutputLayerType> calculateRBM(It begin, It end){
  typedef detail::RBM<VarType, InputLayerType, OutputLayerType>  RBM;
  
}

}


#endif