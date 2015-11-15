#ifndef RBM_H
#define RBM_H

namespace nn{

template<typename VarType, 
	 typename InputLayerType,
	 typename OutputLayerType>
class RBM{
private:
  using InputTmp = typename InputLayerType::template rebindVar<VarType>::type;
  using OutputTmp = typename OutputLayerType::template rebindVar<VarType>::type;
  using InputLayer = INeuralLayer<typename InputTmp::template rebindInputs<OutputTmp::CONST_NEURONS_NUMBER + 1>::type>;
  using OutputLayer = INeuralLayer<typename InputTmp::template rebindInputs<InputTmp::CONST_NEURONS_NUMBER + 1>::type>;

  InputLayer m_input;
  OutputLayer m_output;
  
public:
  RBM(){}
  
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


#endif