#ifndef INEURALLAYER_H
#define INEURALLAYER_H

namespace nn {

template< typename NeuralLayerType>
class INeuralLayer {
public:
    typedef NeuralLayerType NeuralLayer;
    typedef typename NeuralLayerType::Memento Memento;
    typedef typename NeuralLayer::Var Var;
    typedef typename NeuralLayer::Neuron Neuron;
    typedef typename NeuralLayer::const_iterator const_iterator;
    typedef typename NeuralLayer::iterator iterator;
    typedef typename NeuralLayer::reverse_iterator reverse_iterator;
    typedef typename NeuralLayer::const_reverse_iterator const_reverse_iterator;
  
private:
    NeuralLayer m_neuralLayer;

public:
    template<typename... Args>
    INeuralLayer ( Args... args ) :m_neuralLayer ( args... ) {}

    INeuralLayer ( const NeuralLayer& neuralLayer ) : m_neuralLayer ( neuralLayer ) {
    }

    NeuralLayer& operator * () {
        return m_neuralLayer;
    }

    INeuralLayer ( NeuralLayer neuralLayer ) :m_neuralLayer ( neuralLayer ) {}

    const_iterator find ( unsigned int neuronId ) const {
        return m_neuralLayer.find ( neuronId );
    }

    const_iterator begin() const {
        return m_neuralLayer.begin();
    }

    const_iterator end() const {
        return m_neuralLayer.end();
    }

    iterator begin()  {
        return m_neuralLayer.begin();
    }

    iterator end()  {
        return m_neuralLayer.end();
    }

    reverse_iterator rbegin(){
      return m_neuralLayer.rbegin();
    }
    
    reverse_iterator rend(){
      return m_neuralLayer.rend();
    }
   
    const_reverse_iterator rbegin()const{
      return m_neuralLayer.rbegin();
    }
    
    const_reverse_iterator rend()const{
      return m_neuralLayer.rend();
    }
   
    unsigned int size()const {
        return m_neuralLayer.size();
    }

    /**
     * @brief Will set the input value by using the given input id. If the given input is is invalid value will not be set.
     * @param inputId the id of the input.
     * @param value the value which has to be set.
     */
    void setInput ( unsigned int inputId, const Var& value ) {
        m_neuralLayer.setInput ( inputId, value );
    }

    Var getOutput ( unsigned int outputId ) const {
        return m_neuralLayer.getOutput ( outputId );
    }

    NeuralLayer* operator -> () {
        return &m_neuralLayer;
    }

    /**
    * @see {INeuralLayer}
    */
    const Neuron& operator [] ( unsigned int id ) const {
        return m_neuralLayer[id];
    }

    operator NeuralLayerType& () {
        return &m_neuralLayer;
    }
    
    const Var& getBias( unsigned int neuronId )const{
      return m_neuralLayer.getBias(neuronId);
    }

    const Var& getInputWeight ( unsigned int neuronId, unsigned int weightId ) const {
        return m_neuralLayer.getInputWeight( neuronId, weightId );
    }

    const Memento getMemento() const {
        return m_neuralLayer.getMemento();
    }

    /**
     * @brief will set memento to the current layer.
     * @param memento instance of the POD structure which describes the state.
     * @return true if succeed, false otherwise.
     */
    void setMemento ( const Memento& memento ) {
        m_neuralLayer.setMemento(memento);
    }

    /**
     *
     */
    template<typename Layer>
    void calculateOutputs ( Layer& nextLayer ) {
        m_neuralLayer.calculateOutputs( *nextLayer );
    }

    void calculateOutputs () {
        m_neuralLayer.calculateOutputs ();
    }

    ~INeuralLayer() {}
};

template<typename NeuralLayer, typename Var>
NeuralLayer createLayer( const NeuralLayerMemento<Var>& memento )
{
    NeuralLayer layer( memento.getInputsNumber(), memento.getNeuronsNumber() ) ;
    layer.setMemento( memento );
    return layer;
}


}

#endif // INEURALLAYER_H
