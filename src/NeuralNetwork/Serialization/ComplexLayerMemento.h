#ifndef COMPLEXLAYER_MEMENTO_H
#define COMPLEXLAYER_MEMENTO_H
#include <NeuralNetwork/Serialization/PerceptronMemento.h>

namespace nn {

    template< typename Var >
    class ComplexLayerMemento {
      private:
        PerceptronMemento< Var > m_perceptron;

        friend class boost::serialization::access;

        template< class Archive >
        void serialize(Archive& ar, const unsigned int version) {
            ar& BOOST_SERIALIZATION_NVP(m_perceptron);
        }

      public:
        const PerceptronMemento< Var >& getPerceptron() const {
            return m_perceptron;
        }

        void setPerceptron(const PerceptronMemento< Var >& perceptron) {
            m_perceptron = m_perceptron;
        }

        ~ComplexLayerMemento() {
        }
    };
} // namespace nn

#endif
