#pragma once

#include <MPL/TypeTraits.h>

namespace nn::bp {

    template< typename Internal >
    class BPNeuron;

    // TODO please throw this out
    // This code fixes the blowing object when using
    // rebind functionality
    namespace detail {
        template< typename Internal >
        struct unwrapNeuron {
            using type = Internal;
        };

        template< typename Internal >
        struct unwrapNeuron< BPNeuron< Internal > > {
            using type = typename unwrapNeuron< Internal >::type;
        };
    } // namespace detail

    /*
     * Represent the back error propagation Neuron trainer.
     * This class holds a pointer to neuron which should
     * be trained with back error propagation algorithm.
     */
    template< class NeuronType >
    class BPNeuron : public detail::unwrapNeuron< NeuronType >::type {
      public:
        using Internal = typename detail::unwrapNeuron< NeuronType >::type;
        using Neuron = Internal;
        using Var = typename Neuron::Var;
        using Memento = typename Neuron::Memento;
        using OutputFunction = typename Neuron::OutputFunction;
        using Input = typename Neuron::Input;

        template< typename EquationType >
        using use = BPNeuron< typename Internal::template use< EquationType > >;

        template< unsigned int inputs >
        using resize = BPNeuron< typename Internal::template resize< inputs > >;
        using Internal::setMemento;

      public:
        /**
         * @brief Will return the errors delta for the trained neuron.
         * @returns the error deltas value.
         */
        const Var& getDelta(void) const {
            return m_delta;
        }

        void setDelta(const Var& delta) {
            m_delta = delta;
        }

        /*!
         *  Will calculate the differential value.
         *  @return the calculated value.
         */
        template< typename MomentumFunc >
        const Var& calculateDelta(const Var& expectedOutput, MomentumFunc momentum) {
            m_delta =
             momentum(m_delta, m_outputFunction.delta(Neuron::getOutput(), expectedOutput));
            return m_delta;
        }

        const Var calculateDerivate() const {
            return m_outputFunction.derivate(Neuron::getOutput());
        }

      private:
        /**
         * Neurons error delta
         */
        Var m_delta{};

        /**
         * Equation needed in order to calculate the differential value;
         */
        OutputFunction m_outputFunction{};
    };
} // namespace nn::bp
