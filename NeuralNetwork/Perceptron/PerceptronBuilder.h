#pragma once

#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/NeuralLayer/ConvolutionLayer.h"
#include "NeuralNetwork/NeuralLayer/InputLayer.h"
#include "NeuralNetwork/NeuralLayer/NeuralLayer.h"
#include "NeuralNetwork/NeuralLayer/PoolingLayer.h"
#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"
#include "NeuralNetwork/Neuron/Neuron.h"
#include "NeuralNetwork/Neuron/PoolingNeuron.h"
#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include <ConnectionData.h>

#include "Utilities/MPL/Tuple.h"

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <tuple>

namespace nn {

    template< std::size_t W, std::size_t H, typename K >
    struct SlidingWindowConfig {
        static constexpr std::size_t width = W;
        static constexpr std::size_t height = H;
        using Kernel = K;
    };

    template< typename VarType, typename CurrentLayer, typename... PrevLayers >
    struct PerceptronBuilder;

    template< typename VarType, typename InnerBuilder >
    struct ConnectionMatrixBuilder;

    template< typename VarType, typename ConvConfig, typename CurrentLayer, typename... PrevLayers >
    struct ConvBuilder {
        template< std::size_t W, std::size_t H, std::size_t S >
        constexpr auto with_kernel() const {
            using NewConfig =
             SlidingWindowConfig< ConvConfig::width, ConvConfig::height, nn::Kernel< W, H, S > >;
            return ConvBuilder< VarType, NewConfig, CurrentLayer, PrevLayers... >{};
        }

        template< std::size_t W, std::size_t H >
        constexpr auto with_grid() const {
            using NewConfig = SlidingWindowConfig< W, H, typename ConvConfig::Kernel >;
            return ConvBuilder< VarType, NewConfig, CurrentLayer, PrevLayers... >{};
        }

        constexpr auto build() const {
            using Grid =
             nn::SlidingWindow< ConvConfig::width, ConvConfig::height, typename ConvConfig::Kernel >;
            using L =
             ConvolutionLayer< nn::NeuralLayer, Neuron, SigmoidFunction, Grid, VarType >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }
    };

    template< typename VarType, typename CurrentLayer, typename... PrevLayers >
    struct PerceptronBuilder {

        template< std::size_t size >
        constexpr auto dense() const {
            using L = nn::NeuralLayer< Neuron, SigmoidFunction, size >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        template< typename... Neurons >
        constexpr auto dense_complex() const {
            using L = nn::ComplexNeuralLayer< Neurons... >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        template< typename SlidingWindow >
        constexpr auto conv() const {
            using L =
             ConvolutionLayer< nn::NeuralLayer, Neuron, SigmoidFunction, SlidingWindow, VarType >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        constexpr auto conv() const {
            using DefaultConfig = SlidingWindowConfig< 8, 8, nn::Kernel< 3, 3, 1 > >;
            return ConvBuilder< VarType, DefaultConfig, CurrentLayer, PrevLayers... >{};
        }

        template< template< class > class PoolingAlgo, typename SlidingWindow >
        constexpr auto pool() const {
            using L = PoolingLayer< nn::NeuralLayer, PoolingAlgo, SlidingWindow, VarType >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        template< std::size_t size >
        constexpr auto async() const {
            using L = nn::AsyncNeuralLayer< Neuron, SigmoidFunction, size >;
            return PerceptronBuilder< VarType, L, PrevLayers..., CurrentLayer >{};
        }

        template< typename N >
        constexpr auto with_neuron() const {
            using L = typename CurrentLayer::template with_neuron< N >;
            return PerceptronBuilder< VarType, L, PrevLayers... >{};
        }

        /// Wires consecutive input slots of neuron `NeuronId` of the current
        /// layer to the given sources, e.g.
        /// `.with_connection_matrix<0>(connect(1, 3), connect(0, 12))` makes
        /// input 0 read layer 1 neuron 3 and input 1 read layer 0 neuron 12.
        template< std::size_t NeuronId, typename... Srcs >
        auto with_connection_matrix(Srcs... srcs) const {
            bp::ConnectionData connData;
            connData.template add< NeuronId >(sizeof...(PrevLayers), srcs...);
            return ConnectionMatrixBuilder< VarType, PerceptronBuilder >{*this, std::move(connData)};
        }

        /// Overrides only input `Slot` of neuron `NeuronId`; the remaining
        /// slots keep reading their regular predecessor inputs.
        template< std::size_t NeuronId, std::size_t Slot >
        auto with_connection_matrix(Connection src) const {
            bp::ConnectionData connData;
            connData.template set< NeuronId, Slot >(sizeof...(PrevLayers), src);
            return ConnectionMatrixBuilder< VarType, PerceptronBuilder >{*this, std::move(connData)};
        }

        static constexpr std::size_t size() {
            return nn::Perceptron< VarType, PrevLayers..., CurrentLayer >::size();
        }

        using type = nn::Perceptron< VarType, PrevLayers..., CurrentLayer >;
    };

    template< typename VarType, typename InnerBuilder >
    struct ConnectionMatrixBuilder {
        InnerBuilder m_inner;
        bp::ConnectionData m_connections;

        ConnectionMatrixBuilder() = default;
        ConnectionMatrixBuilder(const InnerBuilder& inner, bp::ConnectionData&& conn)
         : m_inner(inner), m_connections(std::move(conn)) {}

        /// See PerceptronBuilder::with_connection_matrix. The target layer is
        /// the one most recently added through the wrapped builder.
        template< std::size_t NeuronId, typename... Srcs >
        auto with_connection_matrix(Srcs... srcs) const {
            auto newConn = m_connections;
            newConn.template add< NeuronId >(numPrevLayers(), srcs...);
            return ConnectionMatrixBuilder< VarType, InnerBuilder >{m_inner, std::move(newConn)};
        }

        /// Overrides only input `Slot` of neuron `NeuronId`; the remaining
        /// slots keep reading their regular predecessor inputs.
        template< std::size_t NeuronId, std::size_t Slot >
        auto with_connection_matrix(Connection src) const {
            auto newConn = m_connections;
            newConn.template set< NeuronId, Slot >(numPrevLayers(), src);
            return ConnectionMatrixBuilder< VarType, InnerBuilder >{m_inner, std::move(newConn)};
        }

        template< std::size_t sz >
        auto dense() const {
            return ConnectionMatrixBuilder< VarType, decltype(m_inner.template dense< sz >()) >{
                m_inner.template dense< sz >(), bp::ConnectionData(m_connections)};
        }

        template< typename SlidingWindow >
        auto conv() const {
            return ConnectionMatrixBuilder< VarType, decltype(m_inner.template conv< SlidingWindow >()) >{
                m_inner.template conv< SlidingWindow >(), bp::ConnectionData(m_connections)};
        }

        auto conv() const {
            return ConnectionMatrixBuilder< VarType, decltype(m_inner.conv()) >{
                m_inner.conv(), bp::ConnectionData(m_connections)};
        }

        template< template< class > class PoolingAlgo, typename SlidingWindow >
        auto pool() const {
            return ConnectionMatrixBuilder< VarType, decltype(m_inner.template pool< PoolingAlgo, SlidingWindow >()) >{
                m_inner.template pool< PoolingAlgo, SlidingWindow >(), bp::ConnectionData(m_connections)};
        }

        template< std::size_t sz >
        auto async() const {
            return ConnectionMatrixBuilder< VarType, decltype(m_inner.template async< sz >()) >{
                m_inner.template async< sz >(), bp::ConnectionData(m_connections)};
        }

        template< typename N >
        auto with_neuron() const {
            return ConnectionMatrixBuilder< VarType, decltype(m_inner.template with_neuron< N >()) >{
                m_inner.template with_neuron< N >(), bp::ConnectionData(m_connections)};
        }

        static constexpr std::size_t size() {
            return InnerBuilder::size();
        }

        using type = typename InnerBuilder::type;

        const bp::ConnectionData& connections() const {
            return m_connections;
        }

      private:
        static constexpr std::size_t numPrevLayers() {
            return InnerBuilder::type::size() - 1;
        }
    };

    template< typename VarType >
    struct PerceptronBuilderBegin {
        template< std::size_t size >
        constexpr auto input() const {
            using L = InputLayer< Neuron, SigmoidFunction, size >;
            return PerceptronBuilder< VarType, L >{};
        }
    };

    template< typename VarType = float >
    constexpr auto build() {
        return PerceptronBuilderBegin< VarType >{};
    }

} // namespace nn
