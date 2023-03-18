#include "NeuralNetwork/NeuralLayer/AsyncNeuralLayer.h"

namespace nn {
    namespace detail {

        boost::asio::thread_pool& pool(std::size_t numberOfThreads) {
            static boost::asio::thread_pool threadPool{numberOfThreads};
            return threadPool;
        }

    } // namespace detail
} // namespace nn
