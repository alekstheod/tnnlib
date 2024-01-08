#include "NeuralNetwork/NeuralLayer/Thread/AsyncNeuralLayer.h"

namespace nn::detail {

    boost::asio::thread_pool& pool(std::size_t numberOfThreads) {
        static boost::asio::thread_pool threadPool{numberOfThreads};
        return threadPool;
    }

} // namespace nn::detail
