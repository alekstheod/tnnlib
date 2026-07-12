#pragma once

#include <cstddef>
#include <cstdint>

namespace nn::gpu {

enum class ActType : uint8_t { Identity, Sigmoid, Relu, Tanh, Softmax };

struct LayerCfg {
    uint32_t out_sz;
    uint32_t in_sz;
    uint32_t tot_w;
    uint32_t w_off;
    uint32_t b_off;
    uint32_t act_off;
    uint32_t del_off;
    uint16_t gw, gh;
    uint8_t stride;
    uint8_t is_conv;
    ActType act;
};

using ErrorFn = bool (*)(unsigned int epoch, float error);

void launch_train_kernel(
    float* dw, float* db, const float* dp,
    uint32_t count, float lr, const LayerCfg* dc,
    uint32_t nl, uint32_t in_sz, float* de,
    uint32_t block_threads, uint32_t shared_bytes);

template<typename BepAlgo>
void calculate(BepAlgo& algo,
               const typename BepAlgo::Prototype* prototypes,
               std::size_t count,
               ErrorFn errorFn);

}
