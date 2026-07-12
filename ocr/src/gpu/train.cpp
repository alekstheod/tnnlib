#include "train.h"

#include "NeuralNetwork/BackPropagation/BepAlgorithm.h"
#include "NeuralNetwork/BackPropagation/ErrorFunction.h"
#include "NeuralNetwork/Perceptron/Perceptron.h"
#include "NeuralNetwork/Perceptron/PerceptronBuilder.h"

#include "NeuralNetwork/ActivationFunction/SigmoidFunction.h"
#include "NeuralNetwork/ActivationFunction/SoftmaxFunction.h"
#include "NeuralNetwork/ActivationFunction/ReluFunction.h"
#include "NeuralNetwork/ActivationFunction/TanhFunction.h"

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>
#include <cstdint>
#include <type_traits>
#include <chrono>

#ifdef TNNLIB_HIP
#include <hip/hip_runtime.h>
#endif

namespace nn::gpu {

namespace {

template<typename BepAlgo>
uint32_t fill(LayerCfg* cfgs) {
    using V = typename BepAlgo::Var;
    using C = typename BepAlgo::BPCtx;
    constexpr auto N = BepAlgo::size();

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        ((cfgs[Is] = [&] {
            using FT = std::tuple_element_t<Is, typename C::Forward>;
            using GT = std::tuple_element_t<Is, typename C::Gradients>;
            constexpr auto os = std::tuple_size_v<FT>;
            constexpr auto ws = std::tuple_size_v<GT>;

            uint32_t is = 0; uint16_t gw = 0, gh = 0;
            uint8_t st = 0; bool cv = false;
            ActType at = ActType::Identity;

            if constexpr (Is > 0) {
                using L = typename BepAlgo::Layers;
                using LT = std::tuple_element_t<Is, L>;
                using AF = typename LT::ActivationFunctions;
                using AF0 = std::tuple_element_t<0, AF>;

                if constexpr (std::is_same_v<AF0, nn::SigmoidFunction<V>>)       at = ActType::Sigmoid;
                else if constexpr (std::is_same_v<AF0, nn::SoftmaxFunction<V>>)   at = ActType::Softmax;
                else if constexpr (std::is_same_v<AF0, nn::ReluFunction<V>>)      at = ActType::Relu;
                else if constexpr (std::is_same_v<AF0, nn::TanhFunction<V>>)      at = ActType::Tanh;

                using PF = std::tuple_element_t<Is - 1, typename C::Forward>;
                constexpr auto ps = std::tuple_size_v<PF>;

                if constexpr (ws % os == 0 && ws / os < ps) {
                    is = ws / os; cv = true;
                    auto k = static_cast<uint32_t>(sqrtf(float(is)));
                    for (uint32_t d = 1; d * d <= ps; ++d) if (ps % d == 0) gw = uint16_t(d);
                    if (gw == 0) gw = uint16_t(sqrtf(float(ps)));
                    gh = uint16_t(ps / gw);
                    st = k > 0 ? uint8_t((gw + k - 1) / (os > 0 ? os : 1)) : 1;
                } else is = ps;
            }

            return LayerCfg{uint32_t(os), is, uint32_t(ws), 0, 0, 0, 0, gw, gh, st, uint8_t(cv ? 1u : 0u), at};
        }()), ...);
    }(std::make_index_sequence<N>{});

    uint32_t wo = 0, bo = 0, ao = 0, dlo = 0;
    for (uint32_t i = 0; i < N; ++i) {
        cfgs[i].w_off = wo; cfgs[i].b_off = bo;
        cfgs[i].act_off = ao; cfgs[i].del_off = dlo;
        wo += cfgs[i].tot_w; bo += cfgs[i].out_sz;
        ao += cfgs[i].out_sz; dlo += cfgs[i].out_sz;
    }
    return N;
}

} // namespace

#ifdef TNNLIB_HIP

template<typename BepAlgo>
void calculate(BepAlgo& algo,
               const typename BepAlgo::Prototype* prototypes,
               std::size_t count,
               ErrorFn errorFn)
{
    if (count == 0) return;

    using V = typename BepAlgo::Var;
    auto& ctx = algo.context();
    constexpr auto N = BepAlgo::size();
    constexpr auto IN = BepAlgo::inputsNumber;
    constexpr auto OUT = BepAlgo::outputsNumber;

    LayerCfg cfgs[N];
    auto nl = fill<BepAlgo>(cfgs);

    uint32_t tw = 0, tb = 0, ta = 0;
    for (uint32_t i = 0; i < nl; ++i) {
        tw += cfgs[i].tot_w; tb += cfgs[i].out_sz;
        ta += cfgs[i].out_sz;
    }

    std::vector<V> hw(tw), hb(tb);
    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (((void)[&] {
            auto& w = std::get<Is>(ctx.weights);
            auto& b = std::get<Is>(ctx.biases);
            std::copy(w.begin(), w.end(), hw.begin() + cfgs[Is].w_off);
            std::copy(b.begin(), b.end(), hb.begin() + cfgs[Is].b_off);
        }()), ...);
    }(std::make_index_sequence<N>{});

    uint32_t ps = IN + OUT;
    std::vector<V> hp(count * ps);
    for (size_t i = 0; i < count; ++i) {
        auto& [inp, outp] = prototypes[i];
        for (size_t j = 0; j < IN; ++j) hp[i * ps + j] = inp[j].value[0];
        std::copy(outp.begin(), outp.end(), hp.data() + i * ps + IN);
    }

    V *dw = nullptr, *db = nullptr, *dp = nullptr, *de = nullptr;
    LayerCfg* dc = nullptr;

    auto ok = [](auto e) { if (e != hipSuccess) { std::cerr << "HIP err " << e << std::endl; std::abort(); }};

    ok(hipMalloc(&dw, tw * sizeof(V)));
    ok(hipMalloc(&db, tb * sizeof(V)));
    ok(hipMalloc(&dp, hp.size() * sizeof(V)));
    ok(hipMalloc(&de, sizeof(V)));
    ok(hipMalloc(&dc, nl * sizeof(LayerCfg)));

    ok(hipMemcpy(dw, hw.data(), tw * sizeof(V), hipMemcpyHostToDevice));
    ok(hipMemcpy(db, hb.data(), tb * sizeof(V), hipMemcpyHostToDevice));
    ok(hipMemcpy(dp, hp.data(), hp.size() * sizeof(V), hipMemcpyHostToDevice));
    ok(hipMemcpy(dc, cfgs, nl * sizeof(LayerCfg), hipMemcpyHostToDevice));

    uint32_t max_out = 0;
    for (uint32_t i = 0; i < nl; ++i)
        if (cfgs[i].out_sz > max_out) max_out = cfgs[i].out_sz;

    uint32_t block_threads = max_out > 256 ? 256 : max_out;
    uint32_t shared_bytes = ta * 2 * sizeof(V);

    V lr = algo.learningRate();
    // Scale learning rate for batch gradient: online_lr * batch_size
    // Online uses per-sample lr=0.01. For batch of 176, scale to 1.76.
    lr = 1.76f;

    // Debug: snapshot initial conv and output weights
    uint32_t conv_off = cfgs[1].w_off;
    uint32_t out_off = cfgs[nl-1].w_off;
    std::vector<V> initial_w(tw);
    ok(hipMemcpy(initial_w.data(), dw, tw * sizeof(V), hipMemcpyDeviceToHost));
    std::cout << "DEBUG conv weights[0..9] at offset " << conv_off << ":";
    for (int i = 0; i < 10; ++i) std::cout << " " << initial_w[conv_off + i];
    std::cout << std::endl;
    std::cout << "DEBUG out weights[0..9] at offset " << out_off << ":";
    for (int i = 0; i < 10; ++i) std::cout << " " << initial_w[out_off + i];
    std::cout << std::endl;

    uint32_t ep = 0;
    V err{0};

    auto t0 = std::chrono::steady_clock::now();

    do {
        ok(hipMemset(de, 0, sizeof(V)));
        launch_train_kernel(dw, db, dp, uint32_t(count),
            lr, dc, nl, uint32_t(IN), de,
            block_threads, shared_bytes);
        ++ep;
        ok(hipMemcpy(&err, de, sizeof(V), hipMemcpyDeviceToHost));
        err /= V(count);
        if (ep == 50) {
            std::vector<V> curr(tw);
            ok(hipMemcpy(curr.data(), dw, tw * sizeof(V), hipMemcpyDeviceToHost));
            std::cout << "DEBUG out after 50 epochs[0..9]:";
            for (int i = 0; i < 10; ++i) std::cout << " " << curr[out_off + i];
            std::cout << std::endl;
            std::cout << "DEBUG out diff from init[0..9]:";
            for (int i = 0; i < 10; ++i) std::cout << " " << (curr[out_off + i] - initial_w[out_off + i]);
            std::cout << std::endl;
        }
        if (ep == 5) {
            std::vector<V> curr(tw);
            ok(hipMemcpy(curr.data(), dw, tw * sizeof(V), hipMemcpyDeviceToHost));
            std::cout << "DEBUG out after 5 epochs[0..9]:";
            for (int i = 0; i < 10; ++i) std::cout << " " << curr[out_off + i];
            std::cout << std::endl;
            std::cout << "DEBUG out diff from init[0..9]:";
            for (int i = 0; i < 10; ++i) std::cout << " " << (curr[out_off + i] - initial_w[out_off + i]);
            std::cout << std::endl;
        }
    } while (errorFn(ep, err));

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::cout << "GPU training: " << ep << " epochs in " << ms << "ms ("
              << (ms / std::max(ep, 1u)) << "ms/epoch)" << std::endl;

    ok(hipMemcpy(hw.data(), dw, tw * sizeof(V), hipMemcpyDeviceToHost));
    ok(hipMemcpy(hb.data(), db, tb * sizeof(V), hipMemcpyDeviceToHost));

    [&]<size_t... Is>(std::index_sequence<Is...>) {
        (((void)[&] {
            auto& w = std::get<Is>(ctx.weights);
            auto& b = std::get<Is>(ctx.biases);
            auto wo = cfgs[Is].w_off;
            std::copy(hw.begin() + wo, hw.begin() + wo + cfgs[Is].tot_w, w.begin());
            std::copy(hb.begin() + cfgs[Is].b_off,
                      hb.begin() + cfgs[Is].b_off + cfgs[Is].out_sz, b.begin());
        }()), ...);
    }(std::make_index_sequence<N>{});

    ok(hipFree(dw)); ok(hipFree(db)); ok(hipFree(dp)); ok(hipFree(de)); ok(hipFree(dc));
}

#else

template<typename BepAlgo>
void calculate(BepAlgo& algo,
               const typename BepAlgo::Prototype* prototypes,
               std::size_t count,
               ErrorFn errorFn)
{
    algo.calculate(prototypes, prototypes + count, errorFn);
}

#endif

// Explicit instantiation for the OCR algorithm type
using OCRPerceptron = decltype(nn::build<float>()
    .input<180>()
    .conv().with_grid<12,15>().with_kernel<3,3,2>().build()
    .dense<30>()
    .dense<10>()
    .with_neuron<nn::Neuron<nn::SoftmaxFunction>>())::type;
using OCRAlgo = nn::bp::BepAlgorithm<OCRPerceptron, nn::bp::CrossEntropyError>;

template void calculate<OCRAlgo>(OCRAlgo&, const OCRAlgo::Prototype*, std::size_t, ErrorFn);

} // namespace nn::gpu
