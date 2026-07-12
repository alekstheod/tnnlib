#include "train.h"

#include <hip/hip_runtime.h>

using nn::gpu::ActType;
using nn::gpu::LayerCfg;

template<typename T> __device__ inline T dsig(T x) { return T{1} / (T{1} + expf(-x)); }
template<typename T> __device__ inline T dsig_d(T x) { return x * (T{1} - x); }
template<typename T> __device__ inline T dth(T x) { return tanhf(x); }
template<typename T> __device__ inline T dth_d(T x) { return T{1} - x * x; }
template<typename T> __device__ inline T drel(T x) { return x > T{0} ? x : T{0}; }
template<typename T> __device__ inline T drel_d(T x) { return x > T{0} ? T{1} : T{0}; }

template<typename T>
__device__ T dact(T x, ActType t) {
    switch (t) {
        case ActType::Sigmoid: return dsig(x);
        case ActType::Tanh:    return dth(x);
        case ActType::Relu:    return drel(x);
        case ActType::Softmax: return x;
        default: return x;
    }
}

template<typename T>
__device__ T dder(T x, ActType t) {
    switch (t) {
        case ActType::Sigmoid: return dsig_d(x);
        case ActType::Tanh:    return dth_d(x);
        case ActType::Relu:    return drel_d(x);
        default: return T{1};
    }
}

template<typename T>
__global__ void train_kernel(
    T* weights, T* biases, const T* prototypes,
    uint32_t count, T lr, const LayerCfg* layers, uint32_t nl,
    uint32_t in_sz, T* error_out)
{
    extern __shared__ T smem[];

    uint32_t p = blockIdx.x;
    uint32_t t = threadIdx.x;

    if (p >= count) return;

    const T* proto = prototypes + p * (in_sz + layers[nl - 1].out_sz);
    const T* expected = proto + in_sz;

    if (t < layers[0].out_sz) smem[layers[0].act_off + t] = proto[t];
    __syncthreads();

    for (uint32_t l = 1; l < nl; ++l) {
        auto& L = layers[l];
        auto& PL = layers[l - 1];
        uint32_t out = L.out_sz, in = L.in_sz;
        const T* w = weights + L.w_off;
        const T* b = biases + L.b_off;
        const T* a_prev = smem + PL.act_off;

        if (t < out) {
            T sum{0};
            if (L.is_conv) {
                auto kw = in;
                auto ks = static_cast<uint32_t>(sqrtf(float(kw)));
                auto sd = L.stride; auto gw = L.gw; auto gh = L.gh;
                auto ow = (gw + sd - 1) / sd;
                auto wy = t / ow; auto wx = t % ow;
                sum = b[t];
                for (uint32_t ky = 0; ky < ks; ++ky)
                    for (uint32_t kx = 0; kx < ks; ++kx) {
                        auto iy = wy * sd + ky; auto ix = wx * sd + kx;
                        if (iy < gh && ix < gw)
                            sum += a_prev[iy * gw + ix] * w[t * kw + ky * ks + kx];
                    }
            } else {
                sum = b[t];
                auto ro = t * in;
                for (uint32_t j = 0; j < in; ++j)
                    sum += a_prev[j] * w[ro + j];
            }
            smem[L.act_off + t] = dact(sum, L.act);
        }

        if (L.act == ActType::Softmax) {
            __syncthreads();
            if (t < out) {
                T* a = smem + L.act_off;
                T mx = a[0];
                for (uint32_t i = 1; i < out; ++i) if (a[i] > mx) mx = a[i];
                T s{0};
                for (uint32_t i = 0; i < out; ++i) s += expf(a[i] - mx);
                a[t] = expf(a[t] - mx) / s;
            }
        }
        __syncthreads();
    }

    if (t == 0) {
        T err{0};
        auto& L = layers[nl - 1];
        const T* a = smem + L.act_off;
        for (uint32_t i = 0; i < L.out_sz; ++i) { T e = a[i] - expected[i]; err += e * e; }
        atomicAdd(error_out, err);
    }

    { auto& L = layers[nl - 1];
      if (t < L.out_sz) {
          T* a = smem + L.act_off;
          T* d = smem + L.del_off;
          d[t] = (a[t] - expected[t]) * dder(a[t], L.act);
      }
      __syncthreads(); }

    for (uint32_t l = nl - 1; l > 0; --l) {
        auto& L = layers[l];
        auto& PL = layers[l - 1];
        uint32_t out = L.out_sz, in = L.in_sz;
        const T* d_next = smem + L.del_off;
        const T* a_cur = smem + L.act_off;

        if (l < nl - 1) {
            auto& NL = layers[l + 1];
            const T* nd = smem + NL.del_off;
            const T* nw = weights + NL.w_off;
            if (t < out) {
                T sum{0};
                for (uint32_t k = 0; k < NL.out_sz; ++k)
                    sum += nd[k] * nw[k * out + t];
                T* d = smem + L.del_off;
                d[t] = sum * dder(a_cur[t], L.act);
            }
            __syncthreads();
        }

        d_next = smem + L.del_off;
        const T* a_prev = smem + PL.act_off;
        T* w = weights + L.w_off;
        T* b = biases + L.b_off;

        if (t < out) {
            T dl = d_next[t];
            if (L.is_conv) {
                auto kw = in;
                auto ks = static_cast<uint32_t>(sqrtf(float(kw)));
                auto sd = L.stride; auto gw = L.gw; auto gh = L.gh;
                auto ow = (gw + sd - 1) / sd;
                auto wy = t / ow; auto wx = t % ow;
                for (uint32_t ky = 0; ky < ks; ++ky)
                    for (uint32_t kx = 0; kx < ks; ++kx) {
                        auto iy = wy * sd + ky; auto ix = wx * sd + kx;
                        if (iy < gh && ix < gw)
                            atomicAdd(w + t * kw + ky * ks + kx, -lr * a_prev[iy * gw + ix] * dl);
                    }
                atomicAdd(b + t, -lr * dl);
            } else {
                auto ro = t * in;
                for (uint32_t j = 0; j < in; ++j)
                    atomicAdd(w + ro + j, -lr * a_prev[j] * dl);
                atomicAdd(b + t, -lr * dl);
            }
        }
        __syncthreads();
    }

}

namespace nn::gpu {

void launch_train_kernel(
    float* dw, float* db, const float* dp,
    uint32_t count, float lr, const LayerCfg* dc,
    uint32_t nl, uint32_t in_sz, float* de,
    uint32_t block_threads, uint32_t shared_bytes)
{
    train_kernel<<<count, block_threads, shared_bytes>>>(
        dw, db, dp, count, lr, dc, nl, in_sz, de);
}

}
