// OpenCL C 3.0 kernel for neural network dot product calculation
__kernel void dot_product(__global const float* weights,
                          __global const float* inputs,
                          __global float* result,
                          const uint sz) {
    const size_t gid = get_global_id(0);
    const size_t offset = gid * sz;
    
    float dot = 0.0f;
    
    // Unrolled loop for better performance when possible
    for (uint i = 0; i < sz; ++i) {
        dot += weights[offset + i] * inputs[offset + i];
    }

    result[gid] = dot;
}
