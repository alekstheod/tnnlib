__kernel void dot_product(__global float* weights,
                          __global float* values,
                          __global float* result,
                          __const unsigned int sz) {
    float dot = 0.f;
    unsigned int i;
    unsigned int idx = get_global_id(0);
    unsigned int offset = idx * sz;
    for(i = 0; i < sz; ++i) {
        dot += weights[offset + i] * values[offset + i];
    }

    result[idx] = dot;
};
