__kernel void dot_product(global float* weights,
                          global float* inputs,
                          global float* result,
                          const unsigned int sz) {
    float dot = 0.f;
    unsigned int idx = get_global_id(0);
    unsigned int offset = idx * sz;
    for(int i = 0; i < sz; ++i) {
        dot += weights[offset + i] * inputs[offset + i];
    }

    result[idx] = dot;
}
