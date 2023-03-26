__kernel void calc_weights(global float* inputs,
                           global float* deltas,
                           global float* weights,
                           float learningRate,
                           const unsigned int sz) {
    unsigned int idx = get_global_id(0);
    unsigned int offset = idx * sz;
    for(int i = 0; i < sz; ++i) {
        weights[offset + i] =
         weights[offset + i] - learningRate * inputs[offset + i] * deltas[idx];
    }
}
