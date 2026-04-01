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

__kernel void calc_output_deltas(global float* outputs,
                                 global float* expected,
                                 global float* deltas,
                                 float momentum,
                                 unsigned int size) {
    unsigned int idx = get_global_id(0);
    float output = outputs[idx];
    float err = output - expected[idx];
    float derivative = 1.0f - output * output;
    float newDelta = err * derivative;
    deltas[idx] = deltas[idx] * momentum + newDelta;
}

__kernel void calc_hidden_deltas(global float* nextDeltas,
                                 global float* weights,
                                 global float* outputs,
                                 global float* deltas,
                                 float momentum,
                                 unsigned int currentSize,
                                 unsigned int nextSize,
                                 unsigned int inputsPerNeuron) {
    unsigned int idx = get_global_id(0);

    float sum = 0.0f;
    for(unsigned int j = 0; j < nextSize; ++j) {
        float w = weights[j * inputsPerNeuron + idx];
        sum += nextDeltas[j] * w;
    }

    float output = outputs[idx];
    float derivative = 1.0f - output * output;
    float newDelta = sum * derivative;
    deltas[idx] = deltas[idx] * momentum + newDelta;
}

__kernel void accumulate_gradients(global float* inputs,
                                   global float* deltas,
                                   global float* weightGradients,
                                   global float* biasGradients,
                                   unsigned int size,
                                   unsigned int inputsPerNeuron) {
    unsigned int neuronIdx = get_global_id(0);
    unsigned int weightOffset = neuronIdx * inputsPerNeuron;

    float delta = deltas[neuronIdx];
    float biasGrad = delta;

    for(unsigned int i = 0; i < inputsPerNeuron; ++i) {
        weightGradients[weightOffset + i] += inputs[weightOffset + i] * delta;
    }

    biasGradients[neuronIdx] += biasGrad;
}

__kernel void calc_output_deltas_and_gradients(global float* outputs,
                                               global float* expected,
                                               global float* inputs,
                                               global float* deltas,
                                               global float* weightGradients,
                                               global float* biasGradients,
                                               float momentum,
                                               unsigned int size,
                                               unsigned int inputsPerNeuron) {
    unsigned int idx = get_global_id(0);
    float output = outputs[idx];
    float err = output - expected[idx];
    float derivative = 1.0f - output * output;
    float newDelta = err * derivative;
    deltas[idx] = deltas[idx] * momentum + newDelta;

    float delta = deltas[idx];
    unsigned int weightOffset = idx * inputsPerNeuron;
    for(unsigned int i = 0; i < inputsPerNeuron; ++i) {
        weightGradients[weightOffset + i] += inputs[weightOffset + i] * delta;
    }
    biasGradients[idx] += delta;
}

__kernel void calc_hidden_deltas_and_gradients(global float* nextDeltas,
                                                global float* weights,
                                                global float* outputs,
                                                global float* inputs,
                                                global float* deltas,
                                                global float* weightGradients,
                                                global float* biasGradients,
                                                float momentum,
                                                unsigned int currentSize,
                                                unsigned int nextSize,
                                                unsigned int inputsPerNeuron) {
    unsigned int idx = get_global_id(0);

    float sum = 0.0f;
    for(unsigned int j = 0; j < nextSize; ++j) {
        float w = weights[j * inputsPerNeuron + idx];
        sum += nextDeltas[j] * w;
    }

    float output = outputs[idx];
    float derivative = 1.0f - output * output;
    float newDelta = sum * derivative;
    deltas[idx] = deltas[idx] * momentum + newDelta;

    float delta = deltas[idx];
    unsigned int weightOffset = idx * inputsPerNeuron;
    for(unsigned int i = 0; i < inputsPerNeuron; ++i) {
        weightGradients[weightOffset + i] += inputs[weightOffset + i] * delta;
    }
    biasGradients[idx] += delta;
}

__kernel void calc_output_and_update(global float* outputs,
                                     global float* expected,
                                     global float* inputs,
                                     global float* weights,
                                     global float* biases,
                                     float learningRate,
                                     float momentum,
                                     unsigned int size,
                                     unsigned int inputsPerNeuron) {
    unsigned int idx = get_global_id(0);
    
    float output = outputs[idx];
    float err = output - expected[idx];
    float derivative = 1.0f - output * output;
    float delta = err * derivative;
    
    float weightUpdate = 0.0f;
    unsigned int weightOffset = idx * inputsPerNeuron;
    for(unsigned int i = 0; i < inputsPerNeuron; ++i) {
        float grad = inputs[weightOffset + i] * delta;
        weightUpdate += grad * grad;
        weights[weightOffset + i] -= learningRate * grad;
    }
    biases[idx] -= learningRate * delta;
}

__kernel void calc_hidden_and_update(global float* nextDeltas,
                                     global float* weights,
                                     global float* outputs,
                                     global float* inputs,
                                     global float* biases,
                                     float learningRate,
                                     float momentum,
                                     unsigned int currentSize,
                                     unsigned int nextSize,
                                     unsigned int inputsPerNeuron) {
    unsigned int idx = get_global_id(0);

    float sum = 0.0f;
    for(unsigned int j = 0; j < nextSize; ++j) {
        float w = weights[j * inputsPerNeuron + idx];
        sum += nextDeltas[j] * w;
    }

    float output = outputs[idx];
    float derivative = 1.0f - output * output;
    float delta = sum * derivative;

    unsigned int weightOffset = idx * inputsPerNeuron;
    for(unsigned int i = 0; i < inputsPerNeuron; ++i) {
        weights[weightOffset + i] -= learningRate * inputs[weightOffset + i] * delta;
    }
    biases[idx] -= learningRate * delta;
}
