extern "C" __global__ void rwkv7_forward(
    int batch_size,
    int token_count,
    const float* receptance,
    const float* sigmoid_decay,
    const float* key,
    const float* value,
    const float* in_context_key,
    const float* in_context_value,
    float* output) {
    constexpr int head_count = 12;
    constexpr int head_size = 64;
    constexpr int embedding_dimension = head_count * head_size;
    const int head = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const int row = static_cast<int>(threadIdx.x);
    if (batch >= batch_size || head >= head_count || row >= head_size) {
        return;
    }

    float state[head_size] = {0.0f};
    __shared__ float shared_receptance[head_size];
    __shared__ float shared_decay[head_size];
    __shared__ float shared_key[head_size];
    __shared__ float shared_context_key[head_size];
    __shared__ float shared_context_value[head_size];

    const int sequence_offset = batch * token_count * embedding_dimension;
    const int head_offset = head * head_size;
    for (int token = 0; token < token_count; ++token) {
        const int token_offset = sequence_offset + token * embedding_dimension + head_offset;
        if (token != 0) {
            __syncthreads();
        }
        shared_receptance[row] = receptance[token_offset + row];
        shared_decay[row] = expf(-0.6065306597126334f * sigmoid_decay[token_offset + row]);
        shared_key[row] = key[token_offset + row];
        shared_context_key[row] = in_context_key[token_offset + row];
        shared_context_value[row] = in_context_value[token_offset + row];
        __syncthreads();

        float projection = 0.0f;
        #pragma unroll
        for (int column = 0; column < head_size; ++column) {
            projection += state[column] * shared_context_key[column];
        }

        const float row_value = value[token_offset + row];
        float mixed = 0.0f;
        #pragma unroll
        for (int column = 0; column < head_size; ++column) {
            const float updated = state[column] * shared_decay[column]
                + projection * shared_context_value[column]
                + row_value * shared_key[column];
            state[column] = updated;
            mixed += updated * shared_receptance[column];
        }
        output[token_offset + row] = mixed;
    }
}
