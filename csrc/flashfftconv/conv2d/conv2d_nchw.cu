#include "shared.h"

const uint BX = 512;
const uint BY = 1;
const uint BZ = 1;

// const uint TILE_SIZE_H = 4;
// const uint TILE_SIZE_W = 4;

constexpr uint THREAD_TILE_SIZE_H = 4;
constexpr uint THREAD_TILE_SIZE_W = 4;
// constexpr uint MAX_PADDING = 1;

__forceinline__ __device__ uint __pos(
    const uint n, const uint c, const uint h, const uint w,
    const uint N, const uint C, const uint H, const uint W)
{
    return n * C * H * W + c * H * W + h * W + w;
}

template <typename T, typename U, uint K>
__global__ void conv2d_kernel(
    const T *__restrict__ input,
    const U *__restrict__ weights,
    // const U *__restrict__ bias,
    T *__restrict__ out,
    const uint padding,
    const uint N,
    const uint C,
    const uint H_IN,
    const uint W_IN,
    const uint H_OUT,
    const uint W_OUT,
    const uint R,
    const uint S)
{
    const uint H = (H_OUT + THREAD_TILE_SIZE_H - 1U) / THREAD_TILE_SIZE_H;
    const uint W = (W_OUT + THREAD_TILE_SIZE_W - 1U) / THREAD_TILE_SIZE_W;

    // linear_idx = n * C * H * W + c * H * W + h * W + w
    const uint linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint n = linear_idx / (C * H * W);
    // n_mod = c * H * W + h_out * W + w_out
    const uint n_mod = linear_idx % (C * H * W);

    const uint c = (n_mod) / (H * W);
    // c_mod = h * W + w_out
    const uint c_mod = (n_mod) % (H * W);

    const uint h_tile_idx = c_mod / W;
    const uint w_tile_idx = c_mod % W;

    T in_local[THREAD_TILE_SIZE_H + K - 1][THREAD_TILE_SIZE_W + K - 1] = {static_cast<T>(0)};
    T w_local[K][K];
    T out_local[THREAD_TILE_SIZE_H][THREAD_TILE_SIZE_W];

#pragma unroll
    for (uint r = 0; r < THREAD_TILE_SIZE_H + K - 1; r++)
    {
#pragma unroll
        for (uint s = 0; s < THREAD_TILE_SIZE_W + K - 1; s++)
        {
            const int h_in = h_tile_idx * THREAD_TILE_SIZE_H + r - (R / 2);
            const int w_in = w_tile_idx * THREAD_TILE_SIZE_W + s - (S / 2);

            if (0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN)
                in_local[r][s] = input[__pos(n, c, h_in, w_in, N, C, H_IN, W_IN)];
        }
    }
    // if (linear_idx == 0)
    // {
    //     printf("in_local:\n");
    //     for (uint r = 0; r < THREAD_TILE_SIZE_H + R - 1; r++)
    //     {
    //         for (uint s = 0; s < THREAD_TILE_SIZE_W + S - 1; s++)
    //             printf("%.2f ", in_local[r][s]);

    //         printf("\n");
    //     }
    // }

#pragma unroll
    for (uint r = 0; r < K; r++)
#pragma unroll
        for (uint s = 0; s < K; s++)
        {
            const uint w_idx = c * K * K + r * K + s;
            set_value(&w_local[r][s], weights[w_idx]);
        }

#pragma unroll
    for (uint h_idx = 0; h_idx < THREAD_TILE_SIZE_H; h_idx++)
#pragma unroll
        for (uint w_idx = 0; w_idx < THREAD_TILE_SIZE_W; w_idx++)
        {
            const uint h_out = h_tile_idx * THREAD_TILE_SIZE_H + h_idx;
            const uint w_out = w_tile_idx * THREAD_TILE_SIZE_W + w_idx;

            if (h_out < H_OUT && w_out < W_OUT)
            {
                T accum{static_cast<T>(0)};

#pragma unroll
                for (int r = 0; r < K; r++)
#pragma unroll
                    for (int s = 0; s < K; s++)
                        accum += in_local[h_idx + r][w_idx + s] * w_local[r][s];

                out_local[h_idx][w_idx] = accum;
            }
        }

#pragma unroll
    for (uint h_idx = 0; h_idx < THREAD_TILE_SIZE_H; h_idx++)
#pragma unroll
        for (uint w_idx = 0; w_idx < THREAD_TILE_SIZE_W; w_idx++)
        {
            const uint h_out = h_tile_idx * THREAD_TILE_SIZE_H + h_idx;
            const uint w_out = w_tile_idx * THREAD_TILE_SIZE_W + w_idx;

            if (h_out < H_OUT && w_out < W_OUT)
            {
                out[__pos(n, c, h_out, w_out, N, C, H_OUT, W_OUT)] = out_local[h_idx][w_idx];
            }
        }
}

torch::Tensor conv2d_cuda_nchw(
    torch::Tensor input,
    torch::Tensor weights,
    // torch::Tensor bias,
    uint padding)
{
    const uint N = input.size(0);
    const uint C = input.size(1);
    const uint H_IN = input.size(2);
    const uint W_IN = input.size(3);

    const uint c_weight = weights.size(0);
    const uint should_one = weights.size(1);
    const uint R = weights.size(2);
    const uint S = weights.size(3);

    TORCH_CHECK(C == c_weight, "input.shape[1] should be equal to weights.shape[0]");
    TORCH_CHECK(should_one == 1, "weights.size[1] should be one");
    TORCH_CHECK(R == S, "Kernel size should be square");
    TORCH_CHECK(R % 2 == 1 && S % 2 == 1, "Kernel size should be odd number");

    uint H_OUT = (H_IN + 2 * padding + 1 - R);
    uint W_OUT = (W_IN + 2 * padding + 1 - S);

    torch::Tensor out = torch::empty({N, C, H_OUT, W_OUT}, input.options());

    dim3 threadsPerBlock(BX, BY, BZ);

    const uint H = (H_OUT + THREAD_TILE_SIZE_H - 1U) / THREAD_TILE_SIZE_H;
    const uint W = (W_OUT + THREAD_TILE_SIZE_W - 1U) / THREAD_TILE_SIZE_W;
    dim3 numBlocks(((N * C * H * W + BX - 1) / BX), 1, 1);

    DISPATCH_FLOAT_AND_HALF_AND_BF16(input.scalar_type(), weights.scalar_type(),
                                     "depthwise conv2d_nchw fwd",
                                     ([&]
                                      { conv2d_kernel<input_t, weight_t, 3U><<<numBlocks, threadsPerBlock>>>(
                                            static_cast<input_t *>(input.data_ptr()),
                                            static_cast<weight_t *>(weights.data_ptr()),
                                            // static_cast<weight_t *>(bias.data_ptr()),
                                            static_cast<input_t *>(out.data_ptr()),
                                            padding,
                                            N,
                                            C,
                                            H_IN,
                                            W_IN,
                                            H_OUT,
                                            W_OUT,
                                            R,
                                            S); }));

    return out;
}