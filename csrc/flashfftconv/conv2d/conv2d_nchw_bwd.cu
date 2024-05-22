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
__global__ void conv2d_kernel_bwd_din(
    const T *__restrict__ dout,
    const T *__restrict__ input,
    const U *__restrict__ weights,
    // const U *__restrict__ bias,
    T *__restrict__ din,
    const uint N,
    const uint C,
    const uint H,
    const uint W,
    const uint NUM_TILE_H,
    const uint NUM_TILE_W)
{
    // linear_idx = n * C * H * W + c * H * W + h * W + w
    const uint linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint n = linear_idx / (C * NUM_TILE_H * NUM_TILE_W);
    // n_mod = c * H * W + h_out * W + w_out
    const uint n_mod = linear_idx % (C * NUM_TILE_H * NUM_TILE_W);

    const uint c = (n_mod) / (NUM_TILE_H * NUM_TILE_W);
    // c_mod = h * W + w_out
    const uint c_mod = (n_mod) % (NUM_TILE_H * NUM_TILE_W);

    const uint h_tile_idx = c_mod / NUM_TILE_W;
    const uint w_tile_idx = c_mod % NUM_TILE_W;

    if (n >= N || c >= C)
        return;

    T dout_local[THREAD_TILE_SIZE_H + K - 1][THREAD_TILE_SIZE_W + K - 1] = {static_cast<T>(0)};
    T w_local[K][K];

    T din_local[THREAD_TILE_SIZE_H][THREAD_TILE_SIZE_W];

#pragma unroll
    for (uint r = 0; r < THREAD_TILE_SIZE_H + K - 1; r++)
    {
#pragma unroll
        for (uint s = 0; s < THREAD_TILE_SIZE_W + K - 1; s++)
        {
            const int h_in = h_tile_idx * THREAD_TILE_SIZE_H + r - (K / 2);
            const int w_in = w_tile_idx * THREAD_TILE_SIZE_W + s - (K / 2);

            if (0 <= h_in && h_in < H && 0 <= w_in && w_in < W)
                dout_local[r][s] = dout[__pos(n, c, h_in, w_in, N, C, H, W)];
        }
    }

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

            if (h_out < H && w_out < W)
            {
                T accum{static_cast<T>(0)};

#pragma unroll
                for (int r = 0; r < K; r++)
#pragma unroll
                    for (int s = 0; s < K; s++)
                        accum += dout_local[h_idx + r][w_idx + s] * w_local[K - r - 1][K - s - 1];

                din_local[h_idx][w_idx] = accum;
            }
        }

#pragma unroll
    for (uint h_idx = 0; h_idx < THREAD_TILE_SIZE_H; h_idx++)
#pragma unroll
        for (uint w_idx = 0; w_idx < THREAD_TILE_SIZE_W; w_idx++)
        {
            const uint h_out = h_tile_idx * THREAD_TILE_SIZE_H + h_idx;
            const uint w_out = w_tile_idx * THREAD_TILE_SIZE_W + w_idx;

            if (h_out < H && w_out < W)
                din[__pos(n, c, h_out, w_out, N, C, H, W)] = din_local[h_idx][w_idx];
        }
}

template <typename T, typename U, uint K>
__global__ void conv2d_kernel_bwd_dweights(
    const T *__restrict__ dout,
    const T *__restrict__ input,
    const U *__restrict__ weights,
    // const U *__restrict__ bias,
    T *__restrict__ dweights,
    const uint N,
    const uint C,
    const uint H,
    const uint W,
    const uint NUM_TILE_H,
    const uint NUM_TILE_W)
{
    // linear_idx = n * C * H * W + c * H * W + h * W + w
    const uint linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

    const uint n = linear_idx / (C * NUM_TILE_H * NUM_TILE_W);
    // n_mod = c * H * W + h_out * W + w_out
    const uint n_mod = linear_idx % (C * NUM_TILE_H * NUM_TILE_W);

    const uint c = (n_mod) / (NUM_TILE_H * NUM_TILE_W);
    // c_mod = h * W + w_out
    const uint c_mod = (n_mod) % (NUM_TILE_H * NUM_TILE_W);

    const uint h_tile_idx = c_mod / NUM_TILE_W;
    const uint w_tile_idx = c_mod % NUM_TILE_W;

    if (n >= N || c >= C)
        return;

    T dout_local[THREAD_TILE_SIZE_H + K - 1][THREAD_TILE_SIZE_W + K - 1] = {static_cast<T>(0)};
    T in_local[THREAD_TILE_SIZE_H + K - 1][THREAD_TILE_SIZE_W + K - 1] = {static_cast<T>(0)};

    T dw_local[K][K];

#pragma unroll
    for (uint r = 0; r < THREAD_TILE_SIZE_H + K - 1; r++)
    {
#pragma unroll
        for (uint s = 0; s < THREAD_TILE_SIZE_W + K - 1; s++)
        {
            const int h_in = h_tile_idx * THREAD_TILE_SIZE_H + r - (K / 2);
            const int w_in = w_tile_idx * THREAD_TILE_SIZE_W + s - (K / 2);

            if (0 <= h_in && h_in < H && 0 <= w_in && w_in < W)
            {
                dout_local[r][s] = dout[__pos(n, c, h_in, w_in, N, C, H, W)];
                in_local[r][s] = input[__pos(n, c, h_in, w_in, N, C, H, W)];

                // printf("rs: [%u][%u] value: %f \n", r, s, in_local[r][s]);
            }
        }
    }

#pragma unroll
    for (uint k1 = 0; k1 < K; k1++)
#pragma unroll
        for (uint k2 = 0; k2 < K; k2++)
        {
            T accum{static_cast<T>(0)};

            for (uint r = K / 2; r < THREAD_TILE_SIZE_H + K - 1 - K / 2; r++)
                for (uint s = K / 2; s < THREAD_TILE_SIZE_W + K - 1 - K / 2; s++)
                {
                    const int i = r - (K / 2) + k1;
                    const int j = s - (K / 2) + k2;

                    accum += dout_local[r][s] * in_local[i][j];
                }

            dw_local[k1][k2] = accum;
            // printf("[%u][%u] %f \n", k1, k2, accum);
        }

#pragma unroll
    for (uint h_idx = 0; h_idx < THREAD_TILE_SIZE_H; h_idx++)
#pragma unroll
        for (uint w_idx = 0; w_idx < THREAD_TILE_SIZE_W; w_idx++)
        {
            const uint h_out = h_tile_idx * THREAD_TILE_SIZE_H + h_idx;
            const uint w_out = w_tile_idx * THREAD_TILE_SIZE_W + w_idx;

            if (h_out < H && w_out < W)
            {
#pragma unroll
                for (uint k1 = 0; k1 < K; k1++)
#pragma unroll
                    for (uint k2 = 0; k2 < K; k2++)
                    {
                        const uint pos = c * K * K * N * NUM_TILE_H * NUM_TILE_W +
                                         k1 * K * N * NUM_TILE_H * NUM_TILE_W +
                                         k2 * N * NUM_TILE_H * NUM_TILE_W +
                                         n * NUM_TILE_H * NUM_TILE_W +
                                         h_tile_idx * NUM_TILE_W +
                                         w_tile_idx;

                        dweights[pos] = dw_local[k1][k2];
                    }
            }
        }
}

std::vector<torch::Tensor> conv2d_cuda_nchw_bwd(
    torch::Tensor dout,
    torch::Tensor input,
    torch::Tensor weights,
    // torch::Tensor bias,
    uint padding)
{
    const uint N = input.size(0);
    const uint C = input.size(1);
    const uint H = input.size(2);
    const uint W = input.size(3);

    TORCH_CHECK(N == dout.size(0), "");
    TORCH_CHECK(C == dout.size(1), "");
    TORCH_CHECK(H == dout.size(2), "");
    TORCH_CHECK(W == dout.size(3), "");

    const uint c_weight = weights.size(0);
    const uint should_one = weights.size(1);
    const uint R = weights.size(2);
    const uint S = weights.size(3);

    TORCH_CHECK(C == c_weight, "input.shape[1] should be equal to weights.shape[0]");
    TORCH_CHECK(should_one == 1, "weights.size[1] should be one");
    TORCH_CHECK(R == S, "Kernel size should be square");
    TORCH_CHECK(R % 2 == 1 && S % 2 == 1, "Kernel size should be odd number");
    TORCH_CHECK(2 * padding + 1 == R, "Kernel size should be equal to 2 * padding + 1")

    const uint NUM_TILE_H = (H + THREAD_TILE_SIZE_H - 1U) / THREAD_TILE_SIZE_H;
    const uint NUM_TILE_W = (W + THREAD_TILE_SIZE_W - 1U) / THREAD_TILE_SIZE_W;

    torch::Tensor din = torch::empty({N, C, H, W}, input.options());
    torch::Tensor dweights = torch::zeros({C, 1, R, S, N, NUM_TILE_H, NUM_TILE_W}, input.options());

    dim3 threadsPerBlock(BX, BY, BZ);
    dim3 numBlocks(((N * C * NUM_TILE_H * NUM_TILE_W + BX - 1) / BX), 1, 1);

    if (R == 3)
    {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(input.scalar_type(), weights.scalar_type(),
                                         "depthwise conv2d_nchw bwd din",
                                         ([&]
                                          { conv2d_kernel_bwd_din<input_t, weight_t, 3U><<<numBlocks, threadsPerBlock>>>(
                                                static_cast<input_t *>(dout.data_ptr()),
                                                static_cast<input_t *>(input.data_ptr()),
                                                static_cast<weight_t *>(weights.data_ptr()),
                                                // static_cast<weight_t *>(bias.data_ptr()),
                                                static_cast<input_t *>(din.data_ptr()),
                                                N,
                                                C,
                                                H,
                                                W,
                                                NUM_TILE_H,
                                                NUM_TILE_W); }));

        DISPATCH_FLOAT_AND_HALF_AND_BF16(input.scalar_type(), weights.scalar_type(),
                                         "depthwise conv2d_nchw bwd dweights",
                                         ([&]
                                          { conv2d_kernel_bwd_dweights<input_t, weight_t, 3U><<<numBlocks, threadsPerBlock>>>(
                                                static_cast<input_t *>(dout.data_ptr()),
                                                static_cast<input_t *>(input.data_ptr()),
                                                static_cast<weight_t *>(weights.data_ptr()),
                                                // static_cast<weight_t *>(bias.data_ptr()),
                                                static_cast<input_t *>(dweights.data_ptr()),
                                                N,
                                                C,
                                                H,
                                                W,
                                                NUM_TILE_H,
                                                NUM_TILE_W); }));
    }
    else if (R == 5)
    {
        DISPATCH_FLOAT_AND_HALF_AND_BF16(input.scalar_type(), weights.scalar_type(),
                                         "depthwise conv2d_nchw bwd din",
                                         ([&]
                                          { conv2d_kernel_bwd_din<input_t, weight_t, 5U><<<numBlocks, threadsPerBlock>>>(
                                                static_cast<input_t *>(dout.data_ptr()),
                                                static_cast<input_t *>(input.data_ptr()),
                                                static_cast<weight_t *>(weights.data_ptr()),
                                                // static_cast<weight_t *>(bias.data_ptr()),
                                                static_cast<input_t *>(din.data_ptr()),
                                                N,
                                                C,
                                                H,
                                                W,
                                                NUM_TILE_H,
                                                NUM_TILE_W); }));

        DISPATCH_FLOAT_AND_HALF_AND_BF16(input.scalar_type(), weights.scalar_type(),
                                         "depthwise conv2d_nchw bwd dweights",
                                         ([&]
                                          { conv2d_kernel_bwd_dweights<input_t, weight_t, 5U><<<numBlocks, threadsPerBlock>>>(
                                                static_cast<input_t *>(dout.data_ptr()),
                                                static_cast<input_t *>(input.data_ptr()),
                                                static_cast<weight_t *>(weights.data_ptr()),
                                                // static_cast<weight_t *>(bias.data_ptr()),
                                                static_cast<input_t *>(dweights.data_ptr()),
                                                N,
                                                C,
                                                H,
                                                W,
                                                NUM_TILE_H,
                                                NUM_TILE_W); }));
    }

    return {din, dweights.sum(std::vector<int64_t>({-3, -2, -1}))};
}