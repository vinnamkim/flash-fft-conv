
// Copyright (c) 2023 Dan Fu, Hermann Kumbong

#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define DISPATCH_FLOAT_AND_HALF_AND_BF16(INPUT_TYPE, WEIGHT_TYPE, NAME, ...)                                                       \
  if ((INPUT_TYPE == at::ScalarType::Float) && (WEIGHT_TYPE == at::ScalarType::Float))                                        \
  {                                                                                                                                \
    using input_t = float;                                                                                                         \
    using weight_t = float;                                                                                                        \
    __VA_ARGS__();                                                                                                                 \
  }                                                                                                                                \
  else                                                                                                                             \
  {                                                                                                                                \
    AT_ERROR(#NAME, " not implemented for input-type '", toString(INPUT_TYPE), "' and weight-type '", toString(WEIGHT_TYPE), "'"); \
  }

template <typename T>
__forceinline__ __device__ void set_value(T *dst, T src)
{
  *dst = src;
}

__forceinline__ __device__ void set_value(__half2 *dst, float2 src)
{
  *dst = __float22half2_rn(src);
}

__forceinline__ __device__ void set_value(float2 *dst, __half2 src)
{
  *dst = __half22float2(src);
}

__forceinline__ __device__ void set_value(__half *dst, float src)
{
  *dst = __float2half(src);
}

__forceinline__ __device__ void set_value(float *dst, __half src)
{
  *dst = __half2float(src);
}

template <typename T, typename U>
__forceinline__ __device__ T multiply(const T v1, const U v2)
{
  return v1 * v2;
}
