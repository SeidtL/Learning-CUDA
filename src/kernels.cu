#include <cassert>
#include <vector>
#include <__clang_cuda_builtin_vars.h>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace detail {

template <typename T>
__device__ 
T warp_trace_reduce(T sum, int warp_size) {
  #pragma unroll
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }
  return sum;
}
template <typename T>
__global__
void trace_kernel(
  const T * __restrict__ ptr, 
  T * __restrict__ gather, 
  int rows, 
  int cols, 
  int warp_size
) {
  extern __shared__ char _smem[];
  T* smem = reinterpret_cast<T*>(_smem);
  auto idx = blockDim.x * blockIdx.x + threadIdx.x;
  auto stride = blockDim.x * gridDim.x;
  auto n = min(rows, cols);

  auto tid = threadIdx.x;

  T sum{};
  for (auto i = idx; i < n; i += stride) {
    sum += ptr[i * cols + i];
  }
  const T warp_sum = warp_trace_reduce(sum, warp_size);

  if (tid % warp_size == 0) {
    smem[tid / warp_size] = warp_sum;
  }
  __syncthreads();

  const auto num_warps = (blockDim.x + warp_size - 1) / warp_size;
  if (tid < warp_size) {
    T block_sum = (tid < num_warps) ? smem[tid] : T{};
    block_sum = warp_trace_reduce(block_sum, warp_size);
    if (tid == 0) {
      atomicAdd(gather, block_sum);
    }
  }
}

template <typename T>
__global__
void flash_atten_kernel(
  const T * __restrict__ q,
  const T * __restrict__ k,
  const T * __restrict__ v,
  T * __restrict__ o,
  int batch_size, 
  int target_seq_len, 
  int src_seq_len, 
  int query_heads, 
  int kv_heads, 
  int head_dim, 
  bool is_causal
) {
  extern __shared__ char _smem[];
  T *const smem = reinterpret_cast<T*>(_smem);
  const auto tidx = threadIdx.x;
  const auto tidy = threadIdx.y;

  auto index_tgt = [&] (int b, int s, int h, int d) {
    return ((b * target_seq_len + s) * query_heads + h) * head_dim + d; 
  };
  
  auto index_src = [&] (int b, int s, int h, int d) {
    return ((b * src_seq_len + s) * kv_heads + h) * head_dim + d; 
  };

  for (int i = 0; i < ) {

  }

}
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function
  
  T *inp;
  T *sum;
  T result{};
  cudaMalloc(&inp, sizeof(T) * rows * cols);
  cudaMalloc(&sum, sizeof(T));

  cudaMemcpy(inp, h_input.data(), sizeof(T) * rows * cols, cudaMemcpyHostToDevice);
  cudaMemcpy(sum, &result, sizeof(T), cudaMemcpyHostToDevice);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int warpSize = prop.warpSize;

  dim3 block {256};
  dim3 grid {static_cast<unsigned int>((std::min(rows, cols) + block.x - 1) / 256)};
  size_t smem_size = ((block.x + warpSize - 1) / warpSize) * sizeof(T);
  detail::trace_kernel<T><<<grid, block, smem_size>>>(
    inp, 
    sum, 
    rows, 
    cols, 
    warpSize
  );

  cudaMemcpy(&result, sum, sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(inp);
  cudaFree(sum);

  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
  assert(query_heads % kv_heads == 0);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
