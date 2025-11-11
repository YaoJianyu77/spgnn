#include "spmm.hpp"
#include <cuda_runtime.h>
#include <cstdio>

namespace spmm {

// 简单的 CUDA 错误检查（返回非零错误码给上层）
static inline int checkCuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[CUDA] %s failed: %s\n", where, cudaGetErrorString(e));
        return static_cast<int>(e);
    }
    return 0;
}

__global__ void spmm_csr_row_kernel(
    int M, int N,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ B, int ldb,
    float* __restrict__ C, int ldc,
    float alpha, float beta)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float* Ci = C + static_cast<size_t>(row) * ldc;

    // 先做 beta 缩放
    if (beta == 0.0f) {
        for (int n = 0; n < N; ++n) Ci[n] = 0.0f;
    } else if (beta != 1.0f) {
        for (int n = 0; n < N; ++n) Ci[n] *= beta;
    }

    if (alpha == 0.0f) return;

    const int start = row_ptr[row];
    const int end   = row_ptr[row + 1];

    for (int p = start; p < end; ++p) {
        const int j = col_ind[p];
        const float a = alpha * values[p];
        const float* Bj = B + static_cast<size_t>(j) * ldb;

        // Ci[0:N] += a * Bj[0:N]
        // 简单标量循环，后续可替换为向量化/分块
        for (int n = 0; n < N; ++n) {
            Ci[n] += a * Bj[n];
        }
    }
}

int spmm_csr_gpu(
    int M, int /*K*/, int N,
    const int* d_row_ptr,
    const int* d_col_ind,
    const float* d_values,
    const float* d_B, int ldb,
    float* d_C, int ldc,
    float alpha,
    float beta,
    int block_size)
{
    if (M <= 0 || N <= 0) return 0;
    if (!d_row_ptr || !d_col_ind || !d_values || !d_B || !d_C) return -1;

    const int grid = (M + block_size - 1) / block_size;

    spmm_csr_row_kernel<<<grid, block_size>>>(
        M, N,
        d_row_ptr, d_col_ind, d_values,
        d_B, ldb,
        d_C, ldc,
        alpha, beta
    );
    // 同步并检查错误
    int err = checkCuda(cudaGetLastError(), "kernel launch");
    if (err) return err;
    return checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

} // namespace spmm
