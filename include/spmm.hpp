#pragma once
#include <cstddef>
#include <cstdint>

namespace spmm {

// A: CSR (M x K)
// B: dense row-major (K x N), row stride = ldb (通常 N)
// C: dense row-major (M x N), row stride = ldc (通常 N)
// 计算: C = alpha * A * B + beta * C
void spmm_csr_cpu(
    int M, int K, int N,
    const int* row_ptr,          // length M+1
    const int* col_ind,          // length nnz
    const float* values,         // length nnz
    const float* B, int ldb,     // ldb >= N
    float* C, int ldc,           // ldc >= N
    float alpha = 1.0f,
    float beta  = 0.0f
);

// GPU 版本要求所有指针为 device 指针（cudaMalloc 分配）
// 返回 cudaError_t（0 表示成功）
int spmm_csr_gpu(
    int M, int K, int N,
    const int* d_row_ptr,        // device, length M+1
    const int* d_col_ind,        // device, length nnz
    const float* d_values,       // device, length nnz
    const float* d_B, int ldb,   // device, ldb >= N
    float* d_C, int ldc,         // device, ldc >= N
    float alpha = 1.0f,
    float beta  = 0.0f,
    int  block_size = 256        // 每个 block 线程数（行粒度）
);

} // namespace spmm
