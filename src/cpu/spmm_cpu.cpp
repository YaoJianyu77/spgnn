#include "spmm.hpp"
#include <cstring>
#include <algorithm>

#if defined(_OPENMP)
  #include <omp.h>
#endif

namespace spmm {

void spmm_csr_cpu(
    int M, int /*K*/, int N,
    const int* row_ptr,
    const int* col_ind,
    const float* values,
    const float* B, int ldb,
    float* C, int ldc,
    float alpha,
    float beta)
{
    // 基线实现：按行遍历 CSR。可选 OpenMP 并行（行级无写冲突）。
    // 约定：调用方保证指针合法、形状匹配。
    // 若 beta == 0，可选择先将 C 清零；这里统一走 beta 缩放路径，方便复用。
    // 行 stride 使用 ldc/ldb，默认等于 N。

    // 先对 C 进行 beta 缩放（避免在内层多次判断）
    if (beta == 0.0f) {
        // 清零
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < M; ++i) {
            float* Ci = C + static_cast<size_t>(i) * ldc;
            std::fill(Ci, Ci + N, 0.0f);
        }
    } else if (beta != 1.0f) {
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < M; ++i) {
            float* Ci = C + static_cast<size_t>(i) * ldc;
            for (int n = 0; n < N; ++n) Ci[n] *= beta;
        }
    }
    // 若 beta == 1 则不处理

    if (alpha == 0.0f) return;

    // 主循环
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < M; ++i) {
        const int start = row_ptr[i];
        const int end   = row_ptr[i + 1];
        float* Ci = C + static_cast<size_t>(i) * ldc;

        for (int p = start; p < end; ++p) {
            const int j = col_ind[p];    // A(i,j) 非零列
            const float a = alpha * values[p];
            const float* Bj = B + static_cast<size_t>(j) * ldb;

            // Ci[0:N] += a * Bj[0:N]
            for (int n = 0; n < N; ++n) {
                Ci[n] += a * Bj[n];
            }
        }
    }
}

} // namespace spmm
