// test/spmm_test.cpp
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

#include "spmm.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
// 即便不是用 nvcc 编译，只要具备 CUDA runtime 头文件和链接，也可包含：
#include <cuda_runtime.h>
#endif

// ===== 工具函数 =====

static bool has_cuda_device() {
    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess) && (count > 0);
}

// 生成 CSR 稀疏矩阵（M x K），按给定稀疏率 density（0~1）随机采样列索引（每行唯一）
static void make_random_csr(int M, int K, float density,
                            std::vector<int>& row_ptr,
                            std::vector<int>& col_ind,
                            std::vector<float>& values,
                            uint64_t seed = 42)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> urand01(0.0f, 1.0f);
    std::uniform_real_distribution<float> valdist(-1.0f, 1.0f);

    row_ptr.assign(M + 1, 0);
    col_ind.clear();
    values.clear();

    const int target_per_row = std::max(1, std::min(K, (int)std::round(density * K)));

    for (int i = 0; i < M; ++i) {
        std::unordered_set<int> cols;
        cols.reserve(target_per_row * 2 + 8);

        // 保证每行唯一列索引
        while ((int)cols.size() < target_per_row) {
            int c = (int)std::floor(urand01(rng) * K);
            if (c >= 0 && c < K) cols.insert(c);
        }

        // 将本行列索引排序，便于后续可复用（非必须）
        std::vector<int> row_cols(cols.begin(), cols.end());
        std::sort(row_cols.begin(), row_cols.end());

        // 追加到全局 CSR
        for (int c : row_cols) {
            col_ind.push_back(c);
            values.push_back(valdist(rng));
        }
        row_ptr[i + 1] = (int)col_ind.size();
    }
}

// 生成 row-major 稠密矩阵 (R x C)，行步长 ld = C
static void make_random_dense(int R, int C, std::vector<float>& mat,
                              uint64_t seed = 123)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> valdist(-1.0f, 1.0f);
    mat.resize((size_t)R * C);
    for (auto& x : mat) x = valdist(rng);
}

// 计算最大绝对/相对误差
struct ErrStats { double max_abs = 0.0; double max_rel = 0.0; };

static ErrStats compare_mats(const std::vector<float>& A,
                             const std::vector<float>& B,
                             int R, int C, double eps = 1e-12)
{
    ErrStats s;
    const size_t n = (size_t)R * C;
    for (size_t i = 0; i < n; ++i) {
        double a = A[i], b = B[i];
        double abs_err = std::abs(a - b);
        double denom = std::max(std::abs(a), std::abs(b));
        double rel_err = abs_err / (denom > eps ? denom : 1.0);
        s.max_abs = std::max(s.max_abs, abs_err);
        s.max_rel = std::max(s.max_rel, rel_err);
    }
    return s;
}

// 将 device 数据复制到 host
static void d2h(void* dst, const void* src, size_t bytes) {
    ASSERT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
}
static void h2d(void* dst, const void* src, size_t bytes) {
    ASSERT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), cudaSuccess);
}

// ===== 正确性测试 =====

TEST(SpMM_Correctness, CPU_vs_GPU_SmallSizes)
{
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device, skip GPU tests.";
    }

    const double atol = 1e-5, rtol = 1e-4;

    struct Case { int M, K, N; float density; };
    std::vector<Case> cases = {
        {8,   16, 4,  0.25f},
        {64,  96, 32, 0.05f},
        {128, 128,64, 0.02f},
    };

    for (auto cs : cases) {
        int M = cs.M, K = cs.K, N = cs.N;
        float density = cs.density;

        std::vector<int> row_ptr, col_ind;
        std::vector<float> values;
        make_random_csr(M, K, density, row_ptr, col_ind, values, /*seed*/ 42);

        std::vector<float> B;
        make_random_dense(K, N, B, /*seed*/ 777);

        // 先测试 beta = 0, alpha = 1
        float alpha = 1.0f, beta = 0.0f;

        // CPU baseline
        std::vector<float> C_cpu((size_t)M * N, 0.0f);
        spmm::spmm_csr_cpu(M, K, N,
                           row_ptr.data(), col_ind.data(), values.data(),
                           B.data(), N,
                           C_cpu.data(), N,
                           alpha, beta);

        // GPU
        int nnz = (int)col_ind.size();
        int *d_row_ptr = nullptr, *d_col_ind = nullptr;
        float *d_values = nullptr, *d_B = nullptr, *d_C = nullptr;

        ASSERT_EQ(cudaMalloc(&d_row_ptr, (size_t)(M + 1) * sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_col_ind, (size_t)nnz * sizeof(int)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_values,  (size_t)nnz * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_B,       (size_t)K * N * sizeof(float)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_C,       (size_t)M * N * sizeof(float)), cudaSuccess);

        h2d(d_row_ptr, row_ptr.data(), (size_t)(M + 1) * sizeof(int));
        h2d(d_col_ind, col_ind.data(), (size_t)nnz * sizeof(int));
        h2d(d_values,  values.data(),  (size_t)nnz * sizeof(float));
        h2d(d_B,       B.data(),       (size_t)K * N * sizeof(float));
        ASSERT_EQ(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)), cudaSuccess);

        ASSERT_EQ(spmm::spmm_csr_gpu(M, K, N,
                                     d_row_ptr, d_col_ind, d_values,
                                     d_B, N,
                                     d_C, N,
                                     alpha, beta,
                                     256), 0);

        std::vector<float> C_gpu((size_t)M * N);
        d2h(C_gpu.data(), d_C, (size_t)M * N * sizeof(float));

        // compare
        auto s = compare_mats(C_cpu, C_gpu, M, N);
        EXPECT_LE(s.max_abs, atol) << "max_abs=" << s.max_abs;
        EXPECT_LE(s.max_rel, rtol) << "max_rel=" << s.max_rel;

        // 再测试随机 alpha/beta
        alpha = 1.3f; beta = 0.7f;

        // 先给 C 随机初值（验证 beta*C 路径）
        std::vector<float> C_init((size_t)M * N);
        make_random_dense(M, N, C_init, /*seed*/ 999);

        std::vector<float> C_cpu2 = C_init;
        spmm::spmm_csr_cpu(M, K, N,
                           row_ptr.data(), col_ind.data(), values.data(),
                           B.data(), N,
                           C_cpu2.data(), N,
                           alpha, beta);

        h2d(d_C, C_init.data(), (size_t)M * N * sizeof(float));
        ASSERT_EQ(spmm::spmm_csr_gpu(M, K, N,
                                     d_row_ptr, d_col_ind, d_values,
                                     d_B, N,
                                     d_C, N,
                                     alpha, beta,
                                     256), 0);
        d2h(C_gpu.data(), d_C, (size_t)M * N * sizeof(float));

        s = compare_mats(C_cpu2, C_gpu, M, N);
        EXPECT_LE(s.max_abs, atol) << "max_abs=" << s.max_abs;
        EXPECT_LE(s.max_rel, rtol) << "max_rel=" << s.max_rel;

        cudaFree(d_row_ptr);
        cudaFree(d_col_ind);
        cudaFree(d_values);
        cudaFree(d_B);
        cudaFree(d_C);
    }
}

// ===== 性能测试（示例）=====
// 注意：性能与编译选项、数据分布、硬件环境紧密相关。这里只给一个可运行模板。
// 你可在 CTest 中将其标记为长时测试，或使用环境变量控制是否运行。

TEST(SpMM_Perf, CPU_GPU_Timing)
{
    if (!has_cuda_device()) {
        GTEST_SKIP() << "No CUDA device, skip GPU perf test.";
    }

    // 规模与稀疏率可自行调整
    const int M = 2048, K = 4096, N = 128;
    const float density = 0.005f;  // ~0.5% 稀疏度
    const int warmup = 5;
    const int iters  = 20;

    std::vector<int> row_ptr, col_ind;
    std::vector<float> values, B, C_cpu((size_t)M * N, 0.0f), C_gpu_host((size_t)M * N, 0.0f);
    make_random_csr(M, K, density, row_ptr, col_ind, values, /*seed*/ 2025);
    make_random_dense(K, N, B, /*seed*/ 7);

    // CPU 计时
    for (int i = 0; i < warmup; ++i) {
        std::fill(C_cpu.begin(), C_cpu.end(), 0.0f);
        spmm::spmm_csr_cpu(M, K, N,
                           row_ptr.data(), col_ind.data(), values.data(),
                           B.data(), N,
                           C_cpu.data(), N, 1.0f, 0.0f);
    }
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        std::fill(C_cpu.begin(), C_cpu.end(), 0.0f);
        spmm::spmm_csr_cpu(M, K, N,
                           row_ptr.data(), col_ind.data(), values.data(),
                           B.data(), N,
                           C_cpu.data(), N, 1.0f, 0.0f);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    // GPU 准备
    int nnz = (int)col_ind.size();
    int *d_row_ptr = nullptr, *d_col_ind = nullptr;
    float *d_values = nullptr, *d_B = nullptr, *d_C = nullptr;
    ASSERT_EQ(cudaMalloc(&d_row_ptr, (size_t)(M + 1) * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_col_ind, (size_t)nnz * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_values,  (size_t)nnz * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_B,       (size_t)K * N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_C,       (size_t)M * N * sizeof(float)), cudaSuccess);

    h2d(d_row_ptr, row_ptr.data(), (size_t)(M + 1) * sizeof(int));
    h2d(d_col_ind, col_ind.data(), (size_t)nnz * sizeof(int));
    h2d(d_values,  values.data(),  (size_t)nnz * sizeof(float));
    h2d(d_B,       B.data(),       (size_t)K * N * sizeof(float));
    ASSERT_EQ(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)), cudaSuccess);

    // GPU 计时（CUDA events）
    cudaEvent_t ev_start, ev_stop;
    ASSERT_EQ(cudaEventCreate(&ev_start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&ev_stop),  cudaSuccess);

    // warmup
    for (int i = 0; i < warmup; ++i) {
        ASSERT_EQ(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)), cudaSuccess);
        ASSERT_EQ(spmm::spmm_csr_gpu(M, K, N,
                                     d_row_ptr, d_col_ind, d_values,
                                     d_B, N,
                                     d_C, N,
                                     1.0f, 0.0f,
                                     256), 0);
    }

    float gpu_ms = 0.0f;
    ASSERT_EQ(cudaEventRecord(ev_start), cudaSuccess);
    for (int i = 0; i < iters; ++i) {
        ASSERT_EQ(cudaMemset(d_C, 0, (size_t)M * N * sizeof(float)), cudaSuccess);
        ASSERT_EQ(spmm::spmm_csr_gpu(M, K, N,
                                     d_row_ptr, d_col_ind, d_values,
                                     d_B, N,
                                     d_C, N,
                                     1.0f, 0.0f,
                                     256), 0);
    }
    ASSERT_EQ(cudaEventRecord(ev_stop), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(ev_stop), cudaSuccess);
    ASSERT_EQ(cudaEventElapsedTime(&gpu_ms, ev_start, ev_stop), cudaSuccess);
    gpu_ms /= iters;

    // 打印（非断言）
    std::cout << "[Perf] CPU avg: " << cpu_ms << " ms, GPU avg: " << gpu_ms << " ms\n";

    // 正确性 sanity：取一次 GPU 结果与 CPU 对比
    d2h(C_gpu_host.data(), d_C, (size_t)M * N * sizeof(float));
    auto s = compare_mats(C_cpu, C_gpu_host, M, N);
    EXPECT_LE(s.max_abs, 1e-4) << "max_abs=" << s.max_abs;
    EXPECT_LE(s.max_rel, 1e-3) << "max_rel=" << s.max_rel;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_B);
    cudaFree(d_C);
}

// ===== main（可选）=====
// 若你的测试框架已提供 gtest_main，可移除此 main。
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
