#pragma once
#include <vector>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace tutil {

inline bool has_cuda() {
    int c=0; return cudaGetDeviceCount(&c)==cudaSuccess && c>0;
}

struct ErrStats { double max_abs=0, max_rel=0; };

inline ErrStats compare(const std::vector<float>& A,
                        const std::vector<float>& B,
                        int R, int C, double eps=1e-12)
{
    ErrStats s;
    size_t n = (size_t)R*C;
    for (size_t i=0;i<n;++i) {
        double a=A[i], b=B[i];
        double ae=std::abs(a-b);
        double re=ae / std::max(std::max(std::abs(a), std::abs(b)), eps);
        if (ae > s.max_abs) s.max_abs = ae;
        if (re > s.max_rel) s.max_rel = re;
    }
    return s;
}

// 生成 CSR (M×K)，按 density 随机列（每行唯一）
inline void make_random_csr(int M,int K,float density,
                            std::vector<int>& rowptr,
                            std::vector<int>& colind,
                            std::vector<float>& vals,
                            uint64_t seed=42)
{
    std::mt19937_64 rng(seed);
    rowptr.assign(M+1,0);
    colind.clear(); vals.clear();

    int rnnz = std::max(1, std::min(K, int(std::round(density*K))));
    for (int i=0;i<M;++i) {
        std::unordered_set<int> used; used.reserve(rnnz*2+8);
        while ((int)used.size()<rnnz) used.insert(int(rng()%K));
        std::vector<int> cols(used.begin(), used.end());
        std::sort(cols.begin(), cols.end());
        for (int c: cols) { colind.push_back(c); }
        for (int t=0; t<(int)cols.size(); ++t) {
            // [-1,1]
            float v = float((rng()%20001)/10000.0 - 1.0);
            vals.push_back(v);
        }
        rowptr[i+1] = (int)colind.size();
    }
}

inline void make_random_dense(int R,int C,std::vector<float>& Mv,uint64_t seed=7){
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1,1);
    Mv.resize((size_t)R*C); for (auto& x: Mv) x=d(rng);
}

// 简易 CUDA 拷贝与事件封装（用于测试）
inline void* dmalloc(size_t bytes) { void* p=nullptr; cudaMalloc(&p, bytes); return p; }
inline void h2d(void* dst, const void* src, size_t b){ cudaMemcpy(dst,src,b,cudaMemcpyHostToDevice); }
inline void d2h(void* dst, const void* src, size_t b){ cudaMemcpy(dst,src,b,cudaMemcpyDeviceToHost); }

} // namespace tutil
