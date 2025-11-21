// benchmarks/backends/our_spmm.hpp
#pragma once
#include "backend_base.hpp"
#include "spmm.hpp"  // 你自己的接口

class OurSpMMBackend : public ISpMMBackend {
public:
    std::string name() const override { return "ours_cpu"; }

    void prepare(const CSRMatrix& A, const DenseMatrix& B) override {
        // 构建你需要的内部格式
    }

    BackendResult run(const BackendConfig& cfg) override {
        // warmup + timing 调用你的 spmm_cpu/spmm_gpu
    }
};
