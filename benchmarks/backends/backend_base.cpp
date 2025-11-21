// benchmarks/backends/backend_base.hpp
#pragma once
#include <string>

struct CSRMatrix { /* ... */ };
struct DenseMatrix { /* ... */ };

struct BackendConfig {
    int warmup_iters = 5;
    int iters = 50;
};

struct BackendResult {
    double avg_ms;
    // 也可以加 GFLOPS, bandwidth 等
};

class ISpMMBackend {
public:
    virtual ~ISpMMBackend() = default;
    virtual std::string name() const = 0;
    virtual void prepare(const CSRMatrix& A, const DenseMatrix& B) = 0;
    virtual BackendResult run(const BackendConfig& cfg) = 0;
};
