// benchmarks/backends/projA_spmm.hpp
#pragma once
#include "backend_base.hpp"
#include "projA_api.hpp"  // 外部项目暴露的接口

class ProjASpMMBackend : public ISpMMBackend {
public:
    std::string name() const override { return "projA_cpu"; }
    // 同样写 prepare / run，内部调用 projA 的 API
};
