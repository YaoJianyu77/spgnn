#pragma once
#include <chrono>
#include <cuda_runtime.h>

namespace tutil {

struct CpuTimer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    void start(){ t0 = clk::now(); }
    double stop_ms() const {
        auto t1 = clk::now();
        return std::chrono::duration<double,std::milli>(t1 - t0).count();
    }
};

struct GpuTimer {
    cudaEvent_t s{}, e{};
    GpuTimer(){ cudaEventCreate(&s); cudaEventCreate(&e); }
    ~GpuTimer(){ cudaEventDestroy(s); cudaEventDestroy(e); }
    void start(){ cudaEventRecord(s); }
    double stop_ms(){
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0; cudaEventElapsedTime(&ms, s, e); return (double)ms;
    }
};

} // namespace tutil
