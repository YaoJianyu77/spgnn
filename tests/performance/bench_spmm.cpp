#include <gtest/gtest.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include "spmm.hpp"
#include "../common/test_utils.hpp"
#include "../common/timers.hpp"

// 轻量 JSON 读取：可用最简单行扫描（为示例简化），工程中可换 nlohmann/json
struct Case { int M,K,N; float density; int iters; };

static std::vector<Case> load_cases(const std::string& path){
    std::vector<Case> v;
    std::ifstream ifs(path); if(!ifs) return v;
    std::string s((std::istreambuf_iterator<char>(ifs)), {});
    // 非健壮解析（仅示例），建议换真正 JSON 解析库
    size_t pos=0;
    while ((pos=s.find("{",pos))!=std::string::npos) {
        Case c{}; 
        auto pick=[&](const char* key)->double{
            auto k=s.find(key,pos); if(k==std::string::npos) return 0;
            auto colon=s.find(":",k); auto end=s.find_first_of(",}",colon);
            return std::stod(s.substr(colon+1,end-colon-1));
        };
        
        c.M=(int)pick("\"M\""); c.K=(int)pick("\"K\""); c.N=(int)pick("\"N\"");
        c.density=(float)pick("\"density\""); c.iters=(int)pick("\"iters\"");
        v.push_back(c); pos++;
    }
    return v;
}

TEST(Perf, BatchCSV)
{
    if (!tutil::has_cuda()) GTEST_SKIP() << "No CUDA device.";

    auto cases = load_cases(std::string(TEST_SOURCE_DIR)+"/performance/cases.json");
    ASSERT_FALSE(cases.empty()) << "No cases loaded.";

    std::ofstream csv("spmm_bench.csv");
    csv << "M,K,N,density,nnz,iters,cpu_ms,gpu_ms,max_abs,max_rel\n";

    for (auto cs : cases) {
        int M=cs.M,K=cs.K,N=cs.N; float dens=cs.density; int iters=cs.iters;

        std::vector<int> rp,ci; std::vector<float> vv,B,Ccpu((size_t)M*N), Cgpu((size_t)M*N);
        tutil::make_random_csr(M,K,dens,rp,ci,vv,123);
        tutil::make_random_dense(K,N,B,7);
        int nnz=(int)ci.size();

        // CPU 计时
        tutil::CpuTimer ct; double acc_cpu=0;
        for(int i=0;i<3;++i){ // warmup
            std::fill(Ccpu.begin(),Ccpu.end(),0.f);
            spmm::spmm_csr_cpu(M,K,N, rp.data(),ci.data(),vv.data(), B.data(),N, Ccpu.data(),N, 1.f,0.f);
        }
        for(int i=0;i<iters;++i){
            std::fill(Ccpu.begin(),Ccpu.end(),0.f);
            ct.start();
            spmm::spmm_csr_cpu(M,K,N, rp.data(),ci.data(),vv.data(), B.data(),N, Ccpu.data(),N, 1.f,0.f);
            acc_cpu += ct.stop_ms();
        }
        double cpu_ms = acc_cpu / iters;

        // GPU 计时
        int *d_rp=(int*)tutil::dmalloc((M+1)*sizeof(int));
        int *d_ci=(int*)tutil::dmalloc(nnz*sizeof(int));
        float *d_vv=(float*)tutil::dmalloc(nnz*sizeof(float));
        float *d_B=(float*)tutil::dmalloc((size_t)K*N*sizeof(float));
        float *d_C=(float*)tutil::dmalloc((size_t)M*N*sizeof(float));
        tutil::h2d(d_rp,rp.data(),(M+1)*sizeof(int));
        tutil::h2d(d_ci,ci.data(),nnz*sizeof(int));
        tutil::h2d(d_vv,vv.data(),nnz*sizeof(float));
        tutil::h2d(d_B,B.data(),(size_t)K*N*sizeof(float));
        cudaMemset(d_C,0,(size_t)M*N*sizeof(float));

        tutil::GpuTimer gt; float acc_gpu=0;
        for(int i=0;i<3;++i){
            cudaMemset(d_C,0,(size_t)M*N*sizeof(float));
            spmm::spmm_csr_gpu(M,K,N, d_rp,d_ci,d_vv, d_B,N, d_C,N, 1.f,0.f, 256);
        }
        for(int i=0;i<iters;++i){
            cudaMemset(d_C,0,(size_t)M*N*sizeof(float));
            gt.start();
            spmm::spmm_csr_gpu(M,K,N, d_rp,d_ci,d_vv, d_B,N, d_C,N, 1.f,0.f, 256);
            acc_gpu += gt.stop_ms();
        }
        double gpu_ms = acc_gpu / iters;

        tutil::d2h(Cgpu.data(), d_C, (size_t)M*N*sizeof(float));
        auto s = tutil::compare(Ccpu, Cgpu, M, N);

        csv << M<<","<<K<<","<<N<<","<<dens<<","<<nnz<<","<<iters<<","
            << cpu_ms << "," << gpu_ms << ","
            << s.max_abs << "," << s.max_rel << "\n";

        cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_vv); cudaFree(d_B); cudaFree(d_C);
    }

    std::cout << "[perf] wrote spmm_bench.csv\n";
}
