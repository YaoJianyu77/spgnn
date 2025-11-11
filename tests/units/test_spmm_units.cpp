#include <gtest/gtest.h>
#include "spmm.hpp"
#include "../common/test_utils.hpp"

TEST(Unit, SmallFixed_alpha_beta)
{
    // A: 3x4 CSR
    std::vector<int> rp = {0,2,4,6};
    std::vector<int> ci = {0,2,2,3,0,3};
    std::vector<float> vv= {1,2,3,4,5,6};
    int M=3,K=4,N=2;

    std::vector<float> B = {1,2, 3,4, 5,6, 7,8}; // (K×N)
    std::vector<float> C(M*N, 0.f);

    spmm::spmm_csr_cpu(M,K,N, rp.data(),ci.data(),vv.data(), B.data(),N, C.data(),N, 1.f,0.f);

    std::vector<float> exp = {11,14, 43,50, 47,52};
    auto s = tutil::compare(C, exp, M, N);
    EXPECT_LE(s.max_abs, 1e-6);
    EXPECT_LE(s.max_rel, 1e-6);

    // alpha/beta 路径
    std::vector<float> Cold = C;
    spmm::spmm_csr_cpu(M,K,N, rp.data(),ci.data(),vv.data(), B.data(),N, C.data(),N, 1.5f,0.5f);
    for (int i=0;i<M*N;++i){
        float want = 1.5f*exp[i] + 0.5f*Cold[i];
        EXPECT_NEAR(C[i], want, 1e-5);
    }
}

TEST(Unit, RandomTiny_CPU_GPU_match)
{
    if (!tutil::has_cuda()) GTEST_SKIP() << "No CUDA device.";

    int M=32,K=48,N=16; float dens=0.1f;
    std::vector<int> rp,ci; std::vector<float> vv,B,Ccpu(M*N,0.f), Cgpu(M*N,0.f);
    tutil::make_random_csr(M,K,dens,rp,ci,vv,123);
    tutil::make_random_dense(K,N,B,7);

    spmm::spmm_csr_cpu(M,K,N, rp.data(),ci.data(),vv.data(), B.data(),N, Ccpu.data(),N, 1.f,0.f);

    int nnz=(int)ci.size();
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

    ASSERT_EQ(spmm::spmm_csr_gpu(M,K,N, d_rp,d_ci,d_vv, d_B,N, d_C,N, 1.f,0.f, 256), 0);
    tutil::d2h(Cgpu.data(), d_C, (size_t)M*N*sizeof(float));

    auto s = tutil::compare(Ccpu, Cgpu, M, N);
    EXPECT_LE(s.max_abs, 1e-4);
    EXPECT_LE(s.max_rel, 1e-3);

    cudaFree(d_rp); cudaFree(d_ci); cudaFree(d_vv); cudaFree(d_B); cudaFree(d_C);
}
