// spgemm_gh200.cu
// cuSPARSE SpGEMM (CSR32) wrapper for GH200/H100 (cc>=8.0), with metrics.
// Call order (Generic API):
//   workEstimation(query)->workEstimation(exec)->compute(query)->compute(exec)->GetSize->alloc C->set C pointers->copy
//
// This is the "simple/original" version: it trusts cusparseSpMatGetSize() for nnz(C).

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                   __FILE__, __LINE__, cudaGetErrorString(_e));               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static inline void CUSPARSE_CHECK(cusparseStatus_t s, const char* file, int line) {
  if (s != CUSPARSE_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuSPARSE error %s:%d: status=%d\n", file, line, (int)s);
    std::exit(1);
  }
}
#define CUSPARSE_CALL(expr) CUSPARSE_CHECK((expr), __FILE__, __LINE__)

template <typename T>
struct CudaTypeTraits;
template <> struct CudaTypeTraits<float>  { static constexpr cudaDataType value_type = CUDA_R_32F; };
template <> struct CudaTypeTraits<double> { static constexpr cudaDataType value_type = CUDA_R_64F; };

enum class MemoryMode {
  Device,   // cudaMalloc
  Managed   // cudaMallocManaged
};

static inline const char* memmode_str(MemoryMode m) {
  return (m == MemoryMode::Device) ? "device" : "managed";
}

static inline double wall_ms_now() {
  using clock = std::chrono::high_resolution_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static inline void mem_get_info(size_t& free_b, size_t& total_b) {
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
}

template <class Fn>
static inline float elapsed_ms_cuda_event(cudaStream_t stream, Fn&& fn) {
  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s, stream));
  fn();
  CUDA_CHECK(cudaEventRecord(e, stream));
  CUDA_CHECK(cudaEventSynchronize(e));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return ms;
}

static void* xmalloc_cuda(size_t bytes, MemoryMode mode) {
  void* p = nullptr;
  if (bytes == 0) return nullptr;
  if (mode == MemoryMode::Device) {
    CUDA_CHECK(cudaMalloc(&p, bytes));
  } else {
    CUDA_CHECK(cudaMallocManaged(&p, bytes, cudaMemAttachGlobal));
  }
  return p;
}

static void uvm_prefetch_and_advise(void* p, size_t bytes, int device, cudaStream_t stream) {
  if (!p || bytes == 0) return;
  CUDA_CHECK(cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, device));
  CUDA_CHECK(cudaMemAdvise(p, bytes, cudaMemAdviseSetAccessedBy, device));
  CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, device, stream));
}

template <typename T>
struct Csr32 {
  int rows = 0, cols = 0;
  int nnz  = 0;
  int* row_offsets = nullptr; // rows+1
  int* col_indices = nullptr; // nnz
  T*   values      = nullptr; // nnz
  MemoryMode mem = MemoryMode::Device;
};

template <typename T>
static void csr_alloc(Csr32<T>& M, int rows, int cols, int nnz, MemoryMode mode) {
  M.rows = rows; M.cols = cols; M.nnz = nnz; M.mem = mode;
  M.row_offsets = (int*)xmalloc_cuda((size_t)(rows + 1) * sizeof(int), mode);
  if (nnz > 0) {
    M.col_indices = (int*)xmalloc_cuda((size_t)nnz * sizeof(int), mode);
    M.values      = (T*)  xmalloc_cuda((size_t)nnz * sizeof(T),   mode);
  } else {
    M.col_indices = nullptr;
    M.values      = nullptr;
  }
}

template <typename T>
static void csr_free(Csr32<T>& M) {
  if (M.row_offsets) CUDA_CHECK(cudaFree(M.row_offsets));
  if (M.col_indices) CUDA_CHECK(cudaFree(M.col_indices));
  if (M.values)      CUDA_CHECK(cudaFree(M.values));
  M = Csr32<T>{};
}

struct SpGemmCallMetrics {
  // sizes
  int m=0,k=0,n=0;
  int nnzA=0, nnzB=0, nnzC=0;

  // algorithm + workspace
  int alg = -1;
  size_t bufferSize1=0, bufferSize2=0;

  // timing (ms)
  float t_desc_ms=0;
  float t_malloc_ws_ms=0;
  float t_work_est_ms=0;
  float t_compute_ms=0;
  float t_malloc_out_ms=0;
  float t_copy_ms=0;
  float t_total_gpu_ms=0;
  float t_wall_total_ms=0;

  // mem sampling
  size_t mem_total=0;
  size_t mem_free_begin=0;
  size_t mem_free_min=0;
  size_t mem_free_end=0;

  // status
  int status = 0; // cusparseStatus_t as int
};

static inline void mem_sample_min(size_t& free_min, size_t& total_out) {
  size_t free_b=0,total_b=0;
  mem_get_info(free_b,total_b);
  total_out = total_b;
  free_min = std::min(free_min, free_b);
}

template <typename T>
static cusparseStatus_t try_spgemm_once(
    cusparseHandle_t handle,
    int m, int k, int n,
    int nnzA, const int* A_row, const int* A_col, const T* A_val,
    int nnzB, const int* B_row, const int* B_col, const T* B_val,
    Csr32<T>& C_out,
    cudaStream_t stream,
    MemoryMode out_mode,
    cusparseSpGEMMAlg_t alg,
    SpGemmCallMetrics* met)
{
  cusparseStatus_t st = CUSPARSE_STATUS_SUCCESS;

  cusparseSpMatDescr_t matA = nullptr;
  cusparseSpMatDescr_t matB = nullptr;
  cusparseSpMatDescr_t matC = nullptr;
  cusparseSpGEMMDescr_t spgemmDescr = nullptr;

  size_t bufferSize1 = 0, bufferSize2 = 0;
  void* dBuffer1 = nullptr;
  void* dBuffer2 = nullptr;

  int64_t C_rows = 0, C_cols = 0, C_nnz64 = 0;

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));

  T alpha = (T)1;
  T beta  = (T)0;

  double wall0 = wall_ms_now();

  size_t free0=0,total0=0;
  mem_get_info(free0,total0);
  size_t free_min = free0;

  auto cleanup = [&]() {
    if (dBuffer1) CUDA_CHECK(cudaFree(dBuffer1));
    if (dBuffer2) CUDA_CHECK(cudaFree(dBuffer2));
    if (spgemmDescr) cusparseSpGEMM_destroyDescr(spgemmDescr);
    if (matA) cusparseDestroySpMat(matA);
    if (matB) cusparseDestroySpMat(matB);
    if (matC) cusparseDestroySpMat(matC);
  };

  double t_desc0 = wall_ms_now();

  st = cusparseCreateCsr(&matA,
                         (int64_t)m, (int64_t)k, (int64_t)nnzA,
                         (void*)A_row, (void*)A_col, (void*)A_val,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO,
                         CudaTypeTraits<T>::value_type);
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }

  st = cusparseCreateCsr(&matB,
                         (int64_t)k, (int64_t)n, (int64_t)nnzB,
                         (void*)B_row, (void*)B_col, (void*)B_val,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO,
                         CudaTypeTraits<T>::value_type);
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }

  // IMPORTANT: matC pointers MUST be nullptr during workEstimation/compute
  st = cusparseCreateCsr(&matC,
                         (int64_t)m, (int64_t)n, (int64_t)0,
                         nullptr, nullptr, nullptr,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO,
                         CudaTypeTraits<T>::value_type);
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }

  st = cusparseSpGEMM_createDescr(&spgemmDescr);
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }

  double t_desc1 = wall_ms_now();
  mem_sample_min(free_min, total0);

  // workEstimation(query)
  st = cusparseSpGEMM_workEstimation(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC,
                                    CudaTypeTraits<T>::value_type,
                                    alg, spgemmDescr,
                                    &bufferSize1, nullptr);
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }

  double t_mws0 = wall_ms_now();
  if (bufferSize1 > 0) CUDA_CHECK(cudaMalloc(&dBuffer1, bufferSize1));
  double t_mws1 = wall_ms_now();
  mem_sample_min(free_min, total0);

  // workEstimation(exec)
  float t_work_est = elapsed_ms_cuda_event(stream, [&](){
    cusparseStatus_t s2 = cusparseSpGEMM_workEstimation(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC,
                                    CudaTypeTraits<T>::value_type,
                                    alg, spgemmDescr,
                                    &bufferSize1, dBuffer1);
    if (s2 != CUSPARSE_STATUS_SUCCESS) st = s2;
  });
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }
  mem_sample_min(free_min, total0);

  // compute(query)
  st = cusparseSpGEMM_compute(handle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, matA, matB, &beta, matC,
                             CudaTypeTraits<T>::value_type,
                             alg, spgemmDescr,
                             &bufferSize2, nullptr);
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }

  double t_mws2 = wall_ms_now();
  if (bufferSize2 > 0) CUDA_CHECK(cudaMalloc(&dBuffer2, bufferSize2));
  double t_mws3 = wall_ms_now();
  mem_sample_min(free_min, total0);

  // compute(exec)
  float t_compute = elapsed_ms_cuda_event(stream, [&](){
    cusparseStatus_t s2 = cusparseSpGEMM_compute(handle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, matA, matB, &beta, matC,
                             CudaTypeTraits<T>::value_type,
                             alg, spgemmDescr,
                             &bufferSize2, dBuffer2);
    if (s2 != CUSPARSE_STATUS_SUCCESS) st = s2;
  });
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }
  mem_sample_min(free_min, total0);

  // GetSize
  st = cusparseSpMatGetSize(matC, &C_rows, &C_cols, &C_nnz64);
  if (st != CUSPARSE_STATUS_SUCCESS) { cleanup(); return st; }
  if (C_rows != m || C_cols != n || C_nnz64 > INT32_MAX) {
    cleanup();
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  }

  // allocate output
  double t_mout0 = wall_ms_now();
  csr_alloc(C_out, m, n, (int)C_nnz64, out_mode);
  double t_mout1 = wall_ms_now();
  mem_sample_min(free_min, total0);

  if (out_mode == MemoryMode::Managed) {
    uvm_prefetch_and_advise(C_out.row_offsets, (size_t)(m+1)*sizeof(int), dev, stream);
    if (C_out.nnz > 0) {
      uvm_prefetch_and_advise(C_out.col_indices, (size_t)C_out.nnz*sizeof(int), dev, stream);
      uvm_prefetch_and_advise(C_out.values,      (size_t)C_out.nnz*sizeof(T),   dev, stream);
    }
  }

  // bind pointers ONLY before copy
  st = cusparseCsrSetPointers(matC, C_out.row_offsets, C_out.col_indices, C_out.values);
  if (st != CUSPARSE_STATUS_SUCCESS) {
    csr_free(C_out);
    cleanup();
    return st;
  }

  // copy
  float t_copy = elapsed_ms_cuda_event(stream, [&](){
    cusparseStatus_t s2 = cusparseSpGEMM_copy(handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matA, matB, &beta, matC,
                          CudaTypeTraits<T>::value_type,
                          alg, spgemmDescr);
    if (s2 != CUSPARSE_STATUS_SUCCESS) st = s2;
  });

  if (st != CUSPARSE_STATUS_SUCCESS) {
    csr_free(C_out);
    cleanup();
    return st;
  }

  cleanup();

  double wall1 = wall_ms_now();
  size_t free1=0,total1=0;
  mem_get_info(free1,total1);
  free_min = std::min(free_min, free1);

  if (met) {
    met->m=m; met->k=k; met->n=n;
    met->nnzA=nnzA; met->nnzB=nnzB; met->nnzC=(int)C_nnz64;

    met->alg = (int)alg;
    met->bufferSize1 = bufferSize1;
    met->bufferSize2 = bufferSize2;

    met->t_desc_ms = (float)(t_desc1 - t_desc0);
    met->t_malloc_ws_ms = (float)((t_mws1 - t_mws0) + (t_mws3 - t_mws2));
    met->t_work_est_ms = t_work_est;
    met->t_compute_ms = t_compute;
    met->t_malloc_out_ms = (float)(t_mout1 - t_mout0);
    met->t_copy_ms = t_copy;
    met->t_total_gpu_ms = t_work_est + t_compute + t_copy;
    met->t_wall_total_ms = (float)(wall1 - wall0);

    met->mem_total = total1;
    met->mem_free_begin = free0;
    met->mem_free_min = free_min;
    met->mem_free_end = free1;
    met->status = (int)st;
  }

  return st;
}

template <typename T>
cusparseStatus_t spgemm_csr_cusparse_alloc_ex(
    cusparseHandle_t handle,
    int m, int k, int n,
    int nnzA, const int* A_row, const int* A_col, const T* A_val,
    int nnzB, const int* B_row, const int* B_col, const T* B_val,
    Csr32<T>& C,
    cudaStream_t stream,
    MemoryMode out_mode,
    SpGemmCallMetrics* met)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "This supports float/double.");

  CUSPARSE_CALL(cusparseSetStream(handle, stream));
  CUSPARSE_CALL(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

  const cusparseSpGEMMAlg_t algs[] = {
    CUSPARSE_SPGEMM_DEFAULT,
    CUSPARSE_SPGEMM_ALG1,
    CUSPARSE_SPGEMM_ALG2,
    CUSPARSE_SPGEMM_ALG3,
  };

  cusparseStatus_t last = CUSPARSE_STATUS_EXECUTION_FAILED;

  for (auto alg : algs) {
    if (C.row_offsets || C.col_indices || C.values) csr_free(C);
    last = try_spgemm_once<T>(handle, m, k, n,
                              nnzA, A_row, A_col, A_val,
                              nnzB, B_row, B_col, B_val,
                              C, stream, out_mode, alg, met);
    if (last == CUSPARSE_STATUS_SUCCESS) return last;
  }
  return last;
}