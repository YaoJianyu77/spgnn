include(FetchContent)

# ===== GoogleTest =====
# 你也可以改为用系统包管理器安装的 GTest
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ===== 可选第三方库（仅用于对比/扩展） =====
# Eigen
if(USE_EIGEN)
  find_package(Eigen3 3.3 QUIET)
  if(NOT Eigen3_FOUND)
    message(WARNING "USE_EIGEN=ON but Eigen3 not found; tests will still build without Eigen.")
  endif()
endif()

# MKL（如果你装了 oneAPI MKL，一般可用 MKLConfig）
if(USE_MKL)
  find_package(MKL QUIET)
  if(NOT MKL_FOUND)
    message(WARNING "USE_MKL=ON but MKL not found; tests will still build without MKL.")
  endif()
endif()

# cuSPARSE（来自 CUDAToolkit）
if(USE_CUSPARSE AND ENABLE_CUDA)
  if(TARGET CUDA::cusparse)
    # ok
  else()
    message(WARNING "USE_CUSPARSE=ON but CUDA::cusparse not found; turn off USE_CUSPARSE or upgrade CUDA.")
  endif()
endif()

# ===== 测试目标 =====
add_executable(spmm_test
  ${PROJECT_SOURCE_DIR}/test/spmm_test.cpp
)

target_link_libraries(spmm_test
  PRIVATE
    gtest_main
    spmm_cpu
)

# GPU & CUDA 运行时
if(ENABLE_CUDA)
  target_link_libraries(spmm_test PRIVATE spmm_gpu CUDA::cudart)
endif()

# 条件性对比库
if(USE_CUSPARSE AND ENABLE_CUDA AND TARGET CUDA::cusparse)
  target_link_libraries(spmm_test PRIVATE CUDA::cusparse)
  target_compile_definitions(spmm_test PRIVATE USE_CUSPARSE=1)
endif()

if(USE_EIGEN AND Eigen3_FOUND)
  target_link_libraries(spmm_test PRIVATE Eigen3::Eigen)
  target_compile_definitions(spmm_test PRIVATE USE_EIGEN=1)
endif()

if(USE_MKL AND MKL_FOUND)
  target_link_libraries(spmm_test PRIVATE MKL::MKL)
  target_compile_definitions(spmm_test PRIVATE USE_MKL=1)
endif()

add_test(NAME spmm_test COMMAND spmm_test)
