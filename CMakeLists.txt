cmake_minimum_required(VERSION 3.20)

project(SpMM LANGUAGES CXX)

# ======= config =======
option(ENABLE_GPU     "Build GPU backend"                ON)
option(ENABLE_CPU     "Build CPU backend"                 ON)
option(ENABLE_OPENMP   "Enable OpenMP for CPU backend"     OFF)
option(SPMM_LIB_KIND_STATIC "Build static libs (otherwise shared)" ON)
option(ENABLE_TESTS    "Build tests (GoogleTest)"          ON)
option(ENABLE_BENCH    "Build benchmarks"                  OFF) # 预留
option(USE_CUSPARSE    "Enable cuSPARSE in tests"          OFF)
option(USE_EIGEN       "Enable Eigen in tests"             OFF)
option(USE_MKL         "Enable MKL in tests"               OFF)

# library kind(STATIC/SHARED)
if(SPMM_LIB_KIND_STATIC)
  set(SPMM_LIB_KIND STATIC)
else()
  set(SPMM_LIB_KIND SHARED)
endif()

# 可传入 GPU 架构（默认覆盖常见数据中心卡）
if(ENABLE_GPU AND NOT DEFINED SPMM_CUDA_ARCHS)
  find_program(NVIDIA_SMI nvidia-smi)
  if(NVIDIA_SMI)
    execute_process(
      COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
      OUTPUT_VARIABLE _caps OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # _caps 可能是多行（多卡），取第一行
    string(REGEX MATCH "^[0-9]+\\.[0-9]+" _cap "${_caps}")
    if(_cap)
      string(REPLACE "." "" _cap_short "${_cap}") # 8.0 -> 80
      message(STATUS "Auto-detected GPU compute capability: ${_cap} -> arch ${_cap_short}")
      set(SPMM_CUDA_ARCHS "${_cap_short}" CACHE STRING "" FORCE)
    else()
      set(SPMM_CUDA_ARCHS "80;90" CACHE STRING "" FORCE)
    endif()
  else()
    set(SPMM_CUDA_ARCHS "80;90" CACHE STRING "" FORCE)
  endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES "${SPMM_CUDA_ARCHS}")

# C++ 标准与通用设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # 便于将来做 Python 扩展或安装
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)    # 方便 IDE/静态分析

# 如果用户没有指定构建类型，默认使用 RelWithDebInfo (高性能 + 可调试)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "注意: 未指定构建类型，默认使用 RelWithDebInfo 模式。")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

# CUDA 语言按需启用
if(ENABLE_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)  # 提供 CUDA::cudart / CUDA::cusparse
endif()

# 子目录
add_subdirectory(src)

if(ENABLE_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

# 预留：基准测试/工具
# if(ENABLE_BENCH AND EXISTS "${PROJECT_SOURCE_DIR}/bench/CMakeLists.txt")
#   add_subdirectory(bench)
# endif()

# # 安装导出（便于下游 find_package）
# include(GNUInstallDirs)

# install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install(EXPORT spmmTargets
#         NAMESPACE spmm::
#         DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spmm)

# # 生成一个最简的 package config（可根据需要再完善）
# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#   "${PROJECT_BINARY_DIR}/SpMMConfigVersion.cmake"
#   VERSION 0.1.0
#   COMPATIBILITY AnyNewerVersion
# )
# configure_file(
#   "${PROJECT_SOURCE_DIR}/cmake/SpMMConfig.cmake.in"
#   "${PROJECT_BINARY_DIR}/SpMMConfig.cmake"
#   @ONLY
# )
# install(FILES
#   "${PROJECT_BINARY_DIR}/SpMMConfig.cmake"
#   "${PROJECT_BINARY_DIR}/SpMMConfigVersion.cmake"
#   DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/spmm
# )