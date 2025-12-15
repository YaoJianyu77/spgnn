# CPU 
if(ENABLE_CPU)
  file(GLOB_RECURSE CPU_SOURCES ${PROJECT_SOURCE_DIR}/src/cpu/*.cpp)
  add_library(spmm_cpu ${SPMM_LIB_KIND}
    ${CPU_SOURCES}
  )
  target_include_directories(spmm_cpu
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )
  target_compile_features(spmm_cpu PUBLIC cxx_std_17)
  target_compile_options(spmm_cpu PRIVATE -O3 -march=native -mtune=native -Wall -Wextra)

# OpenMP
  if(ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
      target_link_libraries(spmm_cpu PUBLIC OpenMP::OpenMP_CXX)
      target_compile_definitions(spmm_cpu PUBLIC SPMM_USE_OPENMP=1)
  else()
      message(WARNING "ENABLE_OPENMP=ON but OpenMP not found; building without it.")
    endif()
  endif()
endif()

# GPU
if(ENABLE_GPU)
  # file(GLOB_RECURSE GPU_SOURCES ${PROJECT_SOURCE_DIR}/src/gpu/*.cu)
  # add_library(spmm_gpu ${SPMM_LIB_KIND}
  #   ${GPU_SOURCES}
  # )
  # target_include_directories(spmm_gpu
  #   PUBLIC
  #     $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  #     $<INSTALL_INTERFACE:include>
  # )
  # target_link_libraries(spmm_gpu PUBLIC CUDA::cudart)
  # set_target_properties(spmm_gpu PROPERTIES
  #   CUDA_SEPARABLE_COMPILATION ON
  #   CUDA_ARCHITECTURES "${SPMM_CUDA_ARCHS}"
  #   POSITION_INDEPENDENT_CODE ON
  # )
  # target_compile_features(spmm_gpu PUBLIC cxx_std_17)
  # target_compile_options(spmm_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
add_executable(spgemm_gpu
  spgemm.cu
)

# 等价于 nvcc -O3 -std=c++17
target_compile_features(spgemm_gpu PRIVATE cxx_std_17)
target_compile_options(spgemm_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# 架构：等价于 -gencode arch=compute_90,code=sm_90 和 compute_90
# CMake 用 CMAKE_CUDA_ARCHITECTURES 表达：
set_property(TARGET spgemm.cu PROPERTY CUDA_ARCHITECTURES 90)

# cuSPARSE
find_package(CUDAToolkit REQUIRED)
target_link_libraries(spgemm PRIVATE CUDA::cusparse)

# 你的 .cu 如果里面有纯 C++ 的 host 代码，也需要 separable compilation 时可打开
# set_target_properties(run_metapath_4hop_gh200 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# # install
# include(GNUInstallDirs)
# if(ENABLE_CPU)
#   install(TARGETS spmm_cpu
#     EXPORT spmmTargets
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#   )
# endif()
# if(ENABLE_GPU)
#   install(TARGETS spmm_gpu
#     EXPORT spmmTargets
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#   )
# endif()
