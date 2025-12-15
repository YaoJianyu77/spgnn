include(FetchContent)

# GTest
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0)
FetchContent_MakeAvailable(googletest)

# 统一测试目标：spmm_tests
file(GLOB_RECURSE TESTS_UNITS_SOURCES ${PROJECT_SOURCE_DIR}/tests/units/*.cpp)
file(GLOB_RECURSE TESTS_PERFORMANCE_SOURCES ${PROJECT_SOURCE_DIR}/tests/performance/*.cpp)
file(GLOB_RECURSE TESTS_INTEGRATION_SOURCES ${PROJECT_SOURCE_DIR}/tests/integration/*.cpp)
add_executable(spmm_tests
  ${TESTS_UNITS_SOURCES}
  ${TESTS_PERFORMANCE_SOURCES}
  ${TESTS_INTEGRATION_SOURCES}
)

target_include_directories(spmm_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/tests/common)
if(ENABLE_CPU)
  target_link_libraries(spmm_tests PRIVATE gtest_main spmm_cpu)
endif()
if(ENABLE_GPU)
  target_link_libraries(spmm_tests PRIVATE spmm_gpu CUDA::cudart)
endif()
if(USE_CUSPARSE AND ENABLE_GPU AND TARGET CUDA::cusparse)
  target_link_libraries(spmm_tests PRIVATE CUDA::cusparse)
  target_compile_definitions(spmm_tests PRIVATE USE_CUSPARSE=1)
endif()

# 让性能用例能找到 cases.json（传一个变量给代码）
target_compile_definitions(spmm_tests PRIVATE TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}/tests")

add_test(NAME spmm_tests COMMAND spmm_tests)
set_tests_properties(spmm_tests PROPERTIES LABELS "unit;perf;integration")
