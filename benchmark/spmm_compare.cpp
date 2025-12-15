// benchmarks/spmm_compare.cpp
#include "backends/our_spmm.hpp"
#include "backends/projA_spmm.hpp"
// ...

int main(int argc, char** argv) {
    // 1. 读取 cases.json / 命令行，构造不同的矩阵规模
    // 2. 对每个 case:
    //      - 创建所有 backend 实例
    //      - backend.prepare(...)
    //      - backend.run(...)
    //      - 打印 CSV/markdown/table

    return 0;
}
