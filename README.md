# SPARSE

**SPARSE** is a lightweight research framework for studying Sparse Matrix–Dense Matrix multiplication (SpMM) on CPUs and GPUs.  
It provides a clean structure for implementing, testing, and benchmarking SpMM kernels and external baselines under a unified interface.

---

## Structure

```text
SPARSE/
├── benchmarks/ # Cross-project benchmarking tools
├── external/   # Baselines as Git submodules
├── include/    # Public interfaces
├── src/        # CPU/GPU implementations
├── tests/      # Unit + integration tests
└── scripts/    # Development utilities
```

The layout separates correctness testing, internal evaluation, and comparative benchmarking, supporting clear and reproducible experimentation.

---

## Getting Started

### Clone

Clone the repository with all submodules:

```bash
git clone --recurse-submodules https://github.com/YaoJianyu77/spgnn.git
cd spgnn
```

Initialize or update submodules if needed:

```bash
git submodule update --init --recursive
```

### Development Setup

To switch all external baselines to the intended research branch (e.g., `bench_patch`):

```bash
./scripts/dev_setup.sh
```

This provides a consistent development environment across multiple submodules.

### Build

```bash
mkdir build && cd build
cmake ..
make -j
```

### Testing

Run correctness and small-scale performance tests:

```bash
ctest --output-on-failure
```

### Benchmarking

Benchmark configurations are defined in JSON and executed via:

```bash
./benchmarks/spmm_compare <cases.json>
```

This enables structured comparison of internal and external SpMM implementations.

---

## License

Released under the MIT License.