# SPARSE (spgnn)

**SPARSE** is a lightweight research framework for studying sparse matrix kernels on modern CPUs and GPUs.
---

## Repository Structure

```text
spgnn/
├── benchmarks/ # Cross-project benchmarking tools (optional)
├── external/   # Baselines as Git submodules (optional)
├── include/    # Public interfaces / headers
├── src/        # CPU/GPU implementations
├── tests/      # Unit + integration tests (optional)
└── tools/      # Development utilities (e.g., DBLP data generator)
```

The layout aims to separate:
- **implementation** (`src/`, `include/`)
- **utilities** (`tools/`)
- **correctness/perf evaluation** (`tests/`, `benchmarks/`)
- **external library** (`external/`)

---

## Requirements

### Build dependencies
- **CMake** (recommended >= 3.20)
- **C++17** compiler
- **NVIDIA CUDA Toolkit** (NVCC + runtime)
- **cuSPARSE** (shipped with CUDA Toolkit)

> For GH200 / SM90, use a CUDA Toolkit version that supports **compute capability 9.0**.

### Python dependencies (DBLP generator)
- **Python** >= 3.8
- `lxml`
- `numpy`

Install:
```bash
pip install lxml numpy
```

---

## Getting Started

### Clone

Base Clone:

```bash
git clone https://github.com/YaoJianyu77/spgnn.git
```

Clone external library:
```bash
git submodule update --init --recursive
```

### (Optional) Development setup for external baselines

If you maintain research branches in submodules (e.g. `bench_patch`), you can provide a script like:

```bash
./scripts/dev_setup.sh
```

This helps keep multi-submodule development consistent.

---

## Build and Run

From the repository root:

```bash
mkdir -p build && cd build
cmake ..
make -j

# generate DBLP CSR into build/
python3 ../tools/dblp_data.py --out_dir dblp

# run
./spgemm_gpu dblp --tag peak_try --warmup 1 --repeat 3 --in_mode device --out_mode device
```