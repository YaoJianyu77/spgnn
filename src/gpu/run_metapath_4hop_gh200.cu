// run_metapath_4hop_gh200.cu
// Read CSR binaries from dblp_data.py, run metapaths on GPU using cuSPARSE SpGEMM.
//
// Usage:
//   ./run_metapath_4hop_gh200 dblp --tag xxx --warmup 1 --repeat 5 --in_mode device --out_mode device
//
// Outputs:
//   dblp/matrix_stats.csv
//   dblp/spgemm_runs.csv
//
// NOTE: CSV tail order is kept to match your awk peak script:
//   ... , mem_total, mem_free_begin, mem_free_min, mem_free_end, in_mode, out_mode

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "spgemm_gh200.cu"

static std::string read_text_or_die(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); std::exit(1); }
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::string s;
  s.resize((size_t)std::max<long>(0, sz));
  size_t got = (sz > 0) ? std::fread(&s[0], 1, (size_t)sz, f) : 0;
  std::fclose(f);
  s.resize(got);
  return s;
}

// minimal JSON int extractor: finds `"KEY": <int>`
static int extract_json_int_or_die(const std::string& json, const std::string& key) {
  std::string pat = "\"" + key + "\"";
  size_t p = json.find(pat);
  if (p == std::string::npos) {
    std::fprintf(stderr, "ERROR: meta.json missing key %s\n", key.c_str());
    std::exit(1);
  }
  p = json.find(':', p);
  if (p == std::string::npos) { std::fprintf(stderr, "ERROR: bad meta.json\n"); std::exit(1); }
  p++;
  while (p < json.size() && (json[p]==' '||json[p]=='\n'||json[p]=='\r'||json[p]=='\t')) p++;
  long long v=0; bool ok=false; bool neg=false;
  if (p < json.size() && json[p]=='-') { neg=true; p++; }
  while (p < json.size() && json[p]>='0' && json[p]<='9') { ok=true; v=v*10+(json[p]-'0'); p++; }
  if (!ok) { std::fprintf(stderr, "ERROR: cannot parse int for %s\n", key.c_str()); std::exit(1); }
  return (int)(neg ? -v : v);
}

static long long extract_json_i64_or_default(const std::string& json, const std::string& key, long long defv) {
  std::string pat = "\"" + key + "\"";
  size_t p = json.find(pat);
  if (p == std::string::npos) return defv;
  p = json.find(':', p);
  if (p == std::string::npos) return defv;
  p++;
  while (p < json.size() && (json[p]==' '||json[p]=='\n'||json[p]=='\r'||json[p]=='\t')) p++;
  long long v=0; bool ok=false; bool neg=false;
  if (p < json.size() && json[p]=='-') { neg=true; p++; }
  while (p < json.size() && json[p]>='0' && json[p]<='9') { ok=true; v=v*10+(json[p]-'0'); p++; }
  if (!ok) return defv;
  return neg ? -v : v;
}

static size_t file_size_bytes(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return 0;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fclose(f);
  return (sz < 0) ? 0 : (size_t)sz;
}

static void read_file_or_die(const std::string& path, void* dst, size_t bytes) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str()); std::exit(1); }
  size_t got = std::fread(dst, 1, bytes, f);
  std::fclose(f);
  if (got != bytes) {
    std::fprintf(stderr, "ERROR: short read %s (got %zu, want %zu)\n", path.c_str(), got, bytes);
    std::exit(1);
  }
}

static MemoryMode parse_memmode(const std::string& s) {
  if (s == "device") return MemoryMode::Device;
  if (s == "managed") return MemoryMode::Managed;
  std::fprintf(stderr, "ERROR: bad mem mode %s (use device|managed)\n", s.c_str());
  std::exit(1);
}

template <typename T>
static Csr32<T> load_csr(const std::string& dir, const std::string& name,
                         int rows, int cols, MemoryMode mode, cudaStream_t stream)
{
  std::string row_path = dir + "/" + name + "_row_offsets.i32";
  std::string col_path = dir + "/" + name + "_col_indices.i32";
  std::string val_path = dir + "/" + name + "_values.f32";

  size_t row_bytes = file_size_bytes(row_path);
  size_t col_bytes = file_size_bytes(col_path);
  size_t val_bytes = file_size_bytes(val_path);

  if (row_bytes == 0) { std::fprintf(stderr, "ERROR: missing %s\n", row_path.c_str()); std::exit(1); }

  int rows_in_file = (int)(row_bytes / sizeof(int)) - 1;
  if (rows_in_file != rows) {
    std::fprintf(stderr, "ERROR: %s rows mismatch: file=%d expected=%d\n",
                 name.c_str(), rows_in_file, rows);
    std::exit(1);
  }
  int nnz = (int)(col_bytes / sizeof(int));
  if (val_bytes != col_bytes) {
    std::fprintf(stderr, "ERROR: %s values bytes != col bytes\n", name.c_str());
    std::exit(1);
  }

  std::vector<int> h_row((size_t)rows + 1);
  read_file_or_die(row_path, h_row.data(), row_bytes);
  if (h_row[(size_t)rows] != nnz) {
    std::fprintf(stderr, "ERROR: %s row_offsets[last]=%d but nnz(file)=%d\n", name.c_str(), h_row[(size_t)rows], nnz);
    std::exit(1);
  }

  std::vector<int>   h_col;
  std::vector<float> h_val;
  if (nnz > 0) {
    h_col.resize((size_t)nnz);
    h_val.resize((size_t)nnz);
    read_file_or_die(col_path, h_col.data(), col_bytes);
    read_file_or_die(val_path, h_val.data(), val_bytes);
  }

  Csr32<T> M;
  csr_alloc(M, rows, cols, nnz, mode);

  if (mode == MemoryMode::Managed) {
    std::memcpy(M.row_offsets, h_row.data(), row_bytes);
    if (nnz > 0) {
      std::memcpy(M.col_indices, h_col.data(), col_bytes);
      std::memcpy(M.values,      h_val.data(), val_bytes);
    }
  } else {
    CUDA_CHECK(cudaMemcpyAsync(M.row_offsets, h_row.data(), row_bytes, cudaMemcpyHostToDevice, stream));
    if (nnz > 0) {
      CUDA_CHECK(cudaMemcpyAsync(M.col_indices, h_col.data(), col_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(M.values,      h_val.data(), val_bytes, cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  return M;
}

static void print_shape(const char* name, const Csr32<float>& M) {
  std::printf("%s: (%d x %d), nnz=%d\n", name, M.rows, M.cols, M.nnz);
}

static bool file_exists_nonempty(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long sz = std::ftell(f);
  std::fclose(f);
  return sz > 0;
}

static void append_line(const std::string& path, const std::string& line) {
  FILE* f = std::fopen(path.c_str(), "ab");
  if (!f) { std::fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); std::exit(1); }
  std::fwrite(line.data(), 1, line.size(), f);
  std::fwrite("\n", 1, 1, f);
  std::fclose(f);
}

static void ensure_csv_headers(const std::string& stats_csv, const std::string& runs_csv) {
  if (!file_exists_nonempty(stats_csv)) {
    append_line(stats_csv, "tag,matrix,rows,cols,nnz,bytes_total,mem_mode");
  }
  if (!file_exists_nonempty(runs_csv)) {
    append_line(runs_csv,
      "tag,op,run,is_warmup,m,k,n,nnzA,nnzB,nnzC,nnzC_ub,nnzC_true,nnzC_alloc,alg,bufferSize1,bufferSize2,"
      "t_desc_ms,t_malloc_ws_ms,t_work_est_ms,t_compute_ms,t_malloc_out_ms,t_copy_ms,t_total_gpu_ms,t_wall_total_ms,"
      "mem_total,mem_free_begin,mem_free_min,mem_free_end,in_mode,out_mode");
  }
}

static void write_matrix_stats(const std::string& stats_csv, const std::string& tag,
                               const char* name, const Csr32<float>& M) {
  long long bytes_total = (long long)(M.rows + 1) * 4 + (long long)M.nnz * 4 + (long long)M.nnz * 4;
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s,%s,%d,%d,%d,%lld,%s",
                tag.c_str(), name, M.rows, M.cols, M.nnz, bytes_total, memmode_str(M.mem));
  append_line(stats_csv, buf);
}

static void append_run_csv(const std::string& runs_csv,
                           const std::string& tag,
                           const std::string& op,
                           int run, int is_warmup,
                           const SpGemmCallMetrics& met,
                           long long nnzC_ub_for_log,
                           MemoryMode in_mode,
                           MemoryMode out_mode)
{
  // in "simple/original" version:
  //   nnzC_true == nnzC_alloc == nnzC (we log the same number)
  long long nnzC_true = (long long)met.nnzC;
  long long nnzC_alloc = (long long)met.nnzC;

  char buf[2048];
  std::snprintf(buf, sizeof(buf),
    "%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%lld,%lld,%d,%zu,%zu,"
    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
    "%zu,%zu,%zu,%zu,%s,%s",
    tag.c_str(), op.c_str(), run, is_warmup,
    met.m, met.k, met.n,
    met.nnzA, met.nnzB, met.nnzC,
    nnzC_ub_for_log, nnzC_true, nnzC_alloc,
    met.alg, met.bufferSize1, met.bufferSize2,
    met.t_desc_ms, met.t_malloc_ws_ms, met.t_work_est_ms, met.t_compute_ms,
    met.t_malloc_out_ms, met.t_copy_ms, met.t_total_gpu_ms, met.t_wall_total_ms,
    met.mem_total, met.mem_free_begin, met.mem_free_min, met.mem_free_end,
    memmode_str(in_mode), memmode_str(out_mode));
  append_line(runs_csv, buf);
}

struct Args {
  std::string dir;
  std::string tag = "default";
  int warmup = 1;
  int repeat = 5;
  MemoryMode in_mode = MemoryMode::Device;
  MemoryMode out_mode = MemoryMode::Device;
};

static Args parse_args(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s dblp_dir [--tag X] [--warmup N] [--repeat N] [--in_mode device|managed] [--out_mode device|managed]\n", argv[0]);
    std::exit(1);
  }
  Args a;
  a.dir = argv[1];
  for (int i = 2; i < argc; i++) {
    std::string k = argv[i];
    auto need = [&](const char* opt) {
      if (i + 1 >= argc) { std::fprintf(stderr, "ERROR: %s needs value\n", opt); std::exit(1); }
      return std::string(argv[++i]);
    };
    if (k == "--tag") a.tag = need("--tag");
    else if (k == "--warmup") a.warmup = std::atoi(need("--warmup").c_str());
    else if (k == "--repeat") a.repeat = std::atoi(need("--repeat").c_str());
    else if (k == "--in_mode") a.in_mode = parse_memmode(need("--in_mode"));
    else if (k == "--out_mode") a.out_mode = parse_memmode(need("--out_mode"));
    else {
      std::fprintf(stderr, "ERROR: unknown arg %s\n", k.c_str());
      std::exit(1);
    }
  }
  return a;
}

static void bench_spgemm(
    cusparseHandle_t handle,
    const std::string& tag,
    const std::string& op,
    const Csr32<float>& A,
    const Csr32<float>& B,
    cudaStream_t stream,
    MemoryMode out_mode,
    int warmup,
    int repeat,
    const std::string& runs_csv,
    long long nnzC_ub_for_log,
    MemoryMode in_mode_for_log)
{
  // warmup
  for (int i = 0; i < warmup; i++) {
    Csr32<float> C;
    SpGemmCallMetrics met{};
    auto st = spgemm_csr_cusparse_alloc_ex<float>(
        handle,
        A.rows, A.cols, B.cols,
        A.nnz, A.row_offsets, A.col_indices, A.values,
        B.nnz, B.row_offsets, B.col_indices, B.values,
        C, stream, out_mode, &met);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      std::fprintf(stderr, "ERROR: %s warmup failed, status=%d\n", op.c_str(), (int)st);
      std::exit(1);
    }
    append_run_csv(runs_csv, tag, op, i, 1, met, nnzC_ub_for_log, in_mode_for_log, out_mode);
    csr_free(C);
  }

  double sum = 0.0;
  for (int i = 0; i < repeat; i++) {
    Csr32<float> C;
    SpGemmCallMetrics met{};
    auto st = spgemm_csr_cusparse_alloc_ex<float>(
        handle,
        A.rows, A.cols, B.cols,
        A.nnz, A.row_offsets, A.col_indices, A.values,
        B.nnz, B.row_offsets, B.col_indices, B.values,
        C, stream, out_mode, &met);
    if (st != CUSPARSE_STATUS_SUCCESS) {
      std::fprintf(stderr, "ERROR: %s failed, status=%d\n", op.c_str(), (int)st);
      std::exit(1);
    }
    append_run_csv(runs_csv, tag, op, warmup + i, 0, met, nnzC_ub_for_log, in_mode_for_log, out_mode);
    sum += met.t_total_gpu_ms;
    csr_free(C);
  }

  double mean = sum / std::max(1, repeat);
  std::printf("[MEAN] %s: mean_total_gpu_ms=%.3f over %d runs (exclude %d warmup)\n",
              op.c_str(), mean, repeat, warmup);
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  size_t free_b=0,total_b=0;
  mem_get_info(free_b,total_b);
  std::printf("[GPU] dev=%d name=%s cc=%d.%d mem=%.2f GiB\n",
              dev, prop.name, prop.major, prop.minor, (double)total_b / 1073741824.0);

  cudaStream_t stream = 0;

  std::string meta = read_text_or_die(args.dir + "/meta.json");
  int P = extract_json_int_or_die(meta, "P");
  int A = extract_json_int_or_die(meta, "A");
  int T = extract_json_int_or_die(meta, "T");
  int C = extract_json_int_or_die(meta, "C");
  long long APCPA_ub = extract_json_i64_or_default(meta, "APCPA_ub", -1);

  std::printf("[META] P=%d A=%d T=%d C=%d\n", P, A, T, C);

  auto PA = load_csr<float>(args.dir, "PA", P, A, args.in_mode, stream);
  auto PT = load_csr<float>(args.dir, "PT", P, T, args.in_mode, stream);
  auto PC = load_csr<float>(args.dir, "PC", P, C, args.in_mode, stream);

  auto AP = load_csr<float>(args.dir, "AP", A, P, args.in_mode, stream);
  auto TP = load_csr<float>(args.dir, "TP", T, P, args.in_mode, stream);
  auto CP = load_csr<float>(args.dir, "CP", C, P, args.in_mode, stream);

  print_shape("PA", PA);
  print_shape("PT", PT);
  print_shape("PC", PC);
  print_shape("AP", AP);
  print_shape("TP", TP);
  print_shape("CP", CP);

  std::string stats_csv = args.dir + "/matrix_stats.csv";
  std::string runs_csv  = args.dir + "/spgemm_runs.csv";
  ensure_csv_headers(stats_csv, runs_csv);

  write_matrix_stats(stats_csv, args.tag, "PA", PA);
  write_matrix_stats(stats_csv, args.tag, "PT", PT);
  write_matrix_stats(stats_csv, args.tag, "PC", PC);
  write_matrix_stats(stats_csv, args.tag, "AP", AP);
  write_matrix_stats(stats_csv, args.tag, "TP", TP);
  write_matrix_stats(stats_csv, args.tag, "CP", CP);

  cusparseHandle_t handle = nullptr;
  CUSPARSE_CALL(cusparseCreate(&handle));
  CUSPARSE_CALL(cusparseSetStream(handle, stream));

  // APA = AP * PA
  bench_spgemm(handle, args.tag, "APA", AP, PA, stream, args.out_mode,
              args.warmup, args.repeat, runs_csv, -1, args.in_mode);
  // X_APPT = AP * PT
  bench_spgemm(handle, args.tag, "X_APPT", AP, PT, stream, args.out_mode,
              args.warmup, args.repeat, runs_csv, -1, args.in_mode);
  // Y_TPPA = TP * PA
  bench_spgemm(handle, args.tag, "Y_TPPA", TP, PA, stream, args.out_mode,
              args.warmup, args.repeat, runs_csv, -1, args.in_mode);

  // APTPA = (AP*PT) * (TP*PA)
  {
    Csr32<float> X, Y;
    auto stx = spgemm_csr_cusparse_alloc_ex<float>(handle, AP.rows, AP.cols, PT.cols,
        AP.nnz, AP.row_offsets, AP.col_indices, AP.values,
        PT.nnz, PT.row_offsets, PT.col_indices, PT.values,
        X, stream, args.out_mode, nullptr);
    if (stx != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "ERROR: build X_APPT failed\n"); std::exit(1); }
    auto sty = spgemm_csr_cusparse_alloc_ex<float>(handle, TP.rows, TP.cols, PA.cols,
        TP.nnz, TP.row_offsets, TP.col_indices, TP.values,
        PA.nnz, PA.row_offsets, PA.col_indices, PA.values,
        Y, stream, args.out_mode, nullptr);
    if (sty != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "ERROR: build Y_TPPA failed\n"); std::exit(1); }

    print_shape("X_APPT", X);
    print_shape("Y_TPPA", Y);

    bench_spgemm(handle, args.tag, "APTPA", X, Y, stream, args.out_mode,
                args.warmup, args.repeat, runs_csv, -1, args.in_mode);

    Csr32<float> Z;
    auto stz = spgemm_csr_cusparse_alloc_ex<float>(handle, X.rows, X.cols, Y.cols,
        X.nnz, X.row_offsets, X.col_indices, X.values,
        Y.nnz, Y.row_offsets, Y.col_indices, Y.values,
        Z, stream, args.out_mode, nullptr);
    if (stz != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "ERROR: build APTPA failed\n"); std::exit(1); }
    print_shape("APTPA", Z);

    csr_free(X); csr_free(Y); csr_free(Z);
  }

  // X_APPC = AP * PC
  bench_spgemm(handle, args.tag, "X_APPC", AP, PC, stream, args.out_mode,
              args.warmup, args.repeat, runs_csv, -1, args.in_mode);
  // Y_CPPA = CP * PA
  bench_spgemm(handle, args.tag, "Y_CPPA", CP, PA, stream, args.out_mode,
              args.warmup, args.repeat, runs_csv, -1, args.in_mode);

  if (APCPA_ub > 0) {
    std::printf("[UB] APCPA nnz upper-bound = %lld\n", APCPA_ub);
  }

  // APCPA = (AP*PC) * (CP*PA)
  {
    Csr32<float> X, Y;
    auto stx = spgemm_csr_cusparse_alloc_ex<float>(handle, AP.rows, AP.cols, PC.cols,
        AP.nnz, AP.row_offsets, AP.col_indices, AP.values,
        PC.nnz, PC.row_offsets, PC.col_indices, PC.values,
        X, stream, args.out_mode, nullptr);
    if (stx != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "ERROR: build X_APPC failed\n"); std::exit(1); }

    auto sty = spgemm_csr_cusparse_alloc_ex<float>(handle, CP.rows, CP.cols, PA.cols,
        CP.nnz, CP.row_offsets, CP.col_indices, CP.values,
        PA.nnz, PA.row_offsets, PA.col_indices, PA.values,
        Y, stream, args.out_mode, nullptr);
    if (sty != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "ERROR: build Y_CPPA failed\n"); std::exit(1); }

    print_shape("X_APPC", X);
    print_shape("Y_CPPA", Y);

    bench_spgemm(handle, args.tag, "APCPA", X, Y, stream, args.out_mode,
                args.warmup, args.repeat, runs_csv, APCPA_ub, args.in_mode);

    Csr32<float> Z;
    auto stz = spgemm_csr_cusparse_alloc_ex<float>(handle, X.rows, X.cols, Y.cols,
        X.nnz, X.row_offsets, X.col_indices, X.values,
        Y.nnz, Y.row_offsets, Y.col_indices, Y.values,
        Z, stream, args.out_mode, nullptr);
    if (stz != CUSPARSE_STATUS_SUCCESS) { std::fprintf(stderr, "ERROR: APCPA SpGEMM failed, status=%d\n", (int)stz); std::exit(1); }
    print_shape("APCPA", Z);

    csr_free(X); csr_free(Y); csr_free(Z);
  }

  csr_free(PA); csr_free(PT); csr_free(PC);
  csr_free(AP); csr_free(TP); csr_free(CP);

  CUSPARSE_CALL(cusparseDestroy(handle));

  std::printf("[DONE] wrote:\n  %s\n  %s\n", stats_csv.c_str(), runs_csv.c_str());
  return 0;
}
