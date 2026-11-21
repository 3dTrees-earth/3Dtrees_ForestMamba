// CUDA kernels that reproduce two torch_cluster 1.6.3 results exactly, faster.
//
// fm_fps: farthest point sampling. torch_cluster launches one 256-thread block
// per cloud, so a single cloud runs on one SM. This version keeps its per-point
// arithmetic (same loop, so nvcc contracts it the same way), its 5e4 initial
// distance and its tie rule (highest distance; on a tie the lowest n % 256, then
// the lowest n), and spreads each step over the whole GPU with one 64-bit
// atomicMax per block. The caller draws the start index exactly like
// torch_cluster (see oneformer3d/fm_kernels.py).
//
// fm_knn1: torch_cluster.knn(x, y, k=1) restricted to candidate rows of x per
// query (-1 = no candidate): same float arithmetic, same rule (smallest
// distance, then lowest index of x).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace {

__device__ __forceinline__ unsigned long long pack(float v, long long n) {
  unsigned int key = ((unsigned int)(n & 255) << 23) | (unsigned int)(n >> 8);
  return ((unsigned long long)__float_as_uint(v) << 32) |
         (unsigned long long)(0xFFFFFFFFu - key);
}

__device__ __forceinline__ long long unpack(unsigned long long p) {
  unsigned int key = 0xFFFFFFFFu - (unsigned int)(p & 0xFFFFFFFFull);
  long long hi = key >> 23, lo = key & ((1u << 23) - 1);
  return lo * 256 + hi;
}

__global__ void fps_step(const float* __restrict__ src, float* __restrict__ dist,
                         const unsigned long long* __restrict__ prev,
                         unsigned long long* __restrict__ cur, const long long start,
                         const int first, const long long N, const long long dim) {
  long long old = first ? start : unpack(*prev);
  unsigned long long best = 0ull;
  bool have = false;
  for (long long n = blockIdx.x * (long long)blockDim.x + threadIdx.x; n < N;
       n += (long long)gridDim.x * blockDim.x) {
    float tmp, dd = 0.f;
    for (long long d = 0; d < dim; d++) {
      tmp = src[dim * old + d] - src[dim * n + d];
      dd += tmp * tmp;
    }
    dd = min(dist[n], dd);
    dist[n] = dd;
    unsigned long long p = pack(dd, n);
    if (!have || p > best) { best = p; have = true; }
  }
  for (int off = 16; off > 0; off >>= 1) {
    unsigned long long o = __shfl_down_sync(0xffffffff, best, off);
    if (o > best) best = o;
  }
  __shared__ unsigned long long warp_best[32];
  int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  if (lane == 0) warp_best[wid] = best;
  __syncthreads();
  if (wid == 0) {
    best = (lane < (blockDim.x + 31) / 32) ? warp_best[lane] : 0ull;
    for (int off = 16; off > 0; off >>= 1) {
      unsigned long long o = __shfl_down_sync(0xffffffff, best, off);
      if (o > best) best = o;
    }
    if (lane == 0 && best) atomicMax(cur, best);
  }
}

__global__ void fps_decode(const unsigned long long* slots, int64_t* out, long long k,
                           long long start) {
  long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i == 0) out[0] = start;
  if (i > 0 && i < k) out[i] = unpack(slots[i - 1]);
}

__global__ void knn1_candidates(const float* __restrict__ x, const float* __restrict__ y,
                                const int64_t* __restrict__ cand, int64_t* __restrict__ out,
                                const long long B, const long long C, const long long dim) {
  long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (q >= B) return;
  float best = 1e10;
  long long best_idx = -1;
  for (long long j = 0; j < C; j++) {
    long long n_x = cand[q * C + j];
    if (n_x < 0) continue;
    float tmp_dist = 0;
    for (long long d = 0; d < dim; d++) {
      tmp_dist += (x[n_x * dim + d] - y[q * dim + d]) * (x[n_x * dim + d] - y[q * dim + d]);
    }
    if (best > tmp_dist || (best == tmp_dist && n_x < best_idx)) {
      best = tmp_dist;
      best_idx = n_x;
    }
  }
  out[q] = best_idx;
}

}  // namespace

torch::Tensor fm_fps(torch::Tensor src, int64_t start, int64_t k) {
  TORCH_CHECK(src.is_cuda() && src.scalar_type() == torch::kFloat && src.dim() == 2,
              "fm_fps expects a CUDA float32 (N, D) tensor");
  TORCH_CHECK(k >= 1 && start >= 0 && start < src.size(0), "fm_fps: bad k or start");
  src = src.contiguous();
  const long long N = src.size(0), dim = src.size(1);
  auto dist = torch::full({N}, 5e4, src.options());
  auto slots = torch::zeros({std::max<int64_t>(k - 1, 1)}, src.options().dtype(torch::kLong));
  auto out = torch::empty({k}, src.options().dtype(torch::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocks = (int)std::min<long long>((N + threads - 1) / threads, 1024);
  auto* s = reinterpret_cast<unsigned long long*>(slots.data_ptr<int64_t>());
  for (int64_t m = 1; m < k; m++) {
    fps_step<<<blocks, threads, 0, stream>>>(src.data_ptr<float>(), dist.data_ptr<float>(),
                                             m > 1 ? s + (m - 2) : s, s + (m - 1), start,
                                             m == 1 ? 1 : 0, N, dim);
  }
  fps_decode<<<(int)((k + 255) / 256), 256, 0, stream>>>(s, out.data_ptr<int64_t>(), k, start);
  return out;
}

torch::Tensor fm_knn1(torch::Tensor x, torch::Tensor y, torch::Tensor cand) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda() && cand.is_cuda(), "fm_knn1 expects CUDA tensors");
  x = x.contiguous().to(torch::kFloat);
  y = y.contiguous().to(torch::kFloat);
  cand = cand.contiguous().to(torch::kLong);
  const long long B = cand.size(0), C = cand.size(1), dim = x.size(1);
  auto out = torch::empty({B}, cand.options());
  if (B == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  knn1_candidates<<<(int)((B + 255) / 256), 256, 0, stream>>>(
      x.data_ptr<float>(), y.data_ptr<float>(), cand.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(), B, C, dim);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fm_fps", &fm_fps, "Farthest point sampling, same picks as torch_cluster.fps");
  m.def("fm_knn1", &fm_knn1, "torch_cluster.knn(k=1) over candidate refs");
}
