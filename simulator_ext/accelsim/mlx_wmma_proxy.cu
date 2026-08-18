#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(error));                                  \
      std::exit(2);                                                             \
    }                                                                           \
  } while (0)

using namespace nvcuda;

__global__ void
wmma_repeat_kernel(const half *a, const half *b, float *c, int repeats)
{
  int tile = blockIdx.x;
  const half *tile_a = a + static_cast<std::size_t>(tile) * 16 * 16;
  const half *tile_b = b + static_cast<std::size_t>(tile) * 16 * 16;
  float *tile_c = c + static_cast<std::size_t>(tile) * 16 * 16;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::load_matrix_sync(a_frag, tile_a, 16);
  wmma::load_matrix_sync(b_frag, tile_b, 16);
  wmma::fill_fragment(c_frag, 0.0f);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(tile_c, c_frag, 16, wmma::mem_row_major);
}

static int
run_wmma(int tiles, int repeats)
{
  if (tiles <= 0 || repeats <= 0) return 2;
  constexpr int elements_per_tile = 16 * 16;
  std::size_t elements = static_cast<std::size_t>(tiles) * elements_per_tile;
  std::vector<half> a(elements, __float2half(0.0625f));
  std::vector<half> b(elements, __float2half(0.0625f));
  std::vector<float> c(elements);
  half *device_a = nullptr;
  half *device_b = nullptr;
  float *device_c = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, elements * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_b, elements * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_c, elements * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, a.data(), elements * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, b.data(), elements * sizeof(half),
                        cudaMemcpyHostToDevice));
  wmma_repeat_kernel<<<tiles, 32>>>(device_a, device_b, device_c, repeats);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), device_c, elements * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double checksum = 0.0;
  for (float value : c) checksum += value;
  double reference = static_cast<double>(tiles) * 16.0 * repeats;
  double relative_error = std::fabs(checksum - reference) / reference;
  long long fma_equivalents =
      static_cast<long long>(tiles) * repeats * 16 * 16 * 16;
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"wmma\",\"count\":%d,"
      "\"parameter\":%d,\"parameter2\":0,\"checksum\":%.9f,"
      "\"reference\":%.9f,\"relative_error\":%.12g,"
      "\"fma_equivalents\":%lld}\n",
      tiles, repeats, checksum, reference, relative_error, fma_equivalents);
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  return 0;
}

int
main(int argc, char **argv)
{
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s wmma TILES REPEATS 0\n", argv[0]);
    return 2;
  }
  std::string operation = argv[1];
  int tiles = std::atoi(argv[2]);
  int repeats = std::atoi(argv[3]);
  int parameter2 = std::atoi(argv[4]);
  if (operation != "wmma" || parameter2 != 0) return 2;
  return run_wmma(tiles, repeats);
}
