#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(error));                                 \
      std::exit(2);                                                             \
    }                                                                          \
  } while (0)

__global__ void
bsmm_stage(const float *input, float *output, int count, int stride)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    int partner = index ^ stride;
    if (partner >= count) partner = index;
    float left = input[index];
    float right = input[partner];
    float mixed0 = fmaf(left, 0.75f, right * 0.25f);
    float mixed1 = fmaf(right, 0.5f, left * 0.5f);
    output[index] = fmaf(mixed0, 0.875f, mixed1 * 0.125f);
  }
}

static double
checksum(const std::vector<float> &values)
{
  double result = 0.0;
  for (float value : values) result += value;
  return result;
}

int
main(int argc, char **argv)
{
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s COUNT STAGES BLOCK_THREADS\n", argv[0]);
    return 2;
  }
  int count = std::atoi(argv[1]);
  int stages = std::atoi(argv[2]);
  int block_threads = std::atoi(argv[3]);
  if (count <= 0 || stages <= 0 || block_threads <= 0 || block_threads > 1024) {
    std::fprintf(stderr, "positive count/stages and block_threads<=1024 required\n");
    return 2;
  }
  std::vector<float> input(count), output(count), reference(count), next(count);
  for (int index = 0; index < count; ++index) {
    input[index] = 0.5f + static_cast<float>(index % 17) * 0.01f;
  }
  reference = input;
  for (int stage = 0; stage < stages; ++stage) {
    int stride = 1 << stage;
    for (int index = 0; index < count; ++index) {
      int partner = index ^ stride;
      if (partner >= count) partner = index;
      float mixed0 = std::fma(
          reference[index], 0.75f, reference[partner] * 0.25f);
      float mixed1 = std::fma(
          reference[partner], 0.5f, reference[index] * 0.5f);
      next[index] = std::fma(mixed0, 0.875f, mixed1 * 0.125f);
    }
    reference.swap(next);
  }

  float *first = nullptr;
  float *second = nullptr;
  CUDA_CHECK(cudaMalloc(&first, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&second, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(
      first, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  int ctas_per_stage = (count + block_threads - 1) / block_threads;
  for (int stage = 0; stage < stages; ++stage) {
    bsmm_stage<<<ctas_per_stage, block_threads>>>(
        first, second, count, 1 << stage);
    CUDA_CHECK(cudaDeviceSynchronize());
    float *temporary = first;
    first = second;
    second = temporary;
  }
  CUDA_CHECK(cudaMemcpy(
      output.data(), first, count * sizeof(float), cudaMemcpyDeviceToHost));
  double expected = checksum(reference);
  double measured = checksum(output);
  double relative = std::fabs(measured - expected) / std::fabs(expected);
  uint64_t scalar_fma =
      static_cast<uint64_t>(count) * static_cast<uint64_t>(stages) * 3ULL;
  std::printf(
      "MLX_FIG24_SCHEDULE_SUMMARY {\"count\":%d,\"stages\":%d,"
      "\"block_threads\":%d,\"ctas_per_stage\":%d,\"total_ctas\":%d,"
      "\"scalar_fma\":%llu,\"checksum\":%.9f,\"reference\":%.9f,"
      "\"relative_error\":%.12g}\n",
      count, stages, block_threads, ctas_per_stage, ctas_per_stage * stages,
      static_cast<unsigned long long>(scalar_fma), measured, expected, relative);
  cudaFree(first);
  cudaFree(second);
  return relative <= 1e-6 ? 0 : 1;
}
