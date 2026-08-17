#include <cuda_runtime.h>

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
                   cudaGetErrorString(error));                                 \
      std::exit(2);                                                             \
    }                                                                          \
  } while (0)

__host__ __device__ static inline float
stage_cosine(int stage)
{
  return 0.75f + static_cast<float>(stage & 3) * 0.03125f;
}

__host__ __device__ static inline float
stage_sine(int stage)
{
  return 0.125f + static_cast<float>(stage & 3) * 0.015625f;
}

__global__ void
fft_pair_stage(const float2 *input, float2 *output, int pairs, int stage)
{
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  float2 left = input[2 * pair];
  float2 right = input[2 * pair + 1];
  float cosine = stage_cosine(stage);
  float sine = stage_sine(stage);
  float p0 = fmaf(right.x, cosine, 0.0f);
  float p1 = fmaf(right.y, sine, 0.0f);
  float p2 = fmaf(right.x, sine, 0.0f);
  float p3 = fmaf(right.y, cosine, 0.0f);
  float real = p0 - p1;
  float imag = p2 + p3;
  output[2 * pair] = make_float2(left.x + real, left.y + imag);
  output[2 * pair + 1] = make_float2(left.x - real, left.y - imag);
}

__global__ void
truncate_half(const float2 *input, float2 *output, int retained_points)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < retained_points) output[index] = input[index];
}

static void
cpu_fft_stage(const std::vector<float2> &input, std::vector<float2> &output,
              int pairs, int stage)
{
  float cosine = stage_cosine(stage);
  float sine = stage_sine(stage);
  for (int pair = 0; pair < pairs; ++pair) {
    float2 left = input[2 * pair];
    float2 right = input[2 * pair + 1];
    float p0 = std::fma(right.x, cosine, 0.0f);
    float p1 = std::fma(right.y, sine, 0.0f);
    float p2 = std::fma(right.x, sine, 0.0f);
    float p3 = std::fma(right.y, cosine, 0.0f);
    float real = p0 - p1;
    float imag = p2 + p3;
    output[2 * pair] = make_float2(left.x + real, left.y + imag);
    output[2 * pair + 1] = make_float2(left.x - real, left.y - imag);
  }
}

static double
checksum(const std::vector<float2> &values)
{
  double result = 0.0;
  for (float2 value : values) result += value.x + value.y;
  return result;
}

static int
run_fftcmp(int pairs, int forward_stages, int inverse_stages, int threads)
{
  if (pairs < 2 || pairs % 2 != 0) return 2;
  std::vector<float2> input(2 * pairs), reference(2 * pairs), next(2 * pairs);
  for (int index = 0; index < 2 * pairs; ++index) {
    input[index] = make_float2(0.25f + (index % 7) * 0.02f,
                               0.125f + (index % 5) * 0.01f);
  }
  reference = input;
  for (int stage = 0; stage < forward_stages; ++stage) {
    cpu_fft_stage(reference, next, pairs, stage);
    reference.swap(next);
  }
  int inverse_pairs = pairs / 2;
  reference.resize(pairs);
  next.resize(pairs);
  for (int stage = 0; stage < inverse_stages; ++stage) {
    cpu_fft_stage(reference, next, inverse_pairs, forward_stages + stage);
    reference.swap(next);
  }

  float2 *first = nullptr, *second = nullptr, *retained = nullptr;
  CUDA_CHECK(cudaMalloc(&first, 2 * pairs * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&second, 2 * pairs * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&retained, pairs * sizeof(float2)));
  CUDA_CHECK(cudaMemcpy(first, input.data(), 2 * pairs * sizeof(float2),
                        cudaMemcpyHostToDevice));
  for (int stage = 0; stage < forward_stages; ++stage) {
    fft_pair_stage<<<(pairs + threads - 1) / threads, threads>>>(first, second,
                                                                 pairs, stage);
    CUDA_CHECK(cudaDeviceSynchronize());
    float2 *temporary = first;
    first = second;
    second = temporary;
  }
  truncate_half<<<(pairs + threads - 1) / threads, threads>>>(first, retained,
                                                               pairs);
  CUDA_CHECK(cudaDeviceSynchronize());
  for (int stage = 0; stage < inverse_stages; ++stage) {
    fft_pair_stage<<<(inverse_pairs + threads - 1) / threads, threads>>>(
        retained, second, inverse_pairs, forward_stages + stage);
    CUDA_CHECK(cudaDeviceSynchronize());
    float2 *temporary = retained;
    retained = second;
    second = temporary;
  }
  std::vector<float2> output(pairs);
  CUDA_CHECK(cudaMemcpy(output.data(), retained, pairs * sizeof(float2),
                        cudaMemcpyDeviceToHost));
  double expected = checksum(reference);
  double measured = checksum(output);
  double error = std::fabs(measured - expected) /
                 std::fmax(1.0, std::fabs(expected));
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"fftcmp\",\"count\":%d,"
      "\"parameter\":%d,\"parameter2\":%d,\"checksum\":%.9f,"
      "\"reference\":%.9f,\"relative_error\":%.12g}\n",
      pairs, forward_stages, inverse_stages, measured, expected, error);
  cudaFree(first);
  cudaFree(second);
  cudaFree(retained);
  return 0;
}

int
main(int argc, char **argv)
{
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s fftcmp COUNT FORWARD_STAGES INVERSE_STAGES\n",
                 argv[0]);
    return 2;
  }
  std::string operation = argv[1];
  int count = std::atoi(argv[2]);
  int forward_stages = std::atoi(argv[3]);
  int inverse_stages = std::atoi(argv[4]);
  if (operation != "fftcmp" || count <= 0 || forward_stages <= 0 ||
      inverse_stages <= 0) {
    return 2;
  }
  return run_fftcmp(count, forward_stages, inverse_stages, 128);
}
