#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(error));                                  \
      std::exit(2);                                                             \
    }                                                                          \
  } while (0)

__global__ void bsmm_service(const float *a, const float *b, float *output,
                             int count, int stages, int repeat)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float left = a[index], right = b[index], accumulator = left;
  for (int iteration = 0; iteration < repeat; ++iteration) {
    for (int stage = 0; stage < stages; ++stage) {
      float alpha = 0.5f + 0.03125f * static_cast<float>(stage + 1);
      accumulator = fmaf(accumulator, alpha, right * (1.0f - alpha));
      accumulator = fmaf(right, 0.375f, accumulator * 0.625f);
      accumulator = fmaf(accumulator, 0.875f, left * 0.125f);
    }
  }
  output[index] = accumulator;
}

__global__ void fft_service(const float *real_input, const float *imag_input,
                            float *output, int count, int stages, int repeat)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float real = real_input[index], imag = imag_input[index];
  for (int iteration = 0; iteration < repeat; ++iteration) {
    for (int stage = 0; stage < stages; ++stage) {
      float cosine = 0.75f + 0.00390625f * static_cast<float>(stage % 16);
      float sine = 0.25f - 0.001953125f * static_cast<float>(stage % 16);
      float next_real = fmaf(real, cosine, -imag * sine) + real_input[index];
      float next_imag = fmaf(real, sine, imag * cosine) + imag_input[index];
      real = next_real * 0.5f;
      imag = next_imag * 0.5f;
    }
  }
  output[index] = real + imag;
}

__global__ void swa_service(const float *query, const float *key,
                            const float *value, float *output, int count,
                            int window, int repeat)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float result = query[index];
  for (int iteration = 0; iteration < repeat; ++iteration) {
    float maximum = -3.402823466e38f;
    for (int offset = 0; offset < window; ++offset) {
      int source = (index + offset) % count;
      float score = fmaf(query[index], key[source], 0.001f * result);
      maximum = fmaxf(maximum, score);
    }
    float denominator = 0.0f, weighted = 0.0f;
    for (int offset = 0; offset < window; ++offset) {
      int source = (index + offset) % count;
      float score = fmaf(query[index], key[source], 0.001f * result);
      float weight = expf(score - maximum);
      denominator += weight;
      weighted = fmaf(weight, value[source], weighted);
    }
    result = weighted / denominator;
  }
  output[index] = result;
}

static void host_bsmm(const std::vector<float> &a, const std::vector<float> &b,
                      std::vector<float> &output, int stages, int repeat)
{
  for (int index = 0; index < static_cast<int>(a.size()); ++index) {
    float left = a[index], right = b[index], accumulator = left;
    for (int iteration = 0; iteration < repeat; ++iteration) {
      for (int stage = 0; stage < stages; ++stage) {
        float alpha = 0.5f + 0.03125f * static_cast<float>(stage + 1);
        accumulator = std::fma(accumulator, alpha, right * (1.0f - alpha));
        accumulator = std::fma(right, 0.375f, accumulator * 0.625f);
        accumulator = std::fma(accumulator, 0.875f, left * 0.125f);
      }
    }
    output[index] = accumulator;
  }
}

static void host_fft(const std::vector<float> &real_input,
                     const std::vector<float> &imag_input,
                     std::vector<float> &output, int stages, int repeat)
{
  for (int index = 0; index < static_cast<int>(real_input.size()); ++index) {
    float real = real_input[index], imag = imag_input[index];
    for (int iteration = 0; iteration < repeat; ++iteration) {
      for (int stage = 0; stage < stages; ++stage) {
        float cosine =
            0.75f + 0.00390625f * static_cast<float>(stage % 16);
        float sine = 0.25f - 0.001953125f * static_cast<float>(stage % 16);
        float next_real =
            std::fma(real, cosine, -imag * sine) + real_input[index];
        float next_imag =
            std::fma(real, sine, imag * cosine) + imag_input[index];
        real = next_real * 0.5f;
        imag = next_imag * 0.5f;
      }
    }
    output[index] = real + imag;
  }
}

static void host_swa(const std::vector<float> &query,
                     const std::vector<float> &key,
                     const std::vector<float> &value,
                     std::vector<float> &output, int window, int repeat)
{
  int count = static_cast<int>(query.size());
  for (int index = 0; index < count; ++index) {
    float result = query[index];
    for (int iteration = 0; iteration < repeat; ++iteration) {
      float maximum = -3.402823466e38f;
      for (int offset = 0; offset < window; ++offset) {
        int source = (index + offset) % count;
        float score = std::fma(query[index], key[source], 0.001f * result);
        maximum = std::fmax(maximum, score);
      }
      float denominator = 0.0f, weighted = 0.0f;
      for (int offset = 0; offset < window; ++offset) {
        int source = (index + offset) % count;
        float score = std::fma(query[index], key[source], 0.001f * result);
        float weight = std::exp(score - maximum);
        denominator += weight;
        weighted = std::fma(weight, value[source], weighted);
      }
      result = weighted / denominator;
    }
    output[index] = result;
  }
}

int main(int argc, char **argv)
{
  if (argc != 8) {
    std::fprintf(stderr,
                 "usage: %s fft|bsmm|swa PARAM COUNT REPEAT WARMUP TRIALS VERIFY\n",
                 argv[0]);
    return 2;
  }
  std::string operation = argv[1];
  int parameter = std::atoi(argv[2]);
  int count = std::atoi(argv[3]);
  int repeat = std::atoi(argv[4]);
  int warmup = std::atoi(argv[5]);
  int trials = std::atoi(argv[6]);
  bool verify = std::atoi(argv[7]) != 0;
  if (parameter <= 0 || count <= 0 || repeat <= 0 || warmup < 0 || trials <= 0)
    return 2;
  std::vector<float> a(count), b(count), c(count), output(count), reference(count);
  for (int index = 0; index < count; ++index) {
    a[index] = 0.25f + 0.001f * static_cast<float>(index % 97);
    b[index] = 0.5f + 0.0015f * static_cast<float>(index % 89);
    c[index] = 0.75f + 0.002f * static_cast<float>(index % 83);
  }
  float *da = nullptr, *db = nullptr, *dc = nullptr, *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&da, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dc, c.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  constexpr int threads = 256;
  int blocks = (count + threads - 1) / threads;
  auto launch = [&]() {
    if (operation == "bsmm")
      bsmm_service<<<blocks, threads>>>(da, db, dout, count, parameter, repeat);
    else if (operation == "fft")
      fft_service<<<blocks, threads>>>(da, db, dout, count, parameter, repeat);
    else if (operation == "swa")
      swa_service<<<blocks, threads>>>(da, db, dc, dout, count, parameter, repeat);
    else {
      std::fprintf(stderr, "unknown operation: %s\n", operation.c_str());
      std::exit(2);
    }
  };
  for (int iteration = 0; iteration < warmup; ++iteration) launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int iteration = 0; iteration < trials; ++iteration) launch();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float average_ms = elapsed_ms / static_cast<float>(trials);
  CUDA_CHECK(cudaMemcpy(output.data(), dout, count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double checksum = 0.0, maximum_error = 0.0;
  for (float result : output) checksum += result;
  if (verify) {
    if (operation == "bsmm") host_bsmm(a, b, reference, parameter, repeat);
    if (operation == "fft") host_fft(a, b, reference, parameter, repeat);
    if (operation == "swa") host_swa(a, b, c, reference, parameter, repeat);
    for (int index = 0; index < count; ++index) {
      maximum_error = std::max(
          maximum_error,
          static_cast<double>(std::fabs(output[index] - reference[index])));
    }
  }
  unsigned long long fma_count = 0;
  if (operation == "bsmm")
    fma_count = 3ULL * count * parameter * repeat;
  if (operation == "fft")
    fma_count = 2ULL * count * parameter * repeat;
  if (operation == "swa")
    fma_count = 2ULL * count * parameter * repeat;
  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  std::printf(
      "FIG24_4090_SUMMARY {\"operation\":\"%s\",\"parameter\":%d,"
      "\"count\":%d,\"repeat\":%d,\"warmup\":%d,\"trials\":%d,"
      "\"verify\":%s,\"average_ms\":%.9f,\"fma_count\":%llu,"
      "\"checksum\":%.9f,\"maximum_absolute_error\":%.12g,"
      "\"gpu_name\":\"%s\",\"compute_capability\":\"%d.%d\"}\n",
      operation.c_str(), parameter, count, repeat, warmup, trials,
      verify ? "true" : "false", average_ms, fma_count, checksum, maximum_error,
      property.name, property.major, property.minor);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dout);
  return verify && maximum_error > 1.0e-5 ? 1 : 0;
}
