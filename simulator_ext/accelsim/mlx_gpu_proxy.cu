#include <cuda_runtime.h>

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
                   cudaGetErrorString(error));                                 \
      std::exit(2);                                                             \
    }                                                                          \
  } while (0)

__global__ void
vector_add(const float *a, const float *b, float *out, int count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) out[index] = a[index] + b[index];
}

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

__global__ void
fft_stage(const float2 *input, float2 *output, int count, int stride)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    int partner = index ^ stride;
    if (partner >= count) partner = index;
    float2 left = input[index];
    float2 right = input[partner];
    int period = stride * 2;
    float angle = -6.2831853071795864769f * static_cast<float>(index % period) /
                  static_cast<float>(period);
    float sine = sinf(angle);
    float cosine = cosf(angle);
    float real = fmaf(right.x, cosine, -right.y * sine);
    float imag = fmaf(right.x, sine, right.y * cosine);
    output[index] = make_float2(left.x + real, left.y + imag);
  }
}

__global__ void
swa_kernel(const float *query, const float *key, const float *value, float *output,
           int count, int window)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float maximum = -3.402823466e38f;
  for (int offset = 0; offset < window; ++offset) {
    int position = (index + offset) % count;
    maximum = fmaxf(maximum, query[index] * key[position]);
  }
  float sum = 0.0f;
  float weighted = 0.0f;
  for (int offset = 0; offset < window; ++offset) {
    int position = (index + offset) % count;
    float weight = expf(query[index] * key[position] - maximum);
    sum += weight;
    weighted = fmaf(weight, value[position], weighted);
  }
  output[index] = weighted / sum;
}

static double
checksum(const std::vector<float> &values)
{
  double result = 0.0;
  for (float value : values) result += value;
  return result;
}

static void
fill_inputs(std::vector<float> &a, std::vector<float> &b, std::vector<float> &c)
{
  for (std::size_t index = 0; index < a.size(); ++index) {
    a[index] = 0.5f + static_cast<float>(index % 17) * 0.01f;
    b[index] = 0.25f + static_cast<float>(index % 13) * 0.015f;
    c[index] = 0.75f + static_cast<float>(index % 11) * 0.02f;
  }
}

static int
run_vector(int count, int block_threads)
{
  std::vector<float> a(count), b(count), c(count), out(count);
  fill_inputs(a, b, c);
  float *da = nullptr, *db = nullptr, *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&da, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  vector_add<<<(count + block_threads - 1) / block_threads, block_threads>>>(
      da, db, dout, count);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), dout, count * sizeof(float), cudaMemcpyDeviceToHost));
  double reference = checksum(a) + checksum(b);
  double measured = checksum(out);
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"vectoradd\",\"count\":%d,"
      "\"parameter\":1,\"checksum\":%.9f,\"reference\":%.9f,"
      "\"relative_error\":%.12g}\n",
      count, measured, reference, std::fabs(measured - reference) / std::fabs(reference));
  cudaFree(da);
  cudaFree(db);
  cudaFree(dout);
  return 0;
}

static int
run_bsmm(int count, int stages, int block_threads)
{
  std::vector<float> input(count), scratch(count), unused(count), output(count);
  fill_inputs(input, scratch, unused);
  std::vector<float> reference = input;
  std::vector<float> next(count);
  for (int stage = 0; stage < stages; ++stage) {
    int stride = 1 << stage;
    for (int index = 0; index < count; ++index) {
      int partner = index ^ stride;
      if (partner >= count) partner = index;
      float mixed0 = std::fma(reference[index], 0.75f, reference[partner] * 0.25f);
      float mixed1 = std::fma(reference[partner], 0.5f, reference[index] * 0.5f);
      next[index] = std::fma(mixed0, 0.875f, mixed1 * 0.125f);
    }
    reference.swap(next);
  }
  float *first = nullptr, *second = nullptr;
  CUDA_CHECK(cudaMalloc(&first, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&second, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(first, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  for (int stage = 0; stage < stages; ++stage) {
    bsmm_stage<<<(count + block_threads - 1) / block_threads, block_threads>>>(
        first, second, count, 1 << stage);
    CUDA_CHECK(cudaDeviceSynchronize());
    float *temporary = first;
    first = second;
    second = temporary;
  }
  CUDA_CHECK(cudaMemcpy(output.data(), first, count * sizeof(float), cudaMemcpyDeviceToHost));
  double expected = checksum(reference);
  double measured = checksum(output);
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"bsmm\",\"count\":%d,"
      "\"parameter\":%d,\"checksum\":%.9f,\"reference\":%.9f,"
      "\"relative_error\":%.12g}\n",
      count, stages, measured, expected, std::fabs(measured - expected) / std::fabs(expected));
  cudaFree(first);
  cudaFree(second);
  return 0;
}

static int
run_fft(int count, int stages, int block_threads)
{
  std::vector<float2> input(count), output(count), reference(count), next(count);
  for (int index = 0; index < count; ++index) {
    input[index] = make_float2(0.25f + (index % 7) * 0.02f,
                               0.125f + (index % 5) * 0.01f);
  }
  reference = input;
  for (int stage = 0; stage < stages; ++stage) {
    int stride = 1 << stage;
    int period = stride * 2;
    for (int index = 0; index < count; ++index) {
      int partner = index ^ stride;
      if (partner >= count) partner = index;
      float angle = -6.2831853071795864769f * static_cast<float>(index % period) /
                    static_cast<float>(period);
      float sine = std::sin(angle), cosine = std::cos(angle);
      float real = std::fma(reference[partner].x, cosine,
                            -reference[partner].y * sine);
      float imag = std::fma(reference[partner].x, sine,
                            reference[partner].y * cosine);
      next[index] = make_float2(reference[index].x + real, reference[index].y + imag);
    }
    reference.swap(next);
  }
  float2 *first = nullptr, *second = nullptr;
  CUDA_CHECK(cudaMalloc(&first, count * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&second, count * sizeof(float2)));
  CUDA_CHECK(cudaMemcpy(first, input.data(), count * sizeof(float2), cudaMemcpyHostToDevice));
  for (int stage = 0; stage < stages; ++stage) {
    fft_stage<<<(count + block_threads - 1) / block_threads, block_threads>>>(
        first, second, count, 1 << stage);
    CUDA_CHECK(cudaDeviceSynchronize());
    float2 *temporary = first;
    first = second;
    second = temporary;
  }
  CUDA_CHECK(cudaMemcpy(output.data(), first, count * sizeof(float2), cudaMemcpyDeviceToHost));
  double expected = 0.0, measured = 0.0;
  for (int index = 0; index < count; ++index) {
    expected += reference[index].x + reference[index].y;
    measured += output[index].x + output[index].y;
  }
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"fft\",\"count\":%d,"
      "\"parameter\":%d,\"checksum\":%.9f,\"reference\":%.9f,"
      "\"relative_error\":%.12g}\n",
      count, stages, measured, expected,
      std::fabs(measured - expected) / std::fmax(1.0, std::fabs(expected)));
  cudaFree(first);
  cudaFree(second);
  return 0;
}

static int
run_swa(int count, int window, int block_threads)
{
  std::vector<float> query(count), key(count), value(count), output(count), reference(count);
  fill_inputs(query, key, value);
  for (int index = 0; index < count; ++index) {
    float maximum = -3.402823466e38f;
    for (int offset = 0; offset < window; ++offset) {
      int position = (index + offset) % count;
      maximum = std::fmax(maximum, query[index] * key[position]);
    }
    float sum = 0.0f, weighted = 0.0f;
    for (int offset = 0; offset < window; ++offset) {
      int position = (index + offset) % count;
      float weight = std::exp(query[index] * key[position] - maximum);
      sum += weight;
      weighted = std::fma(weight, value[position], weighted);
    }
    reference[index] = weighted / sum;
  }
  float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
  CUDA_CHECK(cudaMalloc(&dq, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dk, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dv, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dout, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dq, query.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dk, key.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dv, value.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  swa_kernel<<<(count + block_threads - 1) / block_threads, block_threads>>>(
      dq, dk, dv, dout, count, window);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), dout, count * sizeof(float), cudaMemcpyDeviceToHost));
  double expected = checksum(reference), measured = checksum(output);
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"swa\",\"count\":%d,"
      "\"parameter\":%d,\"checksum\":%.9f,\"reference\":%.9f,"
      "\"relative_error\":%.12g}\n",
      count, window, measured, expected, std::fabs(measured - expected) / std::fabs(expected));
  cudaFree(dq);
  cudaFree(dk);
  cudaFree(dv);
  cudaFree(dout);
  return 0;
}

int
main(int argc, char **argv)
{
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s vectoradd|bsmm|fft|swa COUNT PARAMETER\n", argv[0]);
    return 2;
  }
  std::string operation = argv[1];
  int count = std::atoi(argv[2]);
  int parameter = std::atoi(argv[3]);
  constexpr int block_threads = 128;
  if (operation == "vectoradd") return run_vector(count, block_threads);
  if (operation == "bsmm") return run_bsmm(count, parameter, block_threads);
  if (operation == "fft") return run_fft(count, parameter, block_threads);
  if (operation == "swa") return run_swa(count, parameter, block_threads);
  std::fprintf(stderr, "unknown operation: %s\n", operation.c_str());
  return 2;
}
