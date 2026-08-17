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

__global__ void
fft_pair_stage(const float2 *input, float2 *output, int pairs, int stage)
{
  int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  float2 left = input[2 * pair];
  float2 right = input[2 * pair + 1];
  int period = 1 << ((stage % 12) + 1);
  float angle = -6.2831853071795864769f * static_cast<float>(pair % period) /
                static_cast<float>(period);
  float sine = sinf(angle);
  float cosine = cosf(angle);
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

__global__ void
qk_scores(const float *query, const float *key, float *scores, int count, int inner)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float total = 0.0f;
  int base = index * inner;
  for (int offset = 0; offset < inner; ++offset) {
    total = fmaf(query[base + offset], key[offset], total);
  }
  scores[index] = total;
}

__global__ void
softmax_stats(const float *scores, float *output, int rows, int width)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  int base = row * width;
  float maximum = -3.402823466e38f;
  for (int column = 0; column < width; ++column) {
    maximum = fmaxf(maximum, scores[base + column]);
  }
  float sum = 0.0f;
  for (int column = 0; column < width; ++column) {
    sum += expf(scores[base + column] - maximum);
  }
  output[row] = maximum + sum;
}

__global__ void
sv_outputs(const float *weights, const float *values, const float *normalizer,
           float *output, int count, int width)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  int base = index * width;
  float total = 0.0f;
  for (int offset = 0; offset < width; ++offset) {
    total = fmaf(weights[offset], values[base + offset], total);
  }
  output[index] = total / normalizer[0];
}

static double
float_checksum(const std::vector<float> &values)
{
  double result = 0.0;
  for (float value : values) result += value;
  return result;
}

static double
complex_checksum(const std::vector<float2> &values)
{
  double result = 0.0;
  for (float2 value : values) result += value.x + value.y;
  return result;
}

static void
cpu_fft_stage(const std::vector<float2> &input, std::vector<float2> &output,
              int pairs, int stage)
{
  int period = 1 << ((stage % 12) + 1);
  for (int pair = 0; pair < pairs; ++pair) {
    float2 left = input[2 * pair];
    float2 right = input[2 * pair + 1];
    float angle = -6.2831853071795864769f * static_cast<float>(pair % period) /
                  static_cast<float>(period);
    float sine = std::sin(angle);
    float cosine = std::cos(angle);
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
  double expected = complex_checksum(reference);
  double measured = complex_checksum(output);
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

static int
run_qk(int count, int inner, int threads)
{
  std::vector<float> query(static_cast<std::size_t>(count) * inner);
  std::vector<float> key(inner), reference(count), output(count);
  for (int offset = 0; offset < inner; ++offset) key[offset] = 0.25f + (offset % 7) * 0.01f;
  for (int index = 0; index < count; ++index) {
    float total = 0.0f;
    for (int offset = 0; offset < inner; ++offset) {
      float value = 0.5f + ((index + offset) % 11) * 0.01f;
      query[static_cast<std::size_t>(index) * inner + offset] = value;
      total = std::fma(value, key[offset], total);
    }
    reference[index] = total;
  }
  float *dquery = nullptr, *dkey = nullptr, *doutput = nullptr;
  CUDA_CHECK(cudaMalloc(&dquery, query.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dkey, key.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&doutput, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dquery, query.data(), query.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dkey, key.data(), key.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  qk_scores<<<(count + threads - 1) / threads, threads>>>(dquery, dkey, doutput,
                                                          count, inner);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), doutput, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double expected = float_checksum(reference), measured = float_checksum(output);
  double error = std::fabs(measured - expected) / std::fabs(expected);
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"qk\",\"count\":%d,"
      "\"parameter\":%d,\"parameter2\":0,\"checksum\":%.9f,"
      "\"reference\":%.9f,\"relative_error\":%.12g}\n",
      count, inner, measured, expected, error);
  cudaFree(dquery);
  cudaFree(dkey);
  cudaFree(doutput);
  return 0;
}

static int
run_softmax(int rows, int width, int threads)
{
  std::vector<float> scores(static_cast<std::size_t>(rows) * width), reference(rows),
      output(rows);
  for (int row = 0; row < rows; ++row) {
    float maximum = -3.402823466e38f;
    for (int column = 0; column < width; ++column) {
      float value = 0.125f + ((row + column) % 13) * 0.01f;
      scores[static_cast<std::size_t>(row) * width + column] = value;
      maximum = std::fmax(maximum, value);
    }
    float sum = 0.0f;
    for (int column = 0; column < width; ++column) {
      sum += std::exp(scores[static_cast<std::size_t>(row) * width + column] - maximum);
    }
    reference[row] = maximum + sum;
  }
  float *dscores = nullptr, *doutput = nullptr;
  CUDA_CHECK(cudaMalloc(&dscores, scores.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&doutput, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dscores, scores.data(), scores.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  softmax_stats<<<(rows + threads - 1) / threads, threads>>>(dscores, doutput,
                                                             rows, width);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), doutput, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double expected = float_checksum(reference), measured = float_checksum(output);
  double error = std::fabs(measured - expected) / std::fabs(expected);
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"softmax\",\"count\":%d,"
      "\"parameter\":%d,\"parameter2\":0,\"checksum\":%.9f,"
      "\"reference\":%.9f,\"relative_error\":%.12g}\n",
      rows, width, measured, expected, error);
  cudaFree(dscores);
  cudaFree(doutput);
  return 0;
}

static int
run_sv(int count, int width, int threads)
{
  std::vector<float> weights(width), values(static_cast<std::size_t>(count) * width),
      reference(count), output(count);
  float normalizer = 1.125f;
  for (int offset = 0; offset < width; ++offset) weights[offset] = 1.0f / width;
  for (int index = 0; index < count; ++index) {
    float total = 0.0f;
    for (int offset = 0; offset < width; ++offset) {
      float value = 0.75f + ((index + offset) % 17) * 0.01f;
      values[static_cast<std::size_t>(index) * width + offset] = value;
      total = std::fma(weights[offset], value, total);
    }
    reference[index] = total / normalizer;
  }
  float *dweights = nullptr, *dvalues = nullptr, *dnormalizer = nullptr,
        *doutput = nullptr;
  CUDA_CHECK(cudaMalloc(&dweights, weights.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dvalues, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dnormalizer, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&doutput, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dweights, weights.data(), weights.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dvalues, values.data(), values.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dnormalizer, &normalizer, sizeof(float),
                        cudaMemcpyHostToDevice));
  sv_outputs<<<(count + threads - 1) / threads, threads>>>(
      dweights, dvalues, dnormalizer, doutput, count, width);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), doutput, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double expected = float_checksum(reference), measured = float_checksum(output);
  double error = std::fabs(measured - expected) / std::fabs(expected);
  std::printf(
      "MLX_GPU_PROXY_SUMMARY {\"operator\":\"sv\",\"count\":%d,"
      "\"parameter\":%d,\"parameter2\":0,\"checksum\":%.9f,"
      "\"reference\":%.9f,\"relative_error\":%.12g}\n",
      count, width, measured, expected, error);
  cudaFree(dweights);
  cudaFree(dvalues);
  cudaFree(dnormalizer);
  cudaFree(doutput);
  return 0;
}

int
main(int argc, char **argv)
{
  if (argc != 5) {
    std::fprintf(stderr,
                 "usage: %s fftcmp|qk|softmax|sv COUNT PARAMETER PARAMETER2\n",
                 argv[0]);
    return 2;
  }
  std::string operation = argv[1];
  int count = std::atoi(argv[2]);
  int parameter = std::atoi(argv[3]);
  int parameter2 = std::atoi(argv[4]);
  constexpr int threads = 128;
  if (count <= 0 || parameter <= 0 || parameter2 < 0) return 2;
  if (operation == "fftcmp") return run_fftcmp(count, parameter, parameter2, threads);
  if (operation == "qk") return run_qk(count, parameter, threads);
  if (operation == "softmax") return run_softmax(count, parameter, threads);
  if (operation == "sv") return run_sv(count, parameter, threads);
  std::fprintf(stderr, "unknown operation: %s\n", operation.c_str());
  return 2;
}
