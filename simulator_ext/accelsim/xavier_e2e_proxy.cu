#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

constexpr int kHidden = 8;
constexpr int kFfn = 16;
constexpr float kEpsilon = 1.0e-5f;

__global__ void rmsnorm_kernel(const float *input, float *output, int tokens,
                               int hidden)
{
  int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= tokens) return;
  float sum = 0.0f;
  for (int dimension = 0; dimension < hidden; ++dimension) {
    float value = input[token * hidden + dimension];
    sum = fmaf(value, value, sum);
  }
  float inverse = rsqrtf(sum / static_cast<float>(hidden) + kEpsilon);
  for (int dimension = 0; dimension < hidden; ++dimension) {
    float gamma = 1.0f + 0.01f * static_cast<float>(dimension);
    output[token * hidden + dimension] =
        input[token * hidden + dimension] * inverse * gamma;
  }
}

__global__ void dense_kernel(const float *input, const float *weight,
                             const float *bias, float *output, int rows,
                             int input_dimension, int output_dimension)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= rows * output_dimension) return;
  int row = index / output_dimension;
  int output_index = index % output_dimension;
  float result = bias[output_index];
  for (int input_index = 0; input_index < input_dimension; ++input_index) {
    result = fmaf(input[row * input_dimension + input_index],
                  weight[output_index * input_dimension + input_index], result);
  }
  output[index] = result;
}

__global__ void rope_qk_kernel(const float *query, const float *key,
                               float *rotated_query, float *rotated_key,
                               int tokens, int hidden)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = hidden / 2;
  if (index >= tokens * pairs) return;
  int token = index / pairs;
  int pair = index % pairs;
  int first = token * hidden + 2 * pair;
  int second = first + 1;
  float angle = 0.075f * static_cast<float>((token + 1) * (pair + 1));
  float sine = sinf(angle), cosine = cosf(angle);
  rotated_query[first] = fmaf(query[first], cosine, -query[second] * sine);
  rotated_query[second] = fmaf(query[first], sine, query[second] * cosine);
  rotated_key[first] = fmaf(key[first], cosine, -key[second] * sine);
  rotated_key[second] = fmaf(key[first], sine, key[second] * cosine);
}

__global__ void causal_attention_kernel(const float *query, const float *key,
                                        const float *value, float *output,
                                        int tokens, int hidden)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= tokens * hidden) return;
  int token = index / hidden;
  int dimension = index % hidden;
  float scale = rsqrtf(static_cast<float>(hidden));
  float maximum = -3.402823466e38f;
  for (int source = 0; source <= token; ++source) {
    float score = 0.0f;
    for (int inner = 0; inner < hidden; ++inner) {
      score = fmaf(query[token * hidden + inner], key[source * hidden + inner],
                   score);
    }
    maximum = fmaxf(maximum, score * scale);
  }
  float denominator = 0.0f, weighted = 0.0f;
  for (int source = 0; source <= token; ++source) {
    float score = 0.0f;
    for (int inner = 0; inner < hidden; ++inner) {
      score = fmaf(query[token * hidden + inner], key[source * hidden + inner],
                   score);
    }
    float probability = expf(score * scale - maximum);
    denominator += probability;
    weighted = fmaf(probability, value[source * hidden + dimension], weighted);
  }
  output[index] = weighted / denominator;
}

__global__ void residual_kernel(const float *left, const float *right,
                                float *output, int count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = left[index] + right[index];
}

__global__ void silu_gate_kernel(const float *up, const float *gate,
                                 float *output, int count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float gated = gate[index] / (1.0f + expf(-gate[index]));
  output[index] = up[index] * gated;
}

static float weight_value(int family, int output_index, int input_index)
{
  int code = (family * 17 + output_index * 7 + input_index * 3) % 23;
  return (static_cast<float>(code) - 11.0f) * 0.0125f;
}

static float bias_value(int family, int index)
{
  return static_cast<float>((family * 5 + index * 3) % 11 - 5) * 0.005f;
}

static void fill_parameters(std::vector<std::vector<float>> &weights,
                            std::vector<std::vector<float>> &biases)
{
  const int inputs[7] = {kHidden, kHidden, kHidden, kHidden,
                         kHidden, kHidden, kFfn};
  const int outputs[7] = {kHidden, kHidden, kHidden, kHidden,
                          kFfn, kFfn, kHidden};
  for (int family = 0; family < 7; ++family) {
    weights[family].resize(inputs[family] * outputs[family]);
    biases[family].resize(outputs[family]);
    for (int output = 0; output < outputs[family]; ++output) {
      biases[family][output] = bias_value(family, output);
      for (int input = 0; input < inputs[family]; ++input) {
        weights[family][output * inputs[family] + input] =
            weight_value(family, output, input);
      }
    }
  }
}

static void host_rmsnorm(const std::vector<float> &input,
                         std::vector<float> &output, int tokens)
{
  for (int token = 0; token < tokens; ++token) {
    float sum = 0.0f;
    for (int dimension = 0; dimension < kHidden; ++dimension) {
      float value = input[token * kHidden + dimension];
      sum = std::fma(value, value, sum);
    }
    float inverse = 1.0f / std::sqrt(sum / kHidden + kEpsilon);
    for (int dimension = 0; dimension < kHidden; ++dimension) {
      output[token * kHidden + dimension] =
          input[token * kHidden + dimension] * inverse *
          (1.0f + 0.01f * dimension);
    }
  }
}

static void host_dense(const std::vector<float> &input,
                       const std::vector<float> &weight,
                       const std::vector<float> &bias,
                       std::vector<float> &output, int rows,
                       int input_dimension, int output_dimension)
{
  for (int row = 0; row < rows; ++row) {
    for (int output_index = 0; output_index < output_dimension; ++output_index) {
      float result = bias[output_index];
      for (int input_index = 0; input_index < input_dimension; ++input_index) {
        result = std::fma(input[row * input_dimension + input_index],
                          weight[output_index * input_dimension + input_index],
                          result);
      }
      output[row * output_dimension + output_index] = result;
    }
  }
}

static void host_rope(const std::vector<float> &query,
                      const std::vector<float> &key,
                      std::vector<float> &rotated_query,
                      std::vector<float> &rotated_key, int tokens)
{
  for (int token = 0; token < tokens; ++token) {
    for (int pair = 0; pair < kHidden / 2; ++pair) {
      int first = token * kHidden + 2 * pair;
      int second = first + 1;
      float angle = 0.075f * static_cast<float>((token + 1) * (pair + 1));
      float sine = std::sin(angle), cosine = std::cos(angle);
      rotated_query[first] = std::fma(query[first], cosine, -query[second] * sine);
      rotated_query[second] = std::fma(query[first], sine, query[second] * cosine);
      rotated_key[first] = std::fma(key[first], cosine, -key[second] * sine);
      rotated_key[second] = std::fma(key[first], sine, key[second] * cosine);
    }
  }
}

static void host_attention(const std::vector<float> &query,
                           const std::vector<float> &key,
                           const std::vector<float> &value,
                           std::vector<float> &output, int tokens)
{
  float scale = 1.0f / std::sqrt(static_cast<float>(kHidden));
  std::vector<float> scores(tokens);
  for (int token = 0; token < tokens; ++token) {
    float maximum = -3.402823466e38f;
    for (int source = 0; source <= token; ++source) {
      float score = 0.0f;
      for (int inner = 0; inner < kHidden; ++inner) {
        score = std::fma(query[token * kHidden + inner],
                         key[source * kHidden + inner], score);
      }
      scores[source] = score * scale;
      maximum = std::fmax(maximum, scores[source]);
    }
    float denominator = 0.0f;
    for (int source = 0; source <= token; ++source) {
      scores[source] = std::exp(scores[source] - maximum);
      denominator += scores[source];
    }
    for (int dimension = 0; dimension < kHidden; ++dimension) {
      float weighted = 0.0f;
      for (int source = 0; source <= token; ++source) {
        weighted = std::fma(scores[source], value[source * kHidden + dimension],
                            weighted);
      }
      output[token * kHidden + dimension] = weighted / denominator;
    }
  }
}

static std::vector<float> host_reference(
    int tokens, int layers, const std::vector<float> &input,
    const std::vector<std::vector<float>> &weights,
    const std::vector<std::vector<float>> &biases)
{
  int hidden_count = tokens * kHidden, ffn_count = tokens * kFfn;
  std::vector<float> current = input, norm(hidden_count), query(hidden_count),
                     key(hidden_count), value(hidden_count), rq(hidden_count),
                     rk(hidden_count), attention(hidden_count), projected(hidden_count),
                     residual(hidden_count), norm2(hidden_count), up(ffn_count),
                     gate(ffn_count), activated(ffn_count), down(hidden_count),
                     next(hidden_count);
  for (int layer = 0; layer < layers; ++layer) {
    host_rmsnorm(current, norm, tokens);
    host_dense(norm, weights[0], biases[0], query, tokens, kHidden, kHidden);
    host_dense(norm, weights[1], biases[1], key, tokens, kHidden, kHidden);
    host_dense(norm, weights[2], biases[2], value, tokens, kHidden, kHidden);
    host_rope(query, key, rq, rk, tokens);
    host_attention(rq, rk, value, attention, tokens);
    host_dense(attention, weights[3], biases[3], projected, tokens, kHidden,
               kHidden);
    for (int index = 0; index < hidden_count; ++index)
      residual[index] = current[index] + projected[index];
    host_rmsnorm(residual, norm2, tokens);
    host_dense(norm2, weights[4], biases[4], up, tokens, kHidden, kFfn);
    host_dense(norm2, weights[5], biases[5], gate, tokens, kHidden, kFfn);
    for (int index = 0; index < ffn_count; ++index) {
      activated[index] = up[index] * gate[index] /
                         (1.0f + std::exp(-gate[index]));
    }
    host_dense(activated, weights[6], biases[6], down, tokens, kFfn, kHidden);
    for (int index = 0; index < hidden_count; ++index)
      next[index] = residual[index] + down[index];
    current.swap(next);
  }
  return current;
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s TOKENS LAYERS\n", argv[0]);
    return 2;
  }
  int tokens = std::atoi(argv[1]), layers = std::atoi(argv[2]);
  if (tokens <= 0 || layers <= 0) return 2;
  constexpr int block_threads = 128;
  int hidden_count = tokens * kHidden, ffn_count = tokens * kFfn;
  std::vector<float> input(hidden_count), output(hidden_count);
  for (int index = 0; index < hidden_count; ++index)
    input[index] = 0.2f + 0.01f * static_cast<float>((index * 7) % 31);
  std::vector<std::vector<float>> weights(7), biases(7);
  fill_parameters(weights, biases);
  std::vector<float> reference = host_reference(tokens, layers, input, weights, biases);

  float *current, *next, *norm, *query, *key, *value, *rq, *rk, *attention,
      *projected, *residual, *norm2, *up, *gate, *activated, *down;
  auto allocate = [](float **pointer, int count) {
    CUDA_CHECK(cudaMalloc(pointer, count * sizeof(float)));
  };
  allocate(&current, hidden_count); allocate(&next, hidden_count);
  allocate(&norm, hidden_count); allocate(&query, hidden_count);
  allocate(&key, hidden_count); allocate(&value, hidden_count);
  allocate(&rq, hidden_count); allocate(&rk, hidden_count);
  allocate(&attention, hidden_count); allocate(&projected, hidden_count);
  allocate(&residual, hidden_count); allocate(&norm2, hidden_count);
  allocate(&up, ffn_count); allocate(&gate, ffn_count);
  allocate(&activated, ffn_count); allocate(&down, hidden_count);
  std::vector<float *> device_weights(7, nullptr), device_biases(7, nullptr);
  for (int family = 0; family < 7; ++family) {
    allocate(&device_weights[family], weights[family].size());
    allocate(&device_biases[family], biases[family].size());
    CUDA_CHECK(cudaMemcpy(device_weights[family], weights[family].data(),
                          weights[family].size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_biases[family], biases[family].data(),
                          biases[family].size() * sizeof(float), cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(current, input.data(), hidden_count * sizeof(float),
                        cudaMemcpyHostToDevice));
  int hidden_blocks = (hidden_count + block_threads - 1) / block_threads;
  int ffn_blocks = (ffn_count + block_threads - 1) / block_threads;
  int token_blocks = (tokens + block_threads - 1) / block_threads;
  int rope_blocks = (tokens * kHidden / 2 + block_threads - 1) / block_threads;
  int launches = 0;
  for (int layer = 0; layer < layers; ++layer) {
    rmsnorm_kernel<<<token_blocks, block_threads>>>(current, norm, tokens, kHidden);
    dense_kernel<<<hidden_blocks, block_threads>>>(norm, device_weights[0], device_biases[0], query, tokens, kHidden, kHidden);
    dense_kernel<<<hidden_blocks, block_threads>>>(norm, device_weights[1], device_biases[1], key, tokens, kHidden, kHidden);
    dense_kernel<<<hidden_blocks, block_threads>>>(norm, device_weights[2], device_biases[2], value, tokens, kHidden, kHidden);
    launches += 4;
    rope_qk_kernel<<<rope_blocks, block_threads>>>(query, key, rq, rk, tokens, kHidden); ++launches;
    causal_attention_kernel<<<hidden_blocks, block_threads>>>(rq, rk, value, attention, tokens, kHidden); ++launches;
    dense_kernel<<<hidden_blocks, block_threads>>>(attention, device_weights[3], device_biases[3], projected, tokens, kHidden, kHidden); ++launches;
    residual_kernel<<<hidden_blocks, block_threads>>>(current, projected, residual, hidden_count); ++launches;
    rmsnorm_kernel<<<token_blocks, block_threads>>>(residual, norm2, tokens, kHidden); ++launches;
    dense_kernel<<<ffn_blocks, block_threads>>>(norm2, device_weights[4], device_biases[4], up, tokens, kHidden, kFfn);
    dense_kernel<<<ffn_blocks, block_threads>>>(norm2, device_weights[5], device_biases[5], gate, tokens, kHidden, kFfn);
    launches += 2;
    silu_gate_kernel<<<ffn_blocks, block_threads>>>(up, gate, activated, ffn_count); ++launches;
    dense_kernel<<<hidden_blocks, block_threads>>>(activated, device_weights[6], device_biases[6], down, tokens, kFfn, kHidden); ++launches;
    residual_kernel<<<hidden_blocks, block_threads>>>(residual, down, next, hidden_count); ++launches;
    std::swap(current, next);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), current, hidden_count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double measured = 0.0, expected = 0.0, maximum_error = 0.0;
  for (int index = 0; index < hidden_count; ++index) {
    measured += output[index]; expected += reference[index];
    maximum_error = std::max(maximum_error,
                             static_cast<double>(std::fabs(output[index] - reference[index])));
  }
  double relative_error = std::fabs(measured - expected) /
                          std::fmax(1.0, std::fabs(expected));
  std::printf(
      "XAVIER_E2E_SUMMARY {\"operator\":\"dense_transformer\","
      "\"tokens\":%d,\"hidden\":%d,\"ffn\":%d,\"layers\":%d,"
      "\"kernel_launches\":%d,\"checksum\":%.9f,\"reference\":%.9f,"
      "\"maximum_absolute_error\":%.12g,\"relative_error\":%.12g}\n",
      tokens, kHidden, kFfn, layers, launches, measured, expected, maximum_error,
      relative_error);
  cudaFree(current); cudaFree(next); cudaFree(norm); cudaFree(query);
  cudaFree(key); cudaFree(value); cudaFree(rq); cudaFree(rk);
  cudaFree(attention); cudaFree(projected); cudaFree(residual); cudaFree(norm2);
  cudaFree(up); cudaFree(gate); cudaFree(activated); cudaFree(down);
  for (int family = 0; family < 7; ++family) {
    cudaFree(device_weights[family]); cudaFree(device_biases[family]);
  }
  return maximum_error <= 1.0e-5 ? 0 : 1;
}
