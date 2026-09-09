#ifndef MLX_QA_SPAN_H
#define MLX_QA_SPAN_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
  size_t start, end;
  double score;
  uint64_t candidates;
} mlx_qa_span;
/* Version 1: finite F32 logits, F64 sums, first lexicographic maximum.
 * Returns 0 on success, 1 for invalid input, 2 when no context is eligible.
 * The caller owns n-element buffers. No allocation, framework or libm call. */
int mlx_qa_select_span(const float *start, const float *end,
                       const uint8_t *context, size_t n, size_t max_tokens,
                       mlx_qa_span *result);
#ifdef __cplusplus
}
#endif
#endif
