#include "qa_span.h"
#include <float.h>

int mlx_qa_select_span(const float *start, const float *end,
                       const uint8_t *context, size_t n, size_t max_tokens,
                       mlx_qa_span *result) {
  size_t i, j;
  mlx_qa_span selected = {0, 0, 0.0, 0};
  if (!start || !end || !context || !result || !n || !max_tokens) return 1;
  /* Validate even masked logits. NaN/Inf must not disappear behind masking. */
  for (i = 0; i < n; ++i)
    if (!(start[i] >= -FLT_MAX && start[i] <= FLT_MAX) ||
        !(end[i] >= -FLT_MAX && end[i] <= FLT_MAX) || context[i] > 1) return 1;
  for (i = 0; i < n; ++i) {
    size_t width = n - i < max_tokens ? n - i : max_tokens;
    if (!context[i]) continue;
    for (j = i; j - i < width; ++j) {
      double score;
      if (!context[j]) break;
      score = (double)start[i] + (double)end[j];
      if (selected.candidates == UINT64_MAX) return 1;
      if (!selected.candidates || score > selected.score) {
        selected.start = i;
        selected.end = j;
        selected.score = score;
      }
      ++selected.candidates;
    }
  }
  if (!selected.candidates) return 2;
  *result = selected;
  return 0;
}
