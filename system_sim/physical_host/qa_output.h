#ifndef MLX_HOST_QA_OUTPUT_H
#define MLX_HOST_QA_OUTPUT_H
#include "control_runtime.h"
#include "../../simulator_ext/control_model/qa_span.h"
int mlx_host_qa_capture(const volatile mlx_host_tensor views[4],uint64_t n,
                        float *start,float *end,uint8_t *mask,int64_t *offsets,mlx_qa_span *span);
void mlx_host_qa_emit(uint64_t forward,uint64_t n,const float *start,const float *end,
                     const int64_t *offsets,const mlx_qa_span *span);
void mlx_host_qa_timing(uint64_t begin,uint64_t graph_end,uint64_t post_end);
#endif
