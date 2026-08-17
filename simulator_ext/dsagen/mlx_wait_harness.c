#include <stdint.h>
#include <stdio.h>

#include "interface.h"
#include "timing.h"

#ifndef MLX_HOST_WAIT_ITERATIONS
#define MLX_HOST_WAIT_ITERATIONS 500000
#endif

int
main(void)
{
  struct Arguments *args = init_data();
  printf("[mlx-wait] initialization finished\n");
  uint64_t start = rdcycle();
  run_reference(args);
  printf("[mlx-wait] cpu pass finished, %lu cycles passed!\n", rdcycle() - start);

  start = rdcycle();
  run_accelerator(args, 1);
  printf("[mlx-wait] warm i-cache finished, %lu cycles passed!\n", rdcycle() - start);

  begin_roi();
  run_accelerator(args, 0);
  volatile uint64_t wait_checksum = 0;
  for (uint64_t iteration = 0; iteration < MLX_HOST_WAIT_ITERATIONS; ++iteration) {
    wait_checksum += (iteration ^ (iteration >> 7)) & 0xff;
  }
  end_roi();
  sb_stats();
  printf("[mlx-wait] host wait checksum: %lu\n", wait_checksum);
  printf("[mlx-wait] accelerator finished ...\n");
  if (sanity_check(args)) {
    printf("[mlx-wait] sanity check passed successfully!\n");
    return 0;
  }
  printf("[mlx-wait] sanity check did not pass!\n");
  return 1;
}
