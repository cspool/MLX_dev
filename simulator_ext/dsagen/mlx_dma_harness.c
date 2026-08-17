#include <stdint.h>
#include <stdio.h>

#include "interface.h"
#include "timing.h"

#ifndef MLX_HOST_WAIT_ITERATIONS
#define MLX_HOST_WAIT_ITERATIONS 0
#endif

#define MLX_DMA_BLOCKS 16
#define MLX_DMA_ITERATIONS 4
#define MLX_DMA_BYTES 8
#define MLX_DMA_COLD_BLOCK_STRIDE 4096
#define MLX_DMA_WRITE_BLOCK_STRIDE 256
#define MLX_DMA_ITERATION_STRIDE 64

volatile uint8_t mlx_dma_cold_region[131072] __attribute__((aligned(4096)));
volatile uint8_t mlx_dma_write_region[4096] __attribute__((aligned(4096)));
volatile uint8_t mlx_dma_evict_region[2097152] __attribute__((aligned(4096)));

static uint64_t
selected_write_checksum(void)
{
  uint64_t checksum = 0;
  for (uint64_t block = 0; block < MLX_DMA_BLOCKS; ++block) {
    for (uint64_t iteration = 0; iteration < MLX_DMA_ITERATIONS; ++iteration) {
      uint64_t offset = block * MLX_DMA_WRITE_BLOCK_STRIDE +
                        iteration * MLX_DMA_ITERATION_STRIDE;
      for (uint64_t byte = 0; byte < MLX_DMA_BYTES; ++byte) {
        checksum += mlx_dma_write_region[offset + byte];
      }
    }
  }
  return checksum;
}

int
main(void)
{
  for (uint64_t block = 0; block < MLX_DMA_BLOCKS; ++block) {
    for (uint64_t iteration = 0; iteration < MLX_DMA_ITERATIONS; ++iteration) {
      uint64_t offset = block * MLX_DMA_COLD_BLOCK_STRIDE +
                        iteration * MLX_DMA_ITERATION_STRIDE;
      for (uint64_t byte = 0; byte < MLX_DMA_BYTES; ++byte) {
        mlx_dma_cold_region[offset + byte] = 1;
      }
    }
  }
  uint64_t evict_checksum = 0;
  for (uint64_t offset = 0; offset < sizeof(mlx_dma_evict_region); offset += 64) {
    evict_checksum += mlx_dma_evict_region[offset];
  }
  for (uint64_t offset = 0; offset < sizeof(mlx_dma_write_region); ++offset) {
    mlx_dma_write_region[offset] = 0xa5;
  }
  uint64_t initial_checksum = selected_write_checksum();
  struct Arguments *args = init_data();
  printf("[mlx-dma] initialization finished\n");
  printf("MLX_DMA_GUEST_SYMBOLS {\"cold\":%lu,\"write\":%lu,"
         "\"initial_checksum\":%lu,\"evict_checksum\":%lu}\n",
         (uint64_t)mlx_dma_cold_region, (uint64_t)mlx_dma_write_region,
         initial_checksum, evict_checksum);

  uint64_t start = rdcycle();
  run_reference(args);
  printf("[mlx-dma] cpu pass finished, %lu cycles passed!\n", rdcycle() - start);

  start = rdcycle();
  run_accelerator(args, 1);
  printf("[mlx-dma] warm i-cache finished, %lu cycles passed!\n", rdcycle() - start);

  begin_roi();
  run_accelerator(args, 0);
  volatile uint64_t wait_checksum = 0;
  for (uint64_t iteration = 0; iteration < MLX_HOST_WAIT_ITERATIONS; ++iteration) {
    wait_checksum += (iteration ^ (iteration >> 7)) & 0xff;
  }
  end_roi();

  uint64_t final_checksum = selected_write_checksum();
  printf("MLX_DMA_GUEST_SUMMARY {\"store_checksum\":%lu,"
         "\"wait_checksum\":%lu}\n",
         final_checksum, wait_checksum);
  printf("[mlx-dma] accelerator finished ...\n");
  if (sanity_check(args)) {
    printf("[mlx-dma] sanity check passed successfully!\n");
    return 0;
  }
  printf("[mlx-dma] sanity check did not pass!\n");
  return 1;
}
