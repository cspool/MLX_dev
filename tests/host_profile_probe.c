#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static volatile uint64_t state = 1;

__attribute__((noinline)) void mlx_host_diagnostic_hot_loop(void) {
  for (unsigned i = 0; i < 200000000; ++i)
    state = state * UINT64_C(6364136223846793005) + 1;
}

int main(void) {
  mlx_host_diagnostic_hot_loop();
  printf("%" PRIu64 "\n", state);
  return 0;
}
