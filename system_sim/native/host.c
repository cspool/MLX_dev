#include <stdint.h>

#ifndef MLX_NATIVE_HEADER
#error "MLX_NATIVE_HEADER must name the generated program header"
#endif
#include MLX_NATIVE_HEADER

volatile uint64_t tohost __attribute__((section(".tohost"), aligned(64))) = 0;
volatile uint64_t fromhost __attribute__((section(".tohost"), aligned(64))) = 0;
static uint64_t output[MLX_OUTPUT_BEATS] __attribute__((aligned(64)));

static void print(const char *text) {
  unsigned length=0;
  while (text[length]) ++length;
  volatile uint64_t call[8] __attribute__((aligned(64))) = {64,1,(uintptr_t)text,length,0,0,0,0};
  while (tohost) {}
  asm volatile("fence" ::: "memory");
  tohost=(uintptr_t)call;
  while (!fromhost) {}
  fromhost=0;
  asm volatile("fence" ::: "memory");
}
static void number(uint64_t value) {
  char text[24]; unsigned position=23; text[position]=0;
  do { text[--position]='0'+value%10; value/=10; } while (value);
  print(text+position);
}
static void field(const char *name,uint64_t value) { print(name); number(value); }
static uint64_t cycles(void) {
  uint64_t value; asm volatile("rdcycle %0" : "=r"(value)); return value;
}
static void config(uint64_t address,uint64_t word) {
  asm volatile(".insn r 0x0b, 3, 0, x0, %0, %1" :: "r"(word),"r"(address) : "memory");
}
static uint64_t status(uint64_t index) {
  uint64_t value;
  asm volatile(".insn r 0x0b, 6, 3, %0, %1, x0" : "=r"(value) : "r"(index) : "memory");
  return value;
}

int main(void) {
  if (status(14)!=UINT64_C(0x4d4c5802)) { print("MLX_NATIVE_ELF_ABI_FAIL\n"); return 1; }
  for (unsigned i=0;i<MLX_OUTPUT_BEATS;++i) output[i]=UINT64_C(0xdeadbeefdeadbeef);
  uint64_t config_start=cycles();
  for (unsigned i=0;i<MLX_CONFIG_WORDS;++i) config(mlx_configuration[i].address,mlx_configuration[i].word);
  uint64_t config_end=cycles();
  uint64_t launch_start=cycles(), wait_status;
  asm volatile("fence" ::: "memory");
  asm volatile(".insn r 0x0b, 3, 1, x0, %0, %1" :: "r"((uintptr_t)mlx_input),"r"((uintptr_t)output) : "memory");
  asm volatile(".insn r 0x0b, 4, 2, %0, x0, x0" : "=r"(wait_status) :: "memory");
  asm volatile("fence" ::: "memory");
  uint64_t launch_end=cycles();
  uint64_t result[17];
  for (unsigned i=0;i<17;++i) result[i]=status(i);
  unsigned mismatches=0;
#if MLX_EXPECT_REJECT
  for (unsigned i=0;i<MLX_OUTPUT_BEATS;++i)
    if (output[i]!=UINT64_C(0xdeadbeefdeadbeef)) ++mismatches;
  const unsigned failed=!result[15] || !(wait_status & 16) || result[4] || result[13] || mismatches;
  print(failed ? "MLX_NATIVE_ELF_REJECT_FAIL workload=" : "MLX_NATIVE_ELF_REJECT_PASS workload=");
#else
  for (unsigned i=0;i<MLX_OUTPUT_BEATS;++i) if (output[i]!=mlx_golden[i]) ++mismatches;
  const unsigned failed=mismatches || result[15];
  print(failed ? "MLX_NATIVE_ELF_FAIL workload=" : "MLX_NATIVE_ELF_PASS workload=");
#endif
  print(MLX_WORKLOAD_NAME);
  field(" config=",config_end-config_start);
  field(" launch_wait=",launch_end-launch_start);
  field(" system=",result[1]); field(" dma=",result[3]); field(" kernel=",result[4]);
  field(" instructions=",result[5]); field(" bytes=",result[13]); field(" overlap=",result[16]);
  field(" wait=",wait_status); field(" error=",result[15]); field(" mismatches=",mismatches); print("\n");
  return failed ? 1 : 0;
}
