#pragma once
#include <stdint.h>
#define MLX_CLOCKED_ROCC_MAGIC UINT64_C(0x4d4c58434c4b0001)

static inline uint64_t mlx_clocked_status(uint64_t index){
  uint64_t result;
  __asm__ volatile(".insn r 0x0b, 6, 3, %0, %1, x0":"=r"(result):"r"(index):"memory");return result;
}
static inline void mlx_clocked_launch(uint64_t descriptor,uint64_t bytes){
  __asm__ volatile("fence iorw, iorw":::"memory");
  __asm__ volatile(".insn r 0x0b, 3, 1, x0, %0, %1"::"r"(descriptor),"r"(bytes):"memory");
}
static inline uint64_t mlx_clocked_submit(uint64_t descriptor,uint64_t bytes){
  uint64_t status;mlx_clocked_launch(descriptor,bytes);
  __asm__ volatile(".insn r 0x0b, 4, 2, %0, x0, x0":"=r"(status)::"memory");
  __asm__ volatile("fence iorw, iorw":::"memory");return status;
}
extern volatile uint64_t tohost,fromhost;
static inline void mlx_clocked_pass(void){
  static const char message[]="MLX_CLOCKED_CHAIN_PASS\n";
  volatile uint64_t call[8] __attribute__((aligned(64)))={64,1,(uintptr_t)message,sizeof(message)-1,0,0,0,0};
  while(tohost){}
  __asm__ volatile("fence":::"memory");tohost=(uintptr_t)call;
  while(!fromhost){}
  fromhost=0;__asm__ volatile("fence":::"memory");
}
