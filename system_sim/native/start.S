.section .text.init
.globl _start
_start:
  csrr a0, mhartid
  bnez a0, .Lpark
  .option push
  .option norelax
  la gp, __global_pointer$
  .option pop
  la sp, __stack_top
  la t0, .Ltrap
  csrw mtvec, t0
  la t0, __bss_start
  la t1, __bss_end
.Lclear:
  bgeu t0, t1, .Lmain
  sd zero, 0(t0)
  addi t0, t0, 8
  j .Lclear
.Lmain:
  call main
.Lexit:
  slli a0, a0, 1
  ori a0, a0, 1
  la t0, tohost
  sd a0, 0(t0)
  fence
.Lpark:
  wfi
  j .Lpark
.Ltrap:
  csrr a0, mcause
  addi a0, a0, 64
  j .Lexit
