.option norvc
.option norelax
.section .text.init
.globl _start
_start:
li t0, 0x6000
csrs mstatus, t0
case_1:
csrw fcsr, zero
li x10, 0x8000000000000000
li x11, 0x7fffffffffffffff
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_1_10
fld f10, 0(t0)
la t0, fp_1_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000000
bne x12, t0, fail_1
csrr t2, fflags
li t0, 0
bne t2, t0, fail_1
case_2:
csrw fcsr, zero
li x10, 0x7fffffffffffffff
li x11, 0x8000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_2_10
fld f10, 0(t0)
la t0, fp_2_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000001
bne x12, t0, fail_2
csrr t2, fflags
li t0, 0
bne t2, t0, fail_2
case_3:
csrw fcsr, zero
li x10, 0x1000000000000001
li x11, 0x1000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_3_10
fld f10, 0(t0)
la t0, fp_3_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000001
bne x12, t0, fail_3
csrr t2, fflags
li t0, 0
bne t2, t0, fail_3
case_4:
csrw fcsr, zero
li x10, 0x1000000000000000
li x11, 0x1000000000000001
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_4_10
fld f10, 0(t0)
la t0, fp_4_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000000
bne x12, t0, fail_4
csrr t2, fflags
li t0, 0
bne t2, t0, fail_4
case_5:
csrw fcsr, zero
li x10, 0xffffffffffffffff
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_5_10
fld f10, 0(t0)
la t0, fp_5_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000000
bne x12, t0, fail_5
csrr t2, fflags
li t0, 0
bne t2, t0, fail_5
case_6:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0xffffffffffffffff
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_6_10
fld f10, 0(t0)
la t0, fp_6_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000001
bne x12, t0, fail_6
csrr t2, fflags
li t0, 0
bne t2, t0, fail_6
case_7:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_7_10
fld f10, 0(t0)
la t0, fp_7_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000001
bne x12, t0, fail_7
csrr t2, fflags
li t0, 0
bne t2, t0, fail_7
case_8:
csrw fcsr, zero
li x10, 0x8000000000000000
li x11, 0x8000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_8_10
fld f10, 0(t0)
la t0, fp_8_11
fld f11, 0(t0)
.word 0x00b52633
.word 0x00164613
li t0, 0x0000000000000001
bne x12, t0, fail_8
csrr t2, fflags
li t0, 0
bne t2, t0, fail_8
case_9:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_9_10
fld f10, 0(t0)
la t0, fp_9_11
fld f11, 0(t0)
.word 0xa0a58653
li t0, 0x0000000000000000
bne x12, t0, fail_9
csrr t2, fflags
li t0, 0
bne t2, t0, fail_9
case_10:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_10_10
fld f10, 0(t0)
la t0, fp_10_11
fld f11, 0(t0)
.word 0xa0a58653
li t0, 0x0000000000000001
bne x12, t0, fail_10
csrr t2, fflags
li t0, 0
bne t2, t0, fail_10
case_11:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_11_10
fld f10, 0(t0)
la t0, fp_11_11
fld f11, 0(t0)
.word 0xa0a58653
li t0, 0x0000000000000001
bne x12, t0, fail_11
csrr t2, fflags
li t0, 0
bne t2, t0, fail_11
case_12:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_12_10
fld f10, 0(t0)
la t0, fp_12_11
fld f11, 0(t0)
.word 0xa0a58653
li t0, 0x0000000000000000
bne x12, t0, fail_12
csrr t2, fflags
li t0, 16
bne t2, t0, fail_12
case_13:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_13_10
fld f10, 0(t0)
la t0, fp_13_11
fld f11, 0(t0)
.word 0xa0a58653
li t0, 0x0000000000000000
bne x12, t0, fail_13
csrr t2, fflags
li t0, 16
bne t2, t0, fail_13
case_14:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_14_10
fld f10, 0(t0)
la t0, fp_14_11
fld f11, 0(t0)
.word 0xa0a58653
li t0, 0x0000000000000001
bne x12, t0, fail_14
csrr t2, fflags
li t0, 0
bne t2, t0, fail_14
case_15:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_15_10
fld f10, 0(t0)
la t0, fp_15_11
fld f11, 0(t0)
.word 0x00b57633
li t0, 0x0000000000000000
bne x12, t0, fail_15
csrr t2, fflags
li t0, 0
bne t2, t0, fail_15
case_16:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_16_10
fld f10, 0(t0)
la t0, fp_16_11
fld f11, 0(t0)
.word 0x00a03633
.word 0x00b646b3
li t0, 0x0000000000000000
bne x12, t0, fail_16
li t0, 0x0000000000000000
bne x13, t0, fail_16
csrr t2, fflags
li t0, 0
bne t2, t0, fail_16
case_17:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000001
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_17_10
fld f10, 0(t0)
la t0, fp_17_11
fld f11, 0(t0)
.word 0x00b57633
li t0, 0x0000000000000000
bne x12, t0, fail_17
csrr t2, fflags
li t0, 0
bne t2, t0, fail_17
case_18:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000001
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_18_10
fld f10, 0(t0)
la t0, fp_18_11
fld f11, 0(t0)
.word 0x00a03633
.word 0x00b646b3
li t0, 0x0000000000000000
bne x12, t0, fail_18
li t0, 0x0000000000000001
bne x13, t0, fail_18
csrr t2, fflags
li t0, 0
bne t2, t0, fail_18
case_19:
csrw fcsr, zero
li x10, 0x0000000000000001
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_19_10
fld f10, 0(t0)
la t0, fp_19_11
fld f11, 0(t0)
.word 0x00b57633
li t0, 0x0000000000000000
bne x12, t0, fail_19
csrr t2, fflags
li t0, 0
bne t2, t0, fail_19
case_20:
csrw fcsr, zero
li x10, 0x0000000000000001
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_20_10
fld f10, 0(t0)
la t0, fp_20_11
fld f11, 0(t0)
.word 0x00a03633
.word 0x00b646b3
li t0, 0x0000000000000001
bne x12, t0, fail_20
li t0, 0x0000000000000001
bne x13, t0, fail_20
csrr t2, fflags
li t0, 0
bne t2, t0, fail_20
case_21:
csrw fcsr, zero
li x10, 0x0000000000000001
li x11, 0x0000000000000001
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_21_10
fld f10, 0(t0)
la t0, fp_21_11
fld f11, 0(t0)
.word 0x00b57633
li t0, 0x0000000000000001
bne x12, t0, fail_21
csrr t2, fflags
li t0, 0
bne t2, t0, fail_21
case_22:
csrw fcsr, zero
li x10, 0x0000000000000001
li x11, 0x0000000000000001
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_22_10
fld f10, 0(t0)
la t0, fp_22_11
fld f11, 0(t0)
.word 0x00a03633
.word 0x00b646b3
li t0, 0x0000000000000001
bne x12, t0, fail_22
li t0, 0x0000000000000000
bne x13, t0, fail_22
csrr t2, fflags
li t0, 0
bne t2, t0, fail_22
case_23:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_23_10
fld f10, 0(t0)
la t0, fp_23_11
fld f11, 0(t0)
.word 0x00100613
li t0, 0x0000000000000001
bne x12, t0, fail_23
csrr t2, fflags
li t0, 0
bne t2, t0, fail_23
case_24:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_24_10
fld f10, 0(t0)
la t0, fp_24_11
fld f11, 0(t0)
.word 0x00100613
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
li t0, 0x0000000000000001
bne x12, t0, fail_24
csrr t2, fflags
li t0, 0
bne t2, t0, fail_24
case_25:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_25_10
fld f10, 0(t0)
la t0, fp_25_11
fld f11, 0(t0)
.word 0x00100613
.word 0x00000513
.word 0x00a03533
.word 0x00a67633
li t0, 0x0000000000000000
bne x12, t0, fail_25
csrr t2, fflags
li t0, 0
bne t2, t0, fail_25
case_26:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_26_10
fld f10, 0(t0)
la t0, fp_26_11
fld f11, 0(t0)
.word 0x00100613
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
li t0, 0x0000000000000001
bne x12, t0, fail_26
csrr t2, fflags
li t0, 0
bne t2, t0, fail_26
case_27:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_27_10
fld f10, 0(t0)
la t0, fp_27_11
fld f11, 0(t0)
.word 0x00100613
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x00000513
.word 0x00a03533
.word 0x00a67633
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
li t0, 0x0000000000000000
bne x12, t0, fail_27
csrr t2, fflags
li t0, 0
bne t2, t0, fail_27
case_28:
csrw fcsr, zero
li x10, 0x0000000000000000
li x11, 0x0000000000000000
li x12, 0x0000000000000000
li x20, 0x0000000000000000
li x21, 0x0000000000000000
la t0, fp_28_10
fld f10, 0(t0)
la t0, fp_28_11
fld f11, 0(t0)
.word 0x00100613
.word 0x00100513
.word 0x00a03533
.word 0x00a67633
.word 0x0ff00513
.word 0x00a03533
.word 0x00a67633
li t0, 0x0000000000000001
bne x12, t0, fail_28
csrr t2, fflags
li t0, 0
bne t2, t0, fail_28
li t1, 1
j finish
fail_1:
li t1, 3
j finish
fail_2:
li t1, 5
j finish
fail_3:
li t1, 7
j finish
fail_4:
li t1, 9
j finish
fail_5:
li t1, 11
j finish
fail_6:
li t1, 13
j finish
fail_7:
li t1, 15
j finish
fail_8:
li t1, 17
j finish
fail_9:
li t1, 19
j finish
fail_10:
li t1, 21
j finish
fail_11:
li t1, 23
j finish
fail_12:
li t1, 25
j finish
fail_13:
li t1, 27
j finish
fail_14:
li t1, 29
j finish
fail_15:
li t1, 31
j finish
fail_16:
li t1, 33
j finish
fail_17:
li t1, 35
j finish
fail_18:
li t1, 37
j finish
fail_19:
li t1, 39
j finish
fail_20:
li t1, 41
j finish
fail_21:
li t1, 43
j finish
fail_22:
li t1, 45
j finish
fail_23:
li t1, 47
j finish
fail_24:
li t1, 49
j finish
fail_25:
li t1, 51
j finish
fail_26:
li t1, 53
j finish
fail_27:
li t1, 55
j finish
fail_28:
li t1, 57
j finish
finish:
la t0, tohost
sd t1, 0(t0)
1: j 1b
.section .rodata
.balign 8
fp_1_10:
.dword 0xffffffff00000000
fp_1_11:
.dword 0xffffffff00000000
fp_2_10:
.dword 0xffffffff00000000
fp_2_11:
.dword 0xffffffff00000000
fp_3_10:
.dword 0xffffffff00000000
fp_3_11:
.dword 0xffffffff00000000
fp_4_10:
.dword 0xffffffff00000000
fp_4_11:
.dword 0xffffffff00000000
fp_5_10:
.dword 0xffffffff00000000
fp_5_11:
.dword 0xffffffff00000000
fp_6_10:
.dword 0xffffffff00000000
fp_6_11:
.dword 0xffffffff00000000
fp_7_10:
.dword 0xffffffff00000000
fp_7_11:
.dword 0xffffffff00000000
fp_8_10:
.dword 0xffffffff00000000
fp_8_11:
.dword 0xffffffff00000000
fp_9_10:
.dword 0xffffffffff800000
fp_9_11:
.dword 0xffffffff00000000
fp_10_10:
.dword 0xffffffff80000000
fp_10_11:
.dword 0xffffffff00000000
fp_11_10:
.dword 0xffffffff7f800000
fp_11_11:
.dword 0xffffffff3f800000
fp_12_10:
.dword 0xffffffff7fc00001
fp_12_11:
.dword 0xffffffff3f800000
fp_13_10:
.dword 0xffffffff3f800000
fp_13_11:
.dword 0xffffffff7f800001
fp_14_10:
.dword 0xffffffff40400000
fp_14_11:
.dword 0xffffffff40000000
fp_15_10:
.dword 0xffffffff00000000
fp_15_11:
.dword 0xffffffff00000000
fp_16_10:
.dword 0xffffffff00000000
fp_16_11:
.dword 0xffffffff00000000
fp_17_10:
.dword 0xffffffff00000000
fp_17_11:
.dword 0xffffffff00000000
fp_18_10:
.dword 0xffffffff00000000
fp_18_11:
.dword 0xffffffff00000000
fp_19_10:
.dword 0xffffffff00000000
fp_19_11:
.dword 0xffffffff00000000
fp_20_10:
.dword 0xffffffff00000000
fp_20_11:
.dword 0xffffffff00000000
fp_21_10:
.dword 0xffffffff00000000
fp_21_11:
.dword 0xffffffff00000000
fp_22_10:
.dword 0xffffffff00000000
fp_22_11:
.dword 0xffffffff00000000
fp_23_10:
.dword 0xffffffff00000000
fp_23_11:
.dword 0xffffffff00000000
fp_24_10:
.dword 0xffffffff00000000
fp_24_11:
.dword 0xffffffff00000000
fp_25_10:
.dword 0xffffffff00000000
fp_25_11:
.dword 0xffffffff00000000
fp_26_10:
.dword 0xffffffff00000000
fp_26_11:
.dword 0xffffffff00000000
fp_27_10:
.dword 0xffffffff00000000
fp_27_11:
.dword 0xffffffff00000000
fp_28_10:
.dword 0xffffffff00000000
fp_28_11:
.dword 0xffffffff00000000
.section .tohost,"aw",@progbits
.balign 64
.globl tohost
tohost: .dword 0
.globl fromhost
fromhost: .dword 0
