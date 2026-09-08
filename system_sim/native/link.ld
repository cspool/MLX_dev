OUTPUT_ARCH(riscv)
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text .text.*) }
  . = ALIGN(64);
  .rodata : { *(.rodata .rodata.*) }
  . = ALIGN(64);
  .data : { *(.data .data.*) }
  .sdata : { __global_pointer$ = . + 0x800; *(.sdata .sdata.*) }
  . = ALIGN(64);
  .tohost : { *(.tohost) }
  . = ALIGN(64);
  __bss_start = .;
  .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) }
  . = ALIGN(8);
  __bss_end = .;
  . += 0x10000;
  . = ALIGN(16);
  __stack_top = .;
}
