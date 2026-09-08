OUTPUT_ARCH(riscv)
ENTRY(_start)
PHDRS { low PT_LOAD FLAGS(7); high PT_LOAD FLAGS(6); }
SECTIONS {
  . = 0x80000000;
  .text : { *(.text.init) *(.text .text.*) } :low
  .rodata : { *(.rodata .rodata.*) } :low
  .data : { *(.data .data.*) } :low
  .sdata : { __global_pointer$ = . + 0x800; *(.sdata .sdata.*) } :low
  . = ALIGN(64);
  .tohost : { *(.tohost) } :low
  . = ALIGN(64);
  __bss_start = .;
  .bss : { *(.bss .bss.* .sbss .sbss.*) *(COMMON) } :low
  . = ALIGN(8);
  __bss_end = .;
  . += 0x10000;
  . = ALIGN(16);
  __stack_top = .;
  . = 0x100000000;
  .high : { *(.high .high.*) } :high
}
