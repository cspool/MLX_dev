CHIPYARD_ROOT ?= /root/chipyard
RISCV ?= $(CHIPYARD_ROOT)/esp-tools-install
GCC := $(RISCV)/bin/riscv64-unknown-elf-gcc
OBJDUMP := $(RISCV)/bin/riscv64-unknown-elf-objdump
PROJECT_ROOT := $(abspath ../..)
BUILD_DIR := $(PROJECT_ROOT)/system_sim/build/software
GENERATED_DIR := $(PROJECT_ROOT)/system_sim/generated
CFLAGS := -std=gnu99 -O2 -g -fno-common -fno-builtin-printf -Wall -Wextra \
          -I$(CHIPYARD_ROOT)/tests -I$(PROJECT_ROOT)/system_sim/software \
          -I$(GENERATED_DIR)
LDFLAGS := -static -specs=htif_nano.specs
WORKLOADS := bsmm fft_cmp swa transformer_block
ELFS := $(addprefix $(BUILD_DIR)/mlx-,$(addsuffix .riscv,$(WORKLOADS)))

.PHONY: all clean dumps
all: $(ELFS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/mlx-%.riscv: mlx_system_test.c mlx_runtime.h $(GENERATED_DIR)/%.h | $(BUILD_DIR)
	$(GCC) $(CFLAGS) -DMLX_WORKLOAD_HEADER='"$*.h"' $< $(LDFLAGS) -o $@

dumps: $(ELFS)
	for elf in $(ELFS); do $(OBJDUMP) -D $$elf > $$elf.dump; done

clean:
	$(RM) $(ELFS) $(addsuffix .dump,$(ELFS))
