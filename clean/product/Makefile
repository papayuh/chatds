# SPDX-License-Identifier: MIT
# Product ROM and host runner: explicit clean-only source allowlist.
export WONDERFUL_TOOLCHAIN ?= $(HOME)/opt/wonderful
export BLOCKSDS ?= $(WONDERFUL_TOOLCHAIN)/thirdparty/blocksds/core
PREFIX ?= $(WONDERFUL_TOOLCHAIN)/toolchain/gcc-arm-none-eabi/bin/arm-none-eabi-
BUILD := build
CORE := ../src/format.c ../src/kernels.c ../src/engine.c
PLUMBING := $(addprefix ../plumbing/,calc.c generate.c io.c kb.c retrieve.c)
UI := $(wildcard ../ui/src/*.c) ../ui/adapter/ui_engine_gen.c ../ui/shell/ui_script.c
SOURCES := main.c ../../ds/tokenizer/tok.c $(CORE) $(PLUMBING) $(UI)
HEADERS := $(wildcard ../include/*.h ../src/*.h ../plumbing/*.h ../ui/include/*.h ../ui/src/*.h ../../ds/tokenizer/*.h)
COMMON := -std=c99 -O3 -g -Wall -Wextra -Werror \
 -I../include -I../src -I../plumbing -I../ui/include -I../../ds/tokenizer
ARM := -std=gnu23 -marm -mcpu=arm946e-s+nofp -ffunction-sections -fdata-sections \
 -DARM9 -D__NDS__ -D__BLOCKSDS__ -DCDS_DTCM_POOL \
 -I$(BLOCKSDS)/libs/libnds/include -specs=$(BLOCKSDS)/sys/crts/ds_arm9.specs
.PHONY: all host clean
all: $(BUILD)/chatds.nds
host: $(BUILD)/chatds-host
$(BUILD):
	mkdir -p $@
$(BUILD)/chatds-host: $(SOURCES) $(HEADERS) Makefile | $(BUILD)
	$(CC) $(COMMON) $(SOURCES) -lm -o $@
$(BUILD)/chatds.elf: $(SOURCES) $(HEADERS) Makefile | $(BUILD)
	$(PREFIX)gcc $(COMMON) $(ARM) $(SOURCES) -o $@ \
	 -L$(BLOCKSDS)/libs/libnds/lib -Wl,-Map,$(BUILD)/chatds.map \
	 -Wl,--start-group -lnds9 -lc -lm -Wl,--end-group
$(BUILD)/chatds.nds: $(BUILD)/chatds.elf
	$(BLOCKSDS)/tools/ndstool/ndstool -c $@ \
	 -7 $(BLOCKSDS)/sys/arm7/main_core/arm7_maxmod.elf -9 $< \
	 -b $(BLOCKSDS)/sys/icon.gif 'ChatDS;Clean DSQ8 engine;ChatDS'
clean:
	rm -rf build
