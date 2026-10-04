# SPDX-License-Identifier: MIT
TARGET ?= host
SANITIZE ?= 0
DTCM_POOL ?= 0
export WONDERFUL_TOOLCHAIN ?= $(HOME)/opt/wonderful
export BLOCKSDS ?= $(WONDERFUL_TOOLCHAIN)/thirdparty/blocksds/core
CFLAGS ?= -O3 -g
WARN = -std=c99 -Wall -Wextra -Werror
CPPFLAGS += -Iinclude -Isrc
BUILD = build/$(TARGET)$(if $(filter 1,$(SANITIZE)),-san)$(if $(filter 1,$(DTCM_POOL)),-dtcm)
SOURCES = src/format.c src/kernels.c src/engine.c
OBJECTS = $(patsubst src/%.c,$(BUILD)/%.o,$(SOURCES))
LIBRARY = $(BUILD)/libchatds_engine.a

ifeq ($(TARGET),nds)
PREFIX ?= $(WONDERFUL_TOOLCHAIN)/toolchain/gcc-arm-none-eabi/bin/arm-none-eabi-
CC = $(PREFIX)gcc
AR = $(PREFIX)ar
CFLAGS += -marm -mcpu=arm946e-s+nofp -ffunction-sections -fdata-sections \
 -specs=$(BLOCKSDS)/sys/crts/ds_arm9.specs
CPPFLAGS += -DARM9 -D__NDS__ -D__BLOCKSDS__ -I$(BLOCKSDS)/libs/libnds/include
else ifneq ($(TARGET),host)
$(error TARGET must be host or nds)
endif
ifeq ($(DTCM_POOL),1)
ifneq ($(TARGET),nds)
$(error DTCM_POOL=1 is nds-only)
endif
CPPFLAGS += -DCDS_DTCM_POOL
endif
ifeq ($(SANITIZE),1)
ifneq ($(TARGET),host)
$(error SANITIZE=1 is host-only)
endif
CFLAGS += -fsanitize=address,undefined -fno-omit-frame-pointer
endif

.PHONY: all test sanitize clean
all: $(LIBRARY)
$(BUILD)/%.o: src/%.c src/internal.h include/chatds_engine.h
	@mkdir -p $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -c $< -o $@
$(LIBRARY): $(OBJECTS)
	$(AR) rcs $@ $^
$(BUILD)/test-%: tests/test_%.c $(LIBRARY) src/internal.h include/chatds_engine.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) $< $(LIBRARY) -lm -o $@
ifeq ($(TARGET),host)
test: $(BUILD)/test-engine $(BUILD)/test-kernels
	$(BUILD)/test-engine $(BUILD)/test-model.bin
	$(BUILD)/test-kernels
else
test:
	@echo 'Run make TARGET=host test; emulator validation belongs to the parity runner.' >&2; exit 2
endif
sanitize:
	$(MAKE) TARGET=host SANITIZE=1 test
clean:
	rm -rf build
