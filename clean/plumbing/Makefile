# SPDX-License-Identifier: MIT
# Host build + tests for the ChatDS plumbing; `make arm` cross-compiles it for the ARM9
# (BlocksDS toolchain, see tools/env.sh) and writes per-function stack usage (build/arm/*.su).
CC ?= cc
SRC = kb.c retrieve.c calc.c generate.c io.c
CFLAGS = -std=c99 -O1 -g -Wall -Wextra -Wpedantic -Werror -I../include
SAN = -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
WONDERFUL_TOOLCHAIN ?= $(HOME)/opt/wonderful
ARMCC = $(WONDERFUL_TOOLCHAIN)/toolchain/gcc-arm-none-eabi/bin/arm-none-eabi-gcc
ARMFLAGS = -std=c99 -O2 -Wall -Wextra -Werror -marm -mcpu=arm946e-s -DARM9 -fstack-usage \
	-I../include -I$(WONDERFUL_TOOLCHAIN)/thirdparty/blocksds/core/libs/libnds/include

all: build/cds-prompt build/test-plumbing

# string-level probe: no engine needed
build/cds-prompt: kb.c retrieve.c calc.c cds_prompt.c plumbing.h | build
	$(CC) $(CFLAGS) $(SAN) kb.c retrieve.c calc.c cds_prompt.c -o $@

# test_plumbing.c supplies a stub engine, so the plumbing never links a real one here.
build/test-plumbing: $(SRC) test_plumbing.c plumbing.h | build
	$(CC) $(CFLAGS) -std=gnu99 $(SAN) $(SRC) test_plumbing.c -o $@

build:
	mkdir -p build

arm: $(SRC) plumbing.h
	mkdir -p build/arm
	cd build/arm && for f in $(SRC); do $(ARMCC) $(ARMFLAGS:-I../%=-I../../../%) -c ../../$$f || exit 1; done

test: all
	cd ../.. && python3 -m pytest -q clean/plumbing

clean:
	rm -rf build

.PHONY: all arm test clean
