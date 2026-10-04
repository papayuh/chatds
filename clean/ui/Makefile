# SPDX-License-Identifier: MIT
# Host build of the ChatDS UI: unit tests and a script simulator.
#   make test        build and run the unit tests (SAN=1 adds ASan/UBSan)
#   make sim         build build/ui_sim
#   make rom         build the DS ROM (needs BlocksDS)
#   make rom-test    play tests/scenarios/walk.txt in headless melonDS and compare
#                    every screen with the host simulator
CC ?= cc
CFLAGS += -std=c99 -pedantic -Wall -Wextra -Werror -O2 -g -Iinclude -I../include
ifeq ($(SAN),1)
CFLAGS += -fsanitize=address,undefined -fno-sanitize-recover=all
endif

CORE := $(wildcard src/*.c) $(wildcard shell/*.c)

PLUMBING := $(addprefix ../plumbing/,kb.c retrieve.c calc.c generate.c)

build/test_ui: $(CORE) $(PLUMBING) adapter/ui_engine_gen.c tests/stub_engine.c tests/test_ui.c $(wildcard src/*.h tests/*.h ../plumbing/*.h)
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -Itests -I../plumbing $(CORE) $(PLUMBING) adapter/ui_engine_gen.c tests/stub_engine.c tests/test_ui.c -o $@

build/ui_sim: $(CORE) host/ui_sim.c $(wildcard src/*.h)
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc $(CORE) host/ui_sim.c -o $@

test: build/test_ui
	./build/test_ui

sim: build/ui_sim

rom:
	. ../../tools/env.sh && $(MAKE) -f ds/Makefile

rom-test:
	tests/ds-ui-melon.sh

clean:
	rm -rf build chatds-ui.nds

.PHONY: test sim rom rom-test clean
