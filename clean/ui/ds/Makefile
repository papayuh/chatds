# SPDX-License-Identifier: MIT
# DS ROM for the ChatDS UI, built with BlocksDS. Run from clean/ui:
#   source ../../tools/env.sh && make -f ds/Makefile
# Sources are the UI core and the scripted-input shell plus ds/source.
export WONDERFUL_TOOLCHAIN ?= $(HOME)/opt/wonderful
export BLOCKSDS ?= $(WONDERFUL_TOOLCHAIN)/thirdparty/blocksds/core
export BLOCKSDSEXT ?= $(WONDERFUL_TOOLCHAIN)/thirdparty/blocksds/external

NAME := chatds-ui
GAME_TITLE := ChatDS UI
GAME_SUBTITLE := touch UI test ROM
GAME_AUTHOR := ChatDS
SOURCEDIRS := src shell ds/source
INCLUDEDIRS := include ../include src

include $(BLOCKSDS)/sys/default_makefiles/rom_arm9/Makefile
