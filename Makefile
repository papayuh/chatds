# SPDX-License-Identifier: MIT
# Product targets use only the independent engine, plumbing and UI.
.PHONY: all rom host test clean
all: rom
rom:
	$(MAKE) -C clean/product all
host:
	$(MAKE) -C clean/product host
test:
	python3 tools/run_host_tests.py
clean:
	$(MAKE) -C clean/product clean
