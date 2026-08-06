# High-performance CCU V1 assembler / disassembler
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -flto -Wall -Wextra -Werror -std=c11
CFLAGS  += -Iinclude
LDFLAGS ?= -flto

SRCS := src/isa.c src/asm.c src/vasm.c src/cli.c
OBJS := $(SRCS:src/%.c=build/%.o)

.PHONY: all clean test install

all: build/ccu_v1_asm

build:
	mkdir -p build

build/%.o: src/%.c include/ccu_v1_isa.h include/ccu_v1_asm.h include/ccu_v1_vasm.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/ccu_v1_asm: $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

test: build/ccu_v1_asm
	./build/ccu_v1_asm verify ../examples/all_opcodes.s -w build/verify_out
	./build/ccu_v1_asm as ../examples/all_opcodes.s -o build/all.bin
	./build/ccu_v1_asm dis build/all.bin -o build/all.dis.s
	./build/ccu_v1_asm as build/all.dis.s -o build/all.re.bin
	cmp build/all.bin build/all.re.bin
	./build/ccu_v1_asm verify-vasm ../examples/vars_reuse.s -w build/vasm_verify
	./build/ccu_v1_asm vasm ../examples/vars_reuse.s -o build/vars.bin -m build/vars.meta.json --lowered build/vars.lowered.s
	@python3 -c 'import json; m=json.load(open("build/vars.meta.json"));\
a=[v for v in m["variables"] if v["name"]=="phase_a"][0];\
b=[v for v in m["variables"] if v["name"]=="phase_b"][0];\
assert a["id"]==b["id"], (a,b);\
assert a["live_end"] < b["live_start"], (a,b);\
print("reuse check OK: phase_a/phase_b share xn id", a["id"])'
	@echo "ALL C TESTS PASSED"

clean:
	rm -rf build

install: build/ccu_v1_asm
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 build/ccu_v1_asm $(DESTDIR)/usr/local/bin/ccu_v1_asm
