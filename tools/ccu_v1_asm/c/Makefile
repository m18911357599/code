# High-performance CCU V1 assembler / disassembler
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -flto -Wall -Wextra -Werror -std=c11
CFLAGS  += -Iinclude
LDFLAGS ?= -flto

LIB_SRCS := src/isa.c src/asm.c src/vasm.c src/casm.c src/casm_api.c
LIB_OBJS := $(LIB_SRCS:src/%.c=build/%.o)
CLI_OBJ  := build/cli.o

EXAMPLES := ../examples

.PHONY: all clean test install

all: build/ccu_v1_asm build/casm_host

build:
	mkdir -p build

build/%.o: src/%.c include/ccu_v1_isa.h include/ccu_v1_asm.h include/ccu_v1_vasm.h include/ccu_v1_casm.h \
		include/ccu_v1_casm_api.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/ccu_v1_asm: $(LIB_OBJS) $(CLI_OBJ)
	$(CC) $(CFLAGS) $(LIB_OBJS) $(CLI_OBJ) -o $@ $(LDFLAGS)

# Native C path for loop_main.c:
#   create ctx → begin → call main → instr fill binary → write_file
build/loop_main.o: $(EXAMPLES)/loop_main.c include/ccu_v1_casm_api.h | build
	$(CC) $(CFLAGS) -Dmain=ccu_user_main -c $(EXAMPLES)/loop_main.c -o $@

build/casm_host: src/casm_host.c build/loop_main.o $(LIB_OBJS)
	$(CC) $(CFLAGS) src/casm_host.c build/loop_main.o $(LIB_OBJS) -o $@ $(LDFLAGS)

test: build/ccu_v1_asm build/casm_host
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
	# Native: create ctx → call main → write binary file
	./build/casm_host -o build/loop.bin
	cmp build/loop.bin build/all.bin
	# Text interpreter path still accepts the same source
	./build/ccu_v1_asm verify-casm ../examples/loop_main.c -w build/casm_verify
	cmp build/casm_verify/loop_main.bin build/all.bin
	@echo "ALL C TESTS PASSED"

clean:
	rm -rf build

install: build/ccu_v1_asm
	install -d $(DESTDIR)/usr/local/bin
	install -m 755 build/ccu_v1_asm $(DESTDIR)/usr/local/bin/ccu_v1_asm
